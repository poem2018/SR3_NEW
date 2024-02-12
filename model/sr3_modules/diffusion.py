import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm

import random
import numpy as np
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from torchvision.transforms import functional as trans_fn
from PIL import Image



def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        #opt, ##TODO
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        # super(GaussianDiffusion, self).__init__()  #TODO

        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional  
        

        ##TODO
        # self.opt = opt   

        # self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.data = {}
        self.SR = None

        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):   
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):   #backward inference
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in   #LR_resize   lr
            shape = x.shape
            img = torch.randn(shape, device=device)  #HR_noise
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)   ##??
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        self.SR = x_recon
        self.data['HR'] = x_in['HR']
        self.data['LR']  = x_in['SR']
        self.data['SR']  = self.SR
        ###TODO
        # import pdb; pdb.set_trace()
        # x_rec_downsample = self.downsampling(x_recon, x_in)
        loss_rec = self.loss_func(noise, x_recon)  ###need modify
        # loss_con = self.loss_func(x_in['SR'], x_rec_downsample)    ###is it this way?
        # loss  = 1.0*loss_rec+0*loss_con
        return loss_rec


    # ##TODO
    # def downsampling(self, img, data):
    #     #########TODO: add the esr_gan downsampling#############
    #     """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
    #     """
    #     # training data synthesis
    #     self.gt = img   #.to(self.device)

    #     self.kernel1 = data['kernel1']  #.to(self.device)
    #     self.kernel2 = data['kernel2'] #.to(self.device)
    #     self.sinc_kernel = data['sinc_kernel'] #.to(self.device)

    #     ori_h, ori_w = self.gt.size()[2:4]

    #     # ----------------------- The first degradation process ----------------------- #
    #     # blur
    #     out = filter2D(self.gt, self.kernel1)
    #     # random resize
    #     updown_type = random.choices(['up', 'down', 'keep'], self.opt["datasets"]["downsampling"]['resize_prob'])[0]
    #     if updown_type == 'up':
    #         scale = np.random.uniform(1, self.opt["datasets"]["downsampling"]['resize_range'][1])
    #     elif updown_type == 'down':
    #         scale = np.random.uniform(self.opt["datasets"]["downsampling"]['resize_range'][0], 1)
    #     else:
    #         scale = 1
    #     mode = random.choice(['area', 'bilinear', 'bicubic'])
    #     out = F.interpolate(out, scale_factor=scale, mode=mode)
    #     # add noise
    #     gray_noise_prob = self.opt["datasets"]["downsampling"]['gray_noise_prob']
    #     if np.random.uniform() < self.opt["datasets"]["downsampling"]['gaussian_noise_prob']:
    #         out = random_add_gaussian_noise_pt(
    #             out, sigma_range=self.opt["datasets"]["downsampling"]['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    #     else:
    #         out = random_add_poisson_noise_pt(
    #             out,
    #             scale_range=self.opt["datasets"]["downsampling"]['poisson_scale_range'],
    #             gray_prob=gray_noise_prob,
    #             clip=True,
    #             rounds=False)
    #     # JPEG compression
    #     jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["datasets"]["downsampling"]['jpeg_range'])
    #     out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    #     out = self.jpeger(out, quality=jpeg_p)

    #     # ----------------------- The second degradation process ----------------------- #
    #     # blur
    #     if np.random.uniform() < self.opt["datasets"]["downsampling"]['second_blur_prob']:
    #         out = filter2D(out, self.kernel2)
    #     # random resize
    #     updown_type = random.choices(['up', 'down', 'keep'], self.opt["datasets"]["downsampling"]['resize_prob2'])[0]
    #     if updown_type == 'up':
    #         scale = np.random.uniform(1, self.opt["datasets"]["downsampling"]['resize_range2'][1])
    #     elif updown_type == 'down':
    #         scale = np.random.uniform(self.opt["datasets"]["downsampling"]['resize_range2'][0], 1)
    #     else:
    #         scale = 1
    #     mode = random.choice(['area', 'bilinear', 'bicubic'])
    #     # import pdb; pdb.set_trace()
    #     out = F.interpolate(
    #         out, size=(int(ori_h / self.opt["datasets"]["downsampling"]['scale'] * scale), int(ori_w / self.opt["datasets"]["downsampling"]['scale'] * scale)), mode=mode)
    #     # add noise
    #     gray_noise_prob = self.opt["datasets"]["downsampling"]['gray_noise_prob2']
    #     if np.random.uniform() < self.opt["datasets"]["downsampling"]['gaussian_noise_prob2']:
    #         out = random_add_gaussian_noise_pt(
    #             out, sigma_range=self.opt["datasets"]["downsampling"]['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
    #     else:
    #         out = random_add_poisson_noise_pt(
    #             out,
    #             scale_range=self.opt["datasets"]["downsampling"]['poisson_scale_range2'],
    #             gray_prob=gray_noise_prob,
    #             clip=True,
    #             rounds=False)

    #     # JPEG compression + the final sinc filter
    #     # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
    #     # as one operation.
    #     # We consider two orders:
    #     #   1. [resize back + sinc filter] + JPEG compression
    #     #   2. JPEG compression + [resize back + sinc filter]
    #     # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
    #     if np.random.uniform() < 0.5:
    #         # resize back + the final sinc filter
    #         mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         out = F.interpolate(out, size=(ori_h // self.opt["datasets"]["downsampling"]['scale'], ori_w // self.opt["datasets"]["downsampling"]['scale']), mode=mode)
    #         out = filter2D(out, self.sinc_kernel)
    #         # JPEG compression
    #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["datasets"]["downsampling"]['jpeg_range2'])
    #         out = torch.clamp(out, 0, 1)
    #         out = self.jpeger(out, quality=jpeg_p)
    #     else:
    #         # JPEG compression
    #         jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["datasets"]["downsampling"]['jpeg_range2'])
    #         out = torch.clamp(out, 0, 1)
    #         out = self.jpeger(out, quality=jpeg_p)
    #         # resize back + the final sinc filter
    #         mode = random.choice(['area', 'bilinear', 'bicubic'])
    #         out = F.interpolate(out, size=(ori_h // self.opt["datasets"]["downsampling"]['scale'], ori_w // self.opt["datasets"]["downsampling"]['scale']), mode=mode)
    #         out = filter2D(out, self.sinc_kernel)

    #     # clamp and round
    #     self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    #     # random crop
    #     gt_size = self.opt["datasets"]["downsampling"]['gt_size']
    #     self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt["datasets"]["downsampling"]['scale'])

    #     # training pair pool
    #     # self._dequeue_and_enqueue()  #TODO this need to be added
    #     self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

    #     srsize = data["HR"].size()[-1]
    #     ret_img = trans_fn.resize(self.lq, srsize, Image.BICUBIC, antialias=True)
    #     ret_img = trans_fn.center_crop(ret_img, srsize)

    #     return ret_img

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


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
import torchvision.transforms.functional as TF
import core.metrics as Metrics


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models

        ######TODO modify this when train/use add_tool
        if self.opt['is_control'] ==True:
            #self.opt['path']['resume_state'] = None   #######donot comment when run tool_add_control, comment when finetune train
            self.netG = self.set_device(networks.define_G(opt))
        else:
            self.netG= self.set_device(networks.define_G_without_controlnet(opt))
        
        self.schedule_phase = None
        self.SR = None

        #TODO
        # self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts


        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        print("!!!!!!!!!!!!!!!!!")
        self.load_network()
        self.print_network()

    def feed_data(self, data):

        # # #########TODO: add the esr_gan downsampling#############
        # # """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        # # """
        # # training data synthesis
        # self.gt = data['HR'].to(self.device)

        # vis1 = Metrics.tensor2img(self.gt[0].cpu().detach())
        # Metrics.save_img(vis1,'./dataset/test/test_esrgan0.png')


        # min_max = (-1,1)
        # self.gt = (self.gt - min_max[0])/(min_max[1] - min_max[0])

        # vis1 = Metrics.tensor2img_01(self.gt [0].cpu().detach())
        # Metrics.save_img(vis1,'./dataset/test/test_esrgan1.png')

        # # USM sharpen the GT images
        # # if data['gt_usm'] is True:
        # #     self.gt = self.usm_sharpener(self.gt)

        # self.kernel1 = data['kernel1'].to(self.device)
        # self.kernel2 = data['kernel2'].to(self.device)
        # self.sinc_kernel = data['sinc_kernel'].to(self.device)

        # ori_h, ori_w = self.gt.size()[2:4]
        
        # # import pdb;pdb.set_trace()

        # # ----------------------- The first degradation process ----------------------- #
        # # blur
        # out = filter2D(self.gt, self.kernel1)
        # # random resize
        # updown_type = random.choices(['up', 'down', 'keep'], self.opt["datasets"]["downsampling"]['resize_prob'])[0]
        # if updown_type == 'up':
        #     scale = np.random.uniform(1, self.opt["datasets"]["downsampling"]['resize_range'][1])
        # elif updown_type == 'down':
        #     scale = np.random.uniform(self.opt["datasets"]["downsampling"]['resize_range'][0], 1)
        # else:
        #     scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(out, scale_factor=scale, mode=mode)
        # # add noise
        # gray_noise_prob = self.opt["datasets"]["downsampling"]['gray_noise_prob']
        # if np.random.uniform() < self.opt["datasets"]["downsampling"]['gaussian_noise_prob']:
        #     out = random_add_gaussian_noise_pt(
        #         out, sigma_range=self.opt["datasets"]["downsampling"]['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        # else:
        #     out = random_add_poisson_noise_pt(
        #         out,
        #         scale_range=self.opt["datasets"]["downsampling"]['poisson_scale_range'],
        #         gray_prob=gray_noise_prob,
        #         clip=True,
        #         rounds=False)
        # # JPEG compression
        # jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["datasets"]["downsampling"]['jpeg_range'])
        # out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        # out = self.jpeger(out, quality=jpeg_p)
        

        # vis1 = Metrics.tensor2img_01(out[0].cpu().detach())
        # Metrics.save_img(vis1,'./dataset/test/test_esrgan2.png')

        # # tensor = out[0].cpu()
        # # image = TF.to_pil_image(tensor)
        # # image.save('./dataset/test/test_esrgan5.png', format='PNG')
        # # import pdb;pdb.set_trace()


        # # # ----------------------- The second degradation process ----------------------- #
        # # # blur
        # # if np.random.uniform() < self.opt["datasets"]["downsampling"]['second_blur_prob']:
        # #     out = filter2D(out, self.kernel2)
        # # # random resize
        # # updown_type = random.choices(['up', 'down', 'keep'], self.opt["datasets"]["downsampling"]['resize_prob2'])[0]
        # # if updown_type == 'up':
        # #     scale = np.random.uniform(1, self.opt["datasets"]["downsampling"]['resize_range2'][1])
        # # elif updown_type == 'down':
        # #     scale = np.random.uniform(self.opt["datasets"]["downsampling"]['resize_range2'][0], 1)
        # # else:
        # #     scale = 1
        # # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # # # import pdb; pdb.set_trace()
        # # out = F.interpolate(
        # #     out, size=(int(ori_h / self.opt["datasets"]["downsampling"]['scale'] * scale), int(ori_w / self.opt["datasets"]["downsampling"]['scale'] * scale)), mode=mode)
        # # # add noise
        # # gray_noise_prob = self.opt["datasets"]["downsampling"]['gray_noise_prob2']
        # # if np.random.uniform() < self.opt["datasets"]["downsampling"]['gaussian_noise_prob2']:
        # #     out = random_add_gaussian_noise_pt(
        # #         out, sigma_range=self.opt["datasets"]["downsampling"]['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        # # else:
        # #     out = random_add_poisson_noise_pt(
        # #         out,
        # #         scale_range=self.opt["datasets"]["downsampling"]['poisson_scale_range2'],
        # #         gray_prob=gray_noise_prob,
        # #         clip=True,
        # #         rounds=False)



        # # JPEG compression + the final sinc filter
        # # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # # as one operation.
        # # We consider two orders:
        # #   1. [resize back + sinc filter] + JPEG compression
        # #   2. JPEG compression + [resize back + sinc filter]
        # # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        # if np.random.uniform() < 0.5:
        #     # resize back + the final sinc filter
        #     mode = random.choice(['area', 'bilinear', 'bicubic'])
        #     out = F.interpolate(out, size=(ori_h // self.opt["datasets"]["downsampling"]['scale'], ori_w // self.opt["datasets"]["downsampling"]['scale']), mode=mode)
        #     out = filter2D(out, self.sinc_kernel)
        #     # JPEG compression
        #     jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["datasets"]["downsampling"]['jpeg_range2'])
        #     out = torch.clamp(out, 0, 1)
        #     out = self.jpeger(out, quality=jpeg_p)
        # else:
        #     # JPEG compression
        #     jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt["datasets"]["downsampling"]['jpeg_range2'])
        #     out = torch.clamp(out, 0, 1)
        #     out = self.jpeger(out, quality=jpeg_p)
        #     # resize back + the final sinc filter
        #     mode = random.choice(['area', 'bilinear', 'bicubic'])
        #     out = F.interpolate(out, size=(ori_h // self.opt["datasets"]["downsampling"]['scale'], ori_w // self.opt["datasets"]["downsampling"]['scale']), mode=mode)
        #     out = filter2D(out, self.sinc_kernel)
        

        # vis1 = Metrics.tensor2img_01(out[0].cpu().detach())
        # Metrics.save_img(vis1,'./dataset/test/test_esrgan3.png')

        # # clamp and round
        # self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # # random crop
        # # gt_size = self.opt["datasets"]["downsampling"]['gt_size']
        # # self.gt, self.lq = paired_random_crop(self.gt, self.lq, gt_size, self.opt["datasets"]["downsampling"]['scale'])

        # # training pair pool
        # # self._dequeue_and_enqueue()  #TODO this need to be added
        # self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

        # vis1 = Metrics.tensor2img_01(self.lq[0].cpu().detach())
        # Metrics.save_img(vis1,'./dataset/test/test_esrgan4.png')

        # ##replace to -1,1
        # min_max = (-1,1)
        # self.lq = (self.lq - 0.5)*(min_max[1] - min_max[0])

        # srsize = data["HR"].size()[-1]
        # img = trans_fn.resize(self.lq, srsize)#, Image.BICUBIC, antialias=True)
        # data["LR"] = self.lq
        # data["SR"] = img

        # # import torchvision.transforms.functional as TF
        # # tensor = img[0].cpu()
        # # image = TF.to_pil_image(tensor)
        # # image.save('./dataset/test/test_esrgan.png', format='PNG')

        # vis1 = Metrics.tensor2img_01(img[0].cpu().detach())
        # Metrics.save_img(vis1,'./dataset/test/test_esrgan5.png')

        # import pdb; pdb.set_trace()
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        self.SR = self.netG.SR  ###todo ddpm sr
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        print("testing")
        print(self.data['SR'].size())
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):   ##????
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
         
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

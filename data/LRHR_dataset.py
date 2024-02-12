from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util

import numpy as np
import math
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
import torchvision.transforms.functional as TF
import core.metrics as Metrics

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        # # # manual config TODO: add into config file #########TODO: add the esr_gan downsampling#############
        self.opt = {
        "final_sinc_prob":0.8,
        "blur_kernel_size": 21,
        "kernel_list":['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'], #['iso'],# 
        "kernel_prob": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], #[0.03], #
        "sinc_prob": 0.1,
        "blur_sigma": [0.2, 3], #[0.2,0.21], #
        "betag_range": [0.5, 4],#[0.5,0.51],#
        "betap_range": [1, 2], #[1,1.1],#

        "blur_kernel_size2": 21,
        "kernel_list2": ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'], #['iso'],#['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        "kernel_prob2": [0.45, 0.25, 0.12, 0.03, 0.12, 0.03], #[0.03],#[0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        "sinc_prob2": 0.1,
        "blur_sigma2": [0.2, 1.5], #[0.2,0.21], #
        "betag_range2": [0.5, 4], #[0.5,0.51],#[0.5, 4],
        "betap_range2": [1, 2]} #[1,1.1]}#[1, 2]}


        # self.kernel_range = [7]
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21

        self.blur_kernel_size = self.opt['blur_kernel_size']
        self.kernel_list = self.opt['kernel_list']
        self.kernel_prob = self.opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = self.opt['blur_sigma']
        self.betag_range = self.opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = self.opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = self.opt['sinc_prob']  # the probability for sinc filters

        self.blur_kernel_size2 = self.opt['blur_kernel_size2']
        self.kernel_list2 = self.opt['kernel_list2']
        self.kernel_prob2 = self.opt['kernel_prob2']
        self.blur_sigma2 = self.opt['blur_sigma2']
        self.betag_range2 = self.opt['betag_range2']
        self.betap_range2 = self.opt['betap_range2']
        self.sinc_prob2 = self.opt['sinc_prob2']
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None
        res_data = {}
        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            # print(self.hr_path, self.sr_path, index)
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            crop_area1 = (300,300,556,556)
            img_HR = img_HR.crop(crop_area1)
            img_SR = img_SR.crop(crop_area1)
            #print("!!!!!!!!!!!!!!",img_SR.size)
            # print(np.shape(img_HR))
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            res_data =  {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            res_data = {'HR': img_HR, 'SR': img_SR, 'Index': index}
        # import pdb;pdb.set_trace()
        # import torchvision.transforms.v2.functional as TF


        # img_HR = np.transpose(img_HR, (1, 2, 0))  # HWC, RGB
        # print("!!!",img_HR)
        # tensor1 = res_data['HR'] 
        # tensor2 = img_SR  #res_data['SR'] 
        # vis1 = Metrics.tensor2img(img_HR)
        # Metrics.save_img(vis1, './dataset/test/test1.png')

        # tensor2.save('./dataset/test/test2.png', format='PNG')
        


        # # #########TODO: add the esr_gan downsampling#############
        # # crop or pad to 400
        # # TODO: 400 is hard-coded. You may change it accordingly
        # h, w = img_HR.shape[0:2]

        # # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        # kernel_size = random.choice(self.kernel_range)
        # if np.random.uniform() < self.opt['sinc_prob']:
        #     # this sinc filter setting is for kernels ranging from [7, 21]
        #     if kernel_size < 13:
        #         omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     else:
        #         omega_c = np.random.uniform(np.pi / 5, np.pi)
        #     kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        # else:
        #     kernel = random_mixed_kernels(
        #         self.kernel_list,
        #         self.kernel_prob,
        #         kernel_size,
        #         self.blur_sigma,
        #         self.blur_sigma, [-math.pi, math.pi],
        #         self.betag_range,
        #         self.betap_range,
        #         noise_range=None)
        # # pad kernel
        # pad_size = (21 - kernel_size) // 2
        # kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        # kernel_size = random.choice(self.kernel_range)
        # if np.random.uniform() < self.opt['sinc_prob2']:
        #     if kernel_size < 13:
        #         omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     else:
        #         omega_c = np.random.uniform(np.pi / 5, np.pi)
        #     kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        # else:
        #     kernel2 = random_mixed_kernels(
        #         self.kernel_list2,
        #         self.kernel_prob2,
        #         kernel_size,
        #         self.blur_sigma2,
        #         self.blur_sigma2, [-math.pi, math.pi],
        #         self.betag_range2,
        #         self.betap_range2,
        #         noise_range=None)

        # # pad kernel
        # pad_size = (21 - kernel_size) // 2
        # kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # # ------------------------------------- the final sinc kernel ------------------------------------- #
        # if np.random.uniform() < self.opt['final_sinc_prob']:
        #     kernel_size = random.choice(self.kernel_range)
        #     omega_c = np.random.uniform(np.pi / 3, np.pi)
        #     sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
        #     sinc_kernel = torch.FloatTensor(sinc_kernel)
        # else:
        #     sinc_kernel = self.pulse_tensor

        # # BGR to RGB, HWC to CHW, numpy to tensor
        # # img_HR = img2tensor([img_HR], bgr2rgb=True, float32=True)[0]
        # kernel = torch.FloatTensor(kernel)
        # kernel2 = torch.FloatTensor(kernel2)
        # # res_data['HR'] = img_HR
        # res_data['kernel1'] = kernel
        # res_data['kernel2'] = kernel2
        # res_data['sinc_kernel'] = sinc_kernel
        
        return res_data


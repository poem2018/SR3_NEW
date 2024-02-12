import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as trans_fn
import os
from pathlib import Path
import lmdb
import numpy as np
import time


# 256 - 512 - 1024
# 64 - 256 - 1024
# 16 - 128 - 1024

# def esr_downsample():
#python data/prepare_data.py  --path ../../dataset/CelebAMask-HQ/CelebA-HQ-img/  --out ./dataset/Celeba --size 256,1024 -l
#python data/prepare_data_vlr.py  --path ../../dataset/CelebAMask-HQ/CelebA-HQ-img/  --out ./dataset/Celeba_2x --size 512,1024,256 -l


##use esrgan to get vlr_64, rename to lr_64, lr_256 ->hr_256
#python scripts/generate_multiscale_DF2K.py --input ../SR3/dataset/Celeba_256_1024/lr_256 --output ../SR3/dataset/Celeba_256_1024/vlr_64

##split the train test data
##to get vlr -> sr_vlr  resize_worker

def resize_vlr(img, sizes, lmdb_save):
    # img = img.convert('RGB')
    out = resize_multiple( img, sizes=sizes, resample=Image.BICUBIC, lmdb_save=lmdb_save)
    lr_img, hr_img, sr_img = out
    return lr_img, hr_img, sr_img



def save_test_resize_lr(path, sizes, lmdb_save):
    print("start")
    image_files = [f for f in os.listdir(path)]
    loaded_images = []
    print("path array finish")

    # Load each image
    for image_file in tqdm(image_files, desc="Loading images"):
        image_path = os.path.join(path, image_file)  
        img = Image.open(image_path)
        img_idx = image_path.split("/")[-1]
        img = img.convert('RGB')
        if sizes[0]==img.size[0]:
            out = resize_and_convert(img, size=sizes[1], resample=Image.BICUBIC)
            out.save('../dataset/finetune_8x/lr_{}/{}'.format(sizes[1], img_idx))
            out_r = resize_and_convert(out, size=sizes[0], resample=Image.BICUBIC)
            out_r.save('../dataset/finetune_8x/sr_{}_{}/{}'.format(sizes[1], sizes[0], img_idx))
            # out.save('./dataset/test/sample_64.png')
        else:
            print("wrong size")

    return True
    



def resize_and_convert(img, size, resample):
    if(img.size[0] != size):
        # print("!!!",img.size[0] , size)
        img = trans_fn.resize(img, size, resample)
        img = trans_fn.center_crop(img, size)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='png')
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128, 8), resample=Image.BICUBIC, lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    vlr_img = resize_and_convert(lr_img, sizes[2], resample)
    sr_img = resize_and_convert(vlr_img, sizes[0], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)
        vlr_img = image_convert_bytes(vlr_img)

    return [lr_img, hr_img, sr_img, vlr_img]

def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)
    return img_file.name.split('.')[0], out

class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value

def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img, vlr_img = imgs
        if not wctx.lmdb_save:
            vlr_img.save(
                '{}/lr_{}/{}.png'.format(wctx.out_path, wctx.sizes[2], i.zfill(5)))
            lr_img.save(
                '{}/hr_{}/{}.png'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)))
            sr_img.save(
                '{}/sr_{}_{}/{}.png'.format(wctx.out_path, wctx.sizes[2], wctx.sizes[0], i.zfill(5)))
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[2], i.zfill(5)).encode('utf-8'), vlr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[2], wctx.sizes[0], i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))

def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True

def prepare(img_path, out_path, n_worker, sizes=(16, 128, 8), resample=Image.BICUBIC, lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[2]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[2], sizes[0]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        idx_end = 0
        for file in tqdm(files):
            i = file.name
            path1 = '{}/hr_{}/{}'.format(out_path, sizes[0], i)
            path2 = '{}/lr_{}/{}'.format(out_path, sizes[2], i)
            path3 = '{}/sr_{}_{}/{}'.format(out_path, sizes[2], sizes[0], i)
            if not os.path.exists(path1) or not os.path.exists(path2) or not os.path.exists(path3):
                print(path1,path2,path3)
                i, imgs = resize_fn(file)
                lr_img, hr_img, sr_img, vlr_img = imgs
                if not lmdb_save:
                    lr_img.save(
                        '{}/hr_{}/{}.png'.format(out_path, sizes[0], i.zfill(5)))
                    vlr_img.save(
                        '{}/lr_{}/{}.png'.format(out_path, sizes[2], i.zfill(5)))
                    sr_img.save(
                        '{}/sr_{}_{}/{}.png'.format(out_path, sizes[2], sizes[0], i.zfill(5)))
                else:
                    with env.begin(write=True) as txn:
                        txn.put('hr_{}_{}'.format(
                            sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                        txn.put('lr_{}_{}'.format(
                            sizes[2], i.zfill(5)).encode('utf-8'), vlr_img)
                        txn.put('sr_{}_{}_{}'.format(
                            sizes[2], sizes[0], i.zfill(5)).encode('utf-8'), sr_img)
                total += 1
                print(total)
                if lmdb_save:
                    with env.begin(write=True) as txn:
                        txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))



if __name__ == '__main__':
    print("starttttt")
    # save_test_resize_lr('./dataset/test_Celeba_2x/lr_512/', sizes = (512, 1024), lmdb_save=True) 
    save_test_resize_lr('../dataset/finetune_8x/lr_128/', sizes = (128,16), lmdb_save=True) 
    


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path', '-p', type=str,
#                         default='{}/Dataset/celebahq_256'.format(Path.home()))
#     parser.add_argument('--out', '-o', type=str,
#                         default='./dataset/celebahq')

#     parser.add_argument('--size', type=str, default='64,512')
#     parser.add_argument('--n_worker', type=int, default=3)
#     parser.add_argument('--resample', type=str, default='bicubic')
#     # default save in png format
#     parser.add_argument('--lmdb', '-l', action='store_true')   ##store_true

#     args = parser.parse_args()

#     resample_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}
#     resample = resample_map[args.resample]
#     sizes = [int(s.strip()) for s in args.size.split(',')]

#     # import pdb;pdb.set_trace()
#     args.out = '{}_{}_{}_{}'.format(args.out, sizes[0], sizes[1], sizes[2])
#     prepare(args.path, args.out, args.n_worker,
#             sizes=sizes, resample=resample, lmdb_save=args.lmdb)

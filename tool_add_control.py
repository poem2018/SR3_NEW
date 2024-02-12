import sys
import os

# assert len(sys.argv) == 3, 'Args are wrong.'

# input_path = sys.argv[1]
# output_path = sys.argv[2]

# assert os.path.exists(input_path), 'Input model does not exist.'
# assert not os.path.exists(output_path), 'Output filename already exists.'
# assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
# from share import *



import argparse
import model as Model
import logging
import core.logger as Logger
import core.metrics as Metrics

def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


# model = create_model(config_path='./models/cldm_v15.yaml')  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                            help='JSON file for configuration')
    parser.add_argument('-i','--input_path', type=str, help='Path to the input model file.')
    parser.add_argument('-o','--output_path', type=str, help='Path for the output file.')

    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # import pdb;pdb.set_trace()
    diffusion_without = Model.create_model(opt)
    pretrained_weights = diffusion_without.netG.state_dict()

    opt["is_control"]=True

    optimizer = Model.create_model(opt).optG
    diffusion = Model.create_model(opt)
    scratch_dict = diffusion.netG.state_dict()

    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']


    target_dict = {}
    for k in scratch_dict.keys():
       
        is_control, name = get_node_name(k, 'denoise_fn.unet')
        print(is_control, name)
        if is_control:
            copy_k = 'denoise_fn' + name
        else:
            copy_k = k
        # if is_control:
        #     import pdb;pdb.set_trace()
        if copy_k in pretrained_weights:
            target_dict[k] = pretrained_weights[copy_k].clone()
        else:
            target_dict[k] = scratch_dict[k].clone()
            print(f'These weights are newly added: {k}')


    # print(target_dict.keys())
    diffusion.netG.load_state_dict(target_dict, strict=True)



    gen_path = args.output_path + '_gen.pth'
    opt_path = args.output_path + '_opt.pth'

    #gen
    state_dict = diffusion.netG.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, gen_path)
    # opt
    file_name = args.input_path.split("/")[-1]
    epoch = file_name.split("_E")[-1]
    iter_step = file_name.split("_E")[0][1:]

    opt_state = {'epoch': int(epoch), 'iter': int(iter_step),'scheduler': None, 'optimizer': None}
    opt_state['optimizer'] = optimizer.state_dict()
    torch.save(opt_state, opt_path)

    # logger.info(
    #     'Saved model in [{:s}] ...'.format(gen_path))
    # torch.save(diffusion.state_dict(), args.output_path)
    print('Done.')


    ######do the inference test#######


    val_data = {"LR":torch.randn(1, 3, 256,256),"HR":torch.randn(1, 3, 256,256),"SR":torch.randn(1, 3, 256,256)}
    
    diffusion.feed_data(val_data)
    diffusion.test(continous=True)
    visuals = diffusion.get_current_visuals()

    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
    lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
    fake_img = Metrics.tensor2img(visuals['INF'])  # uint8


    diffusion_without.feed_data(val_data)
    diffusion_without.test(continous=True)
    visuals = diffusion_without.get_current_visuals()

    hr_img_w = Metrics.tensor2img(visuals['HR'])  # uint8
    lr_img_w = Metrics.tensor2img(visuals['LR'])  # uint8
    fake_img_w = Metrics.tensor2img(visuals['INF'])  # uint8

    import pdb;pdb.set_trace()
    

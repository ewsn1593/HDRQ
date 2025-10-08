# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications:
# - Modification of config and checkpoint to support legacy models
# - Add inference mode and HRDA output flag

import argparse
import os
import time
import copy
import mmcv
import torch
import torch.nn as nn
import logging
import yaml
from easydict import EasyDict
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test, single_gpu_ptq_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from tools.brecq.quant.quant_layer import QuantModule, StraightThrough, UniformAffineQuantizer, PreQuantModule

from mmseg.models.backbones.mix_transformer import MixVisionTransformer
from mmseg.models.backbones.resnet import ResNetV1c





# Begin PTQ(brecq) code
import random
import numpy as np
from tools.brecq.quant import *

logger = logging.getLogger('brecq')
logging.basicConfig(level=logging.INFO, format='%(message)s')
# End PTQ(brecq) code

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config

# Begin PTQ(brecq) code
def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_train_samples(train_loader, num_samples):
    train_data = []
    
    batch_size = 1
    for i, data_batch in enumerate(train_loader):
        data = {
            'img':[data_batch['img'].data[0].cuda()],
            'img_metas': data_batch['img_metas'].data,
            'gt_semantic_seg': data_batch['gt_semantic_seg'].data[0],
            #'valid_pseudo_mask': data_batch['valid_pseudo_mask'].data[0],
        }
        if 'valid_pseudo_mask' in data_batch.keys():
            data['valid_pseudo_mask'] = data_batch['valid_pseudo_mask'].data[0]
        # cali_data.append(data_batch['img'][0])
        train_data.append(data)
        if len(train_data) % 16 == 0:
            print("[", len(train_data), "/", num_samples, "]")
        
        if len(train_data) * batch_size == num_samples:
            break
    
    return train_data
# End PTQ(brecq) code

def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    if cfg.model.type == 'MultiResEncoderDecoder':
        cfg.model.type = 'HRDAEncoderDecoder'
    if cfg.model.decode_head.type == 'MultiResAttentionWrapper':
        cfg.model.decode_head.type = 'HRDAHead'
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--inference-mode',
        choices=['same', 'whole', 'slide'],
        default='same',
        help='Inference mode.')
    parser.add_argument(
        '--test-set',
        action='store_true',
        help='Run inference on the test set')
    parser.add_argument(
        '--hrda-out',
        choices=['', 'LR', 'HR', 'ATT'],
        default='',
        help='Extract LR and HR predictions from HRDA architecture.')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)

    # Begin PTQ(brecq) code
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    # parser.add_argument('--arch', default='resnet18', type=str, help='dataset name', choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=1, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    # parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=32, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=5000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    # End PTQ(brecq) code
    
    parser.add_argument('--save_ds', default='default', type=str, help='save ds name')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

# Begin PTA(brecq) code (for counting reconstruction)
recon_total_count = 0
recon_total_count_m = 0
recon_total_count_b = 0

recon_count = 0
# End PTQ(brecq) code

def main():
    args = parse_args()
    
    # Begin PTQ(brecq) code
    seed_all(args.seed)
    # # build imagenet data loader

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg.model.test_cfg.mode = 'slide'
        crsize = cfg.data.train.get('sync_crop_size', cfg.crop_size)
        cfg.model.test_cfg.crop_size = crsize
        cfg.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.hrda_out == 'LR':
        cfg['model']['decode_head']['fixed_attention'] = 0.0
    elif args.hrda_out == 'HR':
        cfg['model']['decode_head']['fixed_attention'] = 1.0
    elif args.hrda_out == 'ATT':
        cfg['model']['decode_head']['debug_output_attention'] = True
    elif args.hrda_out == '':
        pass
    else:
        raise NotImplementedError(args.hrda_out)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    # Begin PTQ(brecq) code (for getting training set (only target))
    cfg.data.train.target.pipeline[2].img_scale = tuple(cfg.data.train.target.pipeline[2].img_scale)
    cfg.data.train.target.pipeline[3].crop_size = tuple(cfg.data.train.target.pipeline[3].crop_size)
    cfg.data.train.target.pipeline[6].size = tuple(cfg.data.train.target.pipeline[6].size)
    train_dataset = build_dataset(cfg.data.train.target)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=True)
    # End PTQ(brecq) code

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    # load model
    #model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    model.eval()

    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'symmetric': True, 'channel_wise': args.channel_wise, 'scale_method': 'mse'} # Symmetric Add
    aq_params = {'n_bits': args.n_bits_a, 'symmetric': False, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant} # Symmetric Add

    # For resnet modification
    aq_params_sym = {'n_bits': args.n_bits_a, 'symmetric': True, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant} # Symmetric Add
    

    

    
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    #### for resnet backbone ####
    if isinstance(qnn.model.backbone, ResNetV1c):
        print('ResNet101-Backbone')
        qnn.model.backbone.stem[0].activation_function = qnn.model.backbone.stem[2]
        qnn.model.backbone.stem[2] = StraightThrough() 

        qnn.model.backbone.stem[3].activation_function = qnn.model.backbone.stem[5]
        qnn.model.backbone.stem[5] = StraightThrough()

        qnn.model.backbone.stem[6].activation_function = qnn.model.backbone.stem[8]
        qnn.model.backbone.stem[8] = StraightThrough()
    #############################
    #### for MiT-B5 backbone ####
    if isinstance(qnn.model.backbone, MixVisionTransformer):
        print('MiT-B5-Backbone')
        qnn.model.backbone.patch_embed1.proj.always_xactquant = True 
    #############################
    if not args.disable_8bit_head_stem:
        ### for ResNet
        if isinstance(qnn.model.backbone, ResNetV1c):
            print('Setting the first and the last layer to 8-bit - for ResNet')
            qnn.set_first_last_layer_to_8bit()
            qnn.model.backbone.stem[0].weight_quantizer.bitwidth_refactor(8)
            #ignore reconstruction of the first layer
            qnn.model.backbone.stem[0].ignore_reconstruction = True
            print('Normally quantize output of first layer - x8bit')
            qnn.model.decode_head.head.conv_seg.weight_quantizer.bitwidth_refactor(8)
            qnn.model.decode_head.scale_attention.conv_seg.weight_quantizer.bitwidth_refactor(8)


            qnn.model.decode_head.head.fuse_layer.conv.activation_function = qnn.model.decode_head.head.fuse_layer.activate
            qnn.model.decode_head.head.fuse_layer.activate = StraightThrough()
            qnn.model.decode_head.scale_attention.fuse_layer.conv.activation_function = qnn.model.decode_head.scale_attention.fuse_layer.activate
            qnn.model.decode_head.scale_attention.fuse_layer.activate = StraightThrough()

            qnn.model.decode_head.head.fuse_layer.conv.act_quantizer.bitwidth_refactor(8)
            qnn.model.decode_head.scale_attention.fuse_layer.conv.act_quantizer.bitwidth_refactor(8)

            ### Embedding layer quantizer change - Must be Symmetric
            qnn.model.decode_head.head.embed_layers['0'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.head.embed_layers['1'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.head.embed_layers['2'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.head.embed_layers['3'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)

            qnn.model.decode_head.scale_attention.embed_layers['0'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.scale_attention.embed_layers['1'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.scale_attention.embed_layers['2'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.scale_attention.embed_layers['3'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)

  
        elif isinstance(qnn.model.backbone, MixVisionTransformer):
            print('Setting the first and the last layer to 8-bit - for MiT')
            qnn.set_first_last_layer_to_8bit()
            qnn.model.backbone.patch_embed1.proj.weight_quantizer.bitwidth_refactor(8)
            #ignore reconstruction of the first layer
            qnn.model.backbone.patch_embed1.proj.ignore_reconstruction = True
            print('Normally quantize output of first layer - x8bit')
            qnn.model.decode_head.head.conv_seg.weight_quantizer.bitwidth_refactor(8)
            qnn.model.decode_head.scale_attention.conv_seg.weight_quantizer.bitwidth_refactor(8)


            qnn.model.decode_head.scale_attention.fuse_layer.conv.activation_function = qnn.model.decode_head.scale_attention.fuse_layer.activate
            qnn.model.decode_head.scale_attention.fuse_layer.activate = StraightThrough()

            qnn.model.decode_head.head.fuse_layer.bottleneck.act_quantizer.bitwidth_refactor(8)
            qnn.model.decode_head.scale_attention.fuse_layer.conv.act_quantizer.bitwidth_refactor(8)

            ### Embedding layer quantizer change - Must be Symmetric
            qnn.model.decode_head.head.embed_layers['0'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.head.embed_layers['1'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.head.embed_layers['2'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.head.embed_layers['3'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)

            qnn.model.decode_head.scale_attention.embed_layers['0'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.scale_attention.embed_layers['1'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.scale_attention.embed_layers['2'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)
            qnn.model.decode_head.scale_attention.embed_layers['3'].proj.act_quantizer = UniformAffineQuantizer(**aq_params_sym)

        else:
            raise NotImplementedError
            
            

    qnn = MMDataParallel(qnn, device_ids=[0])
    qnn.cuda()
    qnn.eval()


    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device
    
    
    # Initialize weight quantization parameters
    qnn.module.set_quant_state(True, False)
    for i in range(1):
        qnn.module.extract_feat_for_ptq(cali_data[i])
    

    # Begin PTQ(brecq) code (for debugging)
    def counting_recon_model(model: nn.Module):
        global recon_total_count
        global recon_total_count_m
        global recon_total_count_b

        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                recon_total_count += 1
                recon_total_count_m += 1
            elif isinstance(module, BaseQuantBlock):
                recon_total_count += 1
                recon_total_count_b += 1
            else:
                counting_recon_model(module)
    # End PTQ(brecq) code

    counting_recon_model(qnn)
    print('Reconstruction counting Info: Total {0}, Layer: {1}, Block: {2}'.format(recon_total_count, recon_total_count_m, recon_total_count_b))

    # Kwargs for weight rounding calibration
    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse')

    # Total Count: 384, Block Count: 0, Layer Count: 384
    # Total 75, Layer: 19, Block: 56 (s1)
    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        global recon_count
        global recon_total_count
        for name, module in model.named_children():
            if isinstance(module, (QuantModule, PreQuantModule)):
                recon_count += 1
                if module.ignore_reconstruction is True:
                    print('<{0}>Ignore reconstruction of layer {1} [{2}/{3}]'.format(time.strftime("%Y-%m-%d %H:%M:%S"), name, recon_count, recon_total_count))
                    continue
                else:
                    print('<{0}>Reconstruction for layer {1} [{2}/{3}]'.format(time.strftime("%Y-%m-%d %H:%M:%S"), name, recon_count, recon_total_count))
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                recon_count += 1
                if module.ignore_reconstruction is True:
                    print('<{0}>Ignore reconstruction of block {1} [{2}/{3}]'.format(time.strftime("%Y-%m-%d %H:%M:%S"), name, recon_count, recon_total_count))
                    continue
                else:
                    print('<{0}>Reconstruction for block {1} [{2}/{3}]'.format(time.strftime("%Y-%m-%d %H:%M:%S"), name, recon_count, recon_total_count))
                    block_reconstruction(qnn, module, **kwargs)
            else:
                recon_model(module)

    # Start calibration
    
    recon_model(qnn)
    qnn.module.set_quant_state(weight_quant=True, act_quant=False)
    
    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.module.set_quant_state(True, True)
        with torch.no_grad():
            for i in range(args.num_samples):
                qnn.module.extract_feat_for_ptq(cali_data[i])

        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.module.disable_network_output_quantization()
        
        qnn.module.model.decode_head.head.conv_seg.disable_act_quant = True
        qnn.module.model.decode_head.scale_attention.conv_seg.disable_act_quant = True

        # Kwargs for activation rounding calibration
        kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p)
        
        recon_model(qnn)
        qnn.module.set_quant_state(weight_quant=True, act_quant=True)

        print('Full quantization (W{}A{})'.format(args.n_bits_w, args.n_bits_a))
        outputs = single_gpu_ptq_test(qnn, data_loader, args.show, args.show_dir, efficient_test, args.opacity)
        rank, _ = get_dist_info()
        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset.format_results(outputs, **kwargs)
            if args.eval:
                dataset.evaluate(outputs, args.eval, **kwargs)

        
    try:
        if args.save_ds == 'idd':
            save_name_ds = 'IDD'
        else:
            save_name_ds = 'CS'

        if isinstance(model.backbone, ResNetV1c):
            name_ = 'R101'
        elif isinstance(model.backbone, MixVisionTransformer):
            name_ = 'MiT'
        else:
            name_ = 'default'
            
        print(f'Save to ./ckpt/{save_name_ds}/BRECQ_PTQTest{name_}_W{args.n_bits_w}A{args.n_bits_a}_fixed_{save_name_ds}_seed{args.seed}.pt')
        torch.save(model, f'./ckpt/{save_name_ds}/BRECQ_PTQTest{name_}_W{args.n_bits_w}A{args.n_bits_a}_fixed_{save_name_ds}_seed{args.seed}.pt')
    except:
        breakpoint()
    
    
if __name__ == '__main__':
    main()
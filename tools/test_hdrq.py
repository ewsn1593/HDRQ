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

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

# Begin PTQ(brecq) code
import random
import numpy as np
from .qdrop.recon import reconstruction
from .qdrop.fold_bn import search_fold_and_remove_bn, StraightThrough
from .qdrop.model import load_model, specials
from .qdrop.quantization.state import enable_calibration_woquantization, enable_quantization, disable_all
from .qdrop.quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer, PreQuantizedLayer
from .qdrop.quantization.fake_quant import QuantizeBase
from .qdrop.quantization.observer import ObserverBase



from mmseg.models.backbones.mix_transformer import MixVisionTransformer
from mmseg.models.backbones.resnet import ResNetV1c



logger = logging.getLogger('HDRQ')
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

def quantize_model(model, config_quant):
    print('always q-output True => Set final layer qoutput false in custom manner')
    def replace_module(module, w_qconfig, a_qconfig, qoutput=True):
        childs = list(iter(module.named_children()))
        st, ed = 0, len(childs)
        prev_quantmodule = None
        while(st < ed):
            tmp_qoutput =  True
            name, child_module = childs[st][0], childs[st][1]
            if type(child_module) in specials:
                setattr(module, name, specials[type(child_module)](child_module, w_qconfig, a_qconfig, tmp_qoutput))
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantizedLayer(child_module, None, w_qconfig, a_qconfig, qoutput=tmp_qoutput))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, w_qconfig, a_qconfig, tmp_qoutput)
            st += 1
    
    replace_module(model, config_quant.w_qconfig, config_quant.a_qconfig, qoutput=False)
    print('Please set first last layer bit in custom manner')
    #w_list, a_list = [], []
    #for name, module in model.named_modules():
    #    if isinstance(module, QuantizeBase) and 'weight' in name:
    #        w_list.append(module)
    #    if isinstance(module, QuantizeBase) and 'act' in name:
    #        a_list.append(module)
    #w_list[0].set_bit(8)
    #w_list[-1].set_bit(8)
    #'the image input has already been in 256, set the last layer\'s input to 8-bit'
    #a_list[-1].set_bit(8)
    logger.info('finish quantize model:\n{}'.format(str(model)))
    
    return model

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
    parser.add_argument('--qdrop_config', default='qdrop_config.yaml', type=str)
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
    # weight calibration parameters
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--num_samples', default=32, type=int, help='size of the calibration dataset')

    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    # End PTQ(brecq) code


    parser.add_argument('--save_ds', default='default', type=str, help='save ds name')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# Begin PTQ(HDRQ) code (for counting reconstruction)
recon_total_count = 0
recon_total_count_m = 0
recon_total_count_b = 0
recon_total_count_p = 0

recon_count = 0
# End PTQ(HDRQ) code

def main():
    args = parse_args()
    
    seed_all(args.seed)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    qdrop_config_path = args.qdrop_config

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

    qdrop_config = parse_config(qdrop_config_path)
    if args.n_bits_a is not None:
        qdrop_config.quant.a_qconfig.bit = args.n_bits_a

    if args.n_bits_w is not None:
        qdrop_config.quant.w_qconfig.bit = args.n_bits_w
    qdrop_config.quant.a_qconfig.quantizer = 'LSQNoiseFakeQuantize'
    qdrop_config.quant.w_qconfig.quantizer = 'HDRQFakeQuantize'

    


    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    search_fold_and_remove_bn(model)

    model = quantize_model(model, qdrop_config.quant)

    sym_a_qconfig = copy.deepcopy(qdrop_config.quant.a_qconfig)
    sym_a_qconfig['symmetric'] = True


    model.cuda()
    ### HDRQ Basemodel Prepare ###
    if isinstance(model.backbone, ResNetV1c):
        base_model = torch.load('/NAS/MS/domain-adaptation-quant/daq/HRDA_ResNet101_Backbone_PT')
    elif isinstance(model.backbone, MixVisionTransformer):
        base_model = torch.load('/NAS/MS/domain-adaptation-quant/daq/HRDA_MITB5_Backbone_PT')
    search_fold_and_remove_bn(base_model)
    quantize_model(base_model, qdrop_config.quant)

    base_model.cuda()
    ######## Copy base weight
    assert(len([n for n,m in model.named_modules()]) == len([n for n,m in base_model.named_modules()]))
    for (n,m), (n_,m_) in zip(model.named_modules(), base_model.named_modules()):
        assert(type(m) == type(m_))
        if hasattr(m, 'weight'):
            if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                m.base_weight = m_.weight.data.detach().clone()
                m.base_weight.requires_grad = False
    ##############################


    #breakpoint()



    #### for resnet backbone ####
    if isinstance(model.backbone, ResNetV1c):
        print('ResNet101-Backbone')
        print('ResNet101-Nothing to do for stem layer in HDRQ')
    #### for MiT-B5 backbone ####
    if isinstance(model.backbone, MixVisionTransformer):
        print('MiT-B5-Backbone')
        model.backbone.patch_embed1.proj.always_xactquant = True 
    #############################
    ##### for All backbones  ####
    # qoutput 
    print('Output Quant False for Last layers')
    model.decode_head.head.conv_seg.qoutput = False
    model.decode_head.scale_attention.conv_seg.qoutput = False
    #############################


    print('First Last Layer 8-bit Setting')
    ### for ResNet
    if isinstance(model.backbone, ResNetV1c):
        print('Setting the first and the last layer to 8-bit - for ResNet')
        model.backbone.stem[0].module.weight_fake_quant.set_bit(8) 
        #ignore reconstruction of the first layer

        model.decode_head.head.conv_seg.module.weight_fake_quant.set_bit(8) 
        model.decode_head.scale_attention.conv_seg.module.weight_fake_quant.set_bit(8) 


        model.decode_head.scale_attention.fuse_layer.conv.layer_post_act_fake_quantize.set_bit(8)
        model.decode_head.head.fuse_layer.conv.layer_post_act_fake_quantize.set_bit(8)

        ### Embedding layer quantizer change - Must be Symmetric
        model.decode_head.head.embed_layers['0'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.head.embed_layers['1'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.head.embed_layers['2'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.head.embed_layers['3'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)

        model.decode_head.scale_attention.embed_layers['0'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.scale_attention.embed_layers['1'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.scale_attention.embed_layers['2'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.scale_attention.embed_layers['3'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)


    elif isinstance(model.backbone, MixVisionTransformer):
        print('Setting the first and the last layer to 8-bit - for MiT')
        model.backbone.patch_embed1.proj.module.weight_fake_quant.set_bit(8)
        #ignore reconstruction of the first layer
        model.decode_head.head.conv_seg.module.weight_fake_quant.set_bit(8) 
        model.decode_head.scale_attention.conv_seg.module.weight_fake_quant.set_bit(8) 


        model.decode_head.head.fuse_layer.bottleneck.layer_post_act_fake_quantize.set_bit(8)
        model.decode_head.scale_attention.fuse_layer.conv.layer_post_act_fake_quantize.set_bit(8)

        ### Embedding layer quantizer change - Must be Symmetric
        model.decode_head.head.embed_layers['0'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.head.embed_layers['1'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.head.embed_layers['2'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.head.embed_layers['3'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)

        model.decode_head.scale_attention.embed_layers['0'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.scale_attention.embed_layers['1'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.scale_attention.embed_layers['2'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)
        model.decode_head.scale_attention.embed_layers['3'].proj.layer_post_act_fake_quantize = Quantizer(None, sym_a_qconfig)

    else:
        raise NotImplementedError


    fp_model = copy.deepcopy(model)
    disable_all(fp_model)
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)
    
    model = MMDataParallel(model, device_ids=[0])
    model.cuda()
    model.eval()
    
    fp_model = MMDataParallel(fp_model, device_ids=[0])
    fp_model.cuda()
    fp_model.eval()


    print(args.save_ds)
    with torch.no_grad():
        st = time.time()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for i in range(args.num_samples // 4):
            print("a", i, time.time())
            model.module.extract_feat_for_ptq(cali_data[i])
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        for i in range(args.num_samples // 8):
            print("w", i, time.time())
            model.module.extract_feat_for_ptq(cali_data[i])
        ed = time.time()
        print('the calibration time is {}'.format(ed - st))

    if hasattr(qdrop_config.quant, 'recon'):
        def counting_recon_model(module: nn.Module, fp_module: nn.Module):
            global recon_total_count
            global recon_total_count_m
            global recon_total_count_b
            global recon_total_count_p
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer)):
                    recon_total_count += 1
                    if isinstance(child_module, QuantizedLayer):
                        recon_total_count_m += 1
                    elif isinstance(child_module, PreQuantizedLayer):
                        recon_total_count_p += 1
                    else: 
                        recon_total_count_b += 1

                    if recon_total_count == 37 or recon_total_count == 42 or recon_total_count == 43:
                        print(name)
                else:
                    counting_recon_model(child_module, getattr(fp_module, name))

        counting_recon_model(model, fp_model)
        print('Reconstruction counting Info: Total {0}, Layer: {1}, Block: {2}, Pre: {3}'.format(recon_total_count, recon_total_count_m, recon_total_count_b, recon_total_count_p))

        enable_quantization(model)
        def recon_model(module: nn.Module, fp_module: nn.Module):
            global recon_total_count
            global recon_count
            """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
            """
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock, PreQuantizedLayer)):
                    recon_count += 1
                    print('<{0}>[{1}/{2}] Begin reconstruction for module:\n{3}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), recon_count, recon_total_count, str(child_module)))
                    reconstruction(model.module, fp_model.module, child_module, getattr(fp_module, name), cali_data, qdrop_config.quant.recon, noise=True)
                else:
                    recon_model(child_module, getattr(fp_module, name))
            
        # Start reconstruction
        recon_model(model, fp_model)
        
    try:
        if args.save_ds == 'idd':
            save_name_ds = 'IDD'
        else:
            save_name_ds = 'CS'

        if isinstance(model.module.backbone, ResNetV1c):
            name_ = 'R101'
        elif isinstance(model.module.backbone, MixVisionTransformer):
            name_ = 'MiT'
        else:
            name_ = 'default'
        print(f'Save to ./ckpt/{save_name_ds}/HDRQ_PTQTest{name_}_W{args.n_bits_w}A{args.n_bits_a}_fixed_{save_name_ds}_seed{args.seed}.pt')
        torch.save(model, f'./ckpt/{save_name_ds}/HDRQ_PTQTest{name_}_W{args.n_bits_w}A{args.n_bits_a}_fixed_{save_name_ds}_seed{args.seed}.pt')
    except:
        breakpoint()

    enable_quantization(model)
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir, efficient_test, args.opacity)
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
    


if __name__ == '__main__':
    main()
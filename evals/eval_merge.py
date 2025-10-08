import argparse
import time
import copy
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import random
import numpy as np

import math

from torch.nn.parallel import DistributedDataParallel as DDP

import mmcv
from mmcv.utils import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.engine import collect_results_cpu, collect_results_gpu

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

from tools.qdrop.fold_bn import StraightThrough

from tools.qdrop.quantization.state import enable_quantization, disable_all

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
    pass

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
    parser.add_argument('config1', help='test config file path1')
    parser.add_argument('config2', help='test config file path2')
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
    parser.add_argument('--local-rank', type=int, default=0)
    
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    
    parser.add_argument('--model_path1', type=str, required=True, help='Model Path1')
    parser.add_argument('--model_path2', type=str, required=True, help='Model Path2')
    parser.add_argument('--world_size', default=1, type=int, help='World Size (N_GPUS)')
    parser.add_argument('--port_num', type=str, default='29700', help='port number')

    # QDROP BRECQ choice
    parser.add_argument('--qmethod', choices=['BRECQ', 'QDROP', 'HDRQ', 'FLEX', 'SMQ', 'FP'], type=str, help='Merge models quantized with which scheme?')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port_num

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # # 작업 그룹 초기화
    # try:
    #     dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # except:
    #     args.port_num = str(int(args.port_num)+100)
    #     setup(args, rank, world_size)  

def cleanup():
    dist.destroy_process_group()

def demo_basic(rank, world_size, args, cfg1, cfg2, dataset1, dataset2, result_queue, cur_iter):
    print(f"Running basic DDP example on rank {rank}.")
    setup(args,rank, world_size)

    seed_all(args.seed + cur_iter)
    
    total_dataset_count = len(dataset1.img_infos)
    dataset_count = total_dataset_count // world_size
    
    begin = 0 + (dataset_count * rank)
    if rank + 1 == world_size:
        end = total_dataset_count
    else:
        end = dataset_count + (dataset_count * rank)
    print("Dataset1 Count: {0}, Rank{1}, {2}-{3}".format(dataset_count, rank, begin, end))
    dataset1.img_infos=dataset1.img_infos[begin:end]
    data_loader1 = build_dataloader(
        dataset1,
        samples_per_gpu=1,
        workers_per_gpu=cfg1.data.workers_per_gpu,
        dist=False,
        shuffle=False)


    total_dataset_count = len(dataset2.img_infos)
    dataset_count = total_dataset_count // world_size
    
    begin = 0 + (dataset_count * rank)
    if rank + 1 == world_size:
        end = total_dataset_count
    else:
        end = dataset_count + (dataset_count * rank)
    print("Dataset2 Count: {0}, Rank{1}, {2}-{3}".format(dataset_count, rank, begin, end))
    dataset2.img_infos=dataset2.img_infos[begin:end]
    data_loader2 = build_dataloader(
        dataset2,
        samples_per_gpu=1,
        workers_per_gpu=cfg2.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # 모델을 생성하고 순위 아이디가 있는 GPU로 전달
    model1= torch.load(args.model_path1, map_location="cuda:{0}".format(rank))    
    model2= torch.load(args.model_path2, map_location="cuda:{0}".format(rank))    

    #################################
    ########## Model Merge ##########
    #################################
    if args.qmethod == 'BRECQ':
        merge_nets_BRECQ(model1, model2)
    elif (args.qmethod == 'QDROP' or args.qmethod == 'HDRQ'):
        merge_nets_QDROP(model1, model2)
    elif args.qmethod == 'FLEX':
        merge_nets_FLEX(model1, model2)
    elif args.qmethod == 'SMQ':
        enable_quantization(model1)
        enable_quantization(model2)
        merge_nets_QDROP(model1, model2)
    else:
        raise NotImplementedError

    model_merged = model1
    if hasattr(model_merged, "module"):
        model_merged = MMDistributedDataParallel(
                model_merged.module,
                device_ids=[rank],
                broadcast_buffers=False)
    else:
        model_merged = MMDistributedDataParallel(
            model_merged,
            device_ids=[rank],
            broadcast_buffers=False)
    model_merged.eval()

    print("Begin Rank {0}".format(rank))
    for i, data in enumerate(data_loader1):
        if i % 10 == 0:
            print("rank {0}, [{1}/{2}]".format(rank, i, len(dataset1)))
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda(rank)
            result = model_merged(return_loss=False, **data)
            if isinstance(result, list):
                result_queue[0][rank].extend(result)
            else:
                result_queue[0][rank].append(result)

    print("Begin Rank {0}".format(rank))
    for i, data in enumerate(data_loader2):
        if i % 10 == 0:
            print("rank {0}, [{1}/{2}]".format(rank, i, len(dataset2)))
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda(rank)
            result = model_merged(return_loss=False, **data)
            if isinstance(result, list):
                result_queue[1][rank].extend(result)
            else:
                result_queue[1][rank].append(result)
        
    cleanup()


def run_demo(demo_fn, args):
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg1 = mmcv.Config.fromfile(args.config1)
    if args.options is not None:
        cfg1.merge_from_dict(args.options)
    cfg1 = update_legacy_cfg(cfg1)
    # set cudnn_benchmark
    if cfg1.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg1.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg1.data.test.pipeline[1].flip = True
    cfg1.model.pretrained = None
    cfg1.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg1.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg1.model.test_cfg.mode = 'slide'
        crsize = cfg1.data.train.get('sync_crop_size', cfg.crop_size)
        cfg1.model.test_cfg.crop_size = crsize
        cfg1.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg1.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg1.data.test[k] = cfg1.data.test[k].replace('val', 'test')

    cfg2 = mmcv.Config.fromfile(args.config2)
    if args.options is not None:
        cfg2.merge_from_dict(args.options)
    cfg2 = update_legacy_cfg(cfg2)
    # set cudnn_benchmark
    if cfg2.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg2.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg2.data.test.pipeline[1].flip = True
    cfg2.model.pretrained = None
    cfg2.data.test.test_mode = True
    if args.inference_mode == 'same':
        # Use pre-defined inference mode
        pass
    elif args.inference_mode == 'whole':
        print('Force whole inference.')
        cfg2.model.test_cfg.mode = 'whole'
    elif args.inference_mode == 'slide':
        print('Force slide inference.')
        cfg2.model.test_cfg.mode = 'slide'
        crsize = cfg2.data.train.get('sync_crop_size', cfg.crop_size)
        cfg2.model.test_cfg.crop_size = crsize
        cfg2.model.test_cfg.stride = [int(e / 2) for e in crsize]
        cfg2.model.test_cfg.batched_slide = True
    else:
        raise NotImplementedError(args.inference_mode)

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg2.data.test[k] = cfg2.data.test[k].replace('val', 'test')
                
    
    
    dataset1 = build_dataset(cfg1.data.test)
    dataset2 = build_dataset(cfg2.data.test)
    # dataset1.img_infos=dataset1.img_infos[0:16]
    # dataset2.img_infos=dataset2.img_infos[0:16]
    
    print("Model1 is", args.model_path1, "world size is", args.world_size)
    world_size = args.world_size
    for cur_iter in range(32):
        beginTime = time.time()
        with mp.Manager() as manager:
            seed_all(args.seed + cur_iter)
            print("Seed is", (args.seed + cur_iter), args.seed, cur_iter)

            result_queue = manager.list()
            result_queue.append(manager.list())
            result_queue.append(manager.list())
            for i in range(world_size):
                result_queue[0].append(manager.list())
                result_queue[1].append(manager.list())
            
            mp.spawn(demo_fn,
                    args=(world_size, args, cfg1, cfg2, dataset1, dataset2, result_queue, cur_iter),
                    nprocs=world_size,
                    join=True)

            results_total_list = copy.deepcopy(result_queue)

            results_list1 = copy.deepcopy(results_total_list[0])
            results1 = np.array(copy.deepcopy(results_list1[0]))
            for i in range(1, len(results_list1)):
                results1 = np.concatenate((results1, copy.deepcopy(results_list1[i])), axis=0)

            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset1.format_results(results1, **kwargs)
            if args.eval:
                eval_result1 = dataset1.evaluate(results1, args.eval, **kwargs)

            results_list2 = copy.deepcopy(results_total_list[1])
            results2 = np.array(copy.deepcopy(results_list2[0]))
            for i in range(1, len(results_list2)):
                results2 = np.concatenate((results2, copy.deepcopy(results_list2[i])), axis=0)

            kwargs = {} if args.eval_options is None else args.eval_options
            if args.format_only:
                dataset2.format_results(results2, **kwargs)
            if args.eval:
                pass
                eval_result2 = dataset2.evaluate(results2, args.eval, **kwargs)

            save_txt_str = (args.model_path1).split('/')[-1]
            f1 = open('./result_txts/{0}_dataset1.txt'.format(save_txt_str), 'a')
            f2 = open('./result_txts/{0}_dataset2.txt'.format(save_txt_str), 'a')
            f3 = open('./result_txts/{0}_harmonic_mean.txt'.format(save_txt_str), 'a')

            for key in eval_result1:
                data1 = eval_result1[key]
                data2 = eval_result2[key]

                if math.isnan(data1) or math.isnan(data2):
                    harmonic_mean = data1
                elif data1 == 0 or data2 == 0:
                    harmonic_mean = 0
                else:
                    harmonic_mean = (2 * data1 * data2) / (data1 + data2)

                f1.write("{0} ".format(data1))
                f2.write("{0} ".format(data2))
                f3.write("{0} ".format(harmonic_mean))

            f1.write("\n")
            f2.write("\n")
            f3.write("\n")

            f1.close()
            f2.close()
            f3.close()

            print(eval_result1['mIoU'], eval_result2['mIoU'])
            print("Iter: ", cur_iter, "Time: ", time.time() - beginTime)

            results_list1.clear()
            results_list2.clear()





# Separte merged net ver.
def merge_nets_BRECQ(src_netFBC, tar_netFBC, lamb=0.5, noise_sampling=True, advanced_sampling=True): # False | True
    for (n,m), (n_, m_) in zip(src_netFBC.named_modules(), tar_netFBC.named_modules()):
        assert(type(m) == type(m_))
        if not isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(m, 'weight'):
                assert(hasattr(m, 'weight_quantizer'))
            if hasattr(m, 'weight_quantizer'): # layer with weight and delta
                # Resricted weight representation 
                if hasattr(m.weight_quantizer, 'alpha'):
                    if m.weight_quantizer.sym:
                        m_quant = torch.clamp(torch.floor(m.weight / m.weight_quantizer.delta)+(m.weight_quantizer.alpha >=0).float(), -(m.weight_quantizer.n_levels), m.weight_quantizer.n_levels - 1).data * m.weight_quantizer.delta
                        m__quant = torch.clamp(torch.floor(m_.weight / m_.weight_quantizer.delta)+(m_.weight_quantizer.alpha >=0).float(), -(m_.weight_quantizer.n_levels), m_.weight_quantizer.n_levels - 1).data * m_.weight_quantizer.delta
                    else:
                        m_quant = torch.clamp(torch.floor(m.weight / m.weight_quantizer.delta)+(m.weight_quantizer.alpha >=0).float(), 0, m.weight_quantizer.n_levels - 1).data * m.weight_quantizer.delta
                        m__quant = torch.clamp(torch.floor(m_.weight / m_.weight_quantizer.delta)+(m_.weight_quantizer.alpha >=0).float(), 0, m_.weight_quantizer.n_levels - 1).data * m_.weight_quantizer.delta
                else:   
                    if m.weight_quantizer.sym:
                        m_quant = torch.clamp(torch.round(m.weight / m.weight_quantizer.delta), -(m.weight_quantizer.n_levels), m.weight_quantizer.n_levels - 1).data * m.weight_quantizer.delta
                        m__quant = torch.clamp(torch.round(m_.weight / m_.weight_quantizer.delta), -(m_.weight_quantizer.n_levels), m_.weight_quantizer.n_levels - 1).data * m_.weight_quantizer.delta
                    else:
                        m_quant = torch.clamp(torch.round(m.weight / m.weight_quantizer.delta), 0, m.weight_quantizer.n_levels - 1).data * m.weight_quantizer.delta
                        m__quant = torch.clamp(torch.round(m_.weight / m_.weight_quantizer.delta), 0, m_.weight_quantizer.n_levels - 1).data * m_.weight_quantizer.delta        
                


                sim_11 = 0
                sim_12 = 0

                if noise_sampling:
                    for i in range(30):
                        noise_m = (torch.rand_like(m.weight) - 0.5) * m.weight_quantizer.delta 
                        noise_m_ = (torch.rand_like(m.weight) - 0.5) * m_.weight_quantizer.delta 

                        mask = ((((m_quant)/m.weight_quantizer.delta) + ((m__quant)/m_.weight_quantizer.delta)) % 2 == 1).int()

                        noise_m = noise_m * mask
                        noise_m_ = noise_m_ * mask
        
                        cur_int = (((m_quant - noise_m) * lamb + (m__quant - noise_m_) * (1-lamb)) / (m.weight_quantizer.delta.data * lamb + m_.weight_quantizer.delta.data * (1-lamb))).round()
                        cur_res = cur_int * (m.weight_quantizer.delta.data * lamb + m_.weight_quantizer.delta.data * (1-lamb))

                        sim_11_ = torch.nn.functional.cosine_similarity(cur_res.flatten()-m_quant.flatten(), ((m_quant + m__quant)/2).flatten()-m_quant.flatten(), dim=0)
                        sim_12_ = torch.nn.functional.cosine_similarity(cur_res.flatten()-m__quant.flatten(), ((m_quant + m__quant)/2).flatten()-m__quant.flatten(), dim=0)
                        
                        sim_11_ = abs(sim_11_)
                        sim_12_ = abs(sim_12_)

                        if (sim_11_ > sim_11 and sim_12_ > sim_12) or i==0: 
                            best_noise_m = noise_m
                            best_noise_m_ = noise_m_ 
                            sim_11 = sim_11_
                            sim_12 = sim_12_

                        if not advanced_sampling: 
                            #print('x advanced sampling?')
                            break

                    noise_m = best_noise_m
                    noise_m_ = best_noise_m_
                else:
                    #print('x noise samping?')
                    noise_m = 0
                    noise_m_ = 0
 
                
                m.weight.data = (m_quant - noise_m) * lamb + (m__quant - noise_m_) * (1-lamb)
                m.weight_quantizer.delta.data = m.weight_quantizer.delta.data * lamb + m_.weight_quantizer.delta.data * (1-lamb)
                if hasattr(m.weight_quantizer, 'alpha'):
                    m.weight_quantizer.round_mode = 'nearest'
                    #m.weight_quantizer.noise = False


            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = m.bias.data * lamb + m_.bias.data* (1-lamb)  
            if hasattr(m, 'act_quantizer'):
                if m.act_quantizer.delta is not None:
                    m.act_quantizer.delta.data = m.act_quantizer.delta.data * lamb + m_.act_quantizer.delta.data * (1-lamb)
  
        
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            assert(isinstance(m, nn.LayerNorm))
            if hasattr(m, 'weight'):
                m.weight.data = m.weight.data * lamb + m_.weight.data * (1-lamb)
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = m.bias.data * lamb + m_.bias.data* (1-lamb)  


# Separte merged net ver.
def merge_nets_QDROP(src_netFBC, tar_netFBC, lamb=0.5, noise_sampling=True, advanced_sampling=True): # False | True
    # Fetaured
    for (n,m), (n_, m_) in zip(src_netFBC.named_modules(), tar_netFBC.named_modules()):
        assert(type(m) == type(m_))
        if not isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(m, 'weight'):
                assert(hasattr(m, 'weight_fake_quant'))
            if hasattr(m, 'weight_fake_quant'): # layer with weight and delta
                # Resricted weight representation 
                if hasattr(m.weight_fake_quant, 'alpha'):
                    assert(not m.weight_fake_quant.adaround)
                    m_quant = torch.clamp(torch.round(m.weight / m.weight_fake_quant.scale), m.weight_fake_quant.quant_min, m.weight_fake_quant.quant_max) * m.weight_fake_quant.scale
                    m__quant = torch.clamp(torch.round(m_.weight / m_.weight_fake_quant.scale), m_.weight_fake_quant.quant_min, m_.weight_fake_quant.quant_max) * m_.weight_fake_quant.scale
                else:   
                    m_quant = torch.clamp(torch.round(m.weight / m.weight_fake_quant.scale), m.weight_fake_quant.quant_min, m.weight_fake_quant.quant_max) * m.weight_fake_quant.scale
                    m__quant = torch.clamp(torch.round(m_.weight / m_.weight_fake_quant.scale), m_.weight_fake_quant.quant_min, m_.weight_fake_quant.quant_max) * m_.weight_fake_quant.scale

                sim_11 = 0
                sim_12 = 0

                if noise_sampling:
                    for i in range(100):
                        noise_m = (torch.rand_like(m.weight) - 0.5) * m.weight_fake_quant.scale
                        noise_m_ = (torch.rand_like(m.weight) - 0.5) * m_.weight_fake_quant.scale

                        mask = ((((m_quant)/m.weight_fake_quant.scale) + ((m__quant)/m_.weight_fake_quant.scale)) % 2 == 1).int()

                        noise_m = noise_m * mask
                        noise_m_ = noise_m_ * mask
        
                        cur_int = (((m_quant - noise_m) * lamb + (m__quant - noise_m_) * (1-lamb)) / (m.weight_fake_quant.scale.data * lamb + m_.weight_fake_quant.scale.data * (1-lamb))).round()
                        cur_res = cur_int * (m.weight_fake_quant.scale.data * lamb + m_.weight_fake_quant.scale.data * (1-lamb))

                        sim_11_ = torch.nn.functional.cosine_similarity(cur_res.flatten()-m_quant.flatten(), ((m_quant + m__quant)/2).flatten()-m_quant.flatten(), dim=0)
                        sim_12_ = torch.nn.functional.cosine_similarity(cur_res.flatten()-m__quant.flatten(), ((m_quant + m__quant)/2).flatten()-m__quant.flatten(), dim=0)
                        
                        sim_11_ = abs(sim_11_)
                        sim_12_ = abs(sim_12_)

                        if (sim_11_ > sim_11 and sim_12_ > sim_12) or i==0: 
                            best_noise_m = noise_m
                            best_noise_m_ = noise_m_ 
                            sim_11 = sim_11_
                            sim_12 = sim_12_

                        if not advanced_sampling: 
                            #print('x advanced sampling?')
                            break

                    noise_m = best_noise_m
                    noise_m_ = best_noise_m_
                else:
                    #print('x noise samping?')
                    noise_m = 0
                    noise_m_ = 0

                m.weight.data = (m_quant - noise_m) * lamb + (m__quant - noise_m_) * (1-lamb)
                m.weight_fake_quant.scale.data = m.weight_fake_quant.scale.data * lamb + m_.weight_fake_quant.scale.data * (1-lamb)



            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = m.bias.data * lamb + m_.bias.data* (1-lamb)  
            if hasattr(m, 'layer_post_act_fake_quantize'):
                if not isinstance(m.layer_post_act_fake_quantize, StraightThrough):
                    if m.layer_post_act_fake_quantize.scale is not None:
                        m.layer_post_act_fake_quantize.scale.data = m.layer_post_act_fake_quantize.scale.data * lamb + m_.layer_post_act_fake_quantize.scale.data * (1-lamb)
            if hasattr(m, 'block_post_act_fake_quantize'):
                if m.block_post_act_fake_quantize.scale is not None:
                    m.block_post_act_fake_quantize.scale.data = m.block_post_act_fake_quantize.scale.data * lamb + m_.block_post_act_fake_quantize.scale.data * (1-lamb)
  
        
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            assert(isinstance(m, nn.LayerNorm))
            if hasattr(m, 'weight'):
                m.weight.data = m.weight.data * lamb + m_.weight.data * (1-lamb)
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = m.bias.data * lamb + m_.bias.data* (1-lamb)  


# Separte merged net ver.
def merge_nets_FLEX(src_netFBC, tar_netFBC, lamb=0.5, noise_sampling=True, advanced_sampling=True): # False | True
    # Fetaured
    for (n,m), (n_, m_) in zip(src_netFBC.named_modules(), tar_netFBC.named_modules()):
        assert(type(m) == type(m_))
        if not isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            if hasattr(m, 'weight'):
                assert(hasattr(m, 'weight_fake_quant'))
            if hasattr(m, 'weight_fake_quant'): # layer with weight and delta
                # Resricted weight representation 
                if hasattr(m.weight_fake_quant, 'alpha'):
                    assert(not m.weight_fake_quant.adaround)
                    scalem = m.weight_fake_quant.scale1 + m.weight_fake_quant.scale2 + m.weight_fake_quant.scale3  if m.weight_fake_quant.scale4 is None else m.weight_fake_quant.scale1 + m.weight_fake_quant.scale2 + m.weight_fake_quant.scale3 + m.weight_fake_quant.scale4
                    scalem = scalem.exp()
                    scalem_ = m_.weight_fake_quant.scale1 + m_.weight_fake_quant.scale2 + m_.weight_fake_quant.scale3 if m.weight_fake_quant.scale4 is None else m_.weight_fake_quant.scale1 + m_.weight_fake_quant.scale2 + m_.weight_fake_quant.scale3 + m_.weight_fake_quant.scale4
                    scalem_ = scalem_.exp()
                    m_quant = torch.clamp(torch.round(m.weight / scalem), m.weight_fake_quant.quant_min, m.weight_fake_quant.quant_max) * m.weight_fake_quant.scale1.exp()
                    m__quant = torch.clamp(torch.round(m_.weight / scalem_), m_.weight_fake_quant.quant_min, m_.weight_fake_quant.quant_max) * m_.weight_fake_quant.scale1.exp()
                else:   
                    scalem = m.weight_fake_quant.scale1 + m.weight_fake_quant.scale2 + m.weight_fake_quant.scale3  if m.weight_fake_quant.scale4 is None else m.weight_fake_quant.scale1 + m.weight_fake_quant.scale2 + m.weight_fake_quant.scale3 + m.weight_fake_quant.scale4
                    scalem = scalem.exp()
                    scalem_ = m_.weight_fake_quant.scale1 + m_.weight_fake_quant.scale2 + m_.weight_fake_quant.scale3 if m.weight_fake_quant.scale4 is None else m_.weight_fake_quant.scale1 + m_.weight_fake_quant.scale2 + m_.weight_fake_quant.scale3 + m_.weight_fake_quant.scale4
                    scalem_ = scalem_.exp()
                    m_quant = torch.clamp(torch.round(m.weight / scalem), m.weight_fake_quant.quant_min, m.weight_fake_quant.quant_max) * m.weight_fake_quant.scale1.exp()
                    m__quant = torch.clamp(torch.round(m_.weight / scalem_), m_.weight_fake_quant.quant_min, m_.weight_fake_quant.quant_max) * m_.weight_fake_quant.scale1.exp()

                sim_11 = 0
                sim_12 = 0
                
                if noise_sampling:
                    for i in range(100):
                        noise_m = (torch.rand_like(m.weight) - 0.5) * m.weight_fake_quant.scale1.exp()
                        noise_m_ = (torch.rand_like(m.weight) - 0.5) * m_.weight_fake_quant.scale1.exp()

                        mask = ((((m_quant)/m.weight_fake_quant.scale1.exp()) + ((m__quant)/m_.weight_fake_quant.scale1.exp())) % 2 == 1).int()

                        noise_m = noise_m * mask
                        noise_m_ = noise_m_ * mask
        
                        cur_int = (((m_quant - noise_m) * lamb + (m__quant - noise_m_) * (1-lamb)) / (m.weight_fake_quant.scale1.data.exp() * lamb + m_.weight_fake_quant.scale1.data.exp() * (1-lamb))).round()
                        cur_res = cur_int * (m.weight_fake_quant.scale1.data.exp() * lamb + m_.weight_fake_quant.scale1.data.exp() * (1-lamb))

                        sim_11_ = torch.nn.functional.cosine_similarity(cur_res.flatten()-m_quant.flatten(), ((m_quant + m__quant)/2).flatten()-m_quant.flatten(), dim=0)
                        sim_12_ = torch.nn.functional.cosine_similarity(cur_res.flatten()-m__quant.flatten(), ((m_quant + m__quant)/2).flatten()-m__quant.flatten(), dim=0)
                        
                        sim_11_ = abs(sim_11_)
                        sim_12_ = abs(sim_12_)

                        if (sim_11_ > sim_11 and sim_12_ > sim_12) or i==0: 
                            best_noise_m = noise_m
                            best_noise_m_ = noise_m_ 
                            sim_11 = sim_11_
                            sim_12 = sim_12_

                        if not advanced_sampling: 
                            #print('x advanced sampling?')
                            break

                    noise_m = best_noise_m
                    noise_m_ = best_noise_m_
                else:
                    #print('x noise samping?')
                    noise_m = 0
                    noise_m_ = 0
            
                m.weight.data = ((m_quant - noise_m) * lamb + (m__quant - noise_m_) * (1-lamb)) 

                ###
                m.weight_fake_quant.scale1.data = torch.log(m.weight_fake_quant.scale1.data.exp() * lamb + m_.weight_fake_quant.scale1.data.exp() * (1-lamb))
                #m.weight_fake_quant.scale2.data = torch.log((scalem + scalem_)/2) - m.weight_fake_quant.scale1.data
                m.weight_fake_quant.scale2.data = torch.zeros_like(m.weight_fake_quant.scale2.data)
                m.weight_fake_quant.scale3.data = torch.zeros_like(m.weight_fake_quant.scale3.data)

                #m.weight.data = m.weight.data * m.weight_fake_quant.scale1.data.exp()

                if m.weight_fake_quant.scale4 is not None:
                    m.weight_fake_quant.scale4.data = torch.zeros_like(m.weight_fake_quant.scale4.data)



            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = m.bias.data * lamb + m_.bias.data* (1-lamb)  
            if hasattr(m, 'layer_post_act_fake_quantize'):
                if not isinstance(m.layer_post_act_fake_quantize, StraightThrough):
                    if m.layer_post_act_fake_quantize.scale is not None:
                        m.layer_post_act_fake_quantize.scale.data = m.layer_post_act_fake_quantize.scale.data * lamb + m_.layer_post_act_fake_quantize.scale.data * (1-lamb)
            if hasattr(m, 'block_post_act_fake_quantize'):
                if m.block_post_act_fake_quantize.scale is not None:
                    m.block_post_act_fake_quantize.scale.data = m.block_post_act_fake_quantize.scale.data * lamb + m_.block_post_act_fake_quantize.scale.data * (1-lamb)
  
        
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            assert(isinstance(m, nn.LayerNorm))
            if hasattr(m, 'weight'):
                m.weight.data = m.weight.data * lamb + m_.weight.data * (1-lamb)
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = m.bias.data * lamb + m_.bias.data* (1-lamb)  
                    
if __name__ == "__main__":
    args = parse_args()
    
    beginTime = time.time()
    run_demo(demo_basic, args)
    print("Time: ", time.time() - beginTime)
    
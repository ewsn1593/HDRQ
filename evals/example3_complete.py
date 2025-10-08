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

from torch.nn.parallel import DistributedDataParallel as DDP

import mmcv
from mmcv.utils import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.engine import collect_results_cpu, collect_results_gpu

from mmseg.apis import multi_gpu_test, single_gpu_test, single_gpu_ptq_test_multi_process
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

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
    # torch.backends.cudnn.deterministic = True

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
    parser.add_argument('--local-rank', type=int, default=0)
    
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    
    parser.add_argument('--model_path', type=str, required=True, help='Model Path')
    parser.add_argument('--world_size', default=1, type=int, help='World Size (N_GPUS)')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29400'

    # 작업 그룹 초기화
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size, args, cfg, data_loader, dataset, result_queue):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    
    total_dataset_count = len(dataset.img_infos)
    dataset_count = total_dataset_count // world_size
    
    begin = 0 + (dataset_count * rank)
    if rank + 1 == world_size:
        end = total_dataset_count
    else:
        end = dataset_count + (dataset_count * rank)
        
    print("Dataset Begin", begin, "Dataset End", end, "Dataset Count", dataset_count)
    dataset.img_infos=dataset.img_infos[begin:end]
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    seed_all(args.seed)

    # 모델을 생성하고 순위 아이디가 있는 GPU로 전달
    model= torch.load(args.model_path, map_location="cuda:{0}".format(rank))    
    model = MMDistributedDataParallel(
            model.module,
            device_ids=[rank],
            broadcast_buffers=False)
    
    model.eval()
    print("Begin Rank {0}".format(rank))
    results = []
    for i, data in enumerate(data_loader):
        print(rank, i, data['img_metas'][0].data[0][0]['filename'])
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda(rank)
            img = data['img']
            img_metas = data['img_metas'][0].data
            result = model.module.model(img, img_metas, return_loss=False)
            # result = model(return_loss=False, **data)
            
            # print(rank, i, data['img_metas'][0].data[0][0]['filename'], "\n", result)
            # print(result)
            if isinstance(result, list):
                results.extend(result)
                result_queue[rank].extend(result)
            else:
                results.append(result)
                result_queue[rank].append(result)


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

    if args.test_set:
        for k in cfg.data.test:
            if isinstance(cfg.data.test[k], str):
                cfg.data.test[k] = cfg.data.test[k].replace('val', 'test')
                
    # Begin PTQ(brecq) code
    seed_all(args.seed)
    # End PTQ(brecq) code
    
    dataset = build_dataset(cfg.data.test)
    # dataset.img_infos=dataset.img_infos[0:50]
    data_loader = None

    print("Model is", args.model_path, "world size is", args.world_size)
    world_size = args.world_size

    with mp.Manager() as manager:
        result_queue = manager.list()
        for i in range(world_size):
            result_queue.append(manager.list())
        
        mp.spawn(demo_fn,
                args=(world_size, args, cfg, data_loader, dataset, result_queue),
                nprocs=world_size,
                join=True)
        
        results_list = copy.deepcopy(result_queue)
        results = np.array(copy.deepcopy(results_list[0]))
        for i in range(1, len(results_list)):
            results = np.concatenate((results, copy.deepcopy(results_list[i])), axis=0)
    
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=True,
    #     shuffle=False)
    
    # # 모델을 생성하고 순위 아이디가 있는 GPU로 전달
    # model_single= torch.load(args.model_path, map_location="cuda:{0}".format(0))
    
    # model_single.eval()
    # results3 = []
    # for i, data in enumerate(data_loader):
    #     print(0, i, data['img_metas'][0].data[0][0]['filename'])
    #     with torch.no_grad():
    #         data['img'][0] = data['img'][0].cuda(0)
    #         img = data['img']
    #         img_metas = data['img_metas'][0].data
    #         result = model_single.module.model(img, img_metas, return_loss=False)
    #         if isinstance(result, list):
    #             results3.extend(result)
    #         else:
    #             results3.append(result)
    
    # for i in range(len(results3)):
    #     print(np.array_equal(results3, results))
                
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(results, **kwargs)
    if args.eval:
        dataset.evaluate(results, args.eval, **kwargs)
        # dataset.evaluate(results3, args.eval, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    
    beginTime = time.time()
    run_demo(demo_basic, args)
    print("Time: ", time.time() - beginTime)
    
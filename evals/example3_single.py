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

import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

import mmcv
from mmcv.utils import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.engine import collect_results_cpu, collect_results_gpu

from mmseg.apis import multi_gpu_test, single_gpu_test, single_gpu_ptq_test, single_gpu_ptq_test_multi_process
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


def demo_basic(rank, world_size, args, cfg, result_queue):
    print(f"Running basic DDP example on rank {rank}.")
    # setup(rank, world_size)

    # 모델을 생성하고 순위 아이디가 있는 GPU로 전달
    model= torch.load("BRECQ_PTQTestR101_W4A4_fixed_CS", map_location="cuda:{0}".format(rank))
    
    # model = MMDistributedDataParallel(
    #         model.module.cuda(rank),
    #         device_ids=[rank],
    #         broadcast_buffers=False)
    
    model.cuda()
    model.eval()
    
    
    dataset = build_dataset(cfg.data.test)
    dataset.img_infos=dataset.img_infos[0:10]
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    print("Begin Rank {0}".format(rank))
    results = []
    for i, data in enumerate(data_loader):
        print(i, rank)
        with torch.no_grad():
            result = model(return_loss=False, **data)
            print(data['img_metas'][0].data[0][0]['filename'])
            print(result)
            results.append(result)
    
    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)
    
    # model = MMDataParallel(model, device_ids=[0])
    # outputs = single_gpu_ptq_test(model, data_loader, args.show, args.show_dir,
    #                               efficient_test, args.opacity)
            
    # breakpoint()
    
    results = np.array(results).squeeze(1)
    if args.out:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(results, **kwargs)
    if args.eval:
        dataset.evaluate(results, args.eval, **kwargs)
    
    
    # results = np.array(results).squeeze(1)
    # kwargs = {} if args.eval_options is None else args.eval_options
    # if args.format_only:
    #     dataset.format_results(results, **kwargs)
    # if args.eval:
    #     dataset.evaluate(results, args.eval, **kwargs)
    
    
    # for i in range(500):
    #     x = torch.rand(1,1,1024,2048)
    #     result_queue.append(x)
    #     del x
    
    # result_queue.put(rank)
    # results = collect_results_gpu(results, len(dataset))
    # if rank == 0:
    #     print(len(results))
    
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     print(len(results))
    # loss_fn = nn.MSELoss()
    # optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # optimizer.zero_grad()
    # outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(rank)
    # loss_fn(outputs, labels).backward()
    # optimizer.step()

    # cleanup()


def run_demo(demo_fn, world_size):
    args = parse_args()

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
    
    result_queue = []
    demo_basic(0, world_size, args, cfg, result_queue)

    # with mp.Manager() as manager:
    #     result_queue = manager.list()
    #     mp.spawn(demo_fn,
    #             args=(world_size, args, cfg, result_queue),
    #             nprocs=world_size,
    #             join=True)
    #     results = copy.deepcopy(result_queue)
    
    kwargs = {} if args.eval_options is None else args.eval_options
    if args.format_only:
        dataset.format_results(outputs, **kwargs)
    if args.eval:
        dataset.evaluate(outputs, args.eval, **kwargs)


if __name__ == "__main__":
    run_demo(demo_basic, 2)
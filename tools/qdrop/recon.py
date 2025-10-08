import math
import numpy as np
import torch
import torch.nn as nn
import logging
from .imagenet_utils import DataSaverHook, StopForwardException
from .quantization.quantized_module import QuantizedModule
from .quantization.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase, HDRQFakeQuantize, LSQNoiseFakeQuantize
logger = logging.getLogger('qdrop')

from mmseg.models.backbones.mix_transformer import MixVisionTransformer

from random import randrange


def save_inp_oup_data(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):
    store_inp = True
    store_oup = True

    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            try:
                _ = model.extract_feat_for_ptq(cali_data[i])
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0].detach())
                else:
                    input_data = data_saver.input_store[0].detach()
                    if isinstance(input_data,tuple):
                        if len(input_data) == 3:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu(),input_data[2].cpu()))
                        else:
                            cached[0].append((input_data[0].cpu(),input_data[1].cpu()))
                    else:
                        cached[0].append(input_data.cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())

    handle.remove()
    torch.cuda.empty_cache()
    return cached

def save_inp_oup_data_mit_backbone(model, module, cali_data: list, store_inp=False, store_oup=False, bs: int = 32, keep_gpu: bool = True):
    store_inp = True
    store_oup = True

    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for i in range(len(cali_data)):
            # print(i,len(cali_data))
            try:
                _ = model.extract_feat_for_ptq(cali_data[i])
            except StopForwardException:
                pass

            if store_inp:
                cur_inp = data_saver.input_store
                if len(cur_inp) == 1:
                    if keep_gpu:
                        cached[0].append((cur_inp[0].detach().to(device),))
                    else:
                        cached[0].append((cur_inp[0].detach().cpu(),))
                elif len(cur_inp) == 3:
                    if keep_gpu:
                        cached[0].append((cur_inp[0].detach().to(device), cur_inp[1], cur_inp[2]))
                    else:
                        cached[0].append((cur_inp[0].detach().cpu(), cur_inp[1], cur_inp[2]))
                else:
                    raise NotImplementedError

            if store_oup:
                cur_out = data_saver.output_store
                if len(cur_out) == 1:
                    if keep_gpu:
                        cached[1].append((cur_out[0].detach().to(device),))
                    else:
                        cached[1].append((cur_out[0].detach().cpu(),))
                elif len(cur_out) == 3:
                    if keep_gpu:
                        cached[1].append((cur_out[0].detach().to(device), cur_out[1], cur_out[2]))
                    else:
                        cached[1].append((cur_out[0].detach().cpu(), cur_out[1], cur_out[2]))
                else:
                    raise NotImplementedError

    handle.remove()
    torch.cuda.empty_cache()
    return cached

class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 module: QuantizedModule,
                 weight: float = 1.,
                 iters: int = 20000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.,
                 noise = False):

        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(iters, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0
        self.noise = noise

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.noise:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
            # logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
            #     float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


def lp_loss(pred, tgt, p=2.0):
    """
    loss function
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()



def reconstruction_resnet_backbone(model, fp_model, module, fp_module, cali_data, config, noise=False):
    device = next(module.parameters()).device
    quant_inp, _ = save_inp_oup_data(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data(fp_model, fp_module, cali_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)
    # prepare for up or down tuning
    w_para, a_para = [], []
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            if isinstance(weight_quantizer, HDRQFakeQuantize) and noise:
                w_para += [layer.weight]
                w_para += [weight_quantizer.scale]
            else:
                w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if isinstance(layer, (LSQFakeQuantize, LSQNoiseFakeQuantize)):
                a_para += [layer.scale]
            if isinstance(layer, LSQPlusFakeQuantize):
                a_para += [layer.scale]
                a_para += [layer.zero_point]
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None

    if noise:
        w_opt = torch.optim.Adam(w_para, lr=1e-4)
        w_scheduler =  CosineWithWarmup(w_opt, warmup_len=config.iters * config.warm_up, warmup_start_multiplier=1e-3, max_epochs=float(config.iters))
    else:
        w_opt = torch.optim.Adam(w_para)
        w_scheduler = None
    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up, noise=noise)


    ### HDRQ ###
    if noise:
        for n,m in module.named_modules():
            if hasattr(m, 'noise'):
                m.noise = True
        print(f'Noise Mode of cur module - {noise}')
        if loss_func.loss_start == 0:
            loss_func.loss_start = 500
            print(f'Loss schedule adjusted-{loss_func.loss_start}')
    ############

    #### [ablation study] ####
    hyp = 5e-2 ##  original:5e-2, hyp01: 5e-3, hyp02: 1e-2, hyp03: 1e-1
    print("[Hyp Info]", hyp)
    ##########################

    sz = len(cali_data)
    for i in range(config.iters):
        ### HDRQ ###
        if i == (config.iters - int(loss_func.loss_start)):
            if noise:
                for n,m in module.named_modules():
                    if hasattr(m, 'noise'):
                        m.noise = False
                print('Noise False')
        ############

        idx = randrange(sz)
        
        if config.drop_prob < 1.0:
            cur_quant_inp = quant_inp[idx].to(device)
            cur_fp_inp = fp_inp[idx].to(device)
            cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
        else:
            cur_inp = quant_inp[idx].to(device)
        cur_fp_oup = fp_oup[idx].to(device)
        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        cur_quant_oup = module(cur_inp)
        err = loss_func(cur_quant_oup, cur_fp_oup)

        if noise:
            weight_reg = 0
            for name, m in module.named_modules():
                if hasattr(m, 'weight') and not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    # for ablation study (set hyp in Line 236-239)
                    weight_reg = weight_reg + (m.weight - m.base_weight).norm()*hyp 
                    # weight_reg = weight_reg + (m.weight - m.base_weight).norm()*5e-2
            
            if loss_func.count % 500 == 0:
                print(f'\tTotal loss:{(err+weight_reg):.2f}(loss:{err:.2f} reg:{weight_reg:.2f})')
            #breakpoint()
            err = err + weight_reg
        #breakpoint()
        err.backward()
        
        w_opt.step()
        if w_scheduler is not None:
            w_scheduler.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

    torch.cuda.empty_cache()
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0

    if noise:
        for n,m in module.named_modules():
            if hasattr(m, 'noise'):
                m.noise = False
        print(f'Noise Mode of cur module - False')
        print()

def reconstruction_mit_backbone(model, fp_model, module, fp_module, cali_data, config, noise=False):
    device = next(module.parameters()).device
    quant_inp, _ = save_inp_oup_data_mit_backbone(model, module, cali_data, store_inp=True, store_oup=False, bs=config.batch_size, keep_gpu=config.keep_gpu)
    fp_inp, fp_oup = save_inp_oup_data_mit_backbone(fp_model, fp_module, cali_data, store_inp=True, store_oup=True, bs=config.batch_size, keep_gpu=config.keep_gpu)

    # prepare for up or down tuning
    w_para, a_para = [], []
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            if isinstance(weight_quantizer, HDRQFakeQuantize) and noise:
                w_para += [layer.weight]
                w_para += [weight_quantizer.scale]
            else:
                w_para += [weight_quantizer.alpha]
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = config.drop_prob
            if isinstance(layer, (LSQFakeQuantize, LSQNoiseFakeQuantize)):
                a_para += [layer.scale]
            if isinstance(layer, LSQPlusFakeQuantize):
                a_para += [layer.scale]
                a_para += [layer.zero_point]
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.iters, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None

    if noise:
        w_opt = torch.optim.Adam(w_para, lr=1e-4)
        w_scheduler =  CosineWithWarmup(w_opt, warmup_len=config.iters * config.warm_up, warmup_start_multiplier=1e-3, max_epochs=float(config.iters))
    else:
        w_opt = torch.optim.Adam(w_para)
        w_scheduler = None
    loss_func = LossFunction(module=module, weight=config.weight, iters=config.iters, b_range=config.b_range,
                             warm_up=config.warm_up, noise=noise)


    ### HDRQ ###
    if noise:
        for n,m in module.named_modules():
            if hasattr(m, 'noise'):
                m.noise = True
        print(f'Noise Mode of cur module - {noise}')
        if loss_func.loss_start == 0:
            loss_func.loss_start = 500
            print(f'Loss schedule adjusted-{loss_func.loss_start}')
    ############

    sz = len(cali_data)
    for i in range(config.iters):
        ### HDRQ ###
        if i == (config.iters - int(loss_func.loss_start)):
            if noise:
                for n,m in module.named_modules():
                    if hasattr(m, 'noise'):
                        m.noise = False
                print('Noise False')
        ############

        idx = randrange(sz)
        
        if len(quant_inp[idx]) == 3:
            if config.drop_prob < 1.0:
                cur_quant_inp, H, W = quant_inp[idx]
                cur_fp_inp, H, W = fp_inp[idx]

                cur_quant_inp = cur_quant_inp.to(device)
                cur_fp_inp = cur_fp_inp.to(device)

                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
            else:
                cur_inp, H, W = quant_inp[idx]
                cur_inp = cur_inp.to(device)

        elif len(quant_inp[idx]) == 1:
            if config.drop_prob < 1.0:
                cur_quant_inp = quant_inp[idx][0].to(device)
                cur_fp_inp = fp_inp[idx][0].to(device)

                cur_quant_inp = cur_quant_inp.to(device)
                cur_fp_inp = cur_fp_inp.to(device)

                cur_inp = torch.where(torch.rand_like(cur_quant_inp) < config.drop_prob, cur_quant_inp, cur_fp_inp)
        else:
            raise NotImplementedError

        if isinstance(fp_oup[idx], tuple):
            cur_fp_oup = fp_oup[idx][0].to(device)
        else:
            cur_fp_oup = fp_oup[idx].to(device)

        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()

        if len(quant_inp[idx]) == 3:
            _, H, W = quant_inp[idx]
            cur_quant_oup = module(cur_inp, H, W)
        elif len(quant_inp[idx]) == 1:
            cur_quant_oup = module(cur_inp)
        else:
            raise NotImplementedError

        if len(cur_quant_oup) == 3:
            err = loss_func(cur_quant_oup[0], cur_fp_oup)
        elif len(cur_quant_oup) == 1:
            err = loss_func(cur_quant_oup, cur_fp_oup)
        else:
            raise NotImplementedError
        

        if noise:
            weight_reg = 0
            for name, m in module.named_modules():
                if hasattr(m, 'weight') and not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    weight_reg = weight_reg + (m.weight - m.base_weight).norm()*5e-2 ##hyp01 : 5e-3 ##hyp02:1e-2 ## original 5e-2 ##hyp03 1e-1
            
            if loss_func.count % 500 == 0:
                print(f'\tTotal loss:{(err+weight_reg):.2f}(loss:{err:.2f} reg:{weight_reg:.2f})')
            #breakpoint()
            err = err + weight_reg

        err.backward()
        w_opt.step()
        if w_scheduler is not None:
            w_scheduler.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

    torch.cuda.empty_cache()
    for name, layer in module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and 'post_act_fake_quantize' in name:
            layer.drop_prob = 1.0

    if noise:
        for n,m in module.named_modules():
            if hasattr(m, 'noise'):
                m.noise = False
        print(f'Noise Mode of cur module - False')
        print()

def reconstruction(model, fp_model, module, fp_module, cali_data, config, noise=False):
    if isinstance(model.backbone, MixVisionTransformer):
        reconstruction_mit_backbone(model, fp_model, module, fp_module, cali_data, config, noise)
    else:
        reconstruction_resnet_backbone(model, fp_model, module, fp_module, cali_data, config, noise)
    

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """ Implements a schedule where the first few epochs are linear warmup, and
    then there's cosine annealing after that."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_len: int,
                 warmup_start_multiplier: float, max_epochs: int, 
                 eta_min: float = 0.0, last_epoch: int = -1):
        if warmup_len < -1:
            raise ValueError("Warmup can't be less than 0.")
        self.warmup_len = warmup_len
        if not (0.0 <= warmup_start_multiplier <= 1.0):
            raise ValueError(
                "Warmup start multiplier must be within [0.0, 1.0].")
        self.warmup_start_multiplier = warmup_start_multiplier
        if max_epochs < 0 or max_epochs < warmup_len:
            raise ValueError("Max epochs must be longer than warm-up.")
        self.max_epochs = max_epochs
        self.cosine_len = self.max_epochs - self.warmup_len
        self.eta_min = eta_min  # Final LR multiplier of cosine annealing
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch > self.max_epochs:
            raise ValueError(
                "Epoch may not be greater than max_epochs={}.".format(
                    self.max_epochs))
        if self.last_epoch < self.warmup_len or self.cosine_len == 0:
            # We're in warm-up, increase LR linearly. End multiplier is implicit 1.0.
            slope = (1.0 - self.warmup_start_multiplier) / self.warmup_len
            lr_multiplier = self.warmup_start_multiplier + slope * self.last_epoch
        else:
            # We're in the cosine annealing part. Note that the implementation
            # is different from the paper in that there's no additive part and
            # the "low" LR is not limited by eta_min. Instead, eta_min is
            # treated as a multiplier as well. The paper implementation is
            # designed for SGDR.
            cosine_epoch = self.last_epoch - self.warmup_len
            lr_multiplier = self.eta_min + (1.0 - self.eta_min) * (
                1 + math.cos(math.pi * cosine_epoch / self.cosine_len)) / 2
        assert lr_multiplier >= 0.0
        return [base_lr * lr_multiplier for base_lr in self.base_lrs]
import torch
import torch.nn.functional as F
from tools.brecq.quant.quant_layer import QuantModule, Union, PreQuantModule
from tools.brecq.quant.quant_model import QuantModel
from tools.brecq.quant.quant_block import BaseQuantBlock

from mmseg.models.backbones.mix_transformer import MixVisionTransformer


def save_inp_oup_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                      asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    if isinstance(model.module.model.backbone, MixVisionTransformer):
        cached_batches.append([])
        cached_batches.append([])

        for i in range(len(cali_data)):
            cur_inp, cur_out = get_inp_out(cali_data[i])

            if len(cur_inp) == 1:
                if keep_gpu:
                    cached_batches[0].append((cur_inp[0].detach().to(device),))
                else:
                    cached_batches[0].append((cur_inp[0].detach().cpu(),))
            elif len(cur_inp) == 3:
                if keep_gpu:
                    cached_batches[0].append((cur_inp[0].detach().to(device), cur_inp[1], cur_inp[2]))
                else:
                    cached_batches[0].append((cur_inp[0].detach().cpu(), cur_inp[1], cur_inp[2]))
            else:
                raise NotNotImpelentedError

            if len(cur_out) == 1:
                if keep_gpu:
                    cached_batches[1].append((cur_out[0].detach().to(device),))
                else:
                    cached_batches[1].append((cur_out[0].detach().cpu(),))
            elif len(cur_out) == 3:
                if keep_gpu:
                    cached_batches[1].append((cur_out[0].detach().to(device), cur_out[1], cur_out[2]))
                else:
                    cached_batches[1].append((cur_out[0].detach().cpu(), cur_out[1], cur_out[2]))
            else:
                raise NotNotImpelentedError

        torch.cuda.empty_cache()
        return cached_batches[0], cached_batches[1]
    else:
        for i in range(len(cali_data)):
            cur_inp, cur_out = get_inp_out(cali_data[i])
            cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

        cached_inps = torch.cat([x[0] for x in cached_batches])
        cached_outs = torch.cat([x[1] for x in cached_batches])
        torch.cuda.empty_cache()
        if keep_gpu:
            cached_inps = cached_inps.to(device)
            cached_outs = cached_outs.to(device)
        return cached_inps, cached_outs

def save_grad_data(model: QuantModel, layer: Union[QuantModule, BaseQuantBlock], cali_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input):
        self.model.eval()
        self.model.module.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                model_input['img'][0] = model_input['img'][0].to(self.device)
                _ = self.model.module.extract_feat_for_ptq(model_input)
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.module.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    model_input['img'][0] = model_input['img'][0].to(self.device)
                    _ = self.model.module.extract_feat_for_ptq(model_input)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()
        
        self.model.module.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        
        if isinstance(self.model.module.model.backbone, MixVisionTransformer):
            return self.data_saver.input_store, self.data_saver.output_store
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()


class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantModule, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantModule, layer: Union[QuantModule, BaseQuantBlock], act_quant: bool = False):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantModule, PreQuantModule, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmseg.models
from tools.brecq.quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough, PreQuantModule
from tools.brecq.models.resnet import BasicBlock, Bottleneck
from tools.brecq.models.regnet import ResBottleneckBlock
from tools.brecq.models.mobilenetv2 import InvertedResidual

import mmseg
from mmseg.ops import resize

class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer

        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()

        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantModule, PreQuantModule)):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu if isinstance(basic_block, mmseg.models.backbones.resnet.BasicBlock) else basic_block.relu1
        self.conv2 = QuantModule(basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = basic_block.relu if isinstance(basic_block, mmseg.models.backbones.resnet.BasicBlock) else basic_block.relu2

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = basic_block.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu if isinstance(bottleneck, mmseg.models.backbones.resnet.Bottleneck) else bottleneck.relu1
        self.conv2 = QuantModule(bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu if isinstance(bottleneck, mmseg.models.backbones.resnet.Bottleneck) else bottleneck.relu2
        self.conv3 = QuantModule(bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu if isinstance(bottleneck, mmseg.models.backbones.resnet.Bottleneck) else bottleneck.relu3

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        # copying all attributes in original block
        self.stride = bottleneck.stride

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)
        self.conv1 = QuantModule(bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = QuantModule(bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = QuantModule(bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu

        if bottleneck.proj_block:
            self.downsample = QuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                          disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__(act_quant_params)

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                QuantModule(inv_res.conv[0], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[3], weight_quant_params, act_quant_params),
                QuantModule(inv_res.conv[6], weight_quant_params, act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out




class QAttention(BaseQuantBlock):
    def __init__(self, attn:mmseg.models.backbones.mix_transformer.Attention, weight_quant_params: dict={}, act_quant_params: dict={}):
        super().__init__(act_quant_params)

        self.sr_ratio = attn.sr_ratio
        self.num_heads = attn.num_heads
        self.dim = attn.dim
        self.scale = attn.scale
           
        
        self.q = PreQuantModule(attn.q, weight_quant_params, act_quant_params, sym=True)
        self.kv = PreQuantModule(attn.kv, weight_quant_params, act_quant_params, sym=True)
        self.proj = PreQuantModule(attn.proj, weight_quant_params, act_quant_params, sym=True)

        sym_act_quant_params = copy.deepcopy(act_quant_params)
        sym_act_quant_params['symmetric'] = True

        self.q_post_act_fake_quantize = UniformAffineQuantizer(**sym_act_quant_params)
        self.k_post_act_fake_quantize = UniformAffineQuantizer(**sym_act_quant_params)
        self.v_post_act_fake_quantize = UniformAffineQuantizer(**sym_act_quant_params)

        self.softmax_post_act_fake_quantize = UniformAffineQuantizer(**act_quant_params)

        if self.sr_ratio > 1:
            self.sr = PreQuantModule(attn.sr, weight_quant_params, act_quant_params, sym=True)
            self.norm = attn.norm 
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        
        # proj
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,3).contiguous()

        # q quant
        if self.use_act_quant:
            q = self.q_post_act_fake_quantize(q)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            
        k, v = kv[0], kv[1]
        
        # kquant vquant
        if self.use_act_quant:
            k = self.k_post_act_fake_quantize(k)
            v = self.v_post_act_fake_quantize(v)

        # attn
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)

        # softmax quant
        if self.use_act_quant:
            attn = self.softmax_post_act_fake_quantize(attn)


        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)

        # proj
        x = self.proj(x)
        

        return x


class QDWConv(BaseQuantBlock):
    def __init__(self, dwb:mmseg.models.backbones.mix_transformer.DWConv, weight_quant_params: dict={}, act_quant_params: dict={}):
        super().__init__(act_quant_params)

        self.dwconv = PreQuantModule(dwb.dwconv, weight_quant_params, act_quant_params, sym=True)


    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x




class QMLPBlock(BaseQuantBlock):
    def __init__(self, mlpb:mmseg.models.backbones.mix_transformer.Mlp, weight_quant_params: dict={}, act_quant_params: dict={}):
        super().__init__(act_quant_params)


        sym_act_quant_params = copy.deepcopy(act_quant_params)
        sym_act_quant_params['symmetric'] = True

        self.fc1 = PreQuantModule(mlpb.fc1, weight_quant_params, act_quant_params, sym=True)
        self.dwconv = QDWConv(mlpb.dwconv, weight_quant_params, act_quant_params)
        self.act = mlpb.act
        self.fc2 =  PreQuantModule(mlpb.fc2, weight_quant_params, act_quant_params, sym=True)


    
    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        
        return x




class QuantMixTransformerBlock(BaseQuantBlock):
    '''
    MixTransformerBlock Implementation
    '''

    def __init__(self, mit:mmseg.models.backbones.mix_transformer.Block, weight_quant_params: dict={}, act_quant_params: dict={}):
        super().__init__(act_quant_params)


        self.attn = QAttention(mit.attn, weight_quant_params, act_quant_params)

        self.norm1 = mit.norm1 
        self.norm2 = mit.norm2

        self.mlp = QMLPBlock(mit.mlp, weight_quant_params, weight_quant_params)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class QuantOverlapPatchEmbed(BaseQuantBlock):
    '''
    MixTransformer OverlapPatchEmbed Implementation
    '''

    def __init__(self, ope:mmseg.models.backbones.mix_transformer.OverlapPatchEmbed, weight_quant_params: dict={}, act_quant_params: dict={}):
        super().__init__(act_quant_params)

        self.proj = PreQuantModule(ope.proj, weight_quant_params, act_quant_params, sym=True)
        self.norm = ope.norm

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W








class QuantASPPWrapper(BaseQuantBlock):
    def __init__(self, aspp:mmseg.models.decode_heads.daformer_head.ASPPWrapper, weight_quant_params: dict={}, act_quant_params: dict={}):
        super().__init__(act_quant_params)
        
        self.image_pool = aspp.image_pool
        self.dilations = aspp.dilations
        self.align_corners = aspp.align_corners
        self.context_layer = aspp.context_layer

        assert(aspp.context_layer is None)

        self.bottleneck = QuantModule(aspp.bottleneck.conv, weight_quant_params, act_quant_params)
        self.bottleneck.activation_function = aspp.bottleneck.activate
        

        
        # Hard coding fit to DAFormer head aspp module
        act_temp = aspp.aspp_modules[0].activate
        aspp.aspp_modules[0] = QuantModule(aspp.aspp_modules[0].conv, weight_quant_params, act_quant_params)
        aspp.aspp_modules[0].activation_function = act_temp


        for i in range(1, len(aspp.aspp_modules)):
            act_temp = aspp.aspp_modules[i].depthwise_conv.activate
            aspp.aspp_modules[i].depthwise_conv.conv = QuantModule(aspp.aspp_modules[i].depthwise_conv.conv, weight_quant_params, act_quant_params)
            aspp.aspp_modules[i].depthwise_conv.activation_function = act_temp

            act_temp = aspp.aspp_modules[i].pointwise_conv.activate
            aspp.aspp_modules[i].pointwise_conv.conv = QuantModule(aspp.aspp_modules[i].pointwise_conv.conv, weight_quant_params, act_quant_params)
            aspp.aspp_modules[i].pointwise_conv.activation_function = act_temp
        
        self.aspp_modules = aspp.aspp_modules

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output






specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    ####################### For MMSEG Comparatibility
    mmseg.models.backbones.resnet.BasicBlock : QuantBasicBlock,
    mmseg.models.backbones.resnet.Bottleneck : QuantBottleneck,
    mmseg.models.backbones.mix_transformer.Block : QuantMixTransformerBlock,
    mmseg.models.backbones.mix_transformer.OverlapPatchEmbed : QuantOverlapPatchEmbed,
    mmseg.models.decode_heads.daformer_head.ASPPWrapper : QuantASPPWrapper
}

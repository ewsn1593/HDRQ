import copy

import torch.nn as nn
import torch
from .resnet import BasicBlock, Bottleneck, resnet18, resnet50  # noqa: F401
from .regnet import ResBottleneckBlock, regnetx_600m, regnetx_3200m  # noqa: F401
from .mobilenetv2 import InvertedResidual, mobilenetv2  # noqa: F401
from .mnasnet import _InvertedResidual, mnasnet  # noqa: F401
from ..quantization.quantized_module import QuantizedLayer, QuantizedBlock, Quantizer, PreQuantizedLayer   # noqa: F401

import mmseg
from mmseg.ops import resize

import timm


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, org_module: BasicBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu if isinstance(org_module, mmseg.models.backbones.resnet.BasicBlock) else org_module.relu1, w_qconfig, a_qconfig)
        self.conv2 = QuantizedLayer(org_module.conv2, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = org_module.relu if isinstance(org_module, mmseg.models.backbones.resnet.BasicBlock) else org_module.relu2
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """
    def __init__(self, org_module: Bottleneck, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu if isinstance(org_module, mmseg.models.backbones.resnet.Bottleneck) else org_module.relu1, w_qconfig, a_qconfig)
        self.conv2_relu = QuantizedLayer(org_module.conv2, org_module.relu if isinstance(org_module, mmseg.models.backbones.resnet.Bottleneck) else org_module.relu2, w_qconfig, a_qconfig)
        self.conv3 = QuantizedLayer(org_module.conv3, None, w_qconfig, a_qconfig, qoutput=False)

        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, w_qconfig, a_qconfig, qoutput=False)
        self.activation = org_module.relu if isinstance(org_module, mmseg.models.backbones.resnet.Bottleneck) else org_module.relu3
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2_relu(out)
        out = self.conv3(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantResBottleneckBlock(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """
    def __init__(self, org_module: ResBottleneckBlock, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.conv1_relu = QuantizedLayer(org_module.f.a, org_module.f.a_relu, w_qconfig, a_qconfig)
        self.conv2_relu = QuantizedLayer(org_module.f.b, org_module.f.b_relu, w_qconfig, a_qconfig)
        self.conv3 = QuantizedLayer(org_module.f.c, None, w_qconfig, a_qconfig, qoutput=False)
        if org_module.proj_block:
            self.downsample = QuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig, qoutput=False)
        else:
            self.downsample = None
        self.activation = org_module.relu
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1_relu(x)
        out = self.conv2_relu(out)
        out = self.conv3(out)
        out += residual
        out = self.activation(out)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class QuantInvertedResidual(QuantizedBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """
    def __init__(self, org_module: InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.use_res_connect = org_module.use_res_connect
        if org_module.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
                QuantizedLayer(org_module.conv[3], None, w_qconfig, a_qconfig, qoutput=False),
            )
        else:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.conv[0], org_module.conv[2], w_qconfig, a_qconfig),
                QuantizedLayer(org_module.conv[3], org_module.conv[5], w_qconfig, a_qconfig),
                QuantizedLayer(org_module.conv[6], None, w_qconfig, a_qconfig, qoutput=False),
            )
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out


class _QuantInvertedResidual(QuantizedBlock):
    # mnasnet
    def __init__(self, org_module: InvertedResidual, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.apply_residual = org_module.apply_residual
        self.conv = nn.Sequential(
            QuantizedLayer(org_module.layers[0], org_module.layers[2], w_qconfig, a_qconfig),
            QuantizedLayer(org_module.layers[3], org_module.layers[5], w_qconfig, a_qconfig),
            QuantizedLayer(org_module.layers[6], None, w_qconfig, a_qconfig, qoutput=False),
        )
        if self.qoutput:
            self.block_post_act_fake_quantize = Quantizer(None, a_qconfig)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.qoutput:
            out = self.block_post_act_fake_quantize(out)
        return out




class QAttention(QuantizedBlock):
    def __init__(self, org_module: mmseg.models.backbones.mix_transformer.Attention, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()

        self.sr_ratio = org_module.sr_ratio
        self.num_heads = org_module.num_heads
        self.dim = org_module.dim
        self.scale = org_module.scale
           
        
        self.q = PreQuantizedLayer(org_module.q, None, w_qconfig, a_qconfig, sym=True)
        self.kv = PreQuantizedLayer(org_module.kv, None, w_qconfig, a_qconfig, sym=True)
        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig, sym=True)

        sym_act_quant_params = copy.deepcopy(a_qconfig)
        sym_act_quant_params['symmetric'] = True

        self.q_post_act_fake_quantize = Quantizer(None, sym_act_quant_params)
        self.k_post_act_fake_quantize = Quantizer(None, sym_act_quant_params)
        self.v_post_act_fake_quantize = Quantizer(None, sym_act_quant_params)

        self.softmax_post_act_fake_quantize = Quantizer(None, a_qconfig)

        if self.sr_ratio > 1:
            self.sr = PreQuantizedLayer(org_module.sr, None, w_qconfig, a_qconfig, sym=True)
            self.norm = org_module.norm 
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        
        # proj
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,3).contiguous()

        # q quant
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
        k = self.k_post_act_fake_quantize(k)
        v = self.v_post_act_fake_quantize(v)

        # attn
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)

        # softmax quant
        attn = self.softmax_post_act_fake_quantize(attn)


        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)

        # proj
        x = self.proj(x)
        

        return x


class QDWConv(QuantizedBlock):
    def __init__(self, org_module: mmseg.models.backbones.mix_transformer.DWConv, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()

        self.dwconv = PreQuantizedLayer(org_module.dwconv, None, w_qconfig, a_qconfig, sym=True)


    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x




class QMLPBlock(QuantizedBlock):
    def __init__(self, org_module: mmseg.models.backbones.mix_transformer.Mlp, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()


        sym_act_quant_params = copy.deepcopy(a_qconfig)
        sym_act_quant_params['symmetric'] = True

        self.fc1 = PreQuantizedLayer(org_module.fc1, None, w_qconfig, a_qconfig, sym=True)
        self.dwconv = QDWConv(org_module.dwconv, w_qconfig, a_qconfig)
        self.act = org_module.act
        self.fc2 =  PreQuantizedLayer(org_module.fc2, None, w_qconfig, a_qconfig, sym=True)


    
    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        
        return x




class QuantMixTransformerBlock(QuantizedBlock):
    '''
    MixTransformerBlock Implementation
    '''

    def __init__(self, org_module: mmseg.models.backbones.mix_transformer.Block, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()


        self.attn = QAttention(org_module.attn, w_qconfig, a_qconfig)

        self.norm1 = org_module.norm1 
        self.norm2 = org_module.norm2

        self.mlp = QMLPBlock(org_module.mlp, w_qconfig, a_qconfig)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class QuantOverlapPatchEmbed(QuantizedBlock):
    '''
    MixTransformer OverlapPatchEmbed Implementation
    '''

    def __init__(self, org_module: mmseg.models.backbones.mix_transformer.OverlapPatchEmbed, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()

        self.proj = PreQuantizedLayer(org_module.proj, None, w_qconfig, a_qconfig, sym=True)
        self.norm = org_module.norm

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W




class QuantASPPWrapper(QuantizedBlock):
    def __init__(self, org_module: mmseg.models.decode_heads.daformer_head.ASPPWrapper, w_qconfig, a_qconfig, qoutput=True):
        super().__init__()
        
        self.image_pool = org_module.image_pool
        self.dilations = org_module.dilations
        self.align_corners = org_module.align_corners
        self.context_layer = org_module.context_layer

        assert(org_module.context_layer is None)

        # QuantizedLayer(org_module.layers[0], org_module.layers[2], w_qconfig, a_qconfig),
        self.bottleneck = QuantizedLayer(org_module.bottleneck.conv, org_module.bottleneck.activate, w_qconfig, a_qconfig)
        
        # Hard coding fit to DAFormer head aspp module
        org_module.aspp_modules[0] = QuantizedLayer(org_module.aspp_modules[0].conv, org_module.aspp_modules[0].activate, w_qconfig, a_qconfig)
       
        for i in range(1, len(org_module.aspp_modules)):
            org_module.aspp_modules[i].depthwise_conv.conv = QuantizedLayer(org_module.aspp_modules[i].depthwise_conv.conv, org_module.aspp_modules[i].depthwise_conv.activate, w_qconfig, a_qconfig)
            org_module.aspp_modules[i].pointwise_conv.conv = QuantizedLayer(org_module.aspp_modules[i].pointwise_conv.conv, org_module.aspp_modules[i].pointwise_conv.activate, w_qconfig, a_qconfig)

        self.aspp_modules = org_module.aspp_modules

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
    _InvertedResidual: _QuantInvertedResidual,
    ####################### For MMSEG Comparatibility
    mmseg.models.backbones.resnet.BasicBlock : QuantBasicBlock,
    mmseg.models.backbones.resnet.Bottleneck : QuantBottleneck,
    mmseg.models.backbones.mix_transformer.Block : QuantMixTransformerBlock,
    mmseg.models.backbones.mix_transformer.OverlapPatchEmbed : QuantOverlapPatchEmbed,
    mmseg.models.decode_heads.daformer_head.ASPPWrapper : QuantASPPWrapper
}


def load_model(config):
    # config['kwargs'] = config.get('kwargs', dict())
    # model = eval(config['type'])(**config['kwargs'])
    # checkpoint = torch.load(config.path, map_location='cpu')
    # if config.type == 'mobilenetv2':
    #     checkpoint = checkpoint['model']
    # model.load_state_dict(checkpoint)
    model = timm.create_model(config['type'], pretrained=True, num_classes=config['kwargs']['num_classes'])
    return model

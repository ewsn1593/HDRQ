import torch


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x






def fake_quantize_per_tensor(x, scale, zero_point, quant_min, quant_max):
    x_int = round_ste(x / scale) #round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant) * scale #(x_quant - zero_point) * scale
    return x_dequant

#######################################################################################
def fake_noise_quantize_per_tensor(x, scale, zero_point, quant_min, quant_max): ###NOTE
    N_BIN = 256
    x_int = round_ste(x / scale) #round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant) * scale #(x_quant - zero_point) * scale


    alpha_max = scale * quant_max
    alpha_min = scale * quant_min
    c1 = x >= alpha_max
    c2 = x <= alpha_min     
    
    with torch.no_grad():                
        diff = (x_dequant - x) / scale 
        sel = diff[torch.logical_not(torch.logical_or(c1, c2))]
        hist = torch.histc(sel, bins=N_BIN, min=-0.5, max=0.5)    
        
        noise_ = torch.multinomial(hist, x.numel(), True) + torch.rand_like(x.view(-1))               
        noise_ = (noise_ / N_BIN - 0.5).view(x.shape)
    return  torch.where(c1, alpha_max, torch.where(c2, alpha_min, x + noise_ * scale)) 
#######################################################################################


def fake_quantize_per_channel_affine(x, scale, zero_point, ch_axis, quant_min, quant_max):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_tensor_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    scale = grad_scale(scale, grad_factor)
    x_int = round_ste(x / scale) #round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant) * scale #(x_quant - zero_point) * scale
    return x_dequant

#######################################################################################
def fake_noise_quantize_learnable_per_tensor_training(x, scale, zero_point, quant_min, quant_max, grad_factor): ###NOTE
    N_BIN = 256
    scale = grad_scale(scale, grad_factor)
    x_int = round_ste(x / scale) #round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant) * scale #(x_quant - zero_point) * scale

    alpha_max = scale * quant_max
    alpha_min = scale * quant_min
    c1 = x >= alpha_max
    c2 = x <= alpha_min     
    
    with torch.no_grad():                
        diff = (x_dequant - x) / scale 
        sel = diff[torch.logical_not(torch.logical_or(c1, c2))]
        hist = torch.histc(sel, bins=N_BIN, min=-0.5, max=0.5)    
        
        noise_ = torch.multinomial(hist, x.numel(), True) + torch.rand_like(x.view(-1))               
        noise_ = (noise_ / N_BIN - 0.5).view(x.shape)
    return  torch.where(c1, alpha_max, torch.where(c2, alpha_min, x + noise_ * scale )) 
#######################################################################################





def fake_quantize_learnable_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_training(x, scale, zero_point, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    scale = grad_scale(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_channel_affine_training(x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)

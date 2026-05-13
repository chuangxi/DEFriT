import torch
import math
import numpy as np
import scipy.io as scio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL.Image import preinit
from sympy import python

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FrFT_forward(obj, frft_order):
    propfield = frft2d(obj, frft_order)
    return propfield


def FrFT_backward(propfield, frft_order):
    obj = frft2d(propfield, -frft_order)
    return obj


def DisFrFT_forward(obj, frft_order):
    propfield = Disfrft2d(obj, frft_order)
    return propfield


def DisFrFT_backward(propfield, frft_order):
    obj = Disfrft2d(propfield, -frft_order)
    return obj


def frft2d(x, a):
    if x.dtype != torch.complex128:
        x = torch.complex(x, torch.zeros_like(x))
    x = frft_dim(x, a)
    x = frft_dim(x.permute(1, 0), a).permute(1, 0)
    return x


def frft_dim(x, a, dim=-1):
    N = x.shape[dim]
    sN = math.sqrt(N)
    a = torch.fmod(a, 4)
    if a == 0:
        return x
    elif a == 2:
        return torch.flip(x, dims=[dim])
    elif a == 1:
        return torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(x), dim=dim)) / sN
    elif a == 3:
        return torch.fft.fftshift(torch.fft.ifft(torch.fft.fftshift(x), dim=dim)) * sN
    else:
        if a > 2:
            a = a - 2
            x = torch.flip(x, dims=[dim])
        if a > 1.5:
            a = a - 1
            x = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(x), dim=dim)) / sN
        if a < 0.5:
            a = a + 1
            x = torch.fft.fftshift(torch.fft.ifft(torch.fft.fftshift(x), dim=dim)) * sN
        alpha = a * torch.pi / 2
        tana2 = torch.tan(alpha / 2)
        sina = torch.sin(alpha)
        f = F.pad(interp_dim(x), (N - 1, N - 1), "constant", 0)
        c_arg = (torch.pi / N / 4 * torch.arange(-2 * N + 2, 2 * N - 1) ** 2).to(x.device)
        chrp_r = torch.cos(tana2 * c_arg)
        chrp_i = -torch.sin(tana2 * c_arg)
        chrp = torch.complex(chrp_r, chrp_i).unsqueeze(0)
        f = chrp * f
        cc = (torch.pi / N / 4) / sina
        c_arg2 = (torch.arange(-4 * N + 4, 4 * N - 3) ** 2).to(x.device)
        ch_r = torch.cos(cc * c_arg2)
        ch_i = torch.sin(cc * c_arg2)
        ch = torch.complex(ch_r, ch_i).unsqueeze(0)
        Faf = fconv_dim(ch, f)
        Faf = Faf[:, 4 * N - 4:8 * N - 7] * torch.sqrt(cc / torch.pi)
        Faf = chrp * Faf
        Faf = Faf[:, N - 1:Faf.shape[dim] - N + 1]
        norm_constant = torch.complex(torch.cos((1 - a) * torch.pi / 4), -torch.sin((1 - a) * torch.pi / 4))
        Faf = norm_constant * Faf[:, ::2]
        return Faf


def interp_dim(x, dim=-1):
    N = x.shape[dim]
    y = torch.zeros_like(x, device=x.device)
    y = F.pad(y, (N - 1, 0), "constant", 0)
    y[:, ::2] = x
    xint = fconv_dim(y, torch.sinc(torch.arange(-2 * N + 3, 2 * N - 2) / 2).unsqueeze(0).to(x.device))
    xint = xint[:, 2 * N - 3:xint.shape[dim] - 2 * N + 3]
    return xint


def fconv_dim(x, y, dim=-1):
    N = x.shape[dim] + y.shape[dim] - 1
    P = 2 ** math.ceil(math.log2(N))
    z = torch.fft.ifft(torch.fft.fft(x, P, dim=dim) * torch.fft.fft(y, P, dim=dim), dim=dim)
    z = z[:, 0:N]
    return z


def Disfrft2d(x, a):
    if x.dtype != torch.complex64:
        x = torch.complex(x, torch.zeros_like(x))
    x = Disfrft(x, a)
    x = Disfrft(x.permute(1, 0), a).permute(1, 0)
    return x


def Disfrft(x, a, dim=-1):
    x = x.to('cuda:0')
    a = a.to('cuda:0')
    device = x.device  # 动态获取输入张量设备
    N = x.shape[dim]

    # 将numpy操作替换为PyTorch GPU操作
    shft = (torch.arange(N, device=device) + (N // 2)) % N

    # 加载预处理矩阵并确保在GPU上
    data = scio.loadmat('./disfrft_matrix_256.mat')
    E = torch.from_numpy(data['E']).to(device).float() # 确保加载时直接到GPU
    E = torch.complex(E, torch.zeros_like(E))  # 直接在GPU上创建复数张量
    # 在GPU上生成所有中间张量
    phase = torch.exp(-1j * torch.pi / 2 * a * torch.arange(N, device=device).view(N, 1))
    indexed_x = x[shft.long() - 1, :]  # 确保索引操作在GPU上

    # 修正矩阵乘法流程
    y = torch.zeros_like(x)
    y[shft.long() - 1, :] = E @ (phase * (E.T @ indexed_x))

    return y


def frft_single(img_torch, frft_orders):
    if img_torch.ndim == 2:  # 处理灰度图（未触发，因为输入是四维）
        img_torch = img_torch.unsqueeze(2).repeat(1, 1, 3)
    else:  # 处理 RGB 图（四维）
        img_torch = img_torch.squeeze(0).permute(1, 2, 0)  # 或使用方法1

    # print(img_torch.shape)  # 输出: torch.Size([256, 256, 3])
    # Separate RGB channels
    r_channel = img_torch[:, :, 0]
    g_channel = img_torch[:, :, 1]
    b_channel = img_torch[:, :, 2]

    # Ensure frft_orders is a tensor
    frft_orders = torch.tensor(frft_orders).to(device)

    # Perform DisFrFT on each channel with corresponding fractional order
    Det_field_r = DisFrFT_forward(r_channel, frft_orders)
    Det_field_g = DisFrFT_forward(g_channel, frft_orders)
    Det_field_b = DisFrFT_forward(b_channel, frft_orders)

    # # Calculate measurements
    # Meas_r = torch.abs(Det_field_r)
    # Meas_g = torch.abs(Det_field_g)
    # Meas_b = torch.abs(Det_field_b)

    # # Ensure measurements are within valid range
    # Meas_r = torch.clamp(Meas_r, 0, 1)
    # Meas_g = torch.clamp(Meas_g, 0, 1)
    # Meas_b = torch.clamp(Meas_b, 0, 1)

    frft_result = torch.stack((Det_field_r, Det_field_g, Det_field_b), dim=-1)

    return frft_result

def frft(img_batch, frft_orders):
    """
    img_batch: shape (B, 3, H, W)
    frft_orders: 单个 float，或长度为3的 list/tuple/tensor
    """
    B = img_batch.shape[0]
    results = []
    
    for i in range(B):
        img = img_batch[i].unsqueeze(0)  # 变成 (1, 3, H, W)
        result = frft_single(img, frft_orders)  # 调用原先的处理函数（改名为 frft_single）
        result = result.permute(2, 0, 1)  # 变成 (3, H, W)，便于堆叠
        results.append(result)
    
    # 拼接回 batch 维度，shape: (B, 3, H, W)
    frft_batch_result = torch.stack(results, dim=0)
    # print(frft_batch_result.shape)
    return frft_batch_result


def ifrft_single(det_fields, frft_orders):
    """
    Perform Inverse Fractional Fourier Transform (iFRFT) on measured fields.

    Parameters:
    - det_fields: Tuple of measured fields for each channel.
    - frft_orders: List of fractional orders for the iFRFT.

    Returns:
    - Obj_hat: Reconstructed image after iFRFT.
    """
    
    if det_fields.ndim == 2:  # 处理灰度图（未触发，因为输入是四维）
        det_fields = det_fields.unsqueeze(2).repeat(1, 1, 3)
    else:  # 处理 RGB 图（四维）
        det_fields = det_fields.squeeze(0).permute(1, 2, 0)  # 或使用方法1
        
    Det_field_r = det_fields[:, :, 0]
    Det_field_g = det_fields[:, :, 1]
    Det_field_b = det_fields[:, :, 2]

    # Ensure frft_orders is a tensor
    frft_orders = torch.tensor(frft_orders).to(device)

    # Perform iFRFT on each channel with corresponding fractional order
    Obj_hat_r = torch.abs(DisFrFT_backward(Det_field_r, frft_orders))
    Obj_hat_g = torch.abs(DisFrFT_backward(Det_field_g, frft_orders))
    Obj_hat_b = torch.abs(DisFrFT_backward(Det_field_b, frft_orders))

    # Clamp values to [0, 1]
    Obj_hat_r = torch.clamp(Obj_hat_r, 0, 1)
    Obj_hat_g = torch.clamp(Obj_hat_g, 0, 1)
    Obj_hat_b = torch.clamp(Obj_hat_b, 0, 1)

    # Stack channels back together
    Obj_hat = torch.stack((Obj_hat_r, Obj_hat_g, Obj_hat_b), dim=-1)

    return Obj_hat

def ifrft(img_batch, frft_orders):
    """
    img_batch: shape (B, 3, H, W)
    frft_orders: 单个 float，或长度为3的 list/tuple/tensor
    """
    B = img_batch.shape[0]
    results = []
    
    for i in range(B):
        img = img_batch[i].unsqueeze(0)  # 变成 (1, 3, H, W)
        result = ifrft_single(img, frft_orders)  # 调用原先的处理函数（改名为 frft_single）
        result = result.permute(2, 0, 1)  # 变成 (3, H, W)，便于堆叠
        results.append(result)
    
    # 拼接回 batch 维度，shape: (B, 3, H, W)
    frft_batch_result = torch.stack(results, dim=0)
    return frft_batch_result

def test():
    x = torch.randn(1, 3, 256, 256).to(device)
    p = 0.12
    pred = frft(x, p)
    pred = ifrft(pred,p)
    print(x.shape, pred.shape)

if __name__ == '__main__':
    test()
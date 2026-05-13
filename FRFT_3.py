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


import torch
import torch.nn.functional as F
import scipy.io as scio
import math

# 判断设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 核心函数: 可导 Discrete FrFT
# -----------------------------
def Disfrft(x, a, dim=-1):
    x = x.to(device)
    a = a.to(device)
    N = x.shape[dim]

    # fft shift index
    shft = (torch.arange(N, device=device) + (N // 2)) % N

    # 加载 MATLAB 矩阵 E
    data = scio.loadmat('./disfrft_matrix_256.mat')
    E = torch.from_numpy(data['E']).to(device).float()
    E = torch.complex(E, torch.zeros_like(E))

    # 相位项（保持 tensor 可导）
    phase = torch.exp(-1j * torch.pi / 2 * a * torch.arange(N, device=device).view(N, 1))

    indexed_x = x[shft.long() - 1, :]  # shift
    y = torch.zeros_like(x, device=device)
    y[shft.long() - 1, :] = E @ (phase * (E.T @ indexed_x))
    return y


def Disfrft2d(x, a):
    if x.dtype != torch.complex64:
        x = torch.complex(x, torch.zeros_like(x))
    x = Disfrft(x, a)
    x = Disfrft(x.permute(1, 0), a).permute(1, 0)
    return x


def DisFrFT_forward(x, a):
    return Disfrft2d(x, a)


def DisFrFT_backward(x, a):
    return Disfrft2d(x, -a)

# --------------------------------
# 单张图像的 FrFT（支持可导 p）
# --------------------------------
def frft_single(x, frft_orders_tensor):
    # x: (1, 3, H, W)
    x = x[0]  # (3, H, W)
    results = []
    for c in range(3):
        x_c = x[c]  # (H, W)
        p_c = frft_orders_tensor[c]  # scalar tensor
        spec_c = DisFrFT_forward(x_c, p_c)
        results.append(spec_c)
    return torch.stack(results, dim=0)  # (3, H, W)


def ifrft_single(x, frft_orders_tensor):
    # x: (1, 3, H, W) complex tensor
    x = x[0]  # (3, H, W)
    results = []
    for c in range(3):
        x_c = x[c]  # (H, W)
        p_c = frft_orders_tensor[c]  # scalar tensor
        rec_c = DisFrFT_backward(x_c, p_c)
        results.append(rec_c)
    return torch.stack(results, dim=0)  # (3, H, W)

# --------------------------------
# Batch 版本：支持 p.shape = (B, 3)
# --------------------------------
def frft(x, p):
    # x: (B, 3, H, W), p: (B, 3)
    B = x.shape[0]
    result = []
    for b in range(B):
        x_b = x[b].unsqueeze(0)  # (1, 3, H, W)
        p_b = p[b]               # (3,) tensor
        result_b = frft_single(x_b, p_b)  # (3, H, W)
        result.append(result_b)
    return torch.stack(result, dim=0)  # (B, 3, H, W)


def ifrft(x, p):
    # x: (B, 3, H, W), complex tensor
    B = x.shape[0]
    result = []
    for b in range(B):
        x_b = x[b].unsqueeze(0)
        p_b = p[b]
        rec_b = ifrft_single(x_b, p_b)  # (3, H, W)
        result.append(rec_b)
    return torch.stack(result, dim=0)

# ------------------
# 测试可导性
# ------------------
def test():
    x = torch.rand(2, 3, 256, 256).to(device)
    x = x.requires_grad_(True)
    p = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.8]], dtype=torch.float32, requires_grad=True).to(device)

    spec = frft(x, p)  # (B, 3, H, W)
    rec = ifrft(spec, p)

    # 可导性测试
    loss = torch.mean(torch.abs(rec.real - x))
    loss.backward()

    print("Loss:", loss.item())
    print("p.grad:", p.grad)
    print("x.grad:", x.grad.mean())

if __name__ == '__main__':
    test()

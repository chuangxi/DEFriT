import torch
import torch.nn as nn
from FRFT_3 import frft, ifrft
import numpy as np
from thop import profile
from DIT import DiT_S_4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 轻量空域微调网络：Residual Refiner
# ---------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.act   = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn2(self.conv2(y))
        return self.act(x + y)

class SpatialRefiner(nn.Module):
    """
    输入通道为 6：cat([x_out_frft(3), x_in(3)])
    输出为 3 通道残差，范围不过度限制，最终由 clamp 截断到 [0,1]
    """
    def __init__(self, in_ch=6, base=48, depth=4):
        super().__init__()
        self.stem = ConvBNAct(in_ch, base, 3, 1, 1)
        self.blocks = nn.Sequential(*[ResBlock(base) for _ in range(depth)])
        self.head = nn.Conv2d(base, 3, 3, 1, 1)  # 残差输出
    def forward(self, x):
        y = self.stem(x)
        y = self.blocks(y)
        res = self.head(y)
        return res

# ---------------------------
# 主网络：FrFT 幅度改 + 空域微调
# ---------------------------
class FrftUNet(nn.Module):
    def __init__(self):
        super(FrftUNet, self).__init__()
        # 频域幅度预测骨干
        self.dit = DiT_S_4(input_size=256)
        # 每图/每通道分数阶 p ∈ [0.4, 0.6]
        self.PNet = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 16, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 1), nn.Sigmoid()  # (B,3,1,1) ∈ (0,1)
        )
        # 空域微调分支
        self.refiner = SpatialRefiner(in_ch=6, base=48, depth=4)

    def forward(self, x):
        B, C, H, W = x.shape

        # ---- 1) 预测分数阶 p：缩放到 [0.4, 0.6]
        p_raw = self.PNet(x).view(B, 3)           # (B,3)
        p = 0.4 + 0.2 * p_raw

        # ---- 2) FrFT 到分数域
        Xp = frft(x, p)                            # complex (B,3,H,W)
        A  = torch.abs(Xp)                         # 幅度
        P  = torch.angle(Xp)                       # 相位

        # ---- 3) 只改幅度谱（相位保持）
        A_pred = self.dit(A)
        real = A_pred * torch.cos(P)
        imag = A_pred * torch.sin(P)
        Xp_mod = torch.complex(real, imag)

        # ---- 4) 逆 FrFT 回空域（频域版输出）
        x_out_frft = ifrft(Xp_mod, p).real
        x_out_frft = torch.clamp(x_out_frft, 0.0, 1.0)

        # ---- 5) 空域残差微调：融合原图与频域输出
        refine_in = torch.cat([x_out_frft, x], dim=1)    # (B,6,H,W)
        residual  = 0.1 * self.refiner(refine_in)
        y_out     = torch.clamp(x_out_frft + residual, 0.0, 1.0)

        # 返回：最终输出、频域输出、中间幅度、残差、p
        return y_out, A_pred,  p

# ---------------------------
# 简单自测 + FLOPs/Params
# ---------------------------
def test():
    x = torch.randn(1, 3, 256, 256).to(device)
    model = FrftUNet().to(device)
    y_out, x_out_frft, A_pred, residual, p = model(x)

    print("Input :", x.shape)
    print("Final :", y_out.shape, " FrFT-only:", x_out_frft.shape, " A_pred:", A_pred.shape, " Residual:", residual.shape)
    print("p in [min,max]:", float(p.min().detach().cpu()), float(p.max().detach().cpu()))

    # thop 统计
    model.eval()
    flops, params = profile(model, (x,))
    print('FLOPs: %.2f M, Params: %.2f M' % (flops / 1e6, params / 1e6))

if __name__ == "__main__":
    test()

import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import sys
import argparse
import time
import dataloader
from model import FrftUNet
import numpy as np
from torch.nn import init
import torch.utils.data as data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from FRFT_3 import frft, ifrft
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def init_weights(net, init_type='xavier_uniform_', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_normal_':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform_':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_normal_':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform_':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    # print('Initialize network with %s' % init_type)
    net.apply(init_func)


def load_pretrained_model(pretrained_path):
    model = FrftUNet()
    pretrained_dict = torch.load(pretrained_path)

    if 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model']

    model.load_state_dict(pretrained_dict, strict=False)
    return model


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化TensorBoard写入器
    writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, "./"))
    model = FrftUNet().cuda()
    init_weights(model, init_type='kaiming_uniform_', gain=1.0)

    # pretrained_nodel_path = r'./snapshots/FrFT-ViT.pth'
    # model = load_pretrained_model(pretrained_nodel_path).cuda()
    
    train_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                               config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path,
                                             config.hazy_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,
                                             num_workers=config.num_workers, pin_memory=True)

    criterion = nn.MSELoss().cuda()
    l1 = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    model.train()

    for epoch in range(config.num_epochs):
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config.num_epochs}",
            dynamic_ncols=True,
            leave=False
        )
    
        for iteration, (img_orig, img_haze) in pbar:
            img_orig = img_orig.cuda(non_blocking=True)  # 作为“干净图/GT”
            img_haze = img_haze.cuda(non_blocking=True)  # 作为“输入图/有雾或水下退化”
    
            # ==== 前向 ====
            clean_image, A_pred, p_pred = model(img_haze)   # 模型会根据 img_haze 预测 p
    
            # ==== 用 “同一个 p_pred” 在 GT 上生成 目标幅度谱 ====
            with torch.no_grad():
                # 重要：阻断梯度，避免 target 通过 p_pred 回传
                p_tgt = p_pred.detach()
                Xp_clean = frft(img_orig, p_tgt)           # 复数谱
                A_tgt = torch.abs(Xp_clean)                # 目标幅度谱（和模型 A_pred 的 p 一致）
    
            # ==== 损失 ====
            loss = (
                0.6 * l1(clean_image, img_orig) +          # 空间复原
                0.2 * l1(A_pred, A_tgt) +                  # 幅度一致（同 p）
                0.2 * (1 - ssim(clean_image, img_orig))    # 结构一致
            )

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

            # 进度条上显示当前 iteration 与 loss
            pbar.set_postfix(iter=iteration + 1, loss=f"{loss.item():.4f}")

            # 你原本的打印/记录逻辑仍可保留（可选）
            if ((iteration + 1) % config.display_iter) == 0:
                # tqdm 已显示，这里可选继续打印
                pass
            if ((iteration + 1) % config.snapshot_iter) == 0:
                writer.add_scalar('Loss/train_iter', loss.item(), epoch * len(train_loader) + iteration)

        # # 每 N 个 epoch 存一版
        # if (epoch + 1) % 20 == 0:
        #     torch.save(model.state_dict(), os.path.join(config.snapshots_folder, f"Epoch_S4_0.45_{epoch + 1}.pth"))

        # === Validation ===
        model.eval()
        with torch.no_grad():
            vbar = tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                desc=f"Val   {epoch+1}/{config.num_epochs}",
                dynamic_ncols=True,
                leave=False
            )
            for iter_val, (img_orig, img_haze) in vbar:
                img_orig = img_orig.cuda(non_blocking=True)
                img_haze = img_haze.cuda(non_blocking=True)

                clean_image, A_pred, p_pred  = model(img_haze)

                torchvision.utils.save_image(
                    torch.cat((img_haze, clean_image, img_orig), 0),
                    os.path.join(config.sample_output_folder, f"{iter_val + 1}.png")
                )
        model.train()

        # 每个 epoch 末保存一次最新权重（会被覆盖）
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "FrFT.pth"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default="mydata3/gt/")  # ground truth image  water/gt/
    parser.add_argument('--hazy_images_path', type=str, default="mydata3/raw/")  # raw image water/raw/
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--grad_clip_norm', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--sample_output_folder', type=str, default="samples/")
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--tensorboard_dir', type=str, default="./tf-logs/")
    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    train(config)

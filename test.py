import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
from tqdm import tqdm 
from model import FrftUNet

def test():
    # 加载预训练模型
    model = FrftUNet().cuda()
    pretrained_path = "./snapshots/FrFT.pth"
    
    pretrained_dict = torch.load(pretrained_path)
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()  # 设置为评估模式

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 根据模型输入尺寸调整
        transforms.ToTensor(),
    ])

    # 创建输出目录
    # input_dir = "input"
    # output_dir = "output"
    input_dir = "/home/t9/桌面/paper2/mydata/SEA-THRU/RAW"
    output_dir = "mydata_output/SEA-THRU"
    os.makedirs(output_dir, exist_ok=True)

    # 遍历测试图片（添加tqdm进度条）
    for img_name in tqdm(os.listdir(input_dir), desc="Processing images", unit="img"):
        if img_name.endswith(".png"):
            # 加载并预处理图像
            img_path = os.path.join(input_dir, img_name)
            img = plt.imread(img_path)[:, :, :3]
            img_tensor = TF.to_tensor(img).unsqueeze(0).cuda()  # 保存为img_tensor变量

            # 模型推理
            with torch.no_grad():  # 禁用梯度计算
                clean_image, _, p = model(img_tensor)
            # print(p)
            # 后处理与保存
            output_image = clean_image.squeeze(0).cpu().permute(1, 2, 0).numpy()  # 转换为HWC格式
            original_img = img_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # 转换为HWC格式
            output_image = (output_image * 255).astype(np.uint8)  # 反归一化
            original_img = (original_img * 255).astype(np.uint8)  # 反归一化

            # 将原始图像张量移回CPU并转换为NumPy数组
            combined_img = np.hstack((original_img, output_image))
            
            save_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".png")
            # Image.fromarray(combined_img).save(save_path)
            Image.fromarray(output_image).save(save_path, format='PNG')

if __name__ == "__main__":
    test()

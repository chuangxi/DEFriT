import os
import sys
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import glob
import random
random.seed(1143)

def populate_by_index(orig_images_path, hazy_images_path):
    """通过索引配对图片，不依赖文件名匹配"""
    # 获取并排序所有图片文件
    orig_files = sorted(glob.glob(os.path.join(orig_images_path, "*")))
    hazy_files = sorted(glob.glob(os.path.join(hazy_images_path, "*")))
    
    # 只保留图片文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    orig_files = [f for f in orig_files if os.path.splitext(f)[1].lower() in image_extensions]
    hazy_files = [f for f in hazy_files if os.path.splitext(f)[1].lower() in image_extensions]
    
    print(f"Found {len(orig_files)} original images, {len(hazy_files)} hazy images")
    
    if len(orig_files) == 0 or len(hazy_files) == 0:
        print("ERROR: No images found!")
        return [], []
    
    # 显示文件名示例
    print("\nFirst 5 original files:")
    for f in orig_files[:5]:
        print(f"  {os.path.basename(f)}")
    
    print("\nFirst 5 hazy files:")
    for f in hazy_files[:5]:
        print(f"  {os.path.basename(f)}")
    
    # 通过索引配对（假设两个文件夹中的图片顺序一致）
    min_len = min(len(orig_files), len(hazy_files))
    all_pairs = [(orig_files[i], hazy_files[i]) for i in range(min_len)]
    
    print(f"\nCreated {len(all_pairs)} pairs by index")
    
    # 随机打乱并分割
    random.shuffle(all_pairs)
    split_idx = int(0.95 * len(all_pairs))
    train_list = all_pairs[:split_idx]
    val_list = all_pairs[split_idx:]
    
    return train_list, val_list

class dehazing_loader(data.Dataset):
    def __init__(self, orig_images_path, hazy_images_path, mode='train'):
        self.train_list, self.val_list = populate_by_index(orig_images_path, hazy_images_path)
        
        if mode == 'train':
            self.data_list = self.train_list
        else:
            self.data_list = self.val_list
        
        print(f"{mode.capitalize()} examples: {len(self.data_list)}")
        
        if len(self.data_list) == 0:
            print("WARNING: No data available!")
    
    def __getitem__(self, index):
        if len(self.data_list) == 0:
            return torch.zeros((3, 256, 256)), torch.zeros((3, 256, 256))
        
        orig_path, hazy_path = self.data_list[index]
        
        try:
            # 读取并转换为tensor
            orig = Image.open(orig_path).convert('RGB')
            hazy = Image.open(hazy_path).convert('RGB')
            
            orig_tensor = TF.to_tensor(orig).float()
            hazy_tensor = TF.to_tensor(hazy).float()
            
            # 裁剪
            orig_tensor = TF.center_crop(orig_tensor, (256, 256))
            hazy_tensor = TF.center_crop(hazy_tensor, (256, 256))
            
        except Exception as e:
            print(f"Error loading {index}: {e}")
            return torch.zeros((3, 256, 256)), torch.zeros((3, 256, 256))
        
        return orig_tensor, hazy_tensor
    
    def __len__(self):
        return len(self.data_list)

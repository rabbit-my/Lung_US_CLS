
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
from autoencoder_256 import AutoEncoderDecoder

class FeatureExtractorDataset(Dataset):
    """优化后的数据集类，假设每个ID文件夹下只有一张图像"""
    def __init__(self, id_list, image_root, mask_root=None, mode='bbox'):
        """
        参数:
            id_list: 唯一ID列表
            image_root: 图像根目录
            mask_root: 掩膜根目录(seg模式需要)
            mode: 'bbox'或'seg'
        """
        self.id_list = id_list
        self.image_root = image_root
        self.mask_root = mask_root
        self.mode = mode
        self.samples = []
        
        # 收集所有样本路径
        for unique_id in self.id_list:
            img_dir = os.path.join(self.image_root, unique_id)
            mask_dir = os.path.join(self.mask_root, unique_id) if mask_root else None
            
            # 检查图像目录是否存在
            if not os.path.exists(img_dir):
                print(f"警告: ID {unique_id} 的图像目录不存在")
                continue
                
            # 获取图像文件(假设每个ID目录下只有一张图像)
            img_files = [f for f in os.listdir(img_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not img_files:
                print(f"警告: ID {unique_id} 的图像目录中没有有效图像")
                continue
                
            img_file = img_files[0]  # 取第一个文件
            img_path = os.path.join(img_dir, img_file)
            
            # 在seg模式下准备掩膜路径
            mask_path = None
            if self.mode == 'seg':
                if not os.path.exists(mask_dir):
                    print(f"警告: ID {unique_id} 的掩膜目录不存在")
                    continue
                    
                # 假设掩膜文件名与图像文件名相同
                mask_path = os.path.join(mask_dir, '1.png')
                if not os.path.exists(mask_path):
                    print(f"警告: 掩膜文件不存在 {mask_path}")
                    continue
                
            self.samples.append((unique_id, img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        unique_id, img_path, mask_path = self.samples[idx]
        
        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"无法加载图像 {img_path}")
            
        # 应用掩膜处理(seg模式)
        if self.mode == 'seg' and mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"无法加载掩膜 {mask_path}")
                
            # 确保掩膜和图像尺寸一致
            if mask.shape != image.shape:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
            # 应用掩膜: 将黑色区域置为0
            image = np.where(mask == 0, 0, image)
        
        # 调整大小到256x256
        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256))
            
        # 归一化并转为tensor
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)  # 添加通道维度
        
        return {
            'image': image,
            'id': unique_id,
            'img_path': img_path
        }

def read_ids_from_excel(excel_path):
    """从Excel读取检查流水号列表"""
    df = pd.read_excel(excel_path)
    ids = df["检查流水号"].astype(str).unique().tolist()
    return [id_.strip() for id_ in ids]  # 去除可能的空格

def extract_features(args, id_list, output_prefix, mode='bbox'):
    """提取特征并保存
    
    参数:
        args: 命令行参数
        id_list: ID列表
        output_prefix: 输出文件前缀
        mode: 'bbox'或'seg'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 准备数据集
    dataset = FeatureExtractorDataset(
        id_list, 
        args.image_root, 
        mask_root=args.mask_root if mode == 'seg' else None,
        mode=mode
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # 加载模型
    model = AutoEncoderDecoder(args.learned_features).to(device)
    model.load_state_dict(torch.load(args.saved_model_path))
    model.eval()
    
    # 提取特征
    features_dict = {}
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            ids = batch['id']
            img_paths = batch['img_path']
            
            _, features = model(images)
            
            for i in range(len(ids)):
                features_dict[ids[i]] = {
                    'features': features[i].cpu().numpy().flatten(),
                    'img_path': img_paths[i]
                }
    
    # 保存为CSV
    features_list = []
    for unique_id, data in features_dict.items():
        row = {
            'id': unique_id,
            'img_path': data['img_path'],
            **{f'feature_{i}': val for i, val in enumerate(data['features'])}
        }
        features_list.append(row)
    
    df = pd.DataFrame(features_list)
    output_csv = os.path.join(args.out_path, mode, f'{output_prefix}_features.csv')
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"{mode.upper()} {output_prefix}特征已保存到: {output_csv}")

def process_fold(args, fold):
    """处理单个fold的数据"""
    print(f"\n{'='*40}")
    print(f"处理 Fold {fold}")
    print(f"{'='*40}")
    
    # 设置当前fold的模型路径
    args.saved_model_path = os.path.join(args.model_root, f"fold{fold}", "autoencoder.pth")
    if not os.path.exists(args.saved_model_path):
        print(f"错误: 模型权重文件不存在 {args.saved_model_path}")
        return
    
    # 设置数据路径
    fold_dir = os.path.join(args.data_root, "fold_split_excel")
    train_excel = os.path.join(fold_dir, f"fold{fold}_train.xlsx")
    test_excel = os.path.join(fold_dir, f"fold{fold}_test.xlsx")
    
    # 读取ID列表
    print("读取训练集ID...")
    train_ids = read_ids_from_excel(train_excel)
    print(f"训练集样本数: {len(train_ids)}")
    
    print("读取测试集ID...")
    test_ids = read_ids_from_excel(test_excel)
    print(f"测试集样本数: {len(test_ids)}")
    
    # 提取并保存bbox特征
    print("\n提取训练集整图特征...")
    extract_features(args, train_ids, f"fold{fold}_train", mode='bbox')
    
    print("\n提取测试集整图特征...")
    extract_features(args, test_ids, f"fold{fold}_test", mode='bbox')
    
    # 提取并保存seg特征
    print("\n提取训练集分割区域特征...")
    extract_features(args, train_ids, f"fold{fold}_train", mode='seg')
    
    print("\n提取测试集分割区域特征...")
    extract_features(args, test_ids, f"fold{fold}_test", mode='seg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='自动编码器特征提取')
    
    # 必需参数
    parser.add_argument("--data_root", type=str, 
                      default="/root/autodl-tmp/lung_data/data",
                      help="数据根目录，包含image/和fold_split_excel/")
    parser.add_argument("--model_root", type=str, 
                      default="/root/data/lungUS_cls/output",
                      help="模型权重根目录")
    parser.add_argument("--out_path", type=str, 
                      default="/root/data/lungUS_cls/pred_autoencoder_fea_csv",
                      help="特征文件输出目录路径")
    
    # 可选参数
    parser.add_argument("--learned_features", type=int, default=1024,
                      help="特征向量维度(默认:1024)")
    parser.add_argument("--batch_size", type=int, default=128,
                      help="批量大小(默认:32)")
    
    args = parser.parse_args()

    # 设置固定路径
    args.image_root = os.path.join(args.data_root, "image")
    args.mask_root = os.path.join(args.data_root, "predicted_masks")
    
    # 创建输出目录结构
    os.makedirs(os.path.join(args.out_path, "bbox"), exist_ok=True)
    os.makedirs(os.path.join(args.out_path, "seg"), exist_ok=True)
    
    # 处理所有5个fold
    for fold in range(1, 6):
        process_fold(args, fold)
    
    print("\n所有特征提取完成！")

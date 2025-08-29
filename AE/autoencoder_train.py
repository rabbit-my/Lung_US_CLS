import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from autoencoder_256 import AutoEncoderDecoder  


import random

def random_augment(image):
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    angle = random.uniform(-15, 15)
    h, w = image.shape
    M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, M_rot, (w, h), flags=cv2.INTER_LINEAR)

    shear_x = random.uniform(-0.1, 0.1)
    shear_y = random.uniform(-0.1, 0.1)
    M_shear = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
    image = cv2.warpAffine(image, M_shear, (w, h), flags=cv2.INTER_LINEAR)

    max_shift = 10
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M_trans, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    return image



class AutoencoderDataset(Dataset):
    def __init__(self, id_list, image_root, mask_root=None, use_mask=False):
        self.id_list = id_list
        self.image_root = image_root
        self.mask_root = mask_root
        self.use_mask = use_mask

        self.samples = []
        for unique_id in self.id_list:
            img_dir = os.path.join(self.image_root, unique_id)
            if self.use_mask and self.mask_root is not None:
                mask_dir = os.path.join(self.mask_root, unique_id)
            else:
                mask_dir = None

            if not os.path.exists(img_dir):
                print(f"Warning: image folder not found for id {unique_id}")
                continue

            img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            for img_path in img_files:
                filename = os.path.basename(img_path)
                if self.use_mask and mask_dir is not None:
                    mask_path = os.path.join(mask_dir, filename)
                    if not os.path.exists(mask_path):
                        print(f"Warning: mask not found for image {img_path}")
                        continue
                else:
                    mask_path = None

                self.samples.append((unique_id, img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        unique_id, img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to load image {img_path}")

        if self.use_mask and mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to load mask {mask_path}")
        else:
            mask = None

        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256))
        if mask is not None and mask.shape != (256, 256):
            mask = cv2.resize(mask, (256, 256))

        # --- 数据增强（仅训练集执行，使用 self.train_mode 区分）---
        if hasattr(self, "train_mode") and self.train_mode:
            image = random_augment(image)


        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).unsqueeze(0)

        sample = {
            'image': image,
            'target': image,
            'id': unique_id
        }

        if self.use_mask and mask is not None:
            mask = mask.astype(np.float32) / 255.0
            mask = torch.tensor(mask).unsqueeze(0)
            sample['mask'] = mask

        return sample




def read_ids_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    ids = df["检查流水号"].astype(str).unique().tolist()
    return ids

def train_epoch(model, device, dataloader, loss_fn, optimizer):
    model.train()
    train_loss = []
    for batch in dataloader:
        images = batch['image'].to(device)
        targets = batch['target'].to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)

def val_epoch(model, device, dataloader, loss_fn):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            outputs, _ = model(images)
            loss = loss_fn(outputs, targets)
            val_loss.append(loss.item())
    return np.mean(val_loss)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/lung_data/data",
                        help="数据根目录，包含 image/ 和 mask/ 和 fold_split_excel/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--learned_features", type=int, default=1024)
    parser.add_argument("--out_path", type=str, default="./output", help="模型和日志保存路径")
    parser.add_argument("--use_mask", action='store_true', help="是否使用mask")
    args = parser.parse_args()

    # Remove the fold argument since we'll iterate through all 5 folds
    image_root = os.path.join(args.data_root, "image")
    mask_root = os.path.join(args.data_root, "mask") if args.use_mask else None

    for fold in range(1, 6):  # Iterate through folds 1 to 5
        print(f"\n{'='*40}")
        print(f"Starting training for Fold {fold}")
        print(f"{'='*40}\n")
        
        # Create fold-specific output directory
        fold_out_path = os.path.join(args.out_path, f"fold{fold}")
        os.makedirs(fold_out_path, exist_ok=True)

        train_excel = os.path.join(args.data_root, "fold_split_excel", f"fold{fold}_train.xlsx")
        val_excel = os.path.join(args.data_root, "fold_split_excel", f"fold{fold}_test.xlsx")

        print("读取训练集ID...")
        train_ids = read_ids_from_excel(train_excel)
        print(f"训练集样本ID数: {len(train_ids)}")

        print("读取验证集ID...")
        val_ids = read_ids_from_excel(val_excel)
        print(f"验证集样本ID数: {len(val_ids)}")

        train_dataset = AutoencoderDataset(train_ids, image_root, mask_root, use_mask=args.use_mask)
        train_dataset.train_mode = True  # 启用增强
        val_dataset = AutoencoderDataset(val_ids, image_root, mask_root, use_mask=args.use_mask)
        val_dataset.train_mode = False  # 验证集不增强

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = AutoEncoderDecoder(args.learned_features).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, device, train_loader, loss_fn, optimizer)
            val_loss = val_epoch(model, device, val_loader, loss_fn)

            print(f"Fold {fold} | Epoch {epoch+1}/{args.epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

            # 保存模型到fold-specific目录
            torch.save(model.state_dict(), os.path.join(fold_out_path, "autoencoder.pth"))

            # 验证阶段保存输入和重建图，保存前5张图片（每个epoch都保存）
            model.eval()
            with torch.no_grad():
                saved_count = 0
                epoch_dir = os.path.join(fold_out_path, f"epoch_{epoch+1}")
                os.makedirs(epoch_dir, exist_ok=True)

                for batch_idx, batch in enumerate(val_loader):
                    images = batch['image'].to(device)
                    outputs, _ = model(images)

                    ids = batch['id']  # 取出当前batch对应的流水号列表
                    for i in range(images.size(0)):
                        if saved_count >= 5:
                            break
                        inp_img = images[i].cpu().numpy()[0] * 255
                        out_img = outputs[i].cpu().numpy()[0] * 255
                        inp_img = inp_img.astype(np.uint8)
                        out_img = out_img.astype(np.uint8)

                        cur_id = ids[i]  # 当前样本的流水号字符串

                        cv2.imwrite(os.path.join(epoch_dir, f"{cur_id}_input.png"), inp_img)
                        cv2.imwrite(os.path.join(epoch_dir, f"{cur_id}_output.png"), out_img)
                        saved_count += 1
                    if saved_count >= 5:
                        break


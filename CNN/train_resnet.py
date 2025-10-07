import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置参数
class Config:
    data_root = '/root/autodl-tmp/lung_data/data'
    image_dir = os.path.join(data_root, 'image')
    excel_dir = os.path.join(data_root, 'fold_split_excel')
    num_folds = 5
    batch_size = 128
    num_epochs = 100
    num_classes = 1  # 二分类
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

# 设置随机种子
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# 训练集数据增强（适合黑白医学图像）
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.RandomVerticalFlip(p=0.5),    # 垂直翻转
    transforms.RandomRotation(15),           # 随机旋转(-15°到15°)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 平移
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 亮度和对比度调整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试集只做基本预处理
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集类
class LungDataset(Dataset):
    def __init__(self, df, label_map, transform=None):
        self.df = df
        self.transform = transform
        self.label_map = label_map
        self.samples = []
        
        # 遍历每个ID，收集所有图像路径和标签
        for _, row in df.iterrows():
            id_folder = os.path.join(Config.image_dir, str(row['检查流水号']))
            if os.path.exists(id_folder):
                label = self.label_map[row['二分类']]
                for img_name in os.listdir(id_folder):
                    img_path = os.path.join(id_folder, img_name)
                    if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float)

# 创建模型
def create_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, Config.num_classes))
    return model.to(Config.device)

# 训练和验证函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_auc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')
        
        # 验证阶段
        val_auc, val_loss = evaluate_model(model, val_loader, criterion)
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), f'best_model.pth')
    
    return best_auc

def evaluate_model(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device).view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(data_loader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return auc, loss

# 主函数
def main():
    # 标签映射
    label_map = {'良性': 0, '恶性': 1}
    
    # 存储每折的结果
    fold_aucs = []
    
    for fold in range(1, Config.num_folds + 1):
        print(f'\n{"="*30}')
        print(f'Fold {fold}/{Config.num_folds}')
        print(f'{"="*30}')
        
        # 加载数据
        train_df = pd.read_excel(os.path.join(Config.excel_dir, f'fold{fold}_train.xlsx'))
        test_df = pd.read_excel(os.path.join(Config.excel_dir, f'fold{fold}_test.xlsx'))
        
        # 创建数据集（训练集使用增强transform，测试集使用基本transform）
        train_dataset = LungDataset(train_df, label_map, train_transform)
        test_dataset = LungDataset(test_df, label_map, test_transform)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=Config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=Config.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 创建模型
        model = create_model()
        
        # 损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # 训练模型
        auc = train_model(
            model, 
            train_loader, 
            test_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            Config.num_epochs
        )
        
        fold_aucs.append(auc)
        print(f'Fold {fold} AUC: {auc:.4f}')
        
        # 释放内存
        del model
        torch.cuda.empty_cache()
    
    # 输出结果
    print('\nFinal Results:')
    for fold, auc in enumerate(fold_aucs, 1):
        print(f'Fold {fold} AUC: {auc:.4f}')
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f'\nAverage AUC: {mean_auc:.4f} ± {std_auc:.4f}')

if __name__ == '__main__':
    main()

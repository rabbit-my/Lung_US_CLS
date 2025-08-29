

import os
import pandas as pd
import SimpleITK as sitk
import six
import logging
import numpy as np
import radiomics
from radiomics import featureextractor
import traceback

# 配置日志
logger = radiomics.logger
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='/root/autodl-tmp/lung_data/data/radiomics_predicted_seg_log.txt', mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 基础路径配置
BASE_PATH = '/root/autodl-tmp/lung_data/data'
EXCEL_PATH = os.path.join(BASE_PATH, 'fold_split_excel', 'lung_data.xlsx')
IMG_BASE = os.path.join(BASE_PATH, 'image')
MASK_BASE = os.path.join(BASE_PATH, 'predicted_masks')
OUTPUT_BASE = os.path.join(BASE_PATH, 'radiomics_features_predicted_seg')  # 特征输出目录

# 创建输出目录
os.makedirs(OUTPUT_BASE, exist_ok=True)

def convert_to_grayscale(image):
    """将图像转换为灰度图（处理RGB向量像素类型）"""
    try:
        # 检查是否为多通道图像（如RGB）
        if image.GetNumberOfComponentsPerPixel() > 1:
            # 方法1：使用VectorIndexSelectionCast选择第一个通道
            image = sitk.VectorIndexSelectionCast(image, 0)
            logger.info(f"成功将图像转换为灰度图，通道数: {image.GetNumberOfComponentsPerPixel()}")
        return image
    except Exception as e:
        logger.error(f"图像灰度转换失败: {str(e)}")
        raise

def process_patient(patient_id):
    """处理单个患者的图像和掩膜"""
    # 构建患者特定路径
    img_dir = os.path.join(IMG_BASE, patient_id)
    mask_dir = os.path.join(MASK_BASE, patient_id)
    output_dir = os.path.join(OUTPUT_BASE, patient_id)
    
    # 创建患者输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取患者的所有图像文件
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    
    if not image_files:
        logger.warning(f"患者 {patient_id} 没有找到图像文件")
        return
    
    # 初始化特征提取器
    settings = {
        'label': 255,
        'binWidth': 25,
        'kernelRadius': 1,
        'maskedkernel': True,
        'verbose': True
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()

    all_features = []
    
    for img_file in image_files:
        try:
            # 构建完整文件路径
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file)  # 假设同名
            
            # 检查文件是否存在
            if not os.path.exists(mask_path):
                logger.warning(f"掩膜文件不存在: {mask_path}")
                continue
            
            # 读取图像和掩膜
            image = sitk.ReadImage(img_path)
            mask = sitk.ReadImage(mask_path)
            
            # 转换为灰度图
            image = convert_to_grayscale(image)
            mask = convert_to_grayscale(mask)  # 确保掩膜也是单通道
            
            # 验证图像和掩膜尺寸匹配
            if image.GetSize() != mask.GetSize():
                logger.warning(f"图像和掩膜尺寸不匹配: {image.GetSize()} vs {mask.GetSize()}")
            
            # 原始图像特征提取
            result = extractor.execute(image, mask)
            
            # 准备特征数据
            feature_dict = {'patient_id': patient_id, 'image': img_file}
            for key, val in six.iteritems(result):
                if not isinstance(val, sitk.Image):  # 忽略图像类型特征
                    feature_dict[key] = val
            all_features.append(feature_dict)
            print(f"成功处理原始图像: {patient_id}/{img_file}")
            
        except Exception as e:
            logger.error(f"处理失败 {patient_id}/{img_file}: {str(e)}")
            traceback.print_exc()
    
    # 保存患者所有图像的特征
    if all_features:
        df = pd.DataFrame(all_features)
        output_file = os.path.join(output_dir, f'{patient_id}_features.csv')
        df.to_csv(output_file, index=False)
        print(f"保存特征到: {output_file}")

def main():
    # 读取Excel获取患者ID列表
    try:
        df = pd.read_excel(EXCEL_PATH)
        patient_ids = df['检查流水号'].astype(str).unique().tolist()
        print(f"找到 {len(patient_ids)} 个患者")
    except Exception as e:
        logger.error(f"读取Excel失败: {str(e)}")
        return
    
    # 处理每个患者
    for i, patient_id in enumerate(patient_ids):
        print(f"\n处理患者 {i+1}/{len(patient_ids)}: {patient_id}")
        try:
            process_patient(patient_id)
        except Exception as e:
            logger.error(f"患者处理失败 {patient_id}: {str(e)}")
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()

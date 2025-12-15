# Credit: This code is modified from the original code {https://github.com/PathologyFoundation/plip/blob/main/reproducibility/generate_validation_datasets}

# =============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
import sys, os, platform, copy, shutil
opj = os.path.join
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFile
import shutil
from functools import partial
import re

import warnings
warnings.filterwarnings("ignore")
import multiprocess as mp
ImageFile.LOAD_TRUNCATED_IMAGES = True

seed = 1
import random
random.seed(seed)

def process_images_in_parallel(image_paths, num_workers=4):
    """并行处理图像调整大小"""
    # Create a pool of workers
    pool = mp.Pool(num_workers)
    
    # Use partial to pass the output size to the resize function
    resizeimg_func = partial(resizeimg)
    
    # Map the resize function to the list of image paths
    pool.map(resizeimg_func, image_paths)
    
    # Close the pool and wait for all workers to finish
    pool.close()
    pool.join()

def resizeimg(fp):
    """调整图像大小为224x224像素"""
    pbar.update(1)
    newsize = 224
    img = Image.open(fp)
    filename = os.path.basename(fp)
    
    if img.size[0] != img.size[1]:
        width, height = img.size
        min_dimension = min(width, height)  # Determine the smallest dimension
        scale_factor = newsize / min_dimension  # Calculate the scale factor needed to make the smallest dimension 224
        # Calculate the new size of the image
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height))  # Resize the image using the calculated size
        # center crop
        left = (new_width - newsize) // 2  # Calculate the coordinates to crop the center of the image
        top = (new_height - newsize) // 2
        right = left + newsize
        bottom = top + newsize
        img_resize = img.crop((left, top, right, bottom))  # Crop the image using the calculated coordinates
    else:
        img_resize = img.resize((newsize, newsize))
    
    img_resize.save(fp)

def parse_mnist_filename(filename):
    """解析MNIST文件名格式：子数据集名_在子数据集中的编号_真实值标签"""
    # 移除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]
    
    # 使用正则表达式匹配格式：子数据集名_编号_标签
    pattern = r'^(.+)_(\d+)_(\d+)$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        subset_name = match.group(1)  # 子数据集名
        index_in_subset = int(match.group(2))  # 在子数据集中的编号
        true_label = int(match.group(3))  # 真实值标签
        return subset_name, index_in_subset, true_label
    else:
        raise ValueError(f"文件名格式不正确: {filename}")

def process_MNIST(root_dir, seed=None, train_ratio=0.7):
    """处理MNIST数据集"""
    
    def prompt_engineering(text=''):
        """生成文本描述"""
        prompt = 'A handwritten digit image of number [].'.replace('[]', text)
        return prompt

    # 收集所有图像文件
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(opj(root, file))
    
    print(f"找到 {len(image_paths)} 个图像文件")
    
    # 解析文件名并创建DataFrame
    data_list = []
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        try:
            subset_name, index_in_subset, true_label = parse_mnist_filename(filename)
            data_list.append({
                'image': img_path,
                'subset_name': subset_name,
                'index_in_subset': index_in_subset,
                'label': true_label,
                'label_text': str(true_label)
            })
        except ValueError as e:
            print(f"跳过文件 {filename}: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    print(f"成功解析 {len(df)} 个文件")
    
    # 按标签分层采样，确保每个数字类别在训练集和测试集中都有代表性
    df_train, df_test = train_test_split(
        df, 
        test_size=1-train_ratio, 
        stratify=df['label'], 
        random_state=seed
    )
    
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    print(f"训练集: {len(df_train)} 个样本")
    print(f"测试集: {len(df_test)} 个样本")
    
    # 显示每个类别的分布
    print("\n训练集类别分布:")
    print(df_train['label'].value_counts().sort_index())
    print("\n测试集类别分布:")
    print(df_test['label'].value_counts().sort_index())
    
    def process_csv(df_in):
        """为每个样本生成文本描述"""
        label_texts = [str(i) for i in range(10)]  # 0-9的数字
        df_all = pd.DataFrame()
        for digit in label_texts:
            df_subtype = df_in.loc[df_in['label_text'] == digit]
            if len(df_subtype) > 0:
                style = 4
                df_subtype[f'text_style_{style}'] = prompt_engineering(digit)
                df_all = pd.concat([df_all, df_subtype], axis=0)
        df_all = df_all.reset_index(drop=True)
        return df_all
    
    train = process_csv(df_train)
    test = process_csv(df_test)
    
    return train, test

if __name__ == '__main__':
    # 检查当前工作目录
    cwd = os.getcwd()
    assert cwd.endswith('mnist'), f"Please make sure this script is in main 'mnist' dataset directory and run it from the 'mnist' directory. Current working directory is: {cwd}"
    
    print(f"当前工作目录: {cwd}")
    
    # 处理MNIST数据集
    df_train, df_test = process_MNIST(cwd, seed=seed, train_ratio=0.7)
    
    # 创建训练集和测试集目录
    train_path = os.path.join(os.getcwd(), 'images', 'train')
    test_path = os.path.join(os.getcwd(), 'images', 'test')
    
    # 确保目录存在
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # 为每个数字类别创建子目录
    for digit in range(10):
        os.makedirs(os.path.join(train_path, str(digit)), exist_ok=True)
        os.makedirs(os.path.join(test_path, str(digit)), exist_ok=True)
    
    # 复制训练集图像到对应类别文件夹
    print('复制训练集图像到对应类别文件夹...')
    for i in tqdm(range(len(df_train)), desc="复制训练集"):
        path = df_train.loc[i, 'image']
        label = df_train.loc[i, 'label_text']
        target_path = os.path.join(train_path, label, os.path.basename(path))
        shutil.copy(path, target_path)
    
    # 复制测试集图像到对应类别文件夹
    print('复制测试集图像到对应类别文件夹...')
    for i in tqdm(range(len(df_test)), desc="复制测试集"):
        path = df_test.loc[i, 'image']
        label = df_test.loc[i, 'label_text']
        target_path = os.path.join(test_path, label, os.path.basename(path))
        shutil.copy(path, target_path)
    
    # 收集所有图像路径用于调整大小
    paths = []
    for root, dirs, files in os.walk(opj(cwd, 'images')):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(opj(root, file))
    
    # 并行调整图像大小
    if paths:
        num_cpus = mp.cpu_count() // 2
        pbar = tqdm(total=len(paths))
        pbar.set_description('调整图像大小')
        process_images_in_parallel(paths, num_workers=num_cpus)
        pbar.close()
    
    print('MNIST数据集预处理完成！')
    print(f'训练集图像保存在: {train_path}')
    print(f'测试集图像保存在: {test_path}')
    print('每个集合都按数字0-9分为10个类别子集') 
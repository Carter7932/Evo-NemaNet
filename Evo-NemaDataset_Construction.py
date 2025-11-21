

"""
数据集平衡与重组算法
功能:
1. 收集所有Train和Val数据
2. 通过数据增强将所有类别扩充到最大类别的样本数量
3. 重新按8:2比例划分Train/Val数据集
4. 保存到新的路径
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict


# =============================
# 配置类
# =============================
@dataclass
class DataBalanceConfig:
    # 输入路径
    TRAIN_DIR: str = "I-Nema_Dataset\Train"
    VAL_DIR: str = "I-Nema_Dataset\Val"
    
    # 输出路径
    OUTPUT_TRAIN_DIR: str = "Evo-Nema_Dataset\Train"
    OUTPUT_VAL_DIR: str = "Evo-Nema_Dataset\Val"
    
    # 数据划分比例
    TRAIN_RATIO: float = 0.8
    
    # 随机种子
    SEED: int = 42
    
    # 图片保存质量
    SAVE_QUALITY: int = 95
    
    # 增强参数
    MIN_AUGMENTATIONS: int = 1
    MAX_AUGMENTATIONS: int = 3
    
    # 旋转参数
    ROTATION_RANGE: tuple = (-15, 15)
    
    # 亮度/对比度/饱和度调整范围
    BRIGHTNESS_RANGE: tuple = (0.7, 1.3)
    CONTRAST_RANGE: tuple = (0.7, 1.3)
    SATURATION_RANGE: tuple = (0.7, 1.3)
    
    # 高斯模糊参数
    BLUR_RADIUS_RANGE: tuple = (0.5, 2.0)
    
    # 随机裁剪比例范围
    CROP_RATIO_RANGE: tuple = (0.8, 1.0)
    
    # 色相调整范围
    HUE_SHIFT_RANGE: tuple = (-10, 10)
    
    # 支持的图片格式
    SUPPORTED_FORMATS: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    def ensure_dirs(self):
        """确保输出目录存在"""
        os.makedirs(self.OUTPUT_TRAIN_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_VAL_DIR, exist_ok=True)
        print(f"创建输出目录: {self.OUTPUT_TRAIN_DIR}, {self.OUTPUT_VAL_DIR}")
        
    def set_random_seed(self):
        """设置随机种子"""
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        print(f"设置随机种子: {self.SEED}")
        
    def print_config(self):
        """打印配置信息"""
        print("[数据平衡配置]")
        for key, value in asdict(self).items():
            print(f"  {key}: {value}")


class DataAugmentor:
    """传统图像数据增强器"""
    
    def __init__(self, config: DataBalanceConfig):
        self.config = config
        self.augmentation_methods = [
            self.horizontal_flip,
            self.vertical_flip,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.brightness_adjust,
            self.contrast_adjust,
            self.saturation_adjust,
            self.gaussian_blur,
            self.random_crop_resize,
            self.color_jitter,
            self.random_rotation,
        ]
    
    def horizontal_flip(self, image: Image.Image) -> Image.Image:
        """水平翻转"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def vertical_flip(self, image: Image.Image) -> Image.Image:
        """垂直翻转"""
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def rotate_90(self, image: Image.Image) -> Image.Image:
        """顺时针旋转90度"""
        return image.transpose(Image.ROTATE_270)
    
    def rotate_180(self, image: Image.Image) -> Image.Image:
        """旋转180度"""
        return image.transpose(Image.ROTATE_180)
    
    def rotate_270(self, image: Image.Image) -> Image.Image:
        """顺时针旋转270度"""
        return image.transpose(Image.ROTATE_90)
    
    def brightness_adjust(self, image: Image.Image) -> Image.Image:
        """亮度调整"""
        factor = random.uniform(*self.config.BRIGHTNESS_RANGE)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def contrast_adjust(self, image: Image.Image) -> Image.Image:
        """对比度调整"""
        factor = random.uniform(*self.config.CONTRAST_RANGE)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def saturation_adjust(self, image: Image.Image) -> Image.Image:
        """饱和度调整"""
        factor = random.uniform(*self.config.SATURATION_RANGE)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def gaussian_blur(self, image: Image.Image) -> Image.Image:
        """高斯模糊"""
        radius = random.uniform(*self.config.BLUR_RADIUS_RANGE)
        return image.filter(ImageFilter.GaussianBlur(radius))
    
    def random_crop_resize(self, image: Image.Image) -> Image.Image:
        """随机裁剪并调整大小"""
        w, h = image.size
        crop_ratio = random.uniform(*self.config.CROP_RATIO_RANGE)
        
        new_w = int(w * crop_ratio)
        new_h = int(h * crop_ratio)
        
        if new_w <= 0 or new_h <= 0:
            return image
            
        left = random.randint(0, max(0, w - new_w))
        top = random.randint(0, max(0, h - new_h))
        right = min(left + new_w, w)
        bottom = min(top + new_h, h)
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((w, h), Image.BILINEAR)
    
    def color_jitter(self, image: Image.Image) -> Image.Image:
        """颜色抖动"""
        try:
            # 随机调整色相
            arr = np.array(image)
            if len(arr.shape) == 3 and arr.shape[2] == 3:
                hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
                hue_shift = random.randint(*self.config.HUE_SHIFT_RANGE)
                hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
                arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                return Image.fromarray(arr.astype(np.uint8))
        except Exception as e:
            print(f"颜色抖动失败: {e}")
        return image
    
    def random_rotation(self, image: Image.Image) -> Image.Image:
        """小角度随机旋转"""
        angle = random.uniform(*self.config.ROTATION_RANGE)
        return image.rotate(angle, fillcolor=(128, 128, 128), expand=False)
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """随机应用一种或多种增强方法"""
        # 随机选择指定数量的增强方法
        max_methods = min(len(self.augmentation_methods), self.config.MAX_AUGMENTATIONS)
        min_methods = min(self.config.MIN_AUGMENTATIONS, max_methods)
        
        num_augmentations = random.randint(min_methods, max_methods)
        selected_methods = random.sample(self.augmentation_methods, num_augmentations)
        
        augmented_image = image.copy()
        for method in selected_methods:
            try:
                augmented_image = method(augmented_image)
            except Exception as e:
                print(f"增强方法失败: {method.__name__}, 错误: {e}")
                continue
        
        return augmented_image


class DatasetBalancer:
    """数据集平衡器"""
    
    def __init__(self, config: DataBalanceConfig):
        self.config = config
        self.train_dir = Path(config.TRAIN_DIR)
        self.val_dir = Path(config.VAL_DIR)
        self.output_train_dir = Path(config.OUTPUT_TRAIN_DIR)
        self.output_val_dir = Path(config.OUTPUT_VAL_DIR)
        self.augmentor = DataAugmentor(config)
        
        # 设置随机种子和创建目录
        self.config.set_random_seed()
        self.config.ensure_dirs()
    
    def collect_all_data(self) -> Dict[str, List[Path]]:
        """收集所有Train和Val数据"""
        print("正在收集所有数据...")
        all_data = defaultdict(list)
        
        # 收集训练数据
        if self.train_dir.exists():
            print(f"扫描训练目录: {self.train_dir}")
            for class_dir in self.train_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_path in class_dir.glob("*"):
                        if img_path.suffix.lower() in self.config.SUPPORTED_FORMATS:
                            all_data[class_name].append(img_path)
        else:
            print(f"警告: 训练目录不存在: {self.train_dir}")
        
        # 收集验证数据
        if self.val_dir.exists():
            print(f"扫描验证目录: {self.val_dir}")
            for class_dir in self.val_dir.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    for img_path in class_dir.glob("*"):
                        if img_path.suffix.lower() in self.config.SUPPORTED_FORMATS:
                            all_data[class_name].append(img_path)
        else:
            print(f"警告: 验证目录不存在: {self.val_dir}")
        
        return dict(all_data)
    
    def analyze_distribution(self, all_data: Dict[str, List[Path]]) -> Tuple[int, Dict[str, int]]:
        """分析数据分布并找到最大样本数"""
        class_counts = {class_name: len(paths) for class_name, paths in all_data.items()}
        max_samples = max(class_counts.values()) if class_counts else 0
        
        print("\n原始数据分布:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} 张图片")
        print(f"最大类别样本数: {max_samples}")
        
        return max_samples, class_counts
    
    def balance_class_data(self, class_name: str, image_paths: List[Path], target_count: int) -> List[Image.Image]:
        """将单个类别的数据平衡到目标数量"""
        print(f"正在平衡类别 '{class_name}': {len(image_paths)} → {target_count}")
        
        balanced_images = []
        
        # 首先加载所有原始图片
        original_images = []
        for img_path in tqdm(image_paths, desc=f"加载 {class_name}", leave=False):
            try:
                img = Image.open(img_path).convert('RGB')
                original_images.append(img)
                balanced_images.append(img.copy())
            except Exception as e:
                print(f"无法加载图片 {img_path}: {e}")
        
        if not original_images:
            print(f"警告: 类别 {class_name} 没有有效图片")
            return balanced_images
        
        # 如果需要增强
        if len(original_images) < target_count:
            need_augment = target_count - len(original_images)
            print(f"  需要增强 {need_augment} 张图片")
            
            # 循环增强直到达到目标数量
            for i in tqdm(range(need_augment), desc=f"增强 {class_name}", leave=False):
                # 随机选择一张原始图片进行增强
                source_img = random.choice(original_images)
                augmented_img = self.augmentor.augment_image(source_img)
                balanced_images.append(augmented_img)
        
        return balanced_images[:target_count]  # 确保不超过目标数量
    
    def balance_all_data(self, all_data: Dict[str, List[Path]], max_samples: int) -> Dict[str, List[Image.Image]]:
        """平衡所有类别的数据"""
        print(f"\n开始数据平衡，目标每类 {max_samples} 张图片...")
        balanced_data = {}
        
        for class_name, image_paths in all_data.items():
            balanced_images = self.balance_class_data(class_name, image_paths, max_samples)
            balanced_data[class_name] = balanced_images
        
        return balanced_data
    
    def split_and_save_data(self, balanced_data: Dict[str, List[Image.Image]]):
        """按指定比例划分数据并保存"""
        print(f"\n开始按 {self.config.TRAIN_RATIO:.0%}:{1-self.config.TRAIN_RATIO:.0%} 比例划分并保存数据...")
        
        for class_name, images in balanced_data.items():
            if not images:
                print(f"警告: 类别 {class_name} 没有图片，跳过")
                continue
                
            # 打乱数据
            images_copy = images.copy()
            random.shuffle(images_copy)
            
            # 计算划分点
            total_count = len(images_copy)
            train_count = int(total_count * self.config.TRAIN_RATIO)
            val_count = total_count - train_count
            
            train_images = images_copy[:train_count]
            val_images = images_copy[train_count:]
            
            print(f"类别 '{class_name}': {total_count} 总计 → {train_count} 训练 + {val_count} 验证")
            
            # 创建类别目录
            train_class_dir = self.output_train_dir / class_name
            val_class_dir = self.output_val_dir / class_name
            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存训练图片
            for i, img in enumerate(tqdm(train_images, desc=f"保存训练 {class_name}", leave=False)):
                try:
                    img_path = train_class_dir / f"{class_name}_train_{i:06d}.jpg"
                    img.save(img_path, 'JPEG', quality=self.config.SAVE_QUALITY)
                except Exception as e:
                    print(f"保存训练图片失败: {e}")
            
            # 保存验证图片
            for i, img in enumerate(tqdm(val_images, desc=f"保存验证 {class_name}", leave=False)):
                try:
                    img_path = val_class_dir / f"{class_name}_val_{i:06d}.jpg"
                    img.save(img_path, 'JPEG', quality=self.config.SAVE_QUALITY)
                except Exception as e:
                    print(f"保存验证图片失败: {e}")
    
    def run(self):
        """运行完整的数据平衡流程"""
        print("=" * 60)
        print("数据集平衡与重组算法")
        print("=" * 60)
        
        # 打印配置
        self.config.print_config()
        print()
        
        # 第1步：收集所有数据
        all_data = self.collect_all_data()
        if not all_data:
            print("错误：没有找到任何数据！")
            return
        
        # 第2步：分析分布
        max_samples, class_counts = self.analyze_distribution(all_data)
        if max_samples == 0:
            print("错误：所有类别都没有样本！")
            return
        
        # 第3步：平衡数据
        balanced_data = self.balance_all_data(all_data, max_samples)
        
        # 第4步：划分并保存
        self.split_and_save_data(balanced_data)
        
        # 总结
        total_original = sum(class_counts.values())
        total_balanced = len(balanced_data) * max_samples
        print("\n" + "=" * 60)
        print("数据平衡完成！")
        print(f"原始总数据量: {total_original}")
        print(f"平衡后总数据量: {total_balanced}")
        print(f"训练集路径: {self.output_train_dir}")
        print(f"验证集路径: {self.output_val_dir}")
        print("=" * 60)


def main():
    """主函数 - 使用Config配置"""
    # 创建配置
    config = DataBalanceConfig(
        TRAIN_DIR="Train_New",              # 原始训练数据路径
        VAL_DIR="Val_New",                  # 原始验证数据路径
        OUTPUT_TRAIN_DIR="Balanced_Train_Filtered",  # 新训练数据保存路径
        OUTPUT_VAL_DIR="Balanced_Val_Filtered",      # 新验证数据保存路径
        TRAIN_RATIO=0.8,               # 训练集比例
        SEED=42,                       # 随机种子
        SAVE_QUALITY=95,               # 图片保存质量
        MIN_AUGMENTATIONS=1,           # 最少增强方法数
        MAX_AUGMENTATIONS=3,           # 最多增强方法数
        ROTATION_RANGE=(-15, 15),      # 旋转角度范围
        BRIGHTNESS_RANGE=(0.7, 1.3),  # 亮度调整范围
        CONTRAST_RANGE=(0.7, 1.3),    # 对比度调整范围
        SATURATION_RANGE=(0.7, 1.3),  # 饱和度调整范围
        BLUR_RADIUS_RANGE=(0.5, 2.0), # 模糊半径范围
        CROP_RATIO_RANGE=(0.8, 1.0),  # 裁剪比例范围
        HUE_SHIFT_RANGE=(-10, 10),    # 色相调整范围
    )
    
    # 创建平衡器并运行
    balancer = DatasetBalancer(config)
    balancer.run()


# 使用示例
if __name__ == "__main__":
    # 方式1：使用默认配置运行
    main()
    
    # 方式2：自定义配置运行
    """
    custom_config = DataBalanceConfig(
        TRAIN_DIR="Original_Train",
        VAL_DIR="Original_Val", 
        OUTPUT_TRAIN_DIR="Enhanced_Train",
        OUTPUT_VAL_DIR="Enhanced_Val",
        TRAIN_RATIO=0.75,  # 75%训练，25%验证
        SEED=123,
        SAVE_QUALITY=90,
        MIN_AUGMENTATIONS=2,  # 至少使用2种增强方法
        MAX_AUGMENTATIONS=4,  # 最多使用4种增强方法
        BRIGHTNESS_RANGE=(0.6, 1.4),  # 更大的亮度调整范围
        ROTATION_RANGE=(-20, 20),     # 更大的旋转角度范围
    )
    
    balancer = DatasetBalancer(custom_config)
    balancer.run()
    """
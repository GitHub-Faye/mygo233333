import os  # 导入操作系统模块
import cv2  # 导入OpenCV计算机视觉库
import numpy as np  # 导入NumPy数值计算库
import shutil  # 导入文件操作模块
import sys  # 导入系统模块
import argparse  # 导入参数解析模块
import zipfile  # 导入压缩文件处理模块
import requests  # 导入HTTP请求模块
from tqdm import tqdm  # 导入进度条模块
from pathlib import Path  # 导入路径处理模块


def download_file(url, dest_path):
    """
    从指定URL下载文件到目标路径
    
    参数:
        url (str): 下载链接
        dest_path (str): 目标路径
    
    返回:
        bool: 下载是否成功
    """
    try:
        # 如果文件已存在，先删除它，以避免断点续传问题
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        print(f"开始下载: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 如果请求不成功则抛出异常
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"下载 {os.path.basename(dest_path)}")
        
        with open(dest_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        
        if total_size != 0 and progress_bar.n != total_size:
            print("错误: 下载未完成")
            # 删除不完整的文件
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False
            
        # 检查下载的文件是否是有效的ZIP文件
        if dest_path.endswith('.zip') and not zipfile.is_zipfile(dest_path):
            print(f"错误: 下载的文件 {dest_path} 不是有效的ZIP文件")
            os.remove(dest_path)
            return False
            
        return True
    except Exception as e:
        print(f"下载错误: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def extract_zip(zip_path, extract_path):
    """
    解压ZIP文件到指定路径
    
    参数:
        zip_path (str): ZIP文件路径
        extract_path (str): 解压目标路径
    
    返回:
        bool: 解压是否成功
    """
    try:
        # 先检查文件是否是有效的ZIP文件
        if not zipfile.is_zipfile(zip_path):
            print(f"错误: {zip_path} 不是有效的ZIP文件，可能下载不完整或损坏")
            # 删除可能损坏的文件
            os.remove(zip_path)
            return False
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取压缩文件中的文件总数
            file_count = len(zip_ref.namelist())
            print(f"正在解压 {os.path.basename(zip_path)}...")
            
            # 创建一个进度条
            with tqdm(total=file_count, desc="解压文件") as pbar:
                for file in zip_ref.namelist():
                    zip_ref.extract(file, extract_path)
                    pbar.update(1)
        return True
    except Exception as e:
        print(f"解压错误: {e}")
        return False


def download_widerface_from_huggingface(output_dir):
    """
    从HuggingFace下载WiderFace数据集
    
    参数:
        output_dir (str): 输出目录
    
    返回:
        str: 数据集根目录路径，如果下载失败则返回None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # WiderFace 数据集下载链接
    # HuggingFace直接链接
    repo = "https://huggingface.co/datasets/wider_face/resolve/main/data"
    train_url = f"{repo}/WIDER_train.zip"
    val_images_url = f"{repo}/WIDER_val.zip"
    test_images_url = f"{repo}/WIDER_test.zip"
    annot_url = f"{repo}/wider_face_split.zip"
    
    # 备用链接 - 官方镜像
    backup_repo = "http://shuoyang1213.me/WIDERFACE"
    backup_train_url = f"{backup_repo}/WIDER_train.zip"
    backup_val_images_url = f"{backup_repo}/WIDER_val.zip"
    backup_test_images_url = f"{backup_repo}/WIDER_test.zip"
    backup_annot_url = f"{backup_repo}/wider_face_split.zip"
    
    # 下载文件路径
    train_zip = os.path.join(output_dir, "WIDER_train.zip")
    val_images_zip = os.path.join(output_dir, "WIDER_val.zip")
    test_images_zip = os.path.join(output_dir, "WIDER_test.zip")
    annot_zip = os.path.join(output_dir, "wider_face_split.zip")
    
    # 下载数据集文件
    print("开始下载WiderFace数据集...")
    
    # 下载标签和分割文件
    if not os.path.exists(annot_zip) or not zipfile.is_zipfile(annot_zip):
        print(f"下载数据集标注文件...")
        if not download_file(annot_url, annot_zip):
            print("尝试备用链接...")
            if not download_file(backup_annot_url, annot_zip):
                print("尝试GitHub备用链接...")
                github_backup_url = "https://github.com/ultralytics/yolov5face/releases/download/v1.0/wider_face_split.zip"
                if not download_file(github_backup_url, annot_zip):
                    return None
    else:
        print(f"数据集标注文件已存在: {annot_zip}")
    
    # 下载训练集图像
    if not os.path.exists(train_zip) or not zipfile.is_zipfile(train_zip):
        print(f"下载训练集图像...")
        if not download_file(train_url, train_zip):
            print("尝试备用链接...")
            if not download_file(backup_train_url, train_zip):
                print("尝试GitHub备用链接...")
                github_backup_url = "https://github.com/ultralytics/yolov5face/releases/download/v1.0/WIDER_train.zip"
                if not download_file(github_backup_url, train_zip):
                    return None
    else:
        print(f"训练集图像文件已存在: {train_zip}")
    
    # 下载验证集图像
    if not os.path.exists(val_images_zip) or not zipfile.is_zipfile(val_images_zip):
        print(f"下载验证集图像...")
        if not download_file(val_images_url, val_images_zip):
            print("尝试备用链接...")
            if not download_file(backup_val_images_url, val_images_zip):
                print("尝试GitHub备用链接...")
                github_backup_url = "https://github.com/ultralytics/yolov5face/releases/download/v1.0/WIDER_val.zip"
                if not download_file(github_backup_url, val_images_zip):
                    return None
    else:
        print(f"验证集图像文件已存在: {val_images_zip}")
    
    # 可选下载测试集图像
    if not os.path.exists(test_images_zip) and args.download_test:
        print(f"下载测试集图像...")
        if not download_file(test_images_url, test_images_zip):
            print("尝试备用链接...")
            if not download_file(backup_test_images_url, test_images_zip):
                print("尝试GitHub备用链接...")
                github_backup_url = "https://github.com/ultralytics/yolov5face/releases/download/v1.0/WIDER_test.zip"
                if not download_file(github_backup_url, test_images_zip):
                    print("警告: 测试集下载失败，将继续处理训练集和验证集")
    
    # 提取文件
    widerface_dir = os.path.join(output_dir, "widerface")
    os.makedirs(widerface_dir, exist_ok=True)
    
    # 解压标签文件
    if not os.path.exists(os.path.join(widerface_dir, "wider_face_split")):
        if not extract_zip(annot_zip, widerface_dir):
            return None
    else:
        print("标签文件已解压")
    
    # 解压训练集图像
    if not os.path.exists(os.path.join(widerface_dir, "WIDER_train")):
        if not extract_zip(train_zip, widerface_dir):
            return None
    else:
        print("训练集图像已解压")
    
    # 解压验证集图像
    if not os.path.exists(os.path.join(widerface_dir, "WIDER_val")):
        if not extract_zip(val_images_zip, widerface_dir):
            return None
    else:
        print("验证集图像已解压")
    
    # 解压测试集图像（如果存在）
    if os.path.exists(test_images_zip) and not os.path.exists(os.path.join(widerface_dir, "WIDER_test")):
        if not extract_zip(test_images_zip, widerface_dir):
            print("警告: 测试集解压失败，将继续处理训练集和验证集")
    
    print(f"WiderFace数据集下载并解压完成。数据集根目录: {widerface_dir}")
    return widerface_dir


def convert_train_set(root_path, save_path, include_landmarks=True, ignore_small=0):
    """
    转换WiderFace训练集为YOLO格式
    
    参数:
        root_path (str): WiderFace数据集根目录
        save_path (str): 保存路径
        include_landmarks (bool): 是否包含关键点（WiderFace不包含关键点，此参数无效）
        ignore_small (int): 忽略小于该尺寸的边界框
    """
    print(f"正在转换训练集到YOLO格式... 保存至: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # 图像目录和标注文件
    image_dir = os.path.join(root_path, "WIDER_train", "images")
    annot_file = os.path.join(root_path, "wider_face_split", "wider_face_train_bbx_gt.txt")
    
    if not os.path.exists(image_dir):
        print(f"错误: 训练集图像目录不存在 {image_dir}")
        return
    
    if not os.path.exists(annot_file):
        print(f"错误: 训练集标注文件不存在 {annot_file}")
        return
    
    # 解析标注文件
    with open(annot_file, "r", encoding="utf-8") as f:
        print("解析训练集标注文件...")
        idx = 0
        processed_images = 0
        
        while True:
            # 读取图像路径
            image_path_line = f.readline().strip()
            if not image_path_line or not image_path_line.endswith(".jpg"):
                break
                
            # 构建完整图像路径
            image_path = os.path.join(image_dir, image_path_line)
            if not os.path.exists(image_path):
                print(f"警告: 图像不存在 {image_path}")
                # 跳过此图像的标注
                try:
                    num_faces = int(f.readline().strip())
                    for _ in range(num_faces):
                        f.readline()  # 跳过人脸标注
                except ValueError:
                    print(f"警告: 无法解析人脸数量，标注文件可能已损坏")
                continue
            
            # 读取图像并获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                # 跳过此图像的标注
                try:
                    num_faces = int(f.readline().strip())
                    for _ in range(num_faces):
                        f.readline()  # 跳过人脸标注
                except ValueError:
                    print(f"警告: 无法解析人脸数量，标注文件可能已损坏")
                continue
                
            height, width, _ = img.shape
            
            # 读取人脸数量
            try:
                num_faces = int(f.readline().strip())
            except ValueError:
                print(f"警告: 无法解析人脸数量，标注文件可能已损坏")
                continue
            
            # 如果没有人脸，仍需读取一行
            if num_faces == 0:
                f.readline()  # 跳过空行
                continue
            
            # 创建输出文件名
            base_name = os.path.basename(image_path)
            save_img_path = os.path.join(save_path, f"{idx:06d}.jpg")
            save_txt_path = os.path.join(save_path, f"{idx:06d}.txt")
            
            # 读取人脸标注并转换为YOLO格式
            valid_faces = 0
            yolo_annotations = []
            
            for _ in range(num_faces):
                line = f.readline().strip()
                line_split = line.split()
                
                # 确保标注格式正确 (至少包含bbox的4个值)
                if len(line_split) >= 4:
                    try:
                        # 解析标注 [x, y, w, h, blur, expression, illumination, invalid, occlusion, pose]
                        values = [int(n) for n in line_split]
                        
                        xmin, ymin, w, h = values[0:4]
                        
                        # 如果框太小则跳过
                        if w < ignore_small or h < ignore_small:
                            continue
                            
                        # 检查标注是否有效
                        if len(values) > 7 and values[7] == 1:  # invalid标记
                            continue
                        
                        # 确保边界框在图像范围内
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        w = min(width - xmin, w)
                        h = min(height - ymin, h)
                        
                        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                        x_center = (xmin + w / 2) / width
                        y_center = (ymin + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height
                        
                        yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                        valid_faces += 1
                        
                    except (ValueError, IndexError) as e:
                        print(f"警告: 无法解析人脸标注 {line}, 错误: {e}")
                else:
                    print(f"警告: 标注格式不正确 {line}")
            
            # 如果有有效的人脸标注，保存图像和标注文件
            if valid_faces > 0:
                cv2.imwrite(save_img_path, img)
                
                with open(save_txt_path, "w") as txt_file:
                    for annot in yolo_annotations:
                        txt_file.write(annot + "\n")
                        
                processed_images += 1
                idx += 1
    
    print(f"训练集转换完成，共处理 {processed_images} 张有效图像")


def convert_val_set(root_path, save_path, ignore_small=0):
    """
    转换WiderFace验证集为YOLO格式
    
    参数:
        root_path (str): WiderFace数据集根目录
        save_path (str): 保存路径
        ignore_small (int): 忽略小于该尺寸的边界框
    """
    print(f"正在转换验证集到YOLO格式... 保存至: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # 图像目录和标注文件
    image_dir = os.path.join(root_path, "WIDER_val", "images")
    annot_file = os.path.join(root_path, "wider_face_split", "wider_face_val_bbx_gt.txt")
    
    if not os.path.exists(image_dir):
        print(f"错误: 验证集图像目录不存在 {image_dir}")
        return
    
    if not os.path.exists(annot_file):
        print(f"错误: 验证集标注文件不存在 {annot_file}")
        return
    
    # 解析标注文件
    with open(annot_file, "r", encoding="utf-8") as f:
        print("解析验证集标注文件...")
        idx = 0
        processed_images = 0
        
        while True:
            # 读取图像路径
            image_path_line = f.readline().strip()
            if not image_path_line or not image_path_line.endswith(".jpg"):
                break
                
            # 构建完整图像路径
            image_path = os.path.join(image_dir, image_path_line)
            if not os.path.exists(image_path):
                print(f"警告: 图像不存在 {image_path}")
                # 跳过此图像的标注
                try:
                    num_faces = int(f.readline().strip())
                    for _ in range(num_faces):
                        f.readline()  # 跳过人脸标注
                except ValueError:
                    print(f"警告: 无法解析人脸数量，标注文件可能已损坏")
                continue
            
            # 读取图像并获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                # 跳过此图像的标注
                try:
                    num_faces = int(f.readline().strip())
                    for _ in range(num_faces):
                        f.readline()  # 跳过人脸标注
                except ValueError:
                    print(f"警告: 无法解析人脸数量，标注文件可能已损坏")
                continue
                
            height, width, _ = img.shape
            
            # 读取人脸数量
            try:
                num_faces = int(f.readline().strip())
            except ValueError:
                print(f"警告: 无法解析人脸数量，标注文件可能已损坏")
                continue
            
            # 如果没有人脸，仍需读取一行
            if num_faces == 0:
                f.readline()  # 跳过空行
                continue
            
            # 创建输出文件名
            base_name = os.path.basename(image_path)
            save_img_path = os.path.join(save_path, f"{idx:06d}.jpg")
            save_txt_path = os.path.join(save_path, f"{idx:06d}.txt")
            
            # 读取人脸标注并转换为YOLO格式
            valid_faces = 0
            yolo_annotations = []
            
            for _ in range(num_faces):
                line = f.readline().strip()
                line_split = line.split()
                
                # 确保标注格式正确 (至少包含bbox的4个值)
                if len(line_split) >= 4:
                    try:
                        # 解析标注 [x, y, w, h, blur, expression, illumination, invalid, occlusion, pose]
                        values = [int(n) for n in line_split]
                        
                        xmin, ymin, w, h = values[0:4]
                        
                        # 如果框太小则跳过
                        if w < ignore_small or h < ignore_small:
                            continue
                            
                        # 检查标注是否有效
                        if len(values) > 7 and values[7] == 1:  # invalid标记
                            continue
                        
                        # 确保边界框在图像范围内
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        w = min(width - xmin, w)
                        h = min(height - ymin, h)
                        
                        # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                        x_center = (xmin + w / 2) / width
                        y_center = (ymin + h / 2) / height
                        w_norm = w / width
                        h_norm = h / height
                        
                        yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                        valid_faces += 1
                        
                    except (ValueError, IndexError) as e:
                        print(f"警告: 无法解析人脸标注 {line}, 错误: {e}")
                else:
                    print(f"警告: 标注格式不正确 {line}")
            
            # 如果有有效的人脸标注，保存图像和标注文件
            if valid_faces > 0:
                cv2.imwrite(save_img_path, img)
                
                with open(save_txt_path, "w") as txt_file:
                    for annot in yolo_annotations:
                        txt_file.write(annot + "\n")
                        
                processed_images += 1
                idx += 1
    
    print(f"验证集转换完成，共处理 {processed_images} 张有效图像")


def create_yaml(dataset_path, class_names=None):
    """
    创建YOLO格式的数据集配置文件
    
    参数:
        dataset_path (str): 数据集根目录
        class_names (list): 类别名称列表，默认为['face']
    """
    if class_names is None:
        class_names = ['face']
    
    # 创建yaml文件路径
    yaml_path = os.path.join(dataset_path, 'widerface.yaml')
    
    # 确保训练和验证集目录存在
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'val')
    
    # 编写yaml内容
    yaml_content = f"""# YOLOv12 face detection dataset
# 数据集配置文件

# 训练和验证集路径
path: {dataset_path}  # 数据集根目录
train: {train_dir}  # 训练集目录
val: {val_dir}  # 验证集目录

# 类别
nc: {len(class_names)}  # 类别数量
names: {class_names}  # 类别名称
"""

    # 写入yaml文件
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"数据集配置文件已创建: {yaml_path}")
    return yaml_path


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将WiderFace数据集转换为YOLOv12格式')
    parser.add_argument('--root', type=str, help='WiderFace数据集根目录')
    parser.add_argument('--output', type=str, default='widerface_yolo', help='输出目录路径，默认为"widerface_yolo"')
    parser.add_argument('--ignore-small', type=int, default=0, help='忽略小于指定尺寸的边界框，默认为0')
    parser.add_argument('--train-only', action='store_true', help='只转换训练集')
    parser.add_argument('--val-only', action='store_true', help='只转换验证集')
    parser.add_argument('--download', action='store_true', help='从HuggingFace下载WiderFace数据集')
    parser.add_argument('--download-test', action='store_true', help='是否也下载测试集')
    parser.add_argument('--download-dir', type=str, default='downloads', help='下载目录路径，默认为"downloads"')
    
    # 解析命令行参数
    global args
    args = parser.parse_args()
    
    # 如果指定了下载，则从HuggingFace下载数据集
    if args.download:
        print("从HuggingFace下载WiderFace数据集...")
        dataset_root = download_widerface_from_huggingface(args.download_dir)
        if dataset_root is None:
            print("数据集下载失败，请检查网络连接或手动下载。")
            return
        args.root = dataset_root
    
    # 确保数据集根目录存在
    if args.root is None:
        print("错误: 未指定数据集根目录。请使用--root参数指定数据集目录，或使用--download从HuggingFace下载数据集。")
        return
    
    if not os.path.isdir(args.root):
        print(f"错误: 数据集目录不存在 {args.root}")
        return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置训练和验证集目录
    train_output = os.path.join(args.output, 'train')
    val_output = os.path.join(args.output, 'val')
    
    # 转换训练集
    if not args.val_only:
        convert_train_set(args.root, train_output, ignore_small=args.ignore_small)
    
    # 转换验证集
    if not args.train_only:
        convert_val_set(args.root, val_output, args.ignore_small)
    
    # 创建数据集配置文件
    create_yaml(args.output)
    
    print(f"数据集转换完成。输出目录: {args.output}")
    print("使用以下命令训练YOLOv12模型:")
    print(f"yolo train model=yolov12.yaml data={os.path.join(args.output, 'widerface.yaml')}")


if __name__ == '__main__':
    main() 
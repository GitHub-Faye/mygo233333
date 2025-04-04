import os.path  # 导入操作系统路径模块
import sys  # 导入系统模块
import torch  # 导入PyTorch深度学习框架
import torch.utils.data as data  # 导入PyTorch数据处理工具
import cv2  # 导入OpenCV计算机视觉库
import numpy as np  # 导入NumPy数值计算库


class WiderFaceDetection(data.Dataset):  # 定义WiderFace数据集类，继承自PyTorch的Dataset类
    def __init__(self, txt_path, preproc=None):  # 初始化函数，接收标签文件路径和预处理函数
        self.preproc = preproc  # 存储预处理函数
        self.imgs_path = []  # 存储图片路径列表
        self.words = []  # 存储标签数据列表
        f = open(txt_path, 'r')  # 打开标签文件
        lines = f.readlines()  # 读取所有行
        isFirst = True  # 标记是否为第一个图片
        labels = []  # 临时存储当前图片的标签
        for line in lines:  # 遍历每一行
            line = line.rstrip()  # 去除行尾空白字符
            if line.startswith('#'):  # 如果行以#开头（表示图片路径）
                if isFirst is True:  # 如果是第一个图片
                    isFirst = False  # 更新标记
                else:  # 如果不是第一个图片
                    labels_copy = labels.copy()  # 复制当前标签列表
                    self.words.append(labels_copy)  # 将标签添加到数据集
                    labels.clear()  # 清空临时标签列表
                path = line[2:]  # 获取图片路径（去除#和空格）
                path = txt_path.replace('label.txt', 'images/') + path  # 构建完整的图片路径
                self.imgs_path.append(path)  # 添加图片路径到列表
            else:  # 如果行不以#开头（表示标签数据）
                line = line.split(' ')  # 分割行数据
                label = [float(x) for x in line]  # 将字符串转换为浮点数
                labels.append(label)  # 添加标签到临时列表

        self.words.append(labels)  # 添加最后一个图片的标签

    def __len__(self):  # 返回数据集长度
        return len(self.imgs_path)  # 返回图片数量

    def __getitem__(self, index):  # 获取数据集中的一项
        img = cv2.imread(self.imgs_path[index])  # 读取图片
        height, width, _ = img.shape  # 获取图片尺寸

        labels = self.words[index]  # 获取对应的标签
        annotations = np.zeros((0, 15))  # 创建标注数组
        if len(labels) == 0:  # 如果没有标签
            return annotations  # 返回空标注
        for idx, label in enumerate(labels):  # 遍历每个标签
            annotation = np.zeros((1, 15))  # 创建单个标注数组
            # 边界框
            annotation[0, 0] = label[0]  # x1坐标
            annotation[0, 1] = label[1]  # y1坐标
            annotation[0, 2] = label[0] + label[2]  # x2坐标
            annotation[0, 3] = label[1] + label[3]  # y2坐标

            # 关键点坐标
            annotation[0, 4] = label[4]    # 左眼x坐标
            annotation[0, 5] = label[5]    # 左眼y坐标
            annotation[0, 6] = label[7]    # 右眼x坐标
            annotation[0, 7] = label[8]    # 右眼y坐标
            annotation[0, 8] = label[10]   # 鼻子x坐标
            annotation[0, 9] = label[11]   # 鼻子y坐标
            annotation[0, 10] = label[13]  # 左嘴角x坐标
            annotation[0, 11] = label[14]  # 左嘴角y坐标
            annotation[0, 12] = label[16]  # 右嘴角x坐标
            annotation[0, 13] = label[17]  # 右嘴角y坐标
            if annotation[0, 4] < 0:  # 如果关键点坐标为负
                annotation[0, 14] = -1  # 标记为无效关键点
            else:
                annotation[0, 14] = 1  # 标记为有效关键点

            annotations = np.append(annotations, annotation, axis=0)  # 添加到标注数组
        target = np.array(annotations)  # 转换为NumPy数组
        if self.preproc is not None:  # 如果有预处理函数
            img, target = self.preproc(img, target)  # 进行预处理

        return torch.from_numpy(img), target  # 返回处理后的图片和标注


def detection_collate(batch):  # 定义数据批次整理函数
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).  # 自定义整理函数，处理具有不同数量目标标注的图片批次

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations  # 参数：图片张量和标注列表的元组

    Return:
        A tuple containing:  # 返回一个元组，包含：
            1) (tensor) batch of images stacked on their 0 dim  # 1) 在0维堆叠的图片批次张量
            2) (list of tensors) annotations for a given image are stacked on 0 dim  # 2) 每个图片的标注堆叠在0维的张量列表
    """
    targets = []  # 存储目标标注
    imgs = []  # 存储图片
    for _, sample in enumerate(batch):  # 遍历批次中的样本
        for _, tup in enumerate(sample):  # 遍历样本中的元素
            if torch.is_tensor(tup):  # 如果是张量（图片）
                imgs.append(tup)  # 添加到图片列表
            elif isinstance(tup, type(np.empty(0))):  # 如果是NumPy数组（标注）
                annos = torch.from_numpy(tup).float()  # 转换为浮点张量
                targets.append(annos)  # 添加到标注列表

    return torch.stack(imgs, 0), targets  # 返回堆叠的图片和标注


if __name__ == '__main__':  # 主程序入口
    if len(sys.argv) == 1:  # 如果没有提供命令行参数
        print('Missing path to WIDERFACE train folder.')  # 打印错误信息
        print('Run command: python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]')  # 打印使用说明
        exit(1)  # 退出程序
    elif len(sys.argv) > 3:  # 如果参数过多
        print('Too many arguments were provided.')  # 打印错误信息
        print('Run command: python3 train2yolo.py /path/to/original/widerface/train [/path/to/save/widerface/train]')  # 打印使用说明
        exit(1)  # 退出程序
    original_path = sys.argv[1]  # 获取原始数据集路径

    if len(sys.argv) == 2:  # 如果只提供了一个参数
        if not os.path.isdir('widerface'):  # 如果widerface目录不存在
            os.mkdir('widerface')  # 创建widerface目录
        if not os.path.isdir('widerface/train'):  # 如果train目录不存在
            os.mkdir('widerface/train')  # 创建train目录

        save_path = 'widerface/train'  # 设置保存路径
    else:  # 如果提供了两个参数
        save_path = sys.argv[2]  # 使用指定的保存路径

    if not os.path.isfile(os.path.join(original_path, 'label.txt')):  # 如果标签文件不存在
        print('Missing label.txt file.')  # 打印错误信息
        exit(1)  # 退出程序

    aa = WiderFaceDetection(os.path.join(original_path, 'label.txt'))  # 创建数据集对象

    for i in range(len(aa.imgs_path)):  # 遍历所有图片
        print(i, aa.imgs_path[i])  # 打印处理进度
        img = cv2.imread(aa.imgs_path[i])  # 读取图片
        base_img = os.path.basename(aa.imgs_path[i])  # 获取图片文件名
        base_txt = os.path.basename(aa.imgs_path[i])[:-4] + ".txt"  # 生成对应的标签文件名
        save_img_path = os.path.join(save_path, base_img)  # 构建图片保存路径
        save_txt_path = os.path.join(save_path, base_txt)  # 构建标签保存路径
        with open(save_txt_path, "w") as f:  # 打开标签文件
            height, width, _ = img.shape  # 获取图片尺寸
            labels = aa.words[i]  # 获取标签数据
            annotations = np.zeros((0, 14))  # 创建标注数组
            if len(labels) == 0:  # 如果没有标签
                continue  # 继续下一个图片
            for idx, label in enumerate(labels):  # 遍历每个标签
                annotation = np.zeros((1, 14))  # 创建单个标注数组
                # 边界框
                label[0] = max(0, label[0])  # 确保x坐标不小于0
                label[1] = max(0, label[1])  # 确保y坐标不小于0
                label[2] = min(width - 1, label[2])  # 确保宽度不超过图片边界
                label[3] = min(height - 1, label[3])  # 确保高度不超过图片边界
                annotation[0, 0] = (label[0] + label[2] / 2) / width  # 计算中心点x坐标（归一化）
                annotation[0, 1] = (label[1] + label[3] / 2) / height  # 计算中心点y坐标（归一化）
                annotation[0, 2] = label[2] / width  # 计算宽度（归一化）
                annotation[0, 3] = label[3] / height  # 计算高度（归一化）
                # 关键点坐标（归一化）
                annotation[0, 4] = label[4] / width  # 左眼x
                annotation[0, 5] = label[5] / height  # 左眼y
                annotation[0, 6] = label[7] / width  # 右眼x
                annotation[0, 7] = label[8] / height  # 右眼y
                annotation[0, 8] = label[10] / width  # 鼻子x
                annotation[0, 9] = label[11] / height  # 鼻子y
                annotation[0, 10] = label[13] / width  # 左嘴角x
                annotation[0, 11] = label[14] / height  # 左嘴角y
                annotation[0, 12] = label[16] / width  # 右嘴角x
                annotation[0, 13] = label[17] / height  # 右嘴角y
                str_label = "0 "  # 创建标签字符串（0表示人脸类别）
                for i in range(len(annotation[0])):  # 遍历标注数据
                    str_label = str_label + " " + str(annotation[0][i])  # 添加到标签字符串
                str_label = str_label.replace('[', '').replace(']', '')  # 移除方括号
                str_label = str_label.replace(',', '') + '\n'  # 移除逗号并添加换行
                f.write(str_label)  # 写入标签文件
        cv2.imwrite(save_img_path, img)  # 保存图片

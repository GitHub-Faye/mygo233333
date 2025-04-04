import os  # 导入操作系统模块
import cv2  # 导入OpenCV计算机视觉库
import numpy as np  # 导入NumPy数值计算库
import shutil  # 导入文件操作模块
import sys  # 导入系统模块
from tqdm import tqdm  # 导入进度条模块


def xywh2xxyy(box):  # 将[x,y,w,h]格式转换为[x1,x2,y1,y2]格式
    x1 = box[0]  # 获取左上角x坐标
    y1 = box[1]  # 获取左上角y坐标
    x2 = box[0] + box[2]  # 计算右下角x坐标
    y2 = box[1] + box[3]  # 计算右下角y坐标
    return x1, x2, y1, y2  # 返回转换后的坐标


def convert(size, box):  # 将边界框坐标转换为YOLO格式
    dw = 1. / (size[0])  # 计算宽度缩放因子
    dh = 1. / (size[1])  # 计算高度缩放因子
    x = (box[0] + box[1]) / 2.0 - 1  # 计算中心点x坐标
    y = (box[2] + box[3]) / 2.0 - 1  # 计算中心点y坐标
    w = box[1] - box[0]  # 计算宽度
    h = box[3] - box[2]  # 计算高度
    x = x * dw  # 归一化x坐标
    w = w * dw  # 归一化宽度
    y = y * dh  # 归一化y坐标
    h = h * dh  # 归一化高度
    return x, y, w, h  # 返回YOLO格式的坐标


def wider2face(root, phase='val', ignore_small=0):  # 将WiderFace数据集转换为YOLO格式
    data = {}  # 存储转换后的数据
    with open('{}/{}/label.txt'.format(root, phase), 'r') as f:  # 打开标签文件
        lines = f.readlines()  # 读取所有行
        for line in tqdm(lines):  # 遍历每一行（显示进度条）
            line = line.strip()  # 去除行首尾空白字符
            if '#' in line:  # 如果是图片路径行
                path = '{}/{}/images/{}'.format(root, phase, line.split()[-1])  # 构建完整图片路径
                img = cv2.imread(path)  # 读取图片
                height, width, _ = img.shape  # 获取图片尺寸
                data[path] = list()  # 初始化该图片的标签列表
            else:  # 如果是标签行
                box = np.array(line.split()[0:4], dtype=np.float32)  # 获取边界框坐标[x,y,w,h]
                if box[2] < ignore_small or box[3] < ignore_small:  # 如果边界框太小
                    continue  # 跳过该标签
                box = convert((width, height), xywh2xxyy(box))  # 转换为YOLO格式
                label = '0 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 -1 -1 -1'.format(round(box[0], 4), round(box[1], 4),
                                                                             round(box[2], 4), round(box[3], 4))  # 生成标签字符串
                data[path].append(label)  # 添加到数据字典
    return data  # 返回转换后的数据


if __name__ == '__main__':  # 主程序入口
    if len(sys.argv) == 1:  # 如果没有提供命令行参数
        print('Missing path to WIDERFACE folder.')  # 打印错误信息
        print('Run command: python3 val2yolo.py /path/to/original/widerface [/path/to/save/widerface/val]')  # 打印使用说明
        exit(1)  # 退出程序
    elif len(sys.argv) > 3:  # 如果参数过多
        print('Too many arguments were provided.')  # 打印错误信息
        print('Run command: python3 val2yolo.py /path/to/original/widerface [/path/to/save/widerface/val]')  # 打印使用说明
        exit(1)  # 退出程序

    root_path = sys.argv[1]  # 获取原始数据集路径
    if not os.path.isfile(os.path.join(root_path, 'val', 'label.txt')):  # 如果标签文件不存在
        print('Missing label.txt file.')  # 打印错误信息
        exit(1)  # 退出程序

    if len(sys.argv) == 2:  # 如果只提供了一个参数
        if not os.path.isdir('widerface'):  # 如果widerface目录不存在
            os.mkdir('widerface')  # 创建widerface目录
        if not os.path.isdir('widerface/val'):  # 如果val目录不存在
            os.mkdir('widerface/val')  # 创建val目录

        save_path = 'widerface/val'  # 设置保存路径
    else:  # 如果提供了两个参数
        save_path = sys.argv[2]  # 使用指定的保存路径

    datas = wider2face(root_path, phase='val')  # 转换数据集
    for idx, data in enumerate(datas.keys()):  # 遍历所有图片
        pict_name = os.path.basename(data)  # 获取图片文件名
        out_img = f'{save_path}/{idx}.jpg'  # 构建输出图片路径
        out_txt = f'{save_path}/{idx}.txt'  # 构建输出标签路径
        shutil.copyfile(data, out_img)  # 复制图片文件
        labels = datas[data]  # 获取标签数据
        f = open(out_txt, 'w')  # 打开标签文件
        for label in labels:  # 遍历标签
            f.write(label + '\n')  # 写入标签
        f.close()  # 关闭文件

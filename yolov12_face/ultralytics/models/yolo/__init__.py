# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - 查看链接获取许可证详情

from ultralytics.models.yolo import classify, detect, obb, pose, segment, world  # 导入YOLO的各个任务模块：分类、检测、面向对象边界框、姿态估计、分割和世界模型

from .model import YOLO, YOLOWorld  # 从model.py导入YOLO和YOLOWorld类

__all__ = "classify", "segment", "detect", "pose", "obb", "world", "YOLO", "YOLOWorld"  # 定义此模块可直接导入的所有组件

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - 查看链接获取许可证详情

from .base import BaseDataset  # 导入基础数据集类
from .build import build_dataloader, build_grounding, build_yolo_dataset, load_inference_source  # 导入构建数据加载器、定位、YOLO数据集和加载推理源的函数
from .dataset import (  # 导入各种数据集类
    ClassificationDataset,  # 分类数据集
    GroundingDataset,  # 定位数据集
    SemanticDataset,  # 语义分割数据集
    YOLOConcatDataset,  # YOLO连接数据集
    YOLODataset,  # YOLO基础数据集
    YOLOMultiModalDataset,  # YOLO多模态数据集
)

__all__ = (  # 定义模块的公共API
    "BaseDataset",  # 基础数据集
    "ClassificationDataset",  # 分类数据集
    "SemanticDataset",  # 语义分割数据集
    "YOLODataset",  # YOLO数据集
    "YOLOMultiModalDataset",  # YOLO多模态数据集
    "YOLOConcatDataset",  # YOLO连接数据集
    "GroundingDataset",  # 定位数据集
    "build_yolo_dataset",  # 构建YOLO数据集函数
    "build_grounding",  # 构建定位函数
    "build_dataloader",  # 构建数据加载器函数
    "load_inference_source",  # 加载推理源函数
)

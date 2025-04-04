# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - 查看链接获取许可证详情

from .tasks import (  # 从tasks模块导入
    BaseModel,  # 基础模型类
    ClassificationModel,  # 分类模型类
    DetectionModel,  # 检测模型类
    SegmentationModel,  # 分割模型类
    attempt_load_one_weight,  # 尝试加载单个权重
    attempt_load_weights,  # 尝试加载多个权重
    guess_model_scale,  # 推测模型尺度
    guess_model_task,  # 推测模型任务
    parse_model,  # 解析模型
    torch_safe_load,  # 安全加载PyTorch模型
    yaml_model_load,  # 从YAML加载模型
)

__all__ = (  # 定义模块的公共API
    "attempt_load_one_weight",  # 尝试加载单个权重函数
    "attempt_load_weights",  # 尝试加载多个权重函数
    "parse_model",  # 解析模型函数
    "yaml_model_load",  # 从YAML加载模型函数
    "guess_model_task",  # 推测模型任务函数
    "guess_model_scale",  # 推测模型尺度函数
    "torch_safe_load",  # 安全加载PyTorch模型函数
    "DetectionModel",  # 检测模型类
    "SegmentationModel",  # 分割模型类
    "ClassificationModel",  # 分类模型类
    "BaseModel",  # 基础模型类
)

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - 查看链接获取许可证详情

__version__ = "8.3.63"  # 设置版本号为8.3.63

import os  # 导入操作系统模块

# Set ENV variables (place before imports)  # 设置环境变量（放在导入之前）
if not os.environ.get("OMP_NUM_THREADS"):  # 如果未设置OMP_NUM_THREADS环境变量
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training  # 设置为1以减少训练期间的CPU使用率

from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOWorld  # 从ultralytics.models导入各种模型类
from ultralytics.utils import ASSETS, SETTINGS  # 从ultralytics.utils导入资源和设置
from ultralytics.utils.checks import check_yolo as checks  # 从ultralytics.utils.checks导入check_yolo并重命名为checks
from ultralytics.utils.downloads import download  # 从ultralytics.utils.downloads导入下载函数

settings = SETTINGS  # 创建settings变量并赋值为SETTINGS
__all__ = (  # 定义模块的公共API
    "__version__",  # 版本号
    "ASSETS",  # 资源
    "YOLO",  # YOLO模型
    "YOLOWorld",  # YOLOWorld模型
    "NAS",  # NAS模型
    "SAM",  # SAM模型
    "FastSAM",  # FastSAM模型
    "RTDETR",  # RTDETR模型
    "checks",  # 检查函数
    "download",  # 下载函数
    "settings",  # 设置
)

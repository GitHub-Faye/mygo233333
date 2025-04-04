# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license  # Ultralytics 🚀 AGPL-3.0 许可证 - 查看链接获取许可证详情

from pathlib import Path  # 导入Path类用于处理文件路径

from ultralytics.engine.model import Model  # 导入基础模型类
from ultralytics.models import yolo  # 导入yolo模块
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel  # 导入各种任务的模型类
from ultralytics.utils import ROOT, yaml_load  # 导入根目录常量和YAML加载函数


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""  # YOLO（只需看一次）目标检测模型

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""  # 初始化YOLO模型，如果模型文件名包含'-world'则切换到YOLOWorld
        path = Path(model)  # 将模型路径转换为Path对象
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model  # 如果是YOLOWorld PyTorch模型
            new_instance = YOLOWorld(path, verbose=verbose)  # 创建YOLOWorld实例
            self.__class__ = type(new_instance)  # 将当前实例的类型改为YOLOWorld
            self.__dict__ = new_instance.__dict__  # 将当前实例的属性字典替换为YOLOWorld实例的属性字典
        else:
            # Continue with default YOLO initialization  # 继续默认的YOLO初始化
            super().__init__(model=model, task=task, verbose=verbose)  # 调用父类初始化方法

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""  # 将头部映射到模型、训练器、验证器和预测器类
        return {
            "classify": {  # 分类任务
                "model": ClassificationModel,  # 分类模型
                "trainer": yolo.classify.ClassificationTrainer,  # 分类训练器
                "validator": yolo.classify.ClassificationValidator,  # 分类验证器
                "predictor": yolo.classify.ClassificationPredictor,  # 分类预测器
            },
            "detect": {  # 检测任务
                "model": DetectionModel,  # 检测模型
                "trainer": yolo.detect.DetectionTrainer,  # 检测训练器
                "validator": yolo.detect.DetectionValidator,  # 检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 检测预测器
            },
            "segment": {  # 分割任务
                "model": SegmentationModel,  # 分割模型
                "trainer": yolo.segment.SegmentationTrainer,  # 分割训练器
                "validator": yolo.segment.SegmentationValidator,  # 分割验证器
                "predictor": yolo.segment.SegmentationPredictor,  # 分割预测器
            },
            "pose": {  # 姿态估计任务
                "model": PoseModel,  # 姿态模型
                "trainer": yolo.pose.PoseTrainer,  # 姿态训练器
                "validator": yolo.pose.PoseValidator,  # 姿态验证器
                "predictor": yolo.pose.PosePredictor,  # 姿态预测器
            },
            "obb": {  # 面向对象边界框任务
                "model": OBBModel,  # OBB模型
                "trainer": yolo.obb.OBBTrainer,  # OBB训练器
                "validator": yolo.obb.OBBValidator,  # OBB验证器
                "predictor": yolo.obb.OBBPredictor,  # OBB预测器
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model."""  # YOLO-World目标检测模型

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        # 初始化YOLOv8-World模型，使用预训练模型文件
        # 
        # 加载YOLOv8-World模型用于目标检测。如果没有提供自定义类名，将分配默认的COCO类名
        # 
        # 参数:
        #     model (str | Path): 预训练模型文件的路径。支持*.pt和*.yaml格式
        #     verbose (bool): 如果为True，则在初始化过程中打印额外信息
        super().__init__(model=model, task="detect", verbose=verbose)  # 调用父类初始化方法，设置任务为"detect"

        # Assign default COCO class names when there are no custom names  # 当没有自定义名称时，分配默认的COCO类名
        if not hasattr(self.model, "names"):  # 如果模型没有names属性
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")  # 从COCO8配置文件加载类名

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""  # 将头部映射到模型、验证器和预测器类
        return {
            "detect": {  # 检测任务
                "model": WorldModel,  # 世界模型
                "validator": yolo.detect.DetectionValidator,  # 检测验证器
                "predictor": yolo.detect.DetectionPredictor,  # 检测预测器
                "trainer": yolo.world.WorldTrainer,  # 世界训练器
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        # 设置类别
        # 
        # 参数:
        #     classes (List(str)): 类别列表，例如 ["person"]
        self.model.set_classes(classes)  # 设置模型的类别
        # Remove background if it's given  # 如果给定了背景类，则移除它
        background = " "  # 背景类用空格表示
        if background in classes:  # 如果背景类在类别列表中
            classes.remove(background)  # 从类别列表中移除背景类
        self.model.names = classes  # 设置模型的names属性为类别列表

        # Reset method class names  # 重置方法类名
        # self.predictor = None  # reset predictor otherwise old names remain  # 重置预测器，否则旧名称会保留
        if self.predictor:  # 如果预测器存在
            self.predictor.model.names = classes  # 更新预测器模型的类名

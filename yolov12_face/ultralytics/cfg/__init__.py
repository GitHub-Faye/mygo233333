# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Ultralytics 🚀 AGPL-3.0 许可证 - https://ultralytics.com/license

import shutil  # 导入用于文件操作的模块
import subprocess  # 导入用于子进程管理的模块
import sys  # 导入系统模块
from pathlib import Path  # 导入路径处理模块
from types import SimpleNamespace  # 导入简单命名空间类型
from typing import Dict, List, Union  # 导入类型提示工具

import cv2  # 导入OpenCV计算机视觉库

from ultralytics.utils import (  # 从ultralytics.utils导入工具函数和常量
    ASSETS,  # 资源文件路径
    DEFAULT_CFG,  # 默认配置
    DEFAULT_CFG_DICT,  # 默认配置字典
    DEFAULT_CFG_PATH,  # 默认配置文件路径
    DEFAULT_SOL_DICT,  # 默认解决方案字典
    IS_VSCODE,  # 是否在VSCode中运行
    LOGGER,  # 日志记录器
    RANK,  # 分布式训练的排名
    ROOT,  # 项目根目录
    RUNS_DIR,  # 运行结果目录
    SETTINGS,  # 设置
    SETTINGS_FILE,  # 设置文件路径
    TESTS_RUNNING,  # 是否正在运行测试
    IterableSimpleNamespace,  # 可迭代的简单命名空间
    __version__,  # 版本号
    checks,  # 检查函数
    colorstr,  # 彩色文本函数
    deprecation_warn,  # 弃用警告函数
    vscode_msg,  # VSCode消息函数
    yaml_load,  # YAML加载函数
    yaml_print,  # YAML打印函数
)

# Define valid solutions
# 定义有效的解决方案
SOLUTION_MAP = {
    "count": ("ObjectCounter", "count"),  # 计数解决方案
    "heatmap": ("Heatmap", "generate_heatmap"),  # 热图解决方案
    "queue": ("QueueManager", "process_queue"),  # 队列管理解决方案
    "speed": ("SpeedEstimator", "estimate_speed"),  # 速度估计解决方案
    "workout": ("AIGym", "monitor"),  # 健身监测解决方案
    "analytics": ("Analytics", "process_data"),  # 数据分析解决方案
    "trackzone": ("TrackZone", "trackzone"),  # 区域跟踪解决方案
    "inference": ("Inference", "inference"),  # 推理解决方案
    "help": None,  # 帮助
}

# Define valid tasks and modes
# 定义有效的任务和模式
MODES = {"train", "val", "predict", "export", "track", "benchmark"}  # 训练、验证、预测、导出、跟踪、基准测试模式
TASKS = {"detect", "segment", "classify", "pose", "obb"}  # 检测、分割、分类、姿态估计、面向对象边界框任务
TASK2DATA = {  # 任务到数据集的映射
    "detect": "coco8.yaml",  # 检测任务对应coco8数据集
    "segment": "coco8-seg.yaml",  # 分割任务对应coco8-seg数据集
    "classify": "imagenet10",  # 分类任务对应imagenet10数据集
    "pose": "coco8-pose.yaml",  # 姿态估计任务对应coco8-pose数据集
    "obb": "dota8.yaml",  # 面向对象边界框任务对应dota8数据集
}
TASK2MODEL = {  # 任务到模型的映射
    "detect": "yolo11n.pt",  # 检测任务对应yolo11n模型
    "segment": "yolo11n-seg.pt",  # 分割任务对应yolo11n-seg模型
    "classify": "yolo11n-cls.pt",  # 分类任务对应yolo11n-cls模型
    "pose": "yolo11n-pose.pt",  # 姿态估计任务对应yolo11n-pose模型
    "obb": "yolo11n-obb.pt",  # 面向对象边界框任务对应yolo11n-obb模型
}
TASK2METRIC = {  # 任务到评估指标的映射
    "detect": "metrics/mAP50-95(B)",  # 检测任务使用mAP50-95(B)指标
    "segment": "metrics/mAP50-95(M)",  # 分割任务使用mAP50-95(M)指标
    "classify": "metrics/accuracy_top1",  # 分类任务使用accuracy_top1指标
    "pose": "metrics/mAP50-95(P)",  # 姿态估计任务使用mAP50-95(P)指标
    "obb": "metrics/mAP50-95(B)",  # 面向对象边界框任务使用mAP50-95(B)指标
}
MODELS = {TASK2MODEL[task] for task in TASKS}  # 所有任务对应的模型集合

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []  # 命令行参数，有时sys.argv为空，则提供默认值
SOLUTIONS_HELP_MSG = f"""
    Arguments received: {str(["yolo"] + ARGV[1:])}. Ultralytics 'yolo solutions' usage overview:
    接收到的参数: {str(["yolo"] + ARGV[1:])}. Ultralytics 'yolo solutions' 使用概述:

        yolo solutions SOLUTION ARGS
        yolo solutions 解决方案 参数

        Where SOLUTION (optional) is one of {list(SOLUTION_MAP.keys())[:-1]}
        其中 解决方案 (可选) 是 {list(SOLUTION_MAP.keys())[:-1]} 之一
              ARGS (optional) are any number of custom 'arg=value' pairs like 'show_in=True' that override defaults 
              参数 (可选) 是任意数量的自定义 'arg=value' 对，如 'show_in=True'，用于覆盖默认值
                  at https://docs.ultralytics.com/usage/cfg
                  详情请参见 https://docs.ultralytics.com/usage/cfg
                
    1. Call object counting solution
    1. 调用对象计数解决方案
        yolo solutions count source="path/to/video/file.mp4" region=[(20, 400), (1080, 400), (1080, 360), (20, 360)]

    2. Call heatmaps solution
    2. 调用热图解决方案
        yolo solutions heatmap colormap=cv2.COLORMAP_PARULA model=yolo11n.pt

    3. Call queue management solution
    3. 调用队列管理解决方案
        yolo solutions queue region=[(20, 400), (1080, 400), (1080, 360), (20, 360)] model=yolo11n.pt

    4. Call workouts monitoring solution for push-ups
    4. 调用俯卧撑健身监测解决方案
        yolo solutions workout model=yolo11n-pose.pt kpts=[6, 8, 10]

    5. Generate analytical graphs
    5. 生成分析图表
        yolo solutions analytics analytics_type="pie"
    
    6. Track objects within specific zones
    6. 在特定区域内跟踪对象
        yolo solutions trackzone source="path/to/video/file.mp4" region=[(150, 150), (1130, 150), (1130, 570), (150, 570)]
        
    7. Streamlit real-time webcam inference GUI
    7. Streamlit实时网络摄像头推理GUI
        yolo streamlit-predict
    """
CLI_HELP_MSG = f"""
    Arguments received: {str(["yolo"] + ARGV[1:])}. Ultralytics 'yolo' commands use the following syntax:
    接收到的参数: {str(["yolo"] + ARGV[1:])}. Ultralytics 'yolo' 命令使用以下语法:

        yolo TASK MODE ARGS
        yolo 任务 模式 参数

        Where   TASK (optional) is one of {TASKS}
        其中    任务 (可选) 是 {TASKS} 之一
                MODE (required) is one of {MODES}
                模式 (必需) 是 {MODES} 之一
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                参数 (可选) 是任意数量的自定义 'arg=value' 对，如 'imgsz=320'，用于覆盖默认值。
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'
                    查看所有参数，请访问 https://docs.ultralytics.com/usage/cfg 或使用 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
    1. 使用0.01的初始学习率训练检测模型10个周期
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
    2. 使用预训练的分割模型在图像尺寸320下预测YouTube视频:
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
    3. 在批量大小1和图像尺寸640下验证预训练的检测模型:
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
    4. 将YOLO11n分类模型导出为ONNX格式，图像尺寸为224x128 (不需要指定任务)
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128

    5. Ultralytics solutions usage
    5. Ultralytics解决方案使用方法
        yolo solutions count or in {list(SOLUTION_MAP.keys())[1:-1]} source="path/to/video/file.mp4"

    6. Run special commands:
    6. 运行特殊命令:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        yolo solutions help

    Docs: https://docs.ultralytics.com
    文档: https://docs.ultralytics.com
    Solutions: https://docs.ultralytics.com/solutions/
    解决方案: https://docs.ultralytics.com/solutions/
    Community: https://community.ultralytics.com
    社区: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Define keys for arg type checks
# 定义参数类型检查的键
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
                    # 整数或浮点数参数，例如 x=2 和 x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
                       # 取值范围在 0.0<=values<=1.0 的小数浮点参数
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {  # integer-only arguments
                 # 仅限整数的参数
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {  # boolean-only arguments
                  # 仅限布尔值的参数
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}


def cfg2dict(cfg):
    """
    Converts a configuration object to a dictionary.
    将配置对象转换为字典。

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration object to be converted. Can be a file path,
            a string, a dictionary, or a SimpleNamespace object.
        cfg (str | Path | Dict | SimpleNamespace): 要转换的配置对象。可以是文件路径、
            字符串、字典或SimpleNamespace对象。

    Returns:
        (Dict): Configuration object in dictionary format.
        (Dict): 字典格式的配置对象。

    Examples:
        Convert a YAML file path to a dictionary:
        将YAML文件路径转换为字典：
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        将SimpleNamespace转换为字典：
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        直接传递已有的字典：
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
        
        - 如果cfg是路径或字符串，它将被作为YAML加载并转换为字典。
        - 如果cfg是SimpleNamespace对象，它将使用vars()转换为字典。
        - 如果cfg已经是字典，它将原样返回。
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict  # 加载字典
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict  # 转换为字典
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary, with optional overrides.
    从文件或字典加载并合并配置数据，可选择性地覆盖某些配置。

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        cfg (str | Path | Dict | SimpleNamespace): 配置数据源。可以是文件路径、字典或
            SimpleNamespace对象。
        overrides (Dict | None): Dictionary containing key-value pairs to override the base configuration.
        overrides (Dict | None): 包含用于覆盖基本配置的键值对的字典。

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.
        (SimpleNamespace): 包含合并后的配置参数的命名空间。

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
                                # 加载默认配置
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})
                                    # 加载配置并使用覆盖项

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
        
        - 如果同时提供了`cfg`和`overrides`，则`overrides`中的值将优先。
        - 特殊处理确保配置的对齐和正确性，例如将数字类型的`project`和`name`转换为字符串，
          并验证配置键和值。
        - 该函数对配置数据执行类型和值检查。
    """
    cfg = cfg2dict(cfg)  # 转换为字典

    # Merge overrides
    # 合并覆盖项
    if overrides:
        overrides = cfg2dict(overrides)  # 转换覆盖项为字典
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore  # 忽略特殊的覆盖键
        check_dict_alignment(cfg, overrides)  # 检查字典对齐
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)  # 合并cfg和overrides字典（优先使用overrides）

    # Special handling for numeric project/name
    # 对数字类型的project/name进行特殊处理
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])  # 转换为字符串
    if cfg.get("name") == "model":  # assign model to 'name' arg  # 将模型分配给'name'参数
        cfg["name"] = str(cfg.get("model", "")).split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")  # 警告信息

    # Type and Value checks
    # 类型和值检查
    check_cfg(cfg)

    # Return instance
    # 返回实例
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg, hard=True):
    """
    Checks configuration argument types and values for the Ultralytics library.
    检查Ultralytics库的配置参数类型和值。

    This function validates the types and values of configuration arguments, ensuring correctness and converting
    them if necessary. It checks for specific key types defined in global variables such as CFG_FLOAT_KEYS,
    CFG_FRACTION_KEYS, CFG_INT_KEYS, and CFG_BOOL_KEYS.
    该函数验证配置参数的类型和值，确保正确性并在必要时进行转换。它检查在全局变量中定义的特定键类型，
    如CFG_FLOAT_KEYS、CFG_FRACTION_KEYS、CFG_INT_KEYS和CFG_BOOL_KEYS。

    Args:
        cfg (Dict): Configuration dictionary to validate.
        cfg (Dict): 要验证的配置字典。
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.
        hard (bool): 如果为True，则对无效类型和值引发异常；如果为False，则尝试转换它们。

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 1.2,  # invalid float (out of 0.0-1.0 range)
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys are checked to be within the range [0.0, 1.0].
        
        - 该函数直接修改输入字典。
        - 忽略None值，因为它们可能来自可选参数。
        - 对于小数键，检查其是否在[0.0, 1.0]范围内。
    """
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args  # None值可能来自可选参数
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)  # 转换为浮点数
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)  # 转换为浮点数
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)  # 转换为整数
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)  # 转换为布尔值


def get_save_dir(args, name=None):
    """
    Returns the directory path for saving outputs, derived from arguments or default settings.
    返回用于保存输出的目录路径，从参数或默认设置派生。

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task',
            'mode', and 'save_dir'.
        args (SimpleNamespace): 包含配置的命名空间对象，如'project'、'name'、'task'、
            'mode'和'save_dir'。
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name'
            or the 'args.mode'.
        name (str | None): 输出目录的可选名称。如果未提供，则默认为'args.name'
            或'args.mode'。

    Returns:
        (Path): Directory path where outputs should be saved.
        (Path): 应保存输出的目录路径。

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        my_project/detect/train
    """
    if getattr(args, "save_dir", None):  # 如果args中有save_dir属性
        save_dir = args.save_dir  # 使用args中的save_dir
    else:
        from ultralytics.utils.files import increment_path  # 导入路径增量函数

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task  # 设置项目路径
        name = name or args.name or f"{args.mode}"  # 设置名称
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)  # 增量路径

    return Path(save_dir)  # 返回路径对象


def _handle_deprecation(custom):
    """
    Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings.
    处理已弃用的配置键，通过弃用警告将它们映射到当前的等效键。

    Args:
        custom (Dict): Configuration dictionary potentially containing deprecated keys.
        custom (Dict): 可能包含已弃用键的配置字典。

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
        
        此函数直接修改输入字典，用当前等效键替换已弃用的键。
        它还在必要时处理值转换，例如为'hide_labels'和'hide_conf'反转布尔值。
    """
    for key in custom.copy().keys():  # 遍历配置字典的副本
        if key == "boxes":
            deprecation_warn(key, "show_boxes")  # 弃用警告
            custom["show_boxes"] = custom.pop("boxes")  # 替换键名
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")  # 弃用警告
            custom["show_labels"] = custom.pop("hide_labels") == "False"  # 反转值含义
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")  # 弃用警告
            custom["show_conf"] = custom.pop("hide_conf") == "False"  # 反转值含义
        if key == "line_thickness":
            deprecation_warn(key, "line_width")  # 弃用警告
            custom["line_width"] = custom.pop("line_thickness")  # 替换键名
        if key == "label_smoothing":
            deprecation_warn(key)  # 弃用警告
            custom.pop("label_smoothing")  # 移除键

    return custom  # 返回处理后的字典


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    Checks alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.
    检查自定义配置字典与基础配置字典之间的对齐，处理已弃用的键并为不匹配的键提供错误消息。

    Args:
        base (Dict): The base configuration dictionary containing valid keys.
        base (Dict): 包含有效键的基础配置字典。
        custom (Dict): The custom configuration dictionary to be checked for alignment.
        custom (Dict): 要检查对齐的自定义配置字典。
        e (Exception | None): Optional error instance passed by the calling function.
        e (Exception | None): 调用函数传递的可选错误实例。

    Raises:
        SystemExit: If mismatched keys are found between the custom and base dictionaries.
        SystemExit: 如果在自定义字典和基础字典之间发现不匹配的键。

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("Mismatched keys found")

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
        
        - 根据与有效键的相似性为不匹配的键建议更正。
        - 自动将自定义配置中的已弃用键替换为更新的等效键。
        - 为每个不匹配的键打印详细的错误消息，以帮助用户更正其配置。
    """
    custom = _handle_deprecation(custom)  # 处理已弃用的键
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))  # 获取基础键和自定义键集合
    if mismatched := [k for k in custom_keys if k not in base_keys]:  # 找出不匹配的键
        from difflib import get_close_matches  # 导入获取接近匹配的函数

        string = ""  # 初始化错误消息字符串
        for x in mismatched:  # 遍历不匹配的键
            matches = get_close_matches(x, base_keys)  # 获取相似的键列表
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]  # 格式化匹配项
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""  # 如果有匹配项，生成建议字符串
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"  # 添加错误消息
        raise SyntaxError(string + CLI_HELP_MSG) from e  # 抛出语法错误


def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' in a list of strings and joins fragments with brackets.
    合并字符串列表中孤立的'='周围的参数，并连接带有括号的片段。

    This function handles the following cases:
    此函数处理以下情况：
    1. ['arg', '=', 'val'] becomes ['arg=val']
       ['arg', '=', 'val'] 变成 ['arg=val']
    2. ['arg=', 'val'] becomes ['arg=val']
       ['arg=', 'val'] 变成 ['arg=val']
    3. ['arg', '=val'] becomes ['arg=val']
       ['arg', '=val'] 变成 ['arg=val']
    4. Joins fragments with brackets, e.g., ['imgsz=[3,', '640,', '640]'] becomes ['imgsz=[3,640,640]']
       连接带有括号的片段，例如，['imgsz=[3,', '640,', '640]'] 变成 ['imgsz=[3,640,640]']

    Args:
        args (List[str]): A list of strings where each element represents an argument or fragment.
        args (List[str]): 字符串列表，其中每个元素表示一个参数或片段。

    Returns:
        List[str]: A list of strings where the arguments around isolated '=' are merged and fragments with brackets are joined.
        List[str]: 一个字符串列表，其中孤立的'='周围的参数被合并，带有括号的片段被连接。

    Examples:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_and_join_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []  # 新参数列表
    current = ""  # 当前处理的字符串
    depth = 0  # 括号深度

    i = 0  # 索引初始化
    while i < len(args):  # 遍历参数列表
        arg = args[i]  # 获取当前参数

        # Handle equals sign merging
        # 处理等号合并
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']  # 合并 ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"  # 将等号和下一个参数添加到上一个参数
            i += 2  # 跳过已处理的两个参数
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']  # 合并 ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")  # 将当前参数和下一个参数合并
            i += 2  # 跳过已处理的两个参数
            continue
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']  # 合并 ['arg', '=val']
            new_args[-1] += arg  # 将当前参数添加到上一个参数
            i += 1  # 跳过当前参数
            continue

        # Handle bracket joining
        # 处理括号连接
        depth += arg.count("[") - arg.count("]")  # 计算括号深度变化
        current += arg  # 添加当前参数到处理字符串
        if depth == 0:  # 如果括号已平衡
            new_args.append(current)  # 将处理的字符串添加到新参数列表
            current = ""  # 重置处理字符串

        i += 1  # 移动到下一个参数

    # Append any remaining current string
    # 添加任何剩余的处理字符串
    if current:  # 如果还有未处理完的字符串
        new_args.append(current)  # 将其添加到新参数列表

    return new_args  # 返回处理后的参数列表


def handle_yolo_hub(args: List[str]) -> None:
    """
    Handles Ultralytics HUB command-line interface (CLI) commands for authentication.

    This function processes Ultralytics HUB CLI commands such as login and logout. It should be called when executing a
    script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments. The first argument should be either 'login'
            or 'logout'. For 'login', an optional second argument can be the API key.

    Examples:
        ```bash
        yolo login YOUR_API_KEY
        ```

    Notes:
        - The function imports the 'hub' module from ultralytics to perform login and logout operations.
        - For the 'login' command, if no API key is provided, an empty string is passed to the login function.
        - The 'logout' command does not require any additional arguments.
    """
    from ultralytics import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""
        # Log in to Ultralytics HUB using the provided API key
        hub.login(key)
    elif args[0] == "logout":
        # Log out from Ultralytics HUB
        hub.logout()


def handle_yolo_settings(args: List[str]) -> None:
    """
    Handles YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset and updating individual settings. It should be
    called when executing a script with arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Examples:
        >>> handle_yolo_settings(["reset"])  # Reset YOLO settings
        >>> handle_yolo_settings(["default_cfg_path=yolo11n.yaml"])  # Update a specific setting

    Notes:
        - If no arguments are provided, the function will display the current settings.
        - The 'reset' command will delete the existing settings file and create new default settings.
        - Other arguments are treated as key-value pairs to update specific settings.
        - The function will check for alignment between the provided settings and the existing ones.
        - After processing, the updated settings will be displayed.
        - For more information on handling YOLO settings, visit:
          https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"  # help URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info("Settings reset successfully")  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        print(SETTINGS)  # print the current settings
        LOGGER.info(f"💡 Learn more about Ultralytics Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")


def handle_yolo_solutions(args: List[str]) -> None:
    """
    Processes YOLO solutions arguments and runs the specified computer vision solutions pipeline.

    Args:
        args (List[str]): Command-line arguments for configuring and running the Ultralytics YOLO
            solutions: https://docs.ultralytics.com/solutions/, It can include solution name, source,
            and other configuration parameters.

    Returns:
        None: The function processes video frames and saves the output but doesn't return any value.

    Examples:
        Run people counting solution with default settings:
        >>> handle_yolo_solutions(["count"])

        Run analytics with custom configuration:
        >>> handle_yolo_solutions(["analytics", "conf=0.25", "source=path/to/video/file.mp4"])

        Run inference with custom configuration, requires Streamlit version 1.29.0 or higher.
        >>> handle_yolo_solutions(["inference", "model=yolo11n.pt"])

    Notes:
        - Default configurations are merged from DEFAULT_SOL_DICT and DEFAULT_CFG_DICT
        - Arguments can be provided in the format 'key=value' or as boolean flags
        - Available solutions are defined in SOLUTION_MAP with their respective classes and methods
        - If an invalid solution is provided, defaults to 'count' solution
        - Output videos are saved in 'runs/solution/{solution_name}' directory
        - For 'analytics' solution, frame numbers are tracked for generating analytical graphs
        - Video processing can be interrupted by pressing 'q'
        - Processes video frames sequentially and saves output in .avi format
        - If no source is specified, downloads and uses a default sample video\
        - The inference solution will be launched using the 'streamlit run' command.
        - The Streamlit app file is located in the Ultralytics package directory.
    """
    full_args_dict = {**DEFAULT_SOL_DICT, **DEFAULT_CFG_DICT}  # arguments dictionary
    overrides = {}

    # check dictionary alignment
    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)  # dict alignment

    # Get solution name
    if args and args[0] in SOLUTION_MAP:
        if args[0] != "help":
            s_n = args.pop(0)  # Extract the solution name directly
        else:
            LOGGER.info(SOLUTIONS_HELP_MSG)
    else:
        LOGGER.warning(
            f"⚠️ No valid solution provided. Using default 'count'. Available: {', '.join(SOLUTION_MAP.keys())}"
        )
        s_n = "count"  # Default solution if none provided

    if args and args[0] == "help":  # Add check for return if user call `yolo solutions help`
        return

    if s_n == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("💡 Loading Ultralytics live inference app...")
        subprocess.run(
            [  # Run subprocess with Streamlit custom argument
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "yolo11n.pt"),
            ]
        )
    else:
        cls, method = SOLUTION_MAP[s_n]  # solution class name, method name and default source

        from ultralytics import solutions  # import ultralytics solutions

        solution = getattr(solutions, cls)(IS_CLI=True, **overrides)  # get solution class i.e ObjectCounter
        process = getattr(
            solution, method
        )  # get specific function of class for processing i.e, count from ObjectCounter

        cap = cv2.VideoCapture(solution.CFG["source"])  # read the video file

        # extract width, height and fps of the video file, create save directory and initialize video writer
        import os  # for directory creation
        from pathlib import Path

        from ultralytics.utils.files import increment_path  # for output directory path update

        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        if s_n == "analytics":  # analytical graphs follow fixed shape for output i.e w=1920, h=1080
            w, h = 1920, 1080
        save_dir = increment_path(Path("runs") / "solutions" / "exp", exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)  # create the output directory
        vw = cv2.VideoWriter(os.path.join(save_dir, "solution.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:  # Process video frames
            f_n = 0  # frame number, required for analytical graphs
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frame = process(frame, f_n := f_n + 1) if s_n == "analytics" else process(frame)
                vw.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()


def parse_key_value_pair(pair: str = "key=value"):
    """
    Parses a key-value pair string into separate key and value components.

    Args:
        pair (str): A string containing a key-value pair in the format "key=value".

    Returns:
        key (str): The parsed key.
        value (str): The parsed value.

    Raises:
        AssertionError: If the value is missing or empty.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolo11n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo11n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    Notes:
        - The function splits the input string on the first '=' character.
        - Leading and trailing whitespace is removed from both key and value.
        - An assertion error is raised if the value is empty after stripping.
    """
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """
    Converts a string representation of a value to its appropriate Python type.

    This function attempts to convert a given string into a Python object of the most appropriate type. It handles
    conversions to None, bool, int, float, and other types that can be evaluated safely.

    Args:
        v (str): The string representation of the value to be converted.

    Returns:
        (Any): The converted value. The type can be None, bool, int, float, or the original string if no conversion
            is applicable.

    Examples:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> smart_value("None")
        None
        >>> smart_value("some_string")
        'some_string'

    Notes:
        - The function uses a case-insensitive comparison for boolean and None values.
        - For other types, it attempts to use Python's eval() function, which can be unsafe if used on untrusted input.
        - If no conversion is possible, the original string is returned.
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return eval(v)
        except Exception:
            return v


def entrypoint(debug=""):
    """
    Ultralytics entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the Ultralytics CLI, parsing command-line arguments and
    executing the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        >>> entrypoint("predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - If no arguments are passed, the function will display the usage help message.
        - For a list of all available commands and their arguments, see the provided help messages and the
          Ultralytics documentation at https://docs.ultralytics.com.
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: yaml_print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "logout": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 'mode' argument is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task == "classify" and mode == "track":
            raise ValueError(
                f"❌ Classification doesn't support 'mode=track'. Valid modes for classification are"
                f" {MODES - {'track'}}.\n{CLI_HELP_MSG}"
            )
        elif task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    "WARNING ⚠️ invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo11n.pt"
        LOGGER.warning(f"WARNING ⚠️ 'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics import RTDETR

        model = RTDETR(model)  # no task argument
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING ⚠️ conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"WARNING ⚠️ 'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    # Recommend VS Code extension
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """
    Copies the default configuration file and creates a new one with '_copy' appended to its name.

    This function duplicates the existing default configuration file (DEFAULT_CFG_PATH) and saves it
    with '_copy' appended to its name in the current working directory. It provides a convenient way
    to create a custom configuration file based on the default settings.

    Examples:
        >>> copy_default_cfg()
        # Output: default.yaml copied to /path/to/current/directory/default_copy.yaml
        # Example YOLO command with this new custom cfg:
        #   yolo cfg='/path/to/current/directory/default_copy.yaml' imgsz=320 batch=8

    Notes:
        - The new configuration file is created in the current working directory.
        - After copying, the function prints a message with the new file's location and an example
          YOLO command demonstrating how to use the new configuration file.
        - This function is useful for users who want to modify the default configuration without
          altering the original file.
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolo11n.pt')
    entrypoint(debug="")

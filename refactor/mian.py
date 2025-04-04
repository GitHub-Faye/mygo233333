import os
import sys
import cv2
import supervision as sv

# 添加项目根目录到系统路径
# 假设refactor目录在yolov12_face项目的根目录下
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 从本地项目导入YOLO类
from yolov12_face.ultralytics.models.yolo.model import YOLO

# 设置HOME变量，如果没有定义的话
HOME = os.path.expanduser("~")

image_path = f"{HOME}/1.jpg"
image = cv2.imread(image_path)

model = YOLO('yolov12l.pt')

results = model(image, verbose=False)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = image.copy()
annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
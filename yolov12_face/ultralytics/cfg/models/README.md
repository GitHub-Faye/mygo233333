## Models
## 模型

Welcome to the [Ultralytics](https://www.ultralytics.com/) Models directory! Here you will find a wide variety of pre-configured model configuration files (`*.yaml`s) that can be used to create custom YOLO models. The models in this directory have been expertly crafted and fine-tuned by the Ultralytics team to provide the best performance for a wide range of object detection and image segmentation tasks.

欢迎来到[Ultralytics](https://www.ultralytics.com/)模型目录！在这里，您将找到各种预配置的模型配置文件（`*.yaml`），可用于创建自定义YOLO模型。此目录中的模型由Ultralytics团队精心制作和调优，为各种目标检测和图像分割任务提供最佳性能。

These model configurations cover a wide range of scenarios, from simple object detection to more complex tasks like instance segmentation and object tracking. They are also designed to run efficiently on a variety of hardware platforms, from CPUs to GPUs. Whether you are a seasoned machine learning practitioner or just getting started with YOLO, this directory provides a great starting point for your custom model development needs.

这些模型配置涵盖了广泛的应用场景，从简单的目标检测到更复杂的任务，如实例分割和目标跟踪。它们还被设计为在各种硬件平台上高效运行，从CPU到GPU。无论您是经验丰富的机器学习从业者还是刚刚开始使用YOLO，该目录都为您的自定义模型开发需求提供了一个很好的起点。

To get started, simply browse through the models in this directory and find one that best suits your needs. Once you've selected a model, you can use the provided `*.yaml` file to train and deploy your custom YOLO model with ease. See full details at the Ultralytics [Docs](https://docs.ultralytics.com/models/), and if you need help or have any questions, feel free to reach out to the Ultralytics team for support. So, don't wait, start creating your custom YOLO model now!

要开始使用，只需浏览此目录中的模型，找到最适合您需求的模型。选择模型后，您可以使用提供的`*.yaml`文件轻松训练和部署您的自定义YOLO模型。在Ultralytics [文档](https://docs.ultralytics.com/models/)中查看完整详细信息，如果您需要帮助或有任何问题，请随时联系Ultralytics团队获取支持。所以，不要等待，现在就开始创建您的自定义YOLO模型吧！

### Usage
### 使用方法

Model `*.yaml` files may be used directly in the [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli/) with a `yolo` command:

模型`*.yaml`文件可以直接在[命令行界面(CLI)](https://docs.ultralytics.com/usage/cli/)中使用`yolo`命令：

```bash
# Train a YOLO11n model using the coco8 dataset for 100 epochs
# 使用coco8数据集训练YOLO11n模型，共100个周期
yolo task=detect mode=train model=yolo11n.yaml data=coco8.yaml epochs=100
```

They may also be used directly in a Python environment, and accept the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

它们也可以直接在Python环境中使用，并接受与上述CLI示例相同的[参数](https://docs.ultralytics.com/usage/cfg/)：

```python
from ultralytics import YOLO

# Initialize a YOLO11n model from a YAML configuration file
# 从YAML配置文件初始化YOLO11n模型
model = YOLO("model.yaml")

# If a pre-trained model is available, use it instead
# 如果有预训练模型可用，则使用预训练模型
# model = YOLO("model.pt")

# Display model information
# 显示模型信息
model.info()

# Train the model using the COCO8 dataset for 100 epochs
# 使用COCO8数据集训练模型，共100个周期
model.train(data="coco8.yaml", epochs=100)
```

## Pre-trained Model Architectures
## 预训练模型架构

Ultralytics supports many model architectures. Visit [Ultralytics Models](https://docs.ultralytics.com/models/) to view detailed information and usage. Any of these models can be used by loading their configurations or pretrained checkpoints if available.

Ultralytics支持多种模型架构。访问[Ultralytics模型](https://docs.ultralytics.com/models/)查看详细信息和使用方法。如果有可用的配置或预训练检查点，可以加载任何这些模型使用。

## Contribute New Models
## 贡献新模型

Have you trained a new YOLO variant or achieved state-of-the-art performance with specific tuning? We'd love to showcase your work in our Models section! Contributions from the community in the form of new models, architectures, or optimizations are highly valued and can significantly enrich our repository.

您是否训练了新的YOLO变体或通过特定调整实现了最先进的性能？我们很乐意在我们的模型部分展示您的工作！社区以新模型、架构或优化形式的贡献被高度重视，可以显著丰富我们的存储库。

By contributing to this section, you're helping us offer a wider array of model choices and configurations to the community. It's a fantastic way to share your knowledge and expertise while making the Ultralytics YOLO ecosystem even more versatile.

通过为此部分做出贡献，您可以帮助我们为社区提供更广泛的模型选择和配置。这是分享您的知识和专业技能的绝佳方式，同时使Ultralytics YOLO生态系统变得更加多样化。

To get started, please consult our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for step-by-step instructions on how to submit a Pull Request (PR) 🛠️. Your contributions are eagerly awaited!

要开始，请查阅我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)，了解如何提交拉取请求(PR)的分步说明 🛠️。我们热切期待您的贡献！

Let's join hands to extend the range and capabilities of the Ultralytics YOLO models 🙏!

让我们携手合作，扩展Ultralytics YOLO模型的范围和功能 🙏！

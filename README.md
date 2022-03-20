# TensorRT c++ BiSeNetV1/2

使用TensorRT c++实现 [BiSeNetV1](https://arxiv.org/abs/1808.00897) 和 [BiSeNetV2](https://arxiv.org/abs/1808.00897)部署

## 环境依赖

1. Opencv3.1
2. TensorRT7.2
3. Cuda10.2

## 代码介绍

1. trt_bisenet.h与trt_bisenet.cpp为主要实现代码，主要包括onnx->tensorrt生成.engine模型、预处理、前向传播、后处理等步骤
2. mat_transform.hpp中为图像预处理相关算法。预处理算法根据项目的不同会稍有区别，这里是我的预处理。
3. gpu_func.cuh与gpu_func.cu为后处理gpu代码。我实现了cpu版与gpu版后处理，这里是gpu版后处理的实现

## 使用方法

具体使用方法可以参照trt_bisenet.cpp中的main()函数。如果初次调用，需要指定onnx模型的路径、生成的trt模型的保存路径以及保存的模型名。
初次调用之后会生成.engine的trt模型，并保存到指定位置，之后再调用，则直接调用.engine模型。
```
# onnx->tensorrt所需要的主要参数
  OnnxInitParam params;
# onnx模型路径
  params.onnx_model_path = "./BiSeNet/checkpoints/onnx/bisenet.onnx";
# 保存生成.engine的模型路径
  params.rt_stream_path = "./"
  params.rt_model_name = "bisenet.engine"
# 是否使用半精度，如果false，则使用fp32精度的
  params.use_fp16 = true;
# 使用的显卡设备
  params.gpu_id = 0;
# 模型分割的类别数
  params.num_classes = 4;
# 设置最大网络输入大小，用于分配内存(显存), 这里需要根据自己的需求设置
  params.max_shape = Shape(1, 3, 640, 640);

# 实例化BiSeNet类，其中会进行模型转化和一些初始化的操作
  BiSeNet model(params);

# 模型前向推理,得到分割的输出，输出为uint8单通道Mat型图像数据，像素值从0~num_classes-1，代表像素的类别
  cv::Mat res = model.Extract(img);
```
#ifndef TRT_ONNX_BASE_H_
#define TRT_ONNX_BASE_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <fstream>
#include "common.hpp"

using namespace std;

struct OnnxInitParam
{
	std::string onnx_model_path;
	std::string rt_stream_path = "./";
	std::string rt_model_name = "bisenetv3.engine";
	bool use_fp16 = false;
	int gpu_id = 0;
	int num_classes;
	Shape max_shape{ 1, 3, 640, 640 };
};

class TRTOnnxBase
{
public:
	TRTOnnxBase() = delete;
	TRTOnnxBase(const OnnxInitParam& params);

protected:
	class Logger : public nvinfer1::ILogger
	{
	public:
		void log(nvinfer1::ILogger::Severity severity, const char* msg)
		{
			switch (severity)
			{
			case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "kINTERNAL_ERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kERROR:
				std::cerr << "kERROR: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kWARNING:
				std::cerr << "kWARNING: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kINFO:
				std::cerr << "kINFO: " << msg << std::endl;
				break;
			case nvinfer1::ILogger::Severity::kVERBOSE:
				std::cerr << "kVERBOSE: " << msg << std::endl;
				break;
			default:
				break;
			}
		}
	};

	// tensorrt推理
	void Forward();

private:
	// void mallocInputOutput(const Shape &input_shape, const Shape &output_shape); 
	// 模型初始化
	void Initial();
	// 加载onnx模型
	void LoadOnnxModel();
	// 加载tensorrt模型
	void LoadGieStreamBuildContext(const std::string& gie_file);
	// 分配执行预测所需的cpu内存与gpu显存
	void mallocInputOutput();
	// 保存序列化的模型
	void SaveRTModel(nvinfer1::IHostMemory* gie_model_stream, const std::string& path);
	// 反序列化tensorrt模型
	void deserializeCudaEngine(const void* blob_data, std::size_t size);

	bool CheckFileExist(const std::string& path)
	{
		std::ifstream check_file(path);
		return check_file.is_open();
	}

private:
	Logger logger;
	nvinfer1::IRuntime* _runtime{ nullptr };
	nvinfer1::ICudaEngine* _engine{ nullptr };
	nvinfer1::IExecutionContext* _context{ nullptr };

	cudaStream_t stream_;

protected:
	std::vector<void*> buffer_queue_;

	float* h_input_tensor_;
	float* d_input_tensor_;
	Shape input_shape_; // 记录每次前向预测的输入样本的shape
	Shape output_shape_; // 记录每次前向预测的输出样本的shape
	float* h_output_tensor_;
	float* d_output_tensor_;

	OnnxInitParam _params;
};

#endif
#ifndef TRT_BISENET_H_
#define TRT_BISENET_H_

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

class BiSeNet
{
public:
	BiSeNet() = delete;
	BiSeNet(const OnnxInitParam& params);
	virtual ~BiSeNet();
	void Initial();

	void LoadOnnxModel();
	void LoadGieStreamBuildContext(const std::string& gie_file);

	cv::Mat Extract(const cv::Mat& img);

private:
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

private: 
	void mallocInputOutput();
	void SaveRTModel(nvinfer1::IHostMemory* gie_model_stream, const std::string& path);

	void deserializeCudaEngine(const void* blob_data, std::size_t size);

	void PreProcessCpu(const cv::Mat& img);

	void Forward();

	cv::Mat PostProcessCpu();
	cv::Mat PostProcessGpu();

	static void softmax(vector<float>& vec);
	static int findMaxIdx(const vector<float>& vec);

	bool CheckFileExist(const std::string& path)
	{
		std::ifstream check_file(path);
		return check_file.is_open();
	}

private:
	OnnxInitParam _params;
	Logger logger;
	nvinfer1::IRuntime* _runtime{ nullptr };
	nvinfer1::ICudaEngine* _engine{ nullptr };
	nvinfer1::IExecutionContext* _context{ nullptr };

	cudaStream_t stream_;

	std::vector<void*> buffer_queue_;

	float* h_input_tensor_;
	float* d_input_tensor_;
	Shape input_shape_t; // 记录每次前向预测的输入样本的shape
	float* h_output_tensor_;
	float* d_output_tensor_;
	Shape output_shape_t; // 记录每次前向预测的输出样本的shape

	std::vector<float> mean_{ 0.485, 0.456, 0.406 };
	std::vector<float> std_{ 0.229, 0.224, 0.225 };
};

#endif
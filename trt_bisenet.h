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

struct OnnxInitParam
{
	std::string onnx_model_path;
	std::string rt_stream_path = "./";
	std::string rt_model_name = "bisenet.engine";
	bool use_fp16 = false;
	int gpu_id = 0;
	int num_classes;
};

bool CheckFileExist(const std::string& path)
{
	std::ifstream check_file(path);
	return check_file.is_open();
}

class Shape
{
public:
	Shape() :num_(0), channels_(0), height_(0), width_(0) {}
	Shape(int num, int channels, int height, int width) :
		num_(num), channels_(channels), height_(height), width_(width) {}
	~Shape() {}

public:
	inline int num() const
	{
		return num_;
	}
	inline int channels() const
	{
		return channels_;
	}
	inline int height() const
	{
		return height_;
	}
	inline int width() const
	{
		return width_;
	}
	inline int count() const
	{
		return num_ * channels_ * height_ * width_;
	}
private:
	int num_;
	int channels_;
	int height_;
	int width_;
};

class Tensor2VecMat
{
public:
	Tensor2VecMat() {}
	vector<cv::Mat> operator()(float* h_src, const Shape& input_shape)  // const std::vector<int>& input_shape
	{
		vector<cv::Mat> input_channels;
		int channels = input_shape.channels();
		int height = input_shape.height();
		int width = input_shape.width();

		for (int i = 0; i < channels; i++)
		{
			cv::Mat channel(height, width, CV_32FC1, h_src);
			input_channels.push_back(channel);
			h_src += height * width;
		}
		return std::move(input_channels);
	}
};

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
	void mallocInputOutput(const Shape &input_shape, const Shape &output_shape);
	void SaveRTModel(nvinfer1::IHostMemory* gie_model_stream, const std::string& path);

	void deserializeCudaEngine(const void* blob_data, std::size_t size);

	void PreProcessCpu(const cv::Mat& img);

	void Forward();

	cv::Mat PostProcessCpu();
	cv::Mat PostProcessGpu();

	static void softmax(vector<float>& vec);
	static int findMaxIdx(const vector<float>& vec);


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
	Shape input_shape_t;
	float* d_output_tensor_;
	Shape output_shape_t;

	std::vector<float> mean_{ 0.485, 0.456, 0.406 };
	std::vector<float> std_{ 0.229, 0.224, 0.225 };
};

#endif
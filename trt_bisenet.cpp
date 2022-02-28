#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <fstream>
#include "mat_transform.hpp"
#include "trt_bisenet.h"
#include "gpu_func.cuh"


BiSeNet::BiSeNet(const OnnxInitParam& params)
{
	cudaSetDevice(params.gpu_id);
	_params = params;

	cudaStreamCreate(&stream_);

	Initial();
}

BiSeNet::~BiSeNet()
{
	cudaStreamSynchronize(stream_);
	if (h_input_tensor_ != nullptr)
		free(h_input_tensor_);
	if (input_tensor_ != nullptr)
		cudaFree(input_tensor_);
	if (output_tensor_ != nullptr)
		cudaFree(output_tensor_);
}

void BiSeNet::Initial()
{
	if (CheckFileExist(_params.rt_stream_path + _params.rt_model_name))
	{
		std::cout << "read rt model..." << std::endl;
		LoadGieStreamBuildContext(_params.rt_stream_path + _params.rt_model_name);
	}
	else
	{
		LoadOnnxModel();
	}
}

void BiSeNet::LoadOnnxModel() 
{
	if (!CheckFileExist(_params.onnx_model_path))
	{
		std::cerr << "onnx file is not found " << _params.onnx_model_path << std::endl;
		exit(0);
	}

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	assert(builder != nullptr);

	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	assert(parser->parseFromFile(_params.onnx_model_path.c_str(), 2));

	nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
	nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
	nvinfer1::ITensor* input = network->getInput(0);
	std::cout << "input name: " << input->getName() << std::endl;
	nvinfer1::Dims input_dims = input->getDimensions();
	std::cout << "batch_size: " << input_dims.d[0]
		<< " channels: " << input_dims.d[1]
		<< " height: " << input_dims.d[2]
		<< " width: " << input_dims.d[3] << std::endl;

	{
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, input_dims.d[1], 1, 1 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 1, input_dims.d[1], 640, 640 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 1, input_dims.d[1], 1024, 1024 });
		build_config->addOptimizationProfile(profile);
	}

	build_config->setMaxWorkspaceSize(1 << 30);
	if (_params.use_fp16)
	{
		if (builder->platformHasFastFp16())
		{
			builder->setHalf2Mode(true);
			std::cout << "useFP16 : " << true << std::endl;
		}
	}
	else
		std::cout << "Using GPU FP32 !" << std::endl;

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *build_config);
	assert(engine != nullptr);

	nvinfer1::IHostMemory* gie_model_stream = engine->serialize();
	SaveRTModel(gie_model_stream, _params.rt_stream_path + _params.rt_model_name);

	deserializeCudaEngine(gie_model_stream->data(), gie_model_stream->size());
}

void BiSeNet::LoadGieStreamBuildContext(const std::string& gie_file)
{
	std::ifstream fgie(gie_file, std::ios_base::in | std::ios_base::binary);
	if (!fgie)
	{
		std::cerr << "Can't read rt model from " << gie_file << std::endl;
		return;
	}

	std::stringstream buffer;
	buffer << fgie.rdbuf();

	std::string stream_model(buffer.str());

	deserializeCudaEngine(stream_model.data(), stream_model.size());
}

void BiSeNet::mallocInputOutput(const std::vector<int> &input_shape, const std::vector<int> &output_shape)
{
	if (!buffer_queue_.empty())
	{
		for (int i = 0; i < buffer_queue_.size(); i++)
		{
			cudaFree(buffer_queue_[i]);
		}

		buffer_queue_.clear();
	}
		
	input_shape_ = input_shape;
	output_shape_ = output_shape;

	int count = 1;
	for (int i = 0; i < input_shape.size(); i++)
		count *= input_shape[i];
	h_input_tensor_ = (float*)malloc(count * sizeof(float));
	cudaMalloc((void**)&input_tensor_, count * sizeof(float));

	count = 1;
	for (int i = 0; i < output_shape.size(); i++)
		count *= output_shape[i];
	cudaMalloc((void**)&output_tensor_, count * sizeof(float));
}

cv::Mat BiSeNet::Extract(const cv::Mat& img)
{
	if (img.empty())
		return img;

	PreProcessCpu(img);
	Forward();

	//cv::Mat res = PostProcessCpu();
	cv::Mat res = PostProcessGpu();
	return std::move(res);
}

//private:
void BiSeNet::SaveRTModel(nvinfer1::IHostMemory* gie_model_stream, const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream->data(), gie_model_stream->size());
	outfile.close();
}

void BiSeNet::deserializeCudaEngine(const void* blob_data, std::size_t size)
{
	_runtime = nvinfer1::createInferRuntime(logger);
	assert(_runtime != nullptr);
	_engine = _runtime->deserializeCudaEngine(blob_data, size, nullptr);
	assert(_engine != nullptr);

	_context = _engine->createExecutionContext();
	assert(_context != nullptr);
}

void BiSeNet::PreProcessCpu(const cv::Mat& img)
{
	cv::Mat img_tmp = img;

	ComposeMatLambda compose({
		LetterResize(cv::Size(640, 640), cv::Scalar(114, 114, 114), 32),
		MatDivConstant(255),
		MatNormalize(mean_, std_),
	});

	cv::Mat sample_float = compose(img_tmp);
	vector<int> input_shape{ 1, sample_float.channels(), sample_float.rows, sample_float.cols };
	vector<int> output_shape{ 1, _params.num_classes, sample_float.rows, sample_float.cols };
	mallocInputOutput(input_shape, output_shape);

	Tensor2VecMat tensor_2_mat;
	std::vector<cv::Mat> channels = tensor_2_mat(h_input_tensor_, input_shape_);
	cv::split(sample_float, channels);
}

void BiSeNet::Forward()
{
	cudaMemcpy(input_tensor_, h_input_tensor_, 
		input_shape_[1] * input_shape_[2] * input_shape_[3] * sizeof(float), 
		cudaMemcpyHostToDevice);

	buffer_queue_.push_back(input_tensor_);
	buffer_queue_.push_back(output_tensor_);
	nvinfer1::Dims4 input_dims{ 1, input_shape_[1], input_shape_[2], input_shape_[3] };
	_context->setBindingDimensions(0, input_dims);
	_context->enqueueV2(buffer_queue_.data(), stream_, nullptr);

	cudaStreamSynchronize(stream_);
}

cv::Mat BiSeNet::PostProcessCpu()
{
	int num = output_shape_[0];
	int channels = output_shape_[1];
	int height = output_shape_[2];
	int width = output_shape_[3];

	float* h_output_tensor = (float*)malloc(num * channels * height * width * sizeof(float));
	cudaMemcpy(h_output_tensor, output_tensor_, 
		num * channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);

	cv::Mat res = cv::Mat::zeros(height, width, CV_8UC1);
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			vector<float> vec;
			for (int c = 0; c < channels; c++)
			{
				int index = row * width + col + c * height * width;
				float val = h_output_tensor[index];
				vec.push_back(val);
			}
			softmax(vec);

			if (vec[2] > 0.5)
				res.at<uchar>(row, col) = uchar(255);
			else
				res.at<uchar>(row, col) = uchar(0);
		}
	}

	free(h_output_tensor);

	return std::move(res);
}

cv::Mat BiSeNet::PostProcessGpu()
{
	int num = output_shape_[0];
	int channels = output_shape_[1];
	int height = output_shape_[2];
	int width = output_shape_[3];

	/*float* gpu_ptr = output.mutable_gpu_data();*/
	/*float *src_ptr = output.mutable_cpu_data();*/
	float* cpu_dst = (float*)malloc(channels * height * width * sizeof(float));
	cout << "start..." << endl;
	segmentation(output_tensor_, channels, height, width, cpu_dst);
	cout << "end ..." << endl;

	cv::Mat res = cv::Mat::zeros(height, width, CV_8UC1);
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			vector<float> vec;
			for (int c = 0; c < channels; c++)
			{
				int index = row * width + col + c * height * width;
				float val = cpu_dst[index];
				vec.push_back(val);
			}

			if (vec[2] > 0.5)
				res.at<uchar>(row, col) = uchar(255);
			else
				res.at<uchar>(row, col) = uchar(0);
		}
	}

	return std::move(res);
}

void BiSeNet::softmax(vector<float>& vec)
{
	float tol = 0.0;
	for (int i = 0; i < vec.size(); i++)
	{
		vec[i] = exp(vec[i]);
		tol += vec[i];
	}

	for (int i = 0; i < vec.size(); i++)
		vec[i] = vec[i] / tol;
}

int main(int argc, char** argv)
{
	OnnxInitParam params;
	params.onnx_model_path = "./BiSeNet/checkpoints/onnx/bisenet.onnx";
	params.use_fp16 = true;
	params.gpu_id = 0;
	params.num_classes = 4;

	BiSeNet model(params);

	cv::Mat img = cv::imread("./BiSeNet/datas/tupian.jpg");

	cv::Mat res = model.Extract(img);
	cv::imshow("res", res);

	cv::waitKey();
}
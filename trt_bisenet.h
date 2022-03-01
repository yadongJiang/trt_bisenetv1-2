#ifndef TRT_BISENET_H_
#define TRT_BISENET_H_

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

class Tensor2VecMat
{
public:
	Tensor2VecMat() {}
	vector<cv::Mat> operator()(float* h_src, const std::vector<int>& input_shape)
	{
		vector<cv::Mat> input_channels;

		for (int i = 0; i < input_shape[1]; i++)
		{
			cv::Mat channel(input_shape[2], input_shape[3], CV_32FC1, h_src);
			input_channels.push_back(channel);
			h_src += input_shape[2] * input_shape[3];
		}
		return std::move(input_channels);
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
	void mallocInputOutput(const std::vector<int>& input_shape, const std::vector<int>& output_shape);
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
	std::vector<int> input_shape_;
	float* d_output_tensor_;
	std::vector<int> output_shape_;

	std::vector<float> mean_{ 0.485, 0.456, 0.406 };
	std::vector<float> std_{ 0.229, 0.224, 0.225 };
};

#endif
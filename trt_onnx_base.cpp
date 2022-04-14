#include "trt_onnx_base.h"

TRTOnnxBase::TRTOnnxBase(const OnnxInitParam& params) : _params(params)
{
	cudaSetDevice(params.gpu_id);

	cudaStreamCreate(&stream_);

	Initial();
}

void TRTOnnxBase::Initial()
{
	if (CheckFileExist(_params.rt_stream_path + _params.rt_model_name))
	{
		std::cout << "read rt model..." << std::endl;
		LoadGieStreamBuildContext(_params.rt_stream_path + _params.rt_model_name);
	}
	else
		LoadOnnxModel();
}

void TRTOnnxBase::LoadOnnxModel()
{
	if (!CheckFileExist(_params.onnx_model_path))
	{
		cout << "onnx_model_path: " << _params.onnx_model_path << endl;
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
	nvinfer1::Dims input_dims = input->getDimensions();
	std::cout << "batch_size: " << input_dims.d[0]
		<< " channels: " << input_dims.d[1]
		<< " height: " << input_dims.d[2]
		<< " width: " << input_dims.d[3] << std::endl;

	{
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, input_dims.d[1], 1, 1 });
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 1, input_dims.d[1], 640, 480 });  // 640
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 1, input_dims.d[1], 640, 640 });
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

	builder->destroy();
	network->destroy();
	parser->destroy();
	build_config->destroy();
	engine->destroy();
}

void TRTOnnxBase::LoadGieStreamBuildContext(const std::string& gie_file)
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

void TRTOnnxBase::deserializeCudaEngine(const void* blob_data, std::size_t size)
{
	_runtime = nvinfer1::createInferRuntime(logger);
	assert(_runtime != nullptr);
	_engine = _runtime->deserializeCudaEngine(blob_data, size, nullptr);
	assert(_engine != nullptr);

	_context = _engine->createExecutionContext();
	assert(_context != nullptr);

	mallocInputOutput();
}

void TRTOnnxBase::mallocInputOutput()
{
	int in_counts = _params.max_shape.count();
	cudaHostAlloc((void**)&h_input_tensor_, in_counts * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_input_tensor_, in_counts * sizeof(float));

	int out_counts = _params.max_shape.num() * _params.num_classes *
		_params.max_shape.height() * _params.max_shape.width();
	cudaHostAlloc((void**)&h_output_tensor_, out_counts * sizeof(float), cudaHostAllocDefault);
	cudaMalloc((void**)&d_output_tensor_, out_counts * sizeof(float));

	buffer_queue_.push_back(d_input_tensor_);
	buffer_queue_.push_back(d_output_tensor_);
}

void TRTOnnxBase::SaveRTModel(nvinfer1::IHostMemory* gie_model_stream, const std::string& path)
{
	std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
	outfile.write((const char*)gie_model_stream->data(), gie_model_stream->size());
	outfile.close();
}

void TRTOnnxBase::Forward()
{
	nvinfer1::Dims4 input_dims{ 1, input_shape_.channels(),
								input_shape_.height(), input_shape_.width() };
	_context->setBindingDimensions(0, input_dims);
	_context->enqueueV2(buffer_queue_.data(), stream_, nullptr);

	cudaStreamSynchronize(stream_);
}
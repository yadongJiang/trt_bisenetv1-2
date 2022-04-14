#include "trt_bisenet.h"
#include "mat_transform.hpp"
#include "gpu_func.cuh"

BiSeNet::BiSeNet(const OnnxInitParam& params) : TRTOnnxBase(params)
{
}

cv::Mat BiSeNet::Extract(const cv::Mat& img)
{
	if (img.empty())
		return img;

	std::lock_guard<std::mutex> lock(mtx_);
	/*PreProcessCpu(img);*/
	ProProcessGPU(img);
	Forward();

	/*cv::Mat res = PostProcessCpu();*/
	cv::Mat res = PostProcessGpu();
	return std::move(res);
}

BiSeNet::~BiSeNet()
{
}

void BiSeNet::PreProcessCpu(const cv::Mat& img)
{
	cv::Mat img_tmp = img;

	ComposeMatLambda compose({
		LetterResize(cv::Size(crop_size_, crop_size_), cv::Scalar(114, 114, 114), 32),
		MatDivConstant(255),
		MatNormalize(mean_, std_),
		});

	cv::Mat sample_float = compose(img_tmp);
	input_shape_.Reshape(1, sample_float.channels(), sample_float.rows, sample_float.cols);
	output_shape_.Reshape(1, _params.num_classes, sample_float.rows, sample_float.cols);

	Tensor2VecMat tensor_2_mat;
	std::vector<cv::Mat> channels = tensor_2_mat(h_input_tensor_, input_shape_);
	cv::split(sample_float, channels);

	cudaMemcpy(d_input_tensor_, h_input_tensor_, input_shape_.count() * sizeof(float), 
				cudaMemcpyHostToDevice);
}

void BiSeNet::ProProcessGPU(const cv::Mat& img)
{
	int src_h = img.rows;
	int src_w = img.cols;
	int channels = img.channels();

	float r = MIN(float(crop_size_) / src_h, float(crop_size_) / src_w);

	int dst_h = int(r * src_h);
	int dst_w = int(r * src_w);

	int pad_h = (crop_size_ - dst_h) % stride_;
	int pad_w = (crop_size_ - dst_w) % stride_;

	dst_h += pad_h;
	dst_w += pad_w;

	input_shape_.Reshape(1, channels, dst_h, dst_w);
	output_shape_.Reshape(1, _params.num_classes, dst_h, dst_w);

	biliresize_normalize(d_input_tensor_, channels, src_h, src_w, dst_h, dst_w,
		pad_h, pad_w, r, img.data, mean_.data(), std_.data());
}

cv::Mat BiSeNet::PostProcessCpu()
{
	int num = output_shape_.num();
	int channels = output_shape_.channels();
	int height = output_shape_.height();
	int width = output_shape_.width();
	int count = output_shape_.count();

	cudaMemcpy(h_output_tensor_, d_output_tensor_,
		count * sizeof(float), cudaMemcpyDeviceToHost);

	cv::Mat res = cv::Mat::zeros(height, width, CV_8UC1);
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			vector<float> vec;
			for (int c = 0; c < channels; c++)
			{
				int index = row * width + col + c * height * width;
				float val = h_output_tensor_[index];
				vec.push_back(val);
			}

			int idx = findMaxIdx(vec);
			if (idx == -1)
				continue;
			res.at<uchar>(row, col) = uchar(idx);
		}
	}

	return std::move(res);
}

cv::Mat BiSeNet::PostProcessGpu()
{
	int num = output_shape_.num();
	int channels = output_shape_.channels();
	int height = output_shape_.height();
	int width = output_shape_.width();

	unsigned char* cpu_dst;
	cudaHostAlloc((void**)&cpu_dst, height * width * sizeof(float), cudaHostAllocDefault);
	//==> segmentation(output_tensor_, channels, height, width, cpu_dst);
	segmentation(d_output_tensor_, channels, height, width, cpu_dst);

	cv::Mat res = cv::Mat(height, width, CV_8UC1, cpu_dst);

	cudaFreeHost(cpu_dst);
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

int BiSeNet::findMaxIdx(const vector<float>& vec)
{
	if (vec.empty())
		return -1;
	auto pos = max_element(vec.begin(), vec.end());
	return std::distance(vec.begin(), pos);
}
#ifndef TRT_BISENET_H_
#define TRT_BISENET_H_

#include "trt_onnx_base.h"

#define MIN(x, y) (x) < (y) ? (x) : (y)

using namespace std;

class BiSeNet : public TRTOnnxBase
{
public:
	BiSeNet() = delete;
	BiSeNet(const OnnxInitParam& params);

	virtual ~BiSeNet();

	cv::Mat Extract(const cv::Mat& img);

private:
	// cpu预处理
	void PreProcessCpu(const cv::Mat& img);
	// gpu预处理
	void ProProcessGPU(const cv::Mat& img);

	// cpu后处理
	cv::Mat PostProcessCpu();
	// gpu后处理
	cv::Mat PostProcessGpu();

	// softmax函数
	static void softmax(vector<float>& vec);
	static int findMaxIdx(const vector<float>& vec);

private:
	int crop_size_ = 640;
	int stride_ = 32;

	std::vector<float> mean_{ 0.485, 0.456, 0.406 };
	std::vector<float> std_{ 0.229, 0.224, 0.225 };
};

#endif
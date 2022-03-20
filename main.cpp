#include "trt_bisenet.h"

int main(int argc, char** argv)
{
	OnnxInitParam params;
	params.onnx_model_path = "E:/BaiduNetdiskDownload/BiSeNetv3/checkpoints/onnx/bisenetv3.onnx";
	params.use_fp16 = true;
	params.gpu_id = 0;
	params.num_classes = 4;
	params.max_shape = Shape(1, 3, 640, 640); // 设置最大网络输入大小，用于分配内存(显存)

	BiSeNet model(params);

	cv::Mat img = cv::imread("E:/BaiduNetdiskDownload/BiSeNetv3/datas/tupian.jpg");

	cv::Mat res = model.Extract(img);
	cv::imshow("res", res);

	res = model.Extract(img);
	cv::imshow("res1", res);
	cv::waitKey();
}
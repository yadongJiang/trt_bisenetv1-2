#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

typedef unsigned char uchar;

__device__ void softmax(float *src, int channels);

/* 分割核函数,获得每一个像素点的预测概率
    src_ptr: 输入数据，模型输出的预测结果
	channels: 模型输出通道数(类别数)
	height: 高
	width: 宽
*/
__global__ void kernel_segmentation_get_logits(float* src_ptr, int channels, int height, int width);

/* 分割核函数，获得每一个像素的分类结果
	src_ptr: 输入数据，模型输出的预测结果
	src_ptr: 输入数据，模型输出的预测结果
	channels: 模型输出通道数(类别数)
	height: 高
	width: 宽
	dev_dst: 保存的分割结果
*/
__global__ void kernel_segmentation_get_cls(float* src_ptr, int channels, int height, int width, unsigned char *dev_dst);

/* 双线性插值核函数
*/
__global__ void kernel_bilinear_resize(float* dst_ptr, int channels, int src_h, int src_w, int dst_h, int dst_w, 
									   int pad_h, int pad_w, float r, unsigned char *src_ptr);

/* 归一化核函数
*/
__global__ void kernel_normalize(float* dst_ptr, int channels, int src_h, int src_w); // , float *mean, float* std

/* kernel_segmentation_get_logits核函数调用接口
*/
void segmentation(float* src_ptr, int channels, int height, int width, float* cpu_dst);

/* kernel_segmentation_get_cls核函数调用接口
*/
void segmentation(float* src_ptr, int channels, int height, int width, unsigned char *cpu_dst);

/* 双线性与归一化对外接口
*/
void biliresize_normalize(float* dst_ptr, int channels, int src_h, int src_w, int dst_h, int dst_w, 
					 int pad_h, int pad_w, float r, unsigned char *src_ptr, float *mean, float* std);

#endif
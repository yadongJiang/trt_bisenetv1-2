#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

__device__ void softmax(float *src, int channels);
__global__ void gpu_segmentation(float* ptr, int channels, int height, int width);
__global__ void gpu_segmentation_with_logits(float* ptr, int channels, int height, int width, unsigned char *dev_dst);

void segmentation(float* src_ptr, int channels, int height, int width, float* cpu_dst);
void segmentation(float* src_ptr, int channels, int height, int width, unsigned char *cpu_dst);

#endif
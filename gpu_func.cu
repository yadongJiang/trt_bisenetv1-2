#include "gpu_func.cuh"
#include <cmath>

__device__ void softmax(float* src, int channels)
{
	float tol = 0.0;
	for (int i = 0; i < channels; i++)
	{
		src[i] = exp(src[i]);
		tol += src[i];
	}
	for (int i = 0; i < channels; i++)
		src[i] = src[i] / tol;
}

__global__ void gpu_segmentation(float* ptr, int channels, int height, int width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = height * width;

	float tol = 0.0;
	for (int c = 0; c < channels; c++)
	{
		float tmp = ptr[y * width + x + c * offset];
		tmp = std::exp(tmp);
		tol += tmp;
		ptr[y * width + x + c * offset] = tmp;
	}

	for (int c = 0; c < channels; c++)
	{
		ptr[y * width + x + c * offset] /= tol;
	}
}

__global__ void gpu_segmentation_with_logits(float* ptr, int channels, int height, int width, unsigned char* dev_dst)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = height * width;

	int idx = 0;
	float max_val = FLT_MIN;
	for (int c = 0; c < channels; c++)
	{
		float tmp = ptr[y * width + x + c * offset];
		if (tmp > max_val)
		{
			max_val = tmp;
			idx = c;
		}
	}
	dev_dst[y * width + x] = idx;
}

void segmentation(float* src_ptr, int channels, int height, int width, float* cpu_dst)
{
	dim3 grid(32, 32);
	dim3 blocks(int(width / 32), int(height / 32));

	gpu_segmentation << <grid, blocks >> > (src_ptr, channels, height, width);

	cudaMemcpy(cpu_dst, src_ptr, channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);
}

void segmentation(float* src_ptr, int channels, int height, int width, unsigned char* cpu_dst)
{
	dim3 grid(int(width / 32), int(height / 32));
	dim3 blocks(32, 32);
	unsigned char* dev_dst;
	cudaMalloc((void**)&dev_dst, height * width * sizeof(unsigned char));
	gpu_segmentation_with_logits << <grid, blocks >> > (src_ptr, channels, height, width, dev_dst);

	cudaMemcpy(cpu_dst, dev_dst, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}
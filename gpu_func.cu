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

void segmentation(float* src_ptr, int channels, int height, int width, float* cpu_dst)
{
	dim3 grid(32, 32);
	dim3 blocks(int(width / 32), int(height / 32));

	gpu_segmentation << <grid, blocks >> > (src_ptr, channels, height, width);

	cudaMemcpy(cpu_dst, src_ptr, channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);
}
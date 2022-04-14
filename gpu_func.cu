#include "gpu_func.cuh"
#include <cmath>

__constant__ float const_mean[3];
__constant__ float const_std[3];

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

__global__ void kernel_segmentation_get_logits(float* ptr, int channels, int height, int width)
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

__global__ void kernel_segmentation_get_cls(float* ptr, int channels,
	int height, int width, unsigned char* dev_dst)
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

	kernel_segmentation_get_logits << <grid, blocks >> > (src_ptr, channels, height, width);

	cudaMemcpy(cpu_dst, src_ptr, channels * height * width * sizeof(float), cudaMemcpyDeviceToHost);
}

void segmentation(float* src_ptr, int channels, int height, int width, unsigned char* cpu_dst)
{
	dim3 grid(int(width / 32), int(height / 32));
	dim3 blocks(32, 32);
	unsigned char* dev_dst;
	cudaMalloc((void**)&dev_dst, height * width * sizeof(unsigned char));
	kernel_segmentation_get_cls << <grid, blocks >> > (src_ptr, channels, height, width, dev_dst);

	cudaMemcpy(cpu_dst, dev_dst, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

__global__ void kernel_bilinear_resize(float* dst_ptr, int channels, int src_h, int src_w,
					int dst_h, int dst_w, int pad_h, int pad_w, float r, uchar* src_ptr)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = y * dst_w + x;

	if (x >= dst_w - pad_w || y >= dst_h - pad_h)
	{
		for (int c = 0; c < channels; c++)
		{
			float value = 114. / 255.;
			dst_ptr[offset + (channels - c - 1) * dst_h * dst_w] = value;
		}
		return;
	}

	float src_x = (x + 0.5) / r - 0.5;
	float src_y = (y + 0.5) / r - 0.5;

	int src_x_0 = int((src_x));
	int src_y_0 = int((src_y));

	int src_x_1 = src_x_0 + 1 < src_w - 1 ? src_x_0 + 1 : src_w - 1;
	int src_y_1 = src_y_0 + 1 < src_h - 1 ? src_y_0 + 1 : src_h - 1;
	for (int c = 0; c < channels; c++)
	{

		unsigned char v00 = src_ptr[(src_y_0 * src_w + src_x_0) * channels + c];
		unsigned char v01 = src_ptr[(src_y_0 * src_w + src_x_1) * channels + c];
		float value0 = (src_x_1 - src_x) * float(v00) + (src_x - src_x_0) * float(v01);

		unsigned char v10 = src_ptr[(src_y_1 * src_w + src_x_0) * channels + c];
		unsigned char v11 = src_ptr[(src_y_1 * src_w + src_x_1) * channels + c];
		float value1 = (src_x_1 - src_x) * float(v10) + (src_x - src_x_0) * float(v11);

		float value = (src_y_1 - src_y) * value0 + (src_y - src_y_0) * value1;

		dst_ptr[offset + (channels - c - 1) * dst_h * dst_w] = value;
	}
}

__global__ void kernel_normalize(float* dst_ptr, int channels, int h, int w) // , float *mean, float* std
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int offset = y * w + x;
	for (int c = 0; c < channels; c++)
	{
		float v = dst_ptr[c * h * w + offset] / 255.0;
		v = v - const_mean[c];
		v = v / const_std[c];
		dst_ptr[c * h * w + offset] = v;
	}
}

void biliresize_normalize(float* dst_ptr, int channels, int src_h, int src_w,
						  int dst_h, int dst_w, int pad_h, int pad_w, float r,
						  uchar* src_ptr, float* mean, float* std)
{
	uchar* src_dev;
	cudaMalloc((void**)&src_dev, src_h * src_w * channels * sizeof(uchar));
	cudaMemcpy(src_dev, src_ptr, src_h * src_w * channels * sizeof(uchar), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(const_mean, mean, channels * sizeof(float));

	cudaMemcpyToSymbol(const_std, std, channels * sizeof(float));


	dim3 grids(int(dst_w / 32), int(dst_h / 32));
	dim3 blocks(32, 32);

	kernel_bilinear_resize << <grids, blocks >> > (dst_ptr, channels, src_h, src_w,
		dst_h, dst_w, pad_h, pad_w, r, src_dev);
	kernel_normalize << <grids, blocks >> > (dst_ptr, channels, dst_h, dst_w);

	cudaFree(src_dev);
}
#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

class Shape
{
public:
	Shape() :num_(0), channels_(0), height_(0), width_(0) {}
	Shape(int num, int channels, int height, int width) :
		num_(num), channels_(channels), height_(height), width_(width) {}
	~Shape() {}

public:
	void Reshape(int num, int channels, int height, int width)
	{
		num_ = num;
		channels_ = channels;
		height_ = height;
		width_ = width;
	}

public:
	inline int num() const
	{
		return num_;
	}
	inline int channels() const
	{
		return channels_;
	}
	inline int height() const
	{
		return height_;
	}
	inline int width() const
	{
		return width_;
	}
	inline int count() const
	{
		return num_ * channels_ * height_ * width_;
	}
private:
	int num_;
	int channels_;
	int height_;
	int width_;
};

class Tensor2VecMat
{
public:
	Tensor2VecMat() {}
	vector<cv::Mat> operator()(float* h_src, const Shape& input_shape)  // const std::vector<int>& input_shape
	{
		vector<cv::Mat> input_channels;
		int channels = input_shape.channels();
		int height = input_shape.height();
		int width = input_shape.width();

		for (int i = 0; i < channels; i++)
		{
			cv::Mat channel(height, width, CV_32FC1, h_src);
			input_channels.push_back(channel);
			h_src += height * width;
		}
		return std::move(input_channels);
	}
};

#endif
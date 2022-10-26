// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <thread>
#include <mutex>
#include <atomic>
#include "net.h"
#include <gpu.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <direct.h>
#include <io.h>
#include <time.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;
int count = 0;
static std::mutex mute;
static std::atomic_bool isOpen;
Size S_mul;
int fps_mul = 0;

void trans_result(string img_name, cv::Mat& org) {
	cv::cvtColor(org, org, CV_BGR2RGB);
	cv::waitKey(0);
	cv::Mat org_img = org;
	string pri = "./sod_results/" + img_name + ".png";
	
	cv::Mat mask = cv::imread("./sod_results/" + img_name + ".png");
	int row = org.rows;
	int col = org.cols;
	string q_save_path = "./trans_results/";
	cv::resize(mask, mask, cv::Size(340, 340), cv::INTER_AREA);
	cv::resize(org, org, cv::Size(340, 340), cv::INTER_AREA);

	cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);

	cv::threshold(mask, mask, 40, 255, cv::THRESH_BINARY);

	for (int i = 0; i < org.rows; i++) {
		for (int j = 0; j < org.cols; j++) {
			if (mask.at<uchar>(i, j) == 0) {
				org.at<cv::Vec3b>(i, j)[0] = 255;
				org.at<cv::Vec3b>(i, j)[1] = 255;
				org.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
	cv::resize(org, org, cv::Size(col, row), cv::INTER_LINEAR);
	cv::imwrite("./trans_results/" + img_name + ".png", org);
}
void show_shape(const char* title, const ncnn::Mat& img) {
	//获取尺寸
	std::cout << title << "\n c, w, h is \n" << img.c << " " << img.w << " " << img.h << "\n";;
}
void visualize(const char* title, const ncnn::Mat& m, int r, int c, int cnt, cv::Mat& orgImg, bool save_out = FALSE)
{
	// 对输入图像进行可视化
	show_shape(title, m);
	string save_path = "./sod_results/";
	std::vector<cv::Mat> normed_feats(m.c);
	for (int i = 0; i < m.c; i++)
	{
		cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));
		
		// cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);
		normed_feats[i] = tmp * 255;
		cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);
	}

	int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
	int th = (m.c - 1) / tw + 1;
	
	cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
	show_map = cv::Scalar(127);
	// tile
	for (int i = 0; i < m.c; i++)
	{
		int ty = i / tw;
		int tx = i % tw;

		normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
	}
	//ncnn::Mat::from_pixels_resize(show_map.data, ncnn::Mat::PIXEL_RGB, show_map.cols, show_map.rows, c, r);
	cv::resize(show_map, show_map, cv::Size(c, r), c, r, cv::INTER_LINEAR);
	//cv::imshow(title, show_map);
	//cv::waitKey(0);
	std::cout << "HERE";
	if (cnt < 10) {
		cv::imwrite(save_path + "000" + to_string(cnt) + ".png", show_map);
		trans_result("000" + to_string(cnt), orgImg);
	}
	else if (cnt < 100) {
		cv::imwrite(save_path + "00" + to_string(cnt) + ".png", show_map);
		trans_result("00" + to_string(cnt), orgImg);
	}
	else if (cnt < 1000) {
		cv::imwrite(save_path + "0" + to_string(cnt) + ".png", show_map);
		trans_result("0" + to_string(cnt), orgImg);
	}
	else {
		cv::imwrite(save_path + to_string(cnt) + ".png", show_map);
		trans_result(to_string(cnt), orgImg);
	}
}

void _visualize(const char* title, const ncnn::Mat& m)
{
	// 对输入图像进行可视化
	std::vector<cv::Mat> normed_feats(m.c);

	for (int i = 0; i < m.c; i++)
	{
		cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

		cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

		cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

		// check NaN
		for (int y = 0; y < m.h; y++)
		{
			const float* tp = tmp.ptr<float>(y);
			uchar* sp = normed_feats[i].ptr<uchar>(y);
			for (int x = 0; x < m.w; x++)
			{
				float v = tp[x];
				if (v != v)
				{
					sp[0] = 0;
					sp[1] = 0;
					sp[2] = 255;
				}

				sp += 3;
			}
		}
	}

	int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
	int th = (m.c - 1) / tw + 1;

	cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
	show_map = cv::Scalar(127);

	// tile
	for (int i = 0; i < m.c; i++)
	{
		int ty = i / tw;
		int tx = i % tw;

		normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
	}

	cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
	cv::imshow(title, show_map);
}


void test_all(string pattern_jpg) {
	// 对文件夹内的所有图像进行显著性检测
	//static int u2net(const cv::Mat& bgr, std::vector<Object>& objects, int r, int c)

	ncnn::Net u2net;

	u2net.opt.use_vulkan_compute = true;
	u2net.load_param("./models/u2netp_v2_sim-opt.param");
	u2net.load_model("./models/u2netp_v2_sim-opt.bin");
	pattern_jpg += "*.jpg";
	//std::string pattern_jpg = "C:\\Users\\Breeze\\Desktop\\A02\\U-2-Net-master\\test_data\\ECSSD\\*.jpg";

	std::vector<cv::String> image_files;
	cv::glob(pattern_jpg, image_files);
	if (image_files.size() == 0) {
		std::cout << "No image files[jpg]" << std::endl;
	}
	int cnt = 0;
	

	//进行图像处理操作......

	for (unsigned int frame = 0; frame < image_files.size(); ++frame) {//image_file.size()代表文件中总共的图片个数
		cnt++;
		printf("inference: %d.jpg\n", cnt);
		clock_t start = clock();//记录起始时间
		cv::Mat image = cv::imread(image_files[frame]);
		cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
		cv::Mat org_img = image;
		int c = image.cols;
		int r = image.rows;
		cv::waitKey(0);
		const int target_size = 320;

		// Rescale
		int img_w = image.cols;
		int img_h = image.rows;
		int new_w = 0, new_h = 0;
		if (img_h > img_w) {
			new_h = target_size * img_h / img_w;
			new_w = target_size;
		}
		else {
			new_h = target_size;
			new_w = target_size * img_w / img_h;
		}

		ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, target_size, target_size);
		
		const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
		const float norm_vals[3] = { 1 / 58.395f, 1 / 57.12f, 1 / 57.375f };
		in.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor ex = u2net.create_extractor();
		ex.set_num_threads(4);
		ex.input("input", in);
		
		ncnn::Mat out;
		ex.extract("output", out);
		clock_t end = clock();
		std::cout << "time cost:" << (end - start)/1000.0 << "s" << endl;
		visualize("title", out, r, c, cnt, org_img);
		//_visualize("title", out);
	}
}


// 该函数正在调试...
static void cameraThreadFunc(int camId, int height, int width, cv::Mat* pFrame)
{
	// 视频帧读取进程函数
	// cv::VideoCapture capture(camId);
	string path = "C:\\Users\\Breeze\\Desktop\\A05\\test\\TEST_01.mp4";
	cv::VideoCapture capture;
	S_mul = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH), (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	fps_mul = capture.get(CV_CAP_PROP_FPS);
	capture.set(cv::CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
	//capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
	//capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
	//capture.set(cv::CAP_PROP_FPS, 30);

	/*if (!capture.isOpened()) {
		isOpen = false;
		std::cout << "Failed to open camera with index " << camId << std::endl;
	}*/

	cv::Mat frame;
	while (isOpen) {
		capture >> frame;
		if (mute.try_lock()) {
			frame.copyTo(*pFrame);
			mute.unlock();
		}
		cv::waitKey(5);
	}
	capture.release();
}


// 该函数正在调试...
cv::Mat frame_process(const char* title, const ncnn::Mat& m, int r, int c, int cnt, cv::Mat& orgImg, bool save_out = FALSE)
{
	std::vector<cv::Mat> normed_feats(m.c);
	for (int i = 0; i < m.c; i++)
	{
		cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

		// cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);
		normed_feats[i] = tmp * 255;
		cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);
	}

	int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
	int th = (m.c - 1) / tw + 1;

	cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
	show_map = cv::Scalar(127);
	// tile
	for (int i = 0; i < m.c; i++)
	{
		int ty = i / tw;
		int tx = i % tw;

		normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
	}


	cv::resize(show_map, show_map, cv::Size(c, r), c, r, cv::INTER_LINEAR);
	if (orgImg.cols != c || orgImg.rows != r) cv::resize(orgImg, orgImg, cv::Size(c, r), c, r, cv::INTER_LINEAR);
	cv::cvtColor(show_map, show_map, cv::COLOR_BGR2GRAY);

	cv::threshold(show_map, show_map, 40, 255, cv::THRESH_BINARY);

	for (int i = 0; i < orgImg.rows; i++) {
		for (int j = 0; j < orgImg.cols; j++) {
			if (show_map.at<uchar>(i, j) == 0) {
				orgImg.at<cv::Vec3b>(i, j)[0] = 255;
				orgImg.at<cv::Vec3b>(i, j)[1] = 255;
				orgImg.at<cv::Vec3b>(i, j)[2] = 255;
			}
		}
	}
	// cv::resize(org, org, cv::Size(col, row), cv::INTER_LINEAR);

	// show_map == mask
	return orgImg;

}


// 该函数正在调试...
void test_video_mulProcess() {
	isOpen = true;
	cv::Mat frame(1920, 1080, CV_8UC3);
	cv::Mat image;
	std::thread thread(cameraThreadFunc, 0, 1920, 1080, &frame);
	int cnt = 0;
	while (isOpen) {
		cnt++;
		mute.lock();
		frame.copyTo(image);
		mute.unlock();
		if (image.empty()) {
			cout << "over";
			break;
		}
		
		ncnn::Net u2net;

		u2net.opt.use_vulkan_compute = false;
		u2net.load_param("./models/u2netp_v2_sim-opt.param");
		u2net.load_model("./models/u2netp_v2_sim-opt.bin");
		
		//进行图像处理操作......
		Size S = Size((int)1920, (int)1080);
		int fps = 25;
		// VideoWriter video("./test.mp4", CV_FOURCC('M', 'J', 'P', 'G'), fps, S, true);
		clock_t start = clock();//记录起始时间
		if (image.empty()) break;
		cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
		cv::Mat org_img = image;
		int c = image.cols;
		int r = image.rows;
		cv::waitKey(0);
		const int target_size = 320;

		// Rescale
		int img_w = image.cols;
		int img_h = image.rows;
		int new_w = 0, new_h = 0;
		if (img_h > img_w) {
			new_h = target_size * img_h / img_w;
			new_w = target_size;
		}
		else {
			new_h = target_size;
			new_w = target_size * img_w / img_h;
		}

		ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, target_size, target_size);

		const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
		const float norm_vals[3] = { 1 / 58.395f, 1 / 57.12f, 1 / 57.375f };
		in.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor ex = u2net.create_extractor();
		ex.set_num_threads(4);
		ex.input("input", in);
		printf("inference: %d", cnt);
		ncnn::Mat out;
		ex.extract("output", out);
		clock_t end = clock();
		cv::Mat frameOut = frame_process("frame", out, r, c, cnt, org_img);
		std::cout << "time cost:" << (end - start) / 1000.0 << "s" << endl;
		//imshow("Frame", frameOut);
		cv::cvtColor(frameOut, frameOut, CV_BGR2RGB);
		// video.write(frameOut);

		cv::waitKey(100);
		cv::imshow("video", frameOut);
		if (cv::waitKey(1) == 'q') {
			break;
		}
	}
	isOpen = false;
	thread.join();
}


void test_video(string path) {
	//static int u2net(const cv::Mat& bgr, std::vector<Object>& objects, int r, int c)
	ncnn::Net u2net;
	u2net.opt.use_vulkan_compute = false;
	u2net.load_param("./models/u2netp_v2_sim-opt.param");
	u2net.load_model("./models/u2netp_v2_sim-opt.bin");
	VideoCapture cap(path);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << endl;
		return;
	}
	int cnt = 0;


	//进行图像处理操作......
	Size S = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
	int fps = cap.get(CV_CAP_PROP_FPS);
	VideoWriter video("./test.mp4", CV_FOURCC('D', 'I', 'V', 'X'), fps, S, true);
	cv::Mat image;
	while (1) {
		cnt++;
		printf("width: %d; height: %d \n", (int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));
		printf("inference: %d.frame\n", cnt);
		clock_t start = clock();//记录起始时间
		cap >> image;
		if (image.empty()) break;
		cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
		cv::Mat org_img = image;
		int c = image.cols;
		int r = image.rows;
		cv::waitKey(0);
		const int target_size = 320;

		// Rescale
		int img_w = image.cols;
		int img_h = image.rows;
		int new_w = 0, new_h = 0;
		if (img_h > img_w) {
			new_h = target_size * img_h / img_w;
			new_w = target_size;
		}
		else {
			new_h = target_size;
			new_w = target_size * img_w / img_h;
		}

		ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_RGB, image.cols, image.rows, target_size, target_size);

		const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
		const float norm_vals[3] = { 1 / 58.395f, 1 / 57.12f, 1 / 57.375f };
		in.substract_mean_normalize(mean_vals, norm_vals);
		ncnn::Extractor ex = u2net.create_extractor();
		ex.set_num_threads(4);
		ex.input("input", in);

		ncnn::Mat out;
		ex.extract("output", out);
		clock_t end = clock();
		cv::Mat frameOut = frame_process("frame", out, r, c, cnt, org_img);
		std::cout << "time cost:" << (end - start) / 1000.0 << "s" << endl;
		//imshow("Frame", frameOut);
		cv::cvtColor(frameOut, frameOut, CV_BGR2RGB);
		video.write(frameOut);
		
	}
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [imagedir]\n or :%s [videopath]", argv[0], argv[0]);
		return -1;
	}
	if (_access("./sod_results", 0) == -1)
	{
		_mkdir("./sod_results");
		std::cout << "mask结果sod_results文件夹不存在，已自动创建\n" << endl;
	}
	else
	{
		std::cout << "mask保存在sod_result中\n" << endl;
	}

	if (_access("./trans_results", 0) == -1)
	{
		_mkdir("./trans_results");
		std::cout << "输出结果trans_results文件夹不存在，已自动创建\n" << endl;
	}
	else
	{
		std::cout << "输出存在trans_results中\n" << endl;
	}
	char* imagespath = argv[1];
	//char* imagespath = "C:\\Users\\Breeze\\Downloads\\DUT-OMRON-image\\DUT-OMRON-image\\";
	string path(imagespath);
	/*cv::Mat m = cv::imread(imagepath, 1);
	if (m.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}*/
	//u2net(m, objects, r, c);

	struct stat s;
	
	if (stat(imagespath, &s) == 0) {
		if (s.st_mode & S_IFDIR) {
			std::cout << "process images!\n" << endl;
			test_all(imagespath);
		}
		else if (s.st_mode & S_IFREG) {
			std::cout << "process video!\n" << endl;
			test_video(path);
		}
		else {
			std::cout << "error:not video not images" << std::endl;
		}
	}
	else {
		std::cout << "error, doesn't exist" << std::endl;
	}
	return 0;
}

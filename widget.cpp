#include "widget.h"
#include "ui_widget.h"
#include "net.h"
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <time.h>
using namespace std;
using namespace cv;
QString IMG_name;
QString IMG_AbsoluteDir;
int time_cost;

Widget::Widget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Widget)
{
    ui->setupUi(this);
    connect(ui->imageButton, SIGNAL(clicked(bool)), this, SLOT(OpenImg()));
    connect(ui->convertButton, SIGNAL(clicked(bool)), this, SLOT(Convert()));
	QFont ft;
	ft.setPointSize(18);
	ft.setBold(true);
	ui->label_5->setGeometry(250, 150, 150, 50);
	ui->label_6->setGeometry(700, 150, 150, 50);
	ui->label_7->setGeometry(1150, 150, 150, 50);
	ui->label_5->setFont(ft);
	ui->label_6->setFont(ft);
	ui->label_7->setFont(ft);
	ft.setBold(false);
	ui->src->setFont(ft);
	ui->imageButton->setFont(ft);
	ui->convertButton->setFont(ft);
	ui->message->setFont(ft);
	ft.setPointSize(12);
	ui->srcEdit->setFont(ft);
	ui->srcEdit->setEnabled(true);
	ui->srcEdit->setText("路径中请不要包含中文字符");
}

void Widget::OpenImg()
{
	ncnn::Mat a;
    QString OpenFile, OpenFilePath;
        QImage image;
        //打开文件夹中的图片文件
        OpenFile = QFileDialog::getOpenFileName(this,
                                                  "Please choose an image file",
                                                  "",
                                                  "Image Files(*.jpg *.png *.bmp *.pgm *.pbm);;All(*.*)");
        if( OpenFile != "" )
        {
            if( image.load(OpenFile) )
            {

                ui->original->setPixmap(QPixmap::fromImage(image).scaled(
                                           ui->original->width(),
                                           ui->original->height(),
                                           Qt::KeepAspectRatio,
                                           Qt::SmoothTransformation));
                ui->message->setText("");
                ui->convertButton->setEnabled(true);
            }
            else
            {
                ui->message->setText("error path!");
                ui->convertButton->setEnabled(false);
            }
        }
        else
        {
            ui->message->setText("error path!");
            ui->convertButton->setEnabled(false);
        }

        //显示所示图片的路径
        QFileInfo OpenFileInfo;
        OpenFileInfo = QFileInfo(OpenFile);
        OpenFilePath = OpenFileInfo.filePath();
		IMG_AbsoluteDir = OpenFilePath;
        ui->srcEdit->setText(OpenFilePath);
		Widget::u2net(OpenFileInfo);
}

void Widget::u2net(QFileInfo orgImgPath) {
	clock_t start = clock();//记录起始时间
	ncnn::Net u2net;
	string img_path, img_name;
	QString im_path = orgImgPath.filePath();
	QString im_name = orgImgPath.baseName();
	IMG_name = im_name;
	img_path = im_path.toStdString();
	img_name = im_name.toStdString();

	u2net.opt.use_vulkan_compute = true;
	u2net.load_param("./models/u2netp_v2_sim-opt.param");
	u2net.load_model("./models/u2netp_v2_sim-opt.bin");
	cv::Mat image = imread(img_path);
	cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
	/*cout << format(image, Formatter::FMT_NUMPY);*/
	/*imshow("1", image);
	cv::waitKey(30);*/
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

	ex.input("input", in);
	ncnn::Mat out;
	ex.extract("output", out);
	string save_path = "./mask_results/";
	QDir q_save_path = "./mask_results/";
	std::vector<cv::Mat> normed_feats(out.c);

	for (int i = 0; i < out.c; i++)
	{
		cv::Mat tmp(out.h, out.w, CV_32FC1, (void*)(const float*)out.channel(i));
		printf("m.h:%d\nm.w:%d", out.h, out.w);
		// cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);
		normed_feats[i] = tmp * 255;
		cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);
		
	}

	int tw = out.w < 10 ? 32 : out.w < 20 ? 16 : out.w < 40 ? 8 : out.w < 80 ? 4 : out.w < 160 ? 2 : 1;
	int th = (out.c - 1) / tw + 1;
	printf("\n tw:%d th:%d\n", tw, th);
	cv::Mat show_map(out.h * th, out.w * tw, CV_8UC3);
	show_map = cv::Scalar(127);
	// tile
	for (int i = 0; i < out.c; i++)
	{
		int ty = i / tw;
		int tx = i % tw;

		normed_feats[i].copyTo(show_map(cv::Rect(tx * out.w, ty * out.h, out.w, out.h)));
	}

	cv::resize(show_map, show_map, cv::Size(c, r), c, r, cv::INTER_LINEAR);
	//cv::imshow(title, show_map);
	//cv::waitKey(0);
	string save_dir = save_path + img_name + ".png";
	bool mk = q_save_path.mkdir("../mask_results");
	string sa = save_path + img_name + ".png";
	cv::imwrite(save_path + img_name + ".png", show_map);
	clock_t end = clock();//记录起始时间
	time_cost = (end - start)/1;
}
void Widget::Convert()
{
    QString OpenFile_mask;
    QString OpenFile_result;
    QImage image;
	// load model;

    //这里应该要插一个函数，生成两张图片。
	
    //mask图片
    OpenFile_mask = "./mask_results/" + IMG_name + ".png";
	//OpenFile_mask = "C:\\Users\\Breeze\\Desktop\\A02\\qtdemo\\untitled7\\mask_results\\0004.png";
    //result图片
    OpenFile_result = "./trans_results/" + IMG_name + ".png";
	bool loaded = image.load(OpenFile_mask);
    if( OpenFile_mask != "" )
    {
        if( image.load(OpenFile_mask) )
        {

            ui->mask->setPixmap(QPixmap::fromImage(image).scaled(
                                       ui->mask->width(),
                                       ui->mask->height(),
                                       Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation));
            ui->message->setText("");
        }
    }
    if( OpenFile_result != "" )
    {
		bool loaded = image.load(OpenFile_result);
        if( image.load(OpenFile_result) )
        {

            ui->result->setPixmap(QPixmap::fromImage(image).scaled(
                                       ui->result->width(),
                                       ui->result->height(),
                                       Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation));
            ui->message->setText("");
        }
    }
    //time接收转化用时，从函数得到
	trans();
    std::string message = "time cost:";
    std::string s_time = std::to_string(time_cost);
    std::string s = "ms";
    std::string ans = message + s_time + s;
    ui->message->setText(QString::fromStdString(ans));
}

Widget::~Widget()
{
    delete ui;
}

void Widget::trans() {
	string org_path = IMG_AbsoluteDir.toStdString();
	cv::Mat mask = cv::imread("./mask_results/"+IMG_name.toStdString()+".png");
	cv::Mat org = cv::imread(org_path);
	int row = org.rows;
	int col = org.cols;
	cv::resize(mask, mask, cv::Size(340, 340), cv::INTER_NEAREST);
	cv::resize(org, org, cv::Size(340, 340), cv::INTER_NEAREST);
	QDir q_save_path = "./trans_results/";
	cvtColor(mask, mask, cv::COLOR_BGR2GRAY);

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
	bool mk = q_save_path.mkdir("../trans_results");
	cv::resize(org, org, cv::Size(col, row), cv::INTER_NEAREST);
	cv::imwrite("./trans_results/" + IMG_name.toStdString() + ".png", org);
}
void visualize(const char* title, const ncnn::Mat& m, int r, int c, int cnt)
{
	string save_path = "C:\\Users\\Breeze\\Desktop\\A02\\U-2-Net-master\\test_data\\ECSSD_result_cpp\\";
	std::vector<cv::Mat> normed_feats(m.c);

	for (int i = 0; i < m.c; i++)
	{
		cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));
		printf("m.h:%d\nm.w:%d", m.h, m.w);
		// cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);
		normed_feats[i] = tmp * 255;
		cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);
		// check NaN
		/*for (int y=0; y<m.h; y++)
		{
			const float* tp = tmp.ptr<float>(y);
			uchar* sp = normed_feats[i].ptr<uchar>(y);
			for (int x=0; x<m.w; x++)
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
		}*/
	}

	int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
	int th = (m.c - 1) / tw + 1;
	printf("\n tw:%d th:%d\n", tw, th);
	cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
	show_map = cv::Scalar(127);
	// tile
	for (int i = 0; i < m.c; i++)
	{
		int ty = i / tw;
		int tx = i % tw;

		normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
	}

	cv::resize(show_map, show_map, cv::Size(c, r), c, r, cv::INTER_NEAREST);
	//cv::imshow(title, show_map);
	//cv::waitKey(0);
	if (cnt < 10) {
		cv::imwrite(save_path + "000" + to_string(cnt) + ".png", show_map);
	}
	else if (cnt < 100) {
		cv::imwrite(save_path + "00" + to_string(cnt) + ".png", show_map);
	}
	else if (cnt < 1000) {
		cv::imwrite(save_path + "0" + to_string(cnt) + ".png", show_map);
	}

}

void Widget::Test_all() {
	//static int u2net(const cv::Mat& bgr, std::vector<Object>& objects, int r, int c)

	ncnn::Net u2net;

	u2net.opt.use_vulkan_compute = true;
	u2net.load_param("C:\\Users\\Breeze\\Desktop\\A02\\U-2-Net-master\\saved_models\\ncnn\\u2netp_v2_sim-opt.param");
	u2net.load_model("C:\\Users\\Breeze\\Desktop\\A02\\U-2-Net-master\\saved_models\\ncnn\\u2netp_v2_sim-opt.bin");

	std::string pattern_jpg = "C:\\Users\\Breeze\\Desktop\\A02\\U-2-Net-master\\test_data\\ECSSD\\*.jpg";

	std::vector<cv::String> image_files;
	cv::glob(pattern_jpg, image_files);
	if (image_files.size() == 0) {
		std::cout << "No image files[jpg]" << std::endl;
	}
	int cnt = 0;
	double time0 = static_cast<double>(cv::getTickCount());//记录起始时间

	//进行图像处理操作......

	for (unsigned int frame = 0; frame < image_files.size(); ++frame) {//image_file.size()代表文件中总共的图片个数
		cnt++;
		printf("inference: %d.jpg\n", cnt);

		cv::Mat image = cv::imread(image_files[frame]);
		cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
		/*cout << format(image, Formatter::FMT_NUMPY);*/
		/*imshow("1", image);
		cv::waitKey(30);*/
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
		const float mean_vals_01[3] = { 1.0f, 1.0f, 1.0f };
		const float norm_vals_01[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
		//in.substract_mean_normalize(mean_vals_01, norm_vals_01);

//		const float mean_vals[3] = {1.0f, 1.0f, 1.0f};
//		const float norm_vals[3] = {0.007843f, 0.007843f, 0.007843f};
		const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
		const float norm_vals[3] = { 1 / 58.395f, 1 / 57.12f, 1 / 57.375f };
		in.substract_mean_normalize(mean_vals, norm_vals);
		//in.substract_mean_normalize(mean_vals, 0);
		//pretty_print(in);
		ncnn::Extractor ex = u2net.create_extractor();

		ex.input("input", in);
		printf("indexing");
		ncnn::Mat out;
		ex.extract("output", out);
		printf("w, h, c :%d, %d, %d", out.w, out.h, out.c);
		// pretty_print(out);
		//printf("%d %d %d\n", out.w, out.h, out.c);
		printf("indexed");
		//printf("%d",out);
		visualize("title", out, r, c, cnt);
	}

	time0 = ((double)cv::getTickCount() - time0) / cv::getTickFrequency();
	printf("运行时间：%d", time0);

}

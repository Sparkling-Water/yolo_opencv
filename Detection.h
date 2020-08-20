#pragma once
#ifndef  __DETECTION_H__
#define  __DETECTION_H__

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string.h>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
using namespace dnn;

class Detection
{
public:
	//构造、析构函数
	Detection();
	~Detection();
	//初始化函数
	void Initialize(int width, int height);
	//读取网络模型
	void ReadModel();
	//行人与车辆检测
	bool Detecting(Mat frame);
	//获取网络输出层名称
	vector<String> GetOutputsNames();
	//对输出进行处理，使用NMS选出最合适的框
	void PostProcess();
	//画检测结果
	void Drawer();
	//画出检测框和相关信息
	void DrawBoxes(int classId, float conf, int left, int top, int right, int bottom);
	//获取Mat对象
	Mat GetFrame();
	//获取图像宽度
	int GetResWidth();
	//获取图像高度
	int GetResHeight();

private:
	//图像属性
	int m_width;			//图像宽度
	int m_height;			//图像高度
	//网络处理相关
	Net m_model;			//网络模型
	Mat m_frame;			//每一帧
	Mat m_blob;				//从每一帧创建一个4D的blob用于网络输入
	vector<Mat> m_outs;		//网络输出
	vector<float> m_confs;	//置信度
	vector<Rect> m_boxes;	//检测框左上角坐标、宽、高
	vector<int> m_classIds;	//类别id
	vector<int> m_perfIndx;	//非极大阈值处理后边界框的下标
	//检测超参数
	int m_inpWidth;			//网络输入图像宽度
	int m_inpHeight;		//网络输入图像高度
	float m_confThro;		//置信度阈值
	float m_NMSThro;		//NMS非极大抑制阈值
	vector<string> m_classes; //类别名称

private:
	//内存释放
	void Dump();
};

#endif

#include "Detection.h"
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

void TestDetection()
{
	string image_path = "./data/test.jpg";
	string save_path = "./data/result.jpg";
    	Mat img = imread(image_path);
	cout<<"width: "<<img.cols<<endl;
	cout<<"height: "<<img.rows<<endl;

	Detection detection = Detection();
	detection.Initialize(img.cols, img.rows);
	detection.Detecting(img);
	imwrite(save_path, detection.GetFrame());
	return;
}


int main()
{
	TestDetection();
	return 0;
}

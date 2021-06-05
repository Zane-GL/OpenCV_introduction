# OpenCV案例实战——人脸识别

[TOC]



##项目一：对图片中的人脸进行检测

*注意事项：需提前准备好此文件  **haarcascade_frontalface_default.xml**（OpenCV库中训练好的人脸识别模型）*

*目录如下*：

***D:\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml***

### 代码展示

```c++
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

int main() {

	string path = "Picture/face03.png";
	Mat img = imread(path);

	CascadeClassifier faceCascade;
    faceCascade.load("Model/haarcascade_frontalface_default.xml");

	if (faceCascade.empty()) {
		cout << "XML file not loaded" << endl;
	}

	vector<Rect> faces;
	faceCascade.detectMultiScale(img, faces, 1.1, 10);

	for (int i = 0; i < faces.size(); i++) {
		Mat imgCrop = img(faces[i]);
		imshow(to_string(i), imgCrop);
		rectangle(img, faces[i].tl(), faces[i].br(), Scalar(0, 255, 0), 3);
	}

	imshow("Image", img);
	waitKey(0);
	return 0;
}


```

### 结果展示

![image-20210605190838419](C:\Users\53421\AppData\Roaming\Typora\typora-user-images\image-20210605190838419.png)

### 要点剖析

#### CascadeClassifier类

**功能介绍：**CascadeClassifier是opencv下objdetect模块中用来做目标检测的级联分类器的一个类

**构造函数——load()**

功能：加载XML分类器文件

```c++
CascadeClassifier faceCascade;
faceCascade.load("Model/haarcascade_frontalface_default.xml");
```

**构造函数——detectMultiScale()**

功能：检测图片中人脸

原型：

```c++
void detectMultiScale(
	const Mat& image,
	CV_OUT vector<Rect>& objects,
	double scaleFactor = 1.1,
	int minNeighbors = 3, 
	int flags = 0,
	Size minSize = Size(),
	Size maxSize = Size()
);
```

参数：

**image**：待检测图片，一般为灰度图像加快检测速度

**objects**：被检测物体的矩形框向量组

**scaleFactor**：表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依						次扩大10%

**minNeighbors**：表示构成检测目标的相邻矩形的最小个数(默认为3个)。如果组成检测目标的小矩							形的个数和小于 min_neighbors - 1 都会被排除。 如果min_neighbors 为 0, 则函							数不做任何操作就返回所有的被检候选矩形框，这种设定值一般用在用户自定义							对检测结果的组合程序上

**flags**：要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，如果设置为						CV_HAAR_DO_CANNY_PRUNING，那么函数将会使用Canny边缘检测来排除边缘			过多或过少的区域

**minSize、maxSize**：用来限制得到的目标区域的范围

实现：

```c++
vector<Rect> faces;
faceCascade.detectMultiScale(img, faces, 1.1, 10);
```


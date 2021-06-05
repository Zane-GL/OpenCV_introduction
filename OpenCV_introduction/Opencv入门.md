# OpenCV入门

**目录**

[TOC]



## 图像读取与显示

### imread()

功能：载入一张图片

原型：

```c++
Mat imread(const string& filename, int flags = 1);
```

参数1：需要载入图片的路径名，例如“C:/daima practice/opencv/mat3/mat3/image4.jpg”

参数2：加载图像的颜色类型。默认为1. 若为0则灰度返回，若为1则原图返回

------

### imshow()

功能：显示一张图片

原型：

```c++
Mat imshow(const String &winname, InputArray mat);
```

参数1：显示的窗口名，可以使用cv::namedWindow函数创建窗口，如不创建，imshow函数将自动创建。

参数2：需要显示的图像

------

### namedWindow()

功能：新建一个显示窗口，可以指定窗口的类型

原型：

```c++
void nameWindow(const string& winname,int flags = WINDOW_AUTOSIZE)；
```

参数1：新建的窗口的名称，可随便取。

参数2：窗口的标识，一般默认为WINDOW_AUTOSIZE 

```c++
//WINDOW_AUTOSIZE 窗口大小自动适应图片大小，并且不可手动更改

//WINDOW_NORMAL 用户可以改变这个窗口大小

//WINDOW_OPENGL 窗口创建的时候会支持OpenGL
```

------

### waitkey()

功能：在一个给定的时间内(单位:ms)等待用户按键触发，如果用户没有按下键，则继续等待(循环)

原型：

```c++
int waitKey	(int delay = 0)
```

参数：一个int类型的参数x，默认值为0。

​          等待x秒，如果在x秒期间，按下任意键，则立刻结束并返回按下键的ASCll码，否则返            回-1。

​          若 x=0，那么会无限等待下去，直到有按键按下

```c++
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
using namespace cv;
using namespace std;
int main(int argc, char* argv[]) {
	//加载图片
	Mat src = imread("D:\\Picture\\01.jpg");
	if (src.empty()) {
		printf("can't load\n");
		return -1;
	}
	//显示图片
	string winName = "OpenCv 01";
	namedWindow(winName, WINDOW_FREERATIO);
	imshow(winName, src);
	waitKey(0);
	return 0;
}
```

## 头文件

**quickopencv.h**

```c++
#pragma once
#include<opencv2/opencv.hpp>
using namespace cv;

class QuickDemo {
public:
	void colorSpace_demo(Mat& imge);//图像色彩空间转换
	void mat_creation_demo(Mat& image);//图像对象的创建与赋值
    void pixel_visit_demo(Mat& image);//图像像素的读写操作
    void operators_demo(Mat& image);//图像像素的算术操作
    void tracking_bar_demo(Mat& image);//滚动条调整图像亮度
    void key_demo(Mat& image);//键盘响应操作
    void bitwise_deme(Mat& image);//图像像素逻辑操作
    void channels_demo(Mat& image);//通道合并与分离
    void inrange_demo(Mat& image);//图像色彩空间转换
    void pixel_statistic_demo(Mat& image);//图像像素值统计
    void drawing_demo(Mat& image);//图像几何绘制
    void random_drawing(Mat& image);//随机数与随机颜色
    void mouse_drawing_demo(Mat &image);//鼠标操作与响应
    void norm_demo(Mat& image);//图像像素类型转换与归一化
    void resize_demo(Mat& image);//图像缩放与插值
    void flip_demo(Mat& image);//图像翻转
    void rotate_demo(Mat& image);//图像旋转
    void video_demo(Mat& image);//视频文件摄像头调用
};
```

## 图像色彩空间转换(1)

### cvtColor()

功能：颜色空间转换函数，可以实现RGB颜色向HSV，HSI等颜色空间转换。也可以转换为灰度图。

原型：

```c++
void cvtColor(InputArray src,OutputArray dst,int code,int dstCn = 0);
参数：
// 输入图  InputArray src
// 输出图  OutputArray dst
// 颜色映射类型  int code
// 输出的通道数  int dstCn=0，可以使用默认值，什么都不写
```

------

### imwrite()

功能：将图片保存在指定目录下

```c++
void QuickDemo::colorSpace_demo(Mat& image) {
	Mat gray, hsv;
    
	cvtColor(image, hsv, COLOR_BGR2HSV);//转hsv
	cvtColor(image, gray, COLOR_BGR2GRAY);//转灰度
    
	imshow("HSV", hsv);
	imshow("灰度", gray);
    
    //将hsv保存在“D:\\Picture\\01hsv.jpg”目录下
	imwrite("D:\\Picture\\01hsv.jpg",hsv);
	imwrite("D:\\Picture\\01gray.jpg", gray);
}
```

## 图像对象的创建与赋值

### .clone()

```c++
m4 = m3.clone();//将m3的克隆赋给m4,改变m4不不改变m3
```

------

### .copty()

```c++
m3.copyTo(m4);//将m3复制给m4，改变m4不不改变m3
```

------

### Mat::zeros(4,5,CV_8UC3)

功能：创建一个4*5大小，每个像素点通道的值为0的图像对像（矩阵）

参数1：图像的高

参数2：图像的宽

参数3：CV_8UC3，表示：8位，unsigned int 型，3通道

​			 CV_32FC1，表示：32位，float型，1通道

也可这样写：Mat::zeros(Size(8, 8), CV_8UC3);

------

### .cols

表示图像的列数（图像的宽度）

------

### .rows

表示图像的行数（图像的高度）

------

### .channels()

表示图像的通道数，要加“（）”

------

### Scalar()

功能：设置图像各通道像素值

参数：1~4个，分别表示B，G，R，A（透明通道）

```c++
void QuickDemo::mat_creation_demo(Mat& image) {
	Mat m1, m2;
	m1 = image.clone();//克隆
	image.copyTo(m2);//复制

	//创建空白图像
	Mat m3 = Mat::zeros(Size(8, 8), CV_8UC3);
     //创建一个8*8大小，每个像素点通道的值为0的图像对像
	//CV_8UC3表示每位是8位的无符号，通道数为3的数据
    
	//打印m3的宽度，高度，通道数
	std::cout << "width: " << m3.cols << "height: " << m3.rows << "channels: " << m3.channels() << std::endl;
    
	//对已知通道数的图像对象赋值
	m3 = Scalar(255, 0, 0);//B、G、R
    
	//打印图像对象m3
	std::cout << m3 << std::endl;
    
	imshow("创建图像m3", m3);
    
	//赋值
	Mat m4;
	m4 = m3;//m3的值会随m4而发生改变
	m4 = Scalar(0, 255, 255);
	m4 = m3.clone();//将m3的克隆赋给m4,改变m4不不改变m3
	m3.copyTo(m4);//将m3复制给m4，改变m4不不改变m3
}
```

## 图像像素的读写操作

### .at

功能：读取图像像素值

```c++
//单通道
image.at<uchar>(row, col);
用at方法时对应的typename做个总结：
CV_8U=0: bool或者uchar
CV_8S=1: schar或者char
CV_16U=2: ushort
CV_16S=3: short
CV_32S=4: int或者unsigned
CV_32F=5: float
CV_64F=6: double
```

```c++
//三通道
//修改row行col列像素点三通道的值
image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
image.at<Vec3b>(row, col)[1] = 255  - bgr[1];
image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
```

```c++
void QuickDemo::pixel_visit_demo(Mat& image) {
	int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	for (int row = 0; row < h; row++) { 
		for (int col = 0; col < w; col++) {
			if (dims == 1) {		//灰度图像（单通道）
				//获取row行col列像素点的字节大小，赋值给pv
				int pv = image.at<uchar>(row, col);
				//修改row行col列像素点
				image.at<uchar>(row, col) = 255 - pv;
			}
			if (dims == 3) {		//彩色图像（3通道）
				//获取row行col列像素点三个通道的字节大小
				Vec3b bgr = image.at<Vec3b>(row, col);
				//修改row行col列像素点三通道的值
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255  - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2];
			}
		}
	}
	String winName = "像素读写演示";
	namedWindow(winName, WINDOW_FREERATIO);
	imshow(winName, image);
}
```

## 图像像素算术操作

**通过图像像素点通道的算术操作来对图像的亮度进行修改**

### add()

功能：对2个图像按像素相加，或对1个图像和1个标量按像素相加

参数1：输入：图像 或者 标量

参数2：增加值（图像 或者 标量）

参数3：输出：相加后的图像

------

### subtract()

功能：对2个图像按像素相减，或对1个图像和1个标量按像素相减

参数1：输入：图像 或者 标量

参数2：减去值（图像 或者 标量）

参数3：输出：相减后的图像

------

### multiply()

功能：对2个图像按像素相乘，或对1个图像和1个标量按像素相乘

参数1：输入：图像 或者 标量

参数2：被乘值（图像 或者 标量）

参数3：输出：相乘后的图像

------

### divide()

功能：对2个图像按像素相除，或对1个图像和1个标量按像素相除

参数1：输入：图像 或者 标量

参数2：被除值（图像 或者 标量）

参数3：输出：相除后的图像

```c++
void QuickDemo::operators_demo(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());

	m = Scalar(2, 2, 2);
	//乘法
	multiply(image, m, dst);
	imshow("乘法操作", dst);
	//加法
	/*int w = image.cols;
	int h = image.rows;
	int dims = image.channels();
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			Vec3b p1 = image.at<Vec3b>(row, col);
			Vec3b p2 = m.at<Vec3b>(row, col);
			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] + p2[0]);
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] + p2[1]);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] - p2[2]);
		}
	}*/
	add(image, m, dst);
	imshow("加法操作", dst);
	//减法
	subtract(image, m, dst);
	imshow("减法操作", dst);*/
	//除法
	divide(image, m, dst);
	imshow("除法操作", dst);
}
```

## 滚动条调整图像亮度与对比度调整

### createTrackbar()

功能：通过改变滑动条的位置来改变函数里面变量的值

> 比如我们需要把程序里面的变量i改变为10，20,30就可以分别滑到10，20,30，可以实时的显示i=10，20,30时的效果图。下面我们通过改变变量 i 的值，实时的把 i 的值输出到屏幕

原型：

```c++
int createTrackbar(const string& trackbarname, const string&winname, int* value,  int count ，TrackbarCallback onChange = 0,  void* userdata = 0);
```

参数1：轨迹条名字

参数2：窗口名字

参数3：滑块初始位置

参数4：表示滑块达到最大位置的值

参数5：默认值为0，指向回调函数

参数6：默认值为0，用户传给回调函数的数据值



```c++
static void on_lightness(int b, void* userdata) {
	Mat image = *((Mat*)userdata);//将userdata强转为Mat类型指针,在对其进行解引用
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	addWeighted(image, 1.0, m, 0, b, dst);
	imshow("亮度与对比度调整", dst);
}

static void on_contrast(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);
	imshow("亮度与对比度调整", dst);
}

void QuickDemo::tracking_bar_demo(Mat& image) {
	namedWindow("亮度与对比度调整", WINDOW_NORMAL);
	int lightness = 50;
	int max_value = 100;
	int contrast_value = 100;
    
	createTrackbar("Value Bar:", "亮度与对比度调整", &lightness, max_value, on_lightness, (void*)(&image));
    
	createTrackbar("Contrast Bar:", "亮度与对比度调整", &contrast_value, 200, on_contrast, (void*)(&image));
	on_lightness(50, &image);
}
```

## 键盘响应操作

```c++
void QuickDemo::key_demo(Mat& image) {
	Mat dst=Mat::zeros(image.size(),image.type());
	while (true){
		int c = waitKey(100);
		if (c == 27) {// ESC键的ASCLL码是27
			break;
		}
		if (c == 49) {// 输入1
			cvtColor(image, dst, COLOR_BGR2GRAY);
		}
		if (c == 50) {// 输入2
			cvtColor(image, dst, COLOR_BGR2HSV);
		}
		if (c == 51) {// 输入3
			dst = Scalar(50, 50, 50);
			add(image, dst, dst);//有bug，多次操作后会溢出
		}
		imshow("键盘响应", dst);
	}
}
```

## 图像像素的逻辑操作

### 与 或 非 异或

```c++
/*
1^1=0
0^0=0
1^0=1
0^1=1

两者相等为0,不等为1.
*/
```

```c++
void QuickDemo::bitwise_deme(Mat& image) {
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);
	imshow("m1", m1);
	imshow("m2", m2);
	Mat dst;
	//bitwise_and(m1, m2, dst);与操作
	//bitwise_or(m1, m2, dst);或操作
	//bitwise_not(image, dst);非操作
	//bitwise_xor(m1,m2,dst);异或操作
	imshow("像素位操作", dst);
}
```

## 通道合并与分离

### split()

功能：通道分离

原型：

```c++
//版本1
void split(const Mat& src,Mat *mvBegin)
//版本2
void split(InputArray m, OutputArrayOfArrays mv);
```

参数1：要进行分离的图像矩阵

参数2：Mat型数组的首地址，或者一个vector<Mat>对象

- *注意：分离出的通道都是黑白灰，而不是红绿蓝。原因是分离后为单通道，相当于分离通道的同时把其他两个通道填充了相同的数值。比如红色通道，分离出红色通道的同时，绿色和蓝色被填充为和红色相同的数值，这样一来就只有黑白灰了。那么红色体现在哪呢？可以进行观察，会发现原图中颜色越接近红色的地方在红色通道越接近白色。*

  *在纯红的地方在红色通道会出现纯白。*

------

### merge()

功能：合并通道

原型：

```c++
//版本1
void merge(const Mat* mv, size_t count, OutputArray dst);
//版本2
void merge(const vector& mv, OutputArray dst );
```

参数1：图像矩阵向量容器

参数2：输出

```c++
void QuickDemo::channels_demo(Mat& image){
    vector<Mat> mv;
    split(image, mv);
    imshow("蓝色",mv[0]);
    imshow("绿色",mv[1]);
    imshow("红色",mv[2]);
    
    Mat dst;
    mv[0] = 0;
    mv[1] = 0;
    merge(mv, dst);
    imshow("红色",dst);
    mixChannels(&image, &dst, 1, from_to, 3);
    imshow("通道混合", dst);
}
```

## 图像色彩空间转换(2)

### inRange()

功能：提取图像指定色彩范围区域

原型：

```c++
void inRange(InputArray src,InputArray lowerb,InputArray upperb,OutputArray dst);
```

参数1：输入要处理的图像，可以为单通道或多通道

参数2：包含下边界的数组或标量

参数3：包含上边界数组或标量

参数4：输出图像，与输入图像src 尺寸相同且为CV_8U 类型

*注意：该函数输出的dst是一幅二值化之后的图像。*

```c++
void QuickDemo::inrange_demo(Mat& image) {
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);

	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40, 40, 200);
	bitwise_not(mask, mask);
	imshow("mask", mask);
	image.copyTo(redback, mask);
	imshow("roi区域提取", redback);
}

```

## 图像像素值统计

人脸识别算法应用

### Point类

```c++
/*
    opencv中提供了点的模板类，分为2维点模板类Point_和3维点模板类Point3_
    
    Point_通过2维图像平面中的x和y坐标确定点的位置
    Point3_通过3维立体图像中的x、y、z坐标确定点的位置
    
    对于点的坐标的类型可以是int、double、float类型
    
    Point_、Point2i、Point互相等价，所以为了方便我们在定义整型点的时候会直接使用Point
*/
```

------

### minMaxLoc()

功能：计算像素点最大值、最小值的位置

原型：

```c++
void minMaxLoc(InputArray src, CV_OUT double* minVal, CV_OUT double* maxVal=0, CV_OUT Point* minLoc=0,CV_OUT Point* maxLoc=0, InputArray mask=noArray());
```

参数1：*src*  输入图像（矩阵）

参数2：*minVal*  最小值，可输入NULL表示不需要

参数3：*maxVal*  最大值，可输入NULL表示不需要

参数4：*minLoc*  最小值的位置，可输入NULL表示不需要，Point类型。

参数5：*maxLoc*  最大值的位置，可输入NULL表示不需要，Point类型。

参数6：*mask*  可选参数

------

### meanStdDev()

功能：计算矩阵的均值和标准偏差

原型：

```c++
void meanStdDev(InputArray src,OutputArray mean, OutputArray stddev, InputArray mask=noArray())
```

参数1：*src*  输入图像

参数2：*mean*  输出参数，计算均值

参数3：*stddev*  输出参数，计算标准差（方差）

参数4：*mask*  可选参数

```c++
void QuickDemo::pixel_statistic_demo(Mat& image) {
	double minv, maxv;
	Point minLoc, maxLoc;
	std::vector<Mat> mv;
	split(image, mv);
	for (int i = 0; i < mv.size(); i++) {
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, Mat());
		std::cout << "NO channels:" << i << "min value:" << minv << "max value:" << maxv << std::endl;
	}
	Mat mean, stddev;
	Mat redback = Mat::zeros(image.size(), image.type());
	meanStdDev(redback, mean, stddev);
	redback = Scalar(40, 40, 200);
	imshow("redback", redback);
	std::cout << "means:" << mean << std::endl;
	std::cout << "stddev:" << stddev << std::endl;
}

```

## 图像几何图形绘制

### Rect类

```c++
//成员变量：x, y, width, higth
//分别为左上角点的坐标和矩形的宽和高
```

------

### rectangle()

功能：绘制矩形

原型：

```c++
void cvRectangle( CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color,int thickness=1, int line_type=8, int shift=0 );
```

参数1：*img*  输入图像

参数2：*pt1*  矩形的一个顶点

参数3：pt2  矩形对角线上的另一个顶点

参数4：*color*  线条颜色 (RGB) 或亮度（灰度图像 ）(grayscale image）

参数5：*thickness*  组成矩形的线条的粗细程度。取负值时（如 CV_FILLED）函数绘制填充了色彩							  的矩形

参数6：*line_type*  线条的类型

参数7：*shift*  坐标点的小数点位数

------

### circle()

功能：绘制圆

原型：

```c++
void cvCircle( CvArr* img, CvPoint center, int radius, CvScalar color,int thickness=1, int line_type=8, int shift=0 );
```

参数1：*img*  输入图像

参数2：*center*  圆心坐标

参数3：*radius*  圆形的半径

参数4：*color*  线条的颜色

参数5：*thickness*  如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充

参数6：*line_type*  线条的类型。

参数7：*shift*  圆心坐标点和半径值的小数点位数

------

### line()

功能：绘制直线

原型：

```c++
void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
```

参数1：*img*  要绘制线段的图像

参数2：*pt1*  线段的起点

参数3：*pt2*  线段的终点

参数4：*color* 线段的颜色

参数5：*thickness*  线条的宽度

参数6： *lineType*  线段的类型

参数7：*shii*ft  坐标点小数的位数

------

### RotatedRect类

```c++
//成员变量
float angle;	//旋转角度，当角度为0、90、180、270等时，矩形就成了一个直立的矩形
Point2f center; //矩形的质心 
Size2f size;    //矩形的边长  
//成员函数
Rect boundingRect () const;			  //返回包含旋转矩形的最小矩形
operator CvBox2D() const;			  //转换到旧式的cvbox2d结构
void points (Point2f pts[]) const;	   //返回矩形的4个顶点
```

------

### ellipse()

功能：绘制椭圆圆弧和椭圆扇形

原型：

```c++
void cvEllipse(CvArr* img,CvPoint center, CvSize axes, double angle,double start_angle, double end_angle, CvScalar color,int thickness=1, int line_type=8, int shift=0);
```

参数1：*img*  输入图像

参数2：*center*  椭圆圆心坐标

参数3：*axes*  轴的长度

参数4：*angle*  偏转的角度

参数5：*start_angle*  圆弧起始角的角度

参数6：*end_angle*  圆弧终结角的角度

参数7：*color*  线条的颜色

参数8：*thickness*  线条的粗细程度

参数9：*line_type*  线条的类型

参数10：*shift*  圆心坐标点和数轴的精度

------

### addWeighted()

功能：将图像叠加混合

原型：

```c++
void addWeighted(InputArray src1, double alpha,InputArray src2, double beta, double gamma, OutputArray dst, int dtype=-1)
```

参数1：src1  输入图或强度值

参数2：alpha  src1的权重

参数3：src2  输入图或强度值，和src1的尺寸和通道数相同

参数4：beta  src2的权重

参数5：gamma  两图相加后再增加的值

参数6：dst  输出图,输出矩阵和输入矩阵有相同的尺寸和通道数

参数7：dtype  可有可无的输出图深度

```c++
void QuickDemo::drawing_demo(Mat& image) {
	Rect rect;
	rect.x = 100;
	rect.y = 100;
	rect.width = 250; 
	rect.height = 300;
	Mat bg = Mat::zeros(image.size(), image.type());
	rectangle(bg, rect, Scalar(0, 0, 255), -1, 8, 0);//绘制矩形
	circle(bg, Point(350, 400), 15, Scalar(255, 0, 0), -1, 8, 0);//绘制圆
	line(bg, Point(100, 100), Point(350, 400), Scalar(0, 255, 0), 4, LINE_AA, 0);//绘制直线
	RotatedRect rrt;
	rrt.center = Point(200, 200);
	rrt.size = Size(100, 200);
	rrt.angle = 0.0;
	ellipse(bg, rrt, Scalar(0, 255, 255), 2, 8);//绘制椭圆
	Mat dst;
	addWeighted(image, 0.7, bg, 0.3, 0, dst);
	imshow("绘制演示", bg);
}
```

## 随机数与随机颜色

### RNG类

```c++
//随机数类
```

### .uniform()

功能：返回指定范围的随机数

原型：

```c++

```

参数1：

参数2：

```c++
void QuickDemo::random_drawing(Mat& image) {
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345);
	while (true){
		int c = waitKey(50);
		if (c == 27) {
			break;
		}
		int x1 = rng.uniform(0, w);
		int y1 = rng.uniform(0, h);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(b,g,r),1,LINE_AA,0);
		imshow("随机绘制直线演示", canvas);
	}
}
```

## 鼠标操作与响应

### setMouseCallback()

功能：鼠标调用函数

原型：

```c++
void setMousecallback(const string& winname, MouseCallback onMouse, void* userdata=0)
```

参数1：*winname*  窗口的名字

参数2：*onMouse*  鼠标响应函数，回调函数。指定窗口里每次鼠标时间发生的时候，被调用的函数指针。 这个函数的原型应该为*void on_Mouse(int event, int x, int y, int flags, void* serdata);*

*参数3：userdate  传给回调函数的参数*

#### onMouse()

功能：回调函数

原型：

```c++
void on_Mouse(int event, int x, int y, int flags, void* userdata);
```

参数1：*event*  CV_EVENT_*变量之一

参数2：*x*  鼠标指针在图像坐标系的x轴坐标

参数3：*y*  鼠标指针在图像坐标系的y轴坐标

参数4：*flags*  CV_EVENT_FLAG的组合

参数5：*userdata*  用户定义的传递到*setMouseCallback*函数调用的参数

#### 所有事件

```c++
enum
{
    CV_EVENT_MOUSEMOVE      =0,   //鼠标移动
    CV_EVENT_LBUTTONDOWN    =1,   //按下左键
    CV_EVENT_RBUTTONDOWN    =2,   //按下右键
    CV_EVENT_MBUTTONDOWN    =3,   //按下中键
    CV_EVENT_LBUTTONUP      =4,   //放开左键
    CV_EVENT_RBUTTONUP      =5,   //放开右键
    CV_EVENT_MBUTTONUP      =6,   //放开中键
    CV_EVENT_LBUTTONDBLCLK  =7,   //左键双击
    CV_EVENT_RBUTTONDBLCLK  =8,   //右键双击
    CV_EVENT_MBUTTONDBLCLK  =9,   //中键双击
    CV_EVENT_MOUSEWHEEL     =10,  //滚轮滚动
    CV_EVENT_MOUSEHWHEEL    =11   //横向滚轮滚动（还好我鼠标是G502可以这么干）
};
enum
{
    CV_EVENT_FLAG_LBUTTON   =1,   //左键拖拽
    CV_EVENT_FLAG_RBUTTON   =2,   //右键拖拽
    CV_EVENT_FLAG_MBUTTON   =4,   //中键拖拽
    CV_EVENT_FLAG_CTRLKEY   =8,   //按住CTRL拖拽
    CV_EVENT_FLAG_SHIFTKEY  =16,  //按住Shift拖拽
    CV_EVENT_FLAG_ALTKEY    =32   //按住ALT拖拽
};
```



```c++
Point sp(-1, -1);
Point ep(-1, -1);
Mat temp;
static void on_draw(int event, int x, int y, int flags, void* userdata) {
	Mat image_on_draw = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN) {
		sp.x = x;
		sp.y = y;
		std::cout << "stant point:" << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP) {
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {
			Rect box(sp.x, sp.y, dx, dy);
			temp.copyTo(image);
			imshow("ROI区域", image(box));
			rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			imshow("鼠标绘制", image);
			sp.x = -1;
			sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("鼠标绘制", image);
			}
		}
	}
}
void QuickDemo::mouse_drawing_demo(Mat& image) {
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	imshow("鼠标绘制", image);
	temp = image.clone();
}
```

## 图像像素类型转换与归一化

### .convertTo()

功能：转换图像的数据类型

原型：

```c++
void convertTo( OutputArray m, int rtype, double alpha=1, double beta=0 ) const;
```

参数1：*m*  目标矩阵。如果m在运算前没有合适的尺寸或类型，将被重新分配

参数2：*rtype*  目标矩阵的类型。因为目标矩阵的通道数与源矩阵一样，所以rtype也可以看做是目标矩阵的位深度。如果rtype为负值，目标矩阵和源矩阵将使用同样的类型。

参数3：*alpha*  尺度变换因子（可选）

参数4：*beta*  附加到尺度变换后的值上的偏移量（可选）

------

### normalize()

功能：归一化数据，该函数分为范围归一化与数据值归一化

原型：

```c++
void cv::normalize(InputArry src,InputOutputArray dst,double alpha=1,double beta=0,int norm_type=NORM_L2,int dtype=-1,InputArray mark=noArry());
```

参数1：src  输入数组；

参数2：dst  输出数组，数组的大小和原数组一致；

参数3：alpha 1.用来规范值，2.规范范围，并且是下限

参数4：beta  只用来规范范围并且是上限

参数5：norm_type  归一化选择的数学公式类型

参数6：dtype  当为负，输出在大小深度通道数都等于输入，当为正，输出只在深度与输如不同，不同的地方游dtype决定

参数7：mark  掩码。选择感兴趣区域，选定后只能对该区域进行操作

```c++
void QuickDemo::norm_demo(Mat& image) {
	Mat dst;
	std::cout << image.type() << std::endl;
	image.convertTo(image,CV_32F);
	std::cout << dst.type() << std::endl;
	normalize(image, dst, 1.0, 0, NORM_MINMAX);
	std::cout << dst.type() << std::endl;
	imshow("图像数据归一化", dst);
}
```

## 图像缩放与插值

### resize()

功能：调整图像大小

原型：

```c++
void resize(InputArray src, OutputArray dst, Size dsize, double fx=0, double fy=0, int interpolation=INTER_LINEAR);
```

参数1：src  输入图像

参数2：dst  目标图像。当参数dsize不为0时，dst的大小为size；否则，它的大小需要根据src的大小，参数fx和fy决定。dst的类型（type）和src图像相同

参数3：dsize  目标图像大小

参数4：fx  水平轴上的比例因子

参数5：fy  垂直轴上的比例因子

*注意：参数dsize和参数(fx, fy)不能够同时为0*

```c++
void QuickDemo::resize_demo(Mat& image) {
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	resize(image, zoomin, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);
	imshow("zoomin", zoomin);
	resize(image, zoomout, Size(w *1.5, h *1.5), 0, 0, INTER_LINEAR);
	imshow("zoomin", zoomout);
}
```

## 图像翻转

### flip()

功能：对图像进行翻转操作

原型：

```c++
void flip(InputArray src, OutputArray dst, int flipCode);
```

参数1：输入图像

参数2：输入图像

参数3：翻转角度

```c++
void QuickDemo::flip_demo(Mat& image) {
	Mat dst;
	//flip(image, dst, 0);//上下翻转
	//flip(image, dst, 1);//左右翻转
	//flip(image, dst, -1);//180°旋转
	imshow("图像翻转", dst);
}
```

## 图像旋转

### getRotationMatrix2D()

功能：使图像绕着某点旋转

原型：

```c++
Mat getRotationMatrix2D(Point2f center, double angle, double scale)
```

参数1：Point2f center  表示旋转的中心点

参数2：double angle  表示旋转的角度

参数3：double scale  图像缩放因子

------

### warpAffine()

功能：对图像进行如旋转、仿射、平移等变换

原型：

```c++
void warpAffine(InputArraysrc,OutputArray dst,InputArray M,Size dsize,int flags=INTER_LINEAR,int borderMode=BORDER_CONSTANT,const Scalar & borderValue = Scalar());
```

参数1：src: 输入图像

参数2：dst: 输出图像，尺寸由dsize指定，图像类型与原图像一致

参数3：M: 2X3的变换矩阵

参数4：dsize: 指定图像输出尺寸

参数5：flags: 插值算法标识符，有默认值INTER_LINEAR，如果插值算法为WARP_INVERSE_MAP, warpAffine函数使用如下矩阵进行图像转换

```c++
void QuickDemo::rotate_demo(Mat& image) {
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
	M = getRotationMatrix2D(Point2f(w / 2, h / 2), 45, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 0, 0));
	imshow("旋转演示", dst);
}
```

## 视频文件摄像头使用

```c++
 void QuickDemo::video_demo(Mat& image) {
	 VideoCapture capture(0);//摄像头
	 VideoCapture capture("D:\\Video\\vtest.avi");//视频
	 Mat frame;
	 while (true){
		 capture.read(frame);
		 flip(frame, frame, 1);
		 if (frame.empty()) {
			 break;
		 }
		 imshow("frame", frame);
		 int c = waitKey(10);
		 if (c == 27) {
			 break;
		 }
	 }
	 capture.release();//释放内存
 }
```

## 视频处理和保存

```c++
 void QuickDemo::video_demo(Mat& image) {
	 //VideoCapture capture(0);//摄像头
	 VideoCapture capture("D:\\Video\\vtest.avi");//视频
	 int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	 int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	 int count = capture.get(CAP_PROP_FRAME_COUNT);
	 double fps = capture.get(CAP_PROP_FPS);
	 std::cout << "frame width: " << frame_width << std::endl;
	 std::cout << "frame height: " << frame_height << std::endl;
	 std::cout << "FPS: " << fps << std::endl;
	 std::cout << "Number of Frames:  " << count << std::endl;
	 VideoWriter writer("D:\\Video\\vtest1.mp4", capture.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);

	 Mat frame;
	 while (true){
		 capture.read(frame);
		 flip(frame, frame, 1);
		 if (frame.empty()) {
			 break;
		 }
		 imshow("frame", frame);
		 colorSpace_demo(frame);
		 writer.write(frame);
		 int c = waitKey(10);
		 if (c == 27) {
			 break;
		 }
	 }
	 capture.release();//释放内存
 }
```

## 图像直方图

# 函数大全

```c++
、cvLoadImage：将图像文件加载至内存；
、cvNamedWindow：在屏幕上创建一个窗口；
、cvShowImage：在一个已创建好的窗口中显示图像；
、cvWaitKey：使程序暂停，等待用户触发一个按键操作；
、cvReleaseImage：释放图像文件所分配的内存；
、cvDestroyWindow：销毁显示图像文件的窗口；
、cvCreateFileCapture：通过参数设置确定要读入的AVI文件；
、cvQueryFrame：用来将下一帧视频文件载入内存；
、cvReleaseCapture：释放CvCapture结构开辟的内存空间；
、cvCreateTrackbar：创建一个滚动条；
、cvSetCaptureProperty：设置CvCapture对象的各种属性；
、cvGetCaptureProperty：查询CvCapture对象的各种属性；
、cvGetSize：当前图像结构的大小；
、cvSmooth：对图像进行平滑处理；
、cvPyrDown：图像金字塔，降采样，图像缩小为原来四分之一；
、cvCanny：Canny边缘检测；
、cvCreateCameraCapture：从摄像设备中读入数据；
、cvCreateVideoWriter：创建一个写入设备以便逐帧将视频流写入视频文件；
、cvWriteFrame：逐帧将视频流写入文件；
、cvReleaseVideoWriter：释放CvVideoWriter结构开辟的内存空间；
、CV_MAT_ELEM：从矩阵中得到一个元素；
、cvAbs：计算数组中所有元素的绝对值；
、cvAbsDiff：计算两个数组差值的绝对值；
、cvAbsDiffS：计算数组和标量差值的绝对值；
、cvAdd：两个数组的元素级的加运算；
、cvAddS：一个数组和一个标量的元素级的相加运算；
、cvAddWeighted：两个数组的元素级的加权相加运算(alpha运算)；
、cvAvg：计算数组中所有元素的平均值；
、cvAvgSdv：计算数组中所有元素的绝对值和标准差；
、cvCalcCovarMatrix：计算一组n维空间向量的协方差；
、cvCmp：对两个数组中的所有元素运用设置的比较操作；
、cvCmpS：对数组和标量运用设置的比较操作；
、cvConvertScale：用可选的缩放值转换数组元素类型；
、cvCopy：把数组中的值复制到另一个数组中；
、cvCountNonZero：计算数组中非0值的个数；
、cvCrossProduct：计算两个三维向量的向量积(叉积)；
、cvCvtColor：将数组的通道从一个颜色空间转换另外一个颜色空间；
、cvDet：计算方阵的行列式；
、cvDiv：用另外一个数组对一个数组进行元素级的除法运算；
、cvDotProduct：计算两个向量的点积；
、cvEigenVV：计算方阵的特征值和特征向量；
、cvFlip：围绕选定轴翻转；
、cvGEMM：矩阵乘法；
、cvGetCol：从一个数组的列中复制元素；
、cvGetCols：从数据的相邻的多列中复制元素；
、cvGetDiag：复制数组中对角线上的所有元素；
、cvGetDims：返回数组的维数；
、cvGetDimSize：返回一个数组的所有维的大小；
、cvGetRow：从一个数组的行中复制元素值；
、cvGetRows：从一个数组的多个相邻的行中复制元素值；
、cvGetSize：得到二维的数组的尺寸，以CvSize返回；
、cvGetSubRect：从一个数组的子区域复制元素值；
、cvInRange：检查一个数组的元素是否在另外两个数组中的值的范围内；
、cvInRangeS：检查一个数组的元素的值是否在另外两个标量的范围内；
、cvInvert：求矩阵的逆；
、cvMahalonobis：计算两个向量间的马氏距离；
、cvMax：在两个数组中进行元素级的取最大值操作；
、cvMaxS：在一个数组和一个标量中进行元素级的取最大值操作；
、cvMerge：把几个单通道图像合并为一个多通道图像；
、cvMin：在两个数组中进行元素级的取最小值操作；
、cvMinS：在一个数组和一个标量中进行元素级的取最小值操作；
、cvMinMaxLoc：寻找数组中的最大最小值；
、cvMul：计算两个数组的元素级的乘积(点乘)；
、cvNot：按位对数组中的每一个元素求反；
、cvNormalize：将数组中元素进行归一化；
、cvOr：对两个数组进行按位或操作；
、cvOrs：在数组与标量之间进行按位或操作；
、cvReduce：通过给定的操作符将二维数组简为向量；
、cvRepeat：以平铺的方式进行数组复制；
、cvSet：用给定值初始化数组；
、cvSetZero：将数组中所有元素初始化为0；
、cvSetIdentity：将数组中对角线上的元素设为1，其他置0；
、cvSolve：求出线性方程组的解；
、cvSplit：将多通道数组分割成多个单通道数组；
、cvSub：两个数组元素级的相减；
、cvSubS：元素级的从数组中减去标量；
、cvSubRS：元素级的从标量中减去数组；
、cvSum：对数组中的所有元素求和；
、cvSVD：二维矩阵的奇异值分解；
、cvSVBkSb：奇异值回代计算；
、cvTrace：计算矩阵迹；
、cvTranspose：矩阵的转置运算；
、cvXor：对两个数组进行按位异或操作；
、cvXorS：在数组和标量之间进行按位异或操作；
、cvZero：将所有数组中的元素置为0；
、cvConvertScaleAbs：计算可选的缩放值的绝对值之后再转换数组元素的类型；
、cvNorm：计算数组的绝对范数， 绝对差分范数或者相对差分范数；
、cvAnd：对两个数组进行按位与操作；
、cvAndS：在数组和标量之间进行按位与操作；
、cvScale：是cvConvertScale的一个宏，可以用来重新调整数组的内容，并且可以将参数从一种数据类型转换为另一种；
、cvT：是函数cvTranspose的缩写；
、cvLine：画直线；
、cvRectangle：画矩形；
、cvCircle：画圆；
、cvEllipse：画椭圆；
、cvEllipseBox：使用外接矩形描述椭圆；
、cvFillPoly、cvFillConvexPoly、cvPolyLine：画多边形；
、cvPutText：在图像上输出一些文本；
、cvInitFont：采用一组参数配置一些用于屏幕输出的基本个特定字体；
、cvSave：矩阵保存；
、cvLoad：矩阵读取；
、cvOpenFileStorage：为读/写打开存储文件；
、cvReleaseFileStorage：释放存储的数据；
、cvStartWriteStruct：开始写入新的数据结构；
、cvEndWriteStruct：结束写入数据结构；
、cvWriteInt：写入整数型；
、cvWriteReal：写入浮点型；
、cvWriteString：写入字符型；
、cvWriteComment：写一个XML或YAML的注释字串；
、cvWrite：写一个对象；
、cvWriteRawData：写入多个数值；
、cvWriteFileNode：将文件节点写入另一个文件存储器；
、cvGetRootFileNode：获取存储器最顶层的节点；
、cvGetFileNodeByName：在映图或存储器中找到相应节点；
、cvGetHashedKey：为名称返回一个惟一的指针；
、cvGetFileNode：在映图或文件存储器中找到节点；
、cvGetFileNodeName：返回文件的节点名；
、cvReadInt：读取一个无名称的整数型；
、cvReadIntByName：读取一个有名称的整数型；
、cvReadReal：读取一个无名称的浮点型；
、cvReadRealByName：读取一个有名称的浮点型；
、cvReadString：从文件节点中寻找字符串；
、cvReadStringByName：找到一个有名称的文件节点并返回它；
、cvRead：将对象解码并返回它的指针；
、cvReadByName：找到对象并解码；
、cvReadRawData：读取多个数值；
、cvStartReadRawData：初始化文件节点序列的读取；
、cvReadRawDataSlice：读取文件节点的内容；
、cvGetModuleInfo：检查IPP库是否已经正常安装并且检验运行是否正常；
、cvResizeWindow：用来调整窗口的大小；
、cvSaveImage：保存图像；
、cvMoveWindow：将窗口移动到其左上角为x,y的位置；
、cvDestroyAllWindow：用来关闭所有窗口并释放窗口相关的内存空间；
、cvGetTrackbarPos：读取滑动条的值；
、cvSetTrackbarPos：设置滑动条的值；
、cvGrabFrame：用于快速将视频帧读入内存；
、cvRetrieveFrame：对读入帧做所有必须的处理；
、cvConvertImage：用于在常用的不同图像格式之间转换；
、cvErode：形态腐蚀；
、cvDilate：形态学膨胀；
、cvMorphologyEx：更通用的形态学函数；
、cvFloodFill：漫水填充算法，用来进一步控制哪些区域将被填充颜色；
、cvResize：放大或缩小图像；
、cvPyrUp：图像金字塔，将现有的图像在每个维度上都放大两倍；
、cvPyrSegmentation：利用金字塔实现图像分割；
、cvThreshold：图像阈值化；
、cvAcc：可以将8位整数类型图像累加为浮点图像；
、cvAdaptiveThreshold：图像自适应阈值；
、cvFilter2D：图像卷积；
、cvCopyMakeBorder：将特定的图像轻微变大，然后以各种方式自动填充图像边界；
、cvSobel：图像边缘检测，Sobel算子；
、cvLaplace：拉普拉斯变换、图像边缘检测；
、cvHoughLines2：霍夫直线变换；
、cvHoughCircles：霍夫圆变换；
、cvRemap：图像重映射，校正标定图像，图像插值；
、cvWarpAffine：稠密仿射变换；
、cvGetQuadrangleSubPix：仿射变换；
、cvGetAffineTransform：仿射映射矩阵的计算；
、cvCloneImage：将整个IplImage结构复制到新的IplImage中；
、cv2DRotationMatrix：仿射映射矩阵的计算；
、cvTransform：稀疏仿射变换；
、cvWarpPerspective：密集透视变换(单应性)；
、cvGetPerspectiveTransform：计算透视映射矩阵；
、cvPerspectiveTransform：稀疏透视变换；
、cvCartToPolar：将数值从笛卡尔空间到极坐标(极性空间)进行映射；
、cvPolarToCart：将数值从极性空间到笛卡尔空间进行映射；
、cvLogPolar：对数极坐标变换；
、cvDFT：离散傅里叶变换；
、cvMulSpectrums：频谱乘法；
、cvDCT：离散余弦变换；
、cvIntegral：计算积分图像；
、cvDistTransform：图像的距离变换；
、cvEqualizeHist：直方图均衡化；
、cvCreateHist：创建一新直方图；
、cvMakeHistHeaderForArray：根据已给出的数据创建直方图；
、cvNormalizeHist：归一化直方图；
、cvThreshHist：直方图阈值函数；
、cvCalcHist：从图像中自动计算直方图；
、cvCompareHist：用于对比两个直方图的相似度；
、cvCalcEMD2：陆地移动距离(EMD)算法；
、cvCalcBackProject：反向投影；
、cvCalcBackProjectPatch：图块的方向投影；
、cvMatchTemplate：模板匹配；
、cvCreateMemStorage：用于创建一个内存存储器；
、cvCreateSeq：创建序列；
、cvSeqInvert：将序列进行逆序操作；
、cvCvtSeqToArray：复制序列的全部或部分到一个连续内存数组中；
、cvFindContours：从二值图像中寻找轮廓；
、cvDrawContours：绘制轮廓；
、cvApproxPoly：使用多边形逼近一个轮廓；
、cvContourPerimeter：轮廓长度；
、cvContoursMoments：计算轮廓矩；
、cvMoments：计算Hu不变矩；
、cvMatchShapes：使用矩进行匹配；
、cvInitLineIterator：对任意直线上的像素进行采样；
、cvSampleLine：对直线采样；
、cvAbsDiff：帧差；
、cvWatershed：分水岭算法；
、cvInpaint：修补图像；
、cvGoodFeaturesToTrack：寻找角点；
、cvFindCornerSubPix：用于发现亚像素精度的角点位置；
、cvCalcOpticalFlowLK：实现非金字塔的Lucas-Kanade稠密光流算法；
、cvMeanShift：mean-shift跟踪算法；
、cvCamShift：camshift跟踪算法；
、cvCreateKalman：创建Kalman滤波器；
、cvCreateConDensation：创建condensation滤波器；
、cvConvertPointsHomogenious：对齐次坐标进行转换；
、cvFindChessboardCorners：定位棋盘角点；
、cvFindHomography：计算单应性矩阵；
、cvRodrigues2：罗德里格斯变换；
、cvFitLine：直线拟合算法；
、cvCalcCovarMatrix：计算协方差矩阵；
、cvInvert：计算协方差矩阵的逆矩阵；
、cvMahalanobis：计算Mahalanobis距离；
、cvKMeans2：K均值；
、cvCloneMat：根据一个已有的矩阵创建一个新矩阵；
、cvPreCornerDetect：计算用于角点检测的特征图；
、cvGetImage：CvMat图像数据格式转换成IplImage图像数据格式；
、cvMatMul：两矩阵相乘；
```
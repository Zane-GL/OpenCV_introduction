#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
 
float w = 255, h = 155;
Mat image, image_copy, matrix, imgWarp;
Point2f pt1[4] = { {0.0,0.0}, {0.0f,0.0f}, {0.0f,0.0f}, {0.0f,0.0f} }, pt2[4] = { {0.0f,0.0f},{w,0.0f}, {0.0f,h},{w,h} };
string winName = "Image";

static void onMouse(int event, int x, int y, int flags, void* userdata) {
	static int i = 0;
	switch (event)
	{
		case EVENT_LBUTTONDOWN://×ó»÷
		{
			pt1[i] = Point(x, y);
			circle(image_copy, pt1[i], 1, Scalar(0, 0, 255), 10);
			i++;
			cout << i << endl;
			if (i == 4) {
				matrix = getPerspectiveTransform(pt1, pt2);
				warpPerspective(image_copy, imgWarp, matrix, Size(w, h));
				imshow("Image Warp", imgWarp);
				break;
			}	
		}
	}
	imshow(winName, image_copy);
}
int main06() {
	system("color 2F");

	string path = "Picture/020.jpg";
	image= imread(path);
	image.copyTo(image_copy);

	
	namedWindow(winName, WINDOW_AUTOSIZE);
	imshow(winName, image);

	setMouseCallback(winName, onMouse, (void*)(&image));

	
	waitKey(0);
	return 0;

}
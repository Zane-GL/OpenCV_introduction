#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/objdetect.hpp>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

int main() {

	string path = "Picture/face05.jpg";
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


#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define STEP_SIZEE 1
#define STEP_SIZE 1


Mat rotateImage(const Mat& src, double angle, double scale, Mat center_rotate) {
	//cout << "enter" << endl;
	int temp1 = center_rotate.at<int>(0);
	int temp2 = center_rotate.at<int>(1);
	//cout << "center rotate 1 " << center_rotate.at<int>(0) << "2 " << center_rotate.at<int>(0) << endl;
	Point2f center(center_rotate.at<int>(0),center_rotate.at<int>(1));
	Mat rotMat = getRotationMatrix2D(center, angle, scale);

	Mat dst;
	warpAffine(src, dst, rotMat, Size(1280,720));
	return dst;
}

Mat blendImages(const Mat& img1, const Mat& img2, const Mat& img3, int off_x1, int off_y1, int off_x2, int off_y2,
		int off_x3, int off_y3, int canvas_width, int canvas_height, Mat mask_left, Mat mask_center_0,
		Mat mask_center_1, Mat mask_right) {

	int w1 = img1.cols;
	int h1 = img1.rows;
	int w2 = img2.cols;
	int h2 = img2.rows;
	int w3 = img3.cols;
	int h3 = img3.rows;
	//cout << "width " << w1 << " height "  << h1 << endl;
	// Calculate the canvas size to hold both images, considering the offsets
	int c1 = canvas_width;  
	int c2 = canvas_height;  
	//int c = c1+c1;
	//int v = c2+c2;
	// Create an empty canvas with black background
	Mat canvas1(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas2(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	Mat canvas3(Size(c1, c2), img1.type(), Scalar(0, 0, 0));

	Mat canvas4(Size(c1, c2), img1.type(), Scalar(0, 0, 0));
	
	Mat canvas1_float,canvas2_float,canvas3_float;

	// Place img1 on the canvas at (off_x1, off_y1)
	Mat roi1 = img1(Rect(0, 0, w1-off_x1, h1-off_y1));
	Mat roi4 = canvas1(Rect(off_x1,off_y1, w1-off_x1, h1-off_y1));
	roi1.copyTo(roi4);

	
	canvas1.convertTo(canvas1, CV_32F);
	multiply(canvas1,mask_center_0,canvas1);
	multiply(canvas1,mask_center_1,canvas1);
	canvas1.convertTo(canvas1, CV_8U);

	//imshow("one",roi1);
	//imshow("final",canvas1);
	
	// Place img2 on the canvas at (off_x2, off_y2)
	Mat roi2 = img2(Rect(0, 0, (w2-off_x2), (h2-off_y2)));
	Mat roi5 = canvas2(Rect(off_x2, off_y2, (w2-off_x2), (h2-off_y2)));
	roi2.copyTo(roi5);

	canvas2.convertTo(canvas2, CV_32F);
	multiply(canvas2,mask_left,canvas2);
	canvas2.convertTo(canvas2, CV_8U);	

	Mat roi3 = img3(Rect(0, 0, (w3-off_x3), (h3-off_y3)));
	Mat roi6 = canvas3(Rect(off_x3, off_y3, (w3-off_x3), (h3-off_y3)));
	roi3.copyTo(roi6);
	
	canvas3.convertTo(canvas3, CV_32F);
	multiply(canvas3,mask_right,canvas3);
	canvas3.convertTo(canvas3, CV_8U);	

	imwrite("f1.jpg",canvas1);
	imwrite("f2.jpg",canvas2);
	imwrite("f3.jpg",canvas3);
	
	add(canvas1,canvas2,canvas2);
	add(canvas2,canvas3,canvas4);

	return canvas4;
}

int main(int argc, char **argv) {

	/*if(argc < 8){
		cout << "Usage: ./main <img-1> <img-1-trans-mat> <img-2> <img-2-trans-mat> <img-3> <img-3-trans-mat> <coordinates.yml>" << endl;
		return -1;
	}*/
//	VideoCapture cap1("/home/rushi/1_MY_BEV/web_cam_data/video_left.mkv");
// 	VideoCapture cap2("/home/rushi/1_MY_BEV/web_cam_data/video_right.mkv");
// 	VideoCapture cap3("/home/rushi/1_MY_BEV/web_cam_data/video_left.mkv");

	VideoCapture cap1(2, cv::CAP_V4L2);
	VideoCapture cap2(0, cv::CAP_V4L2);
	VideoCapture cap3(5, cv::CAP_V4L2);
	
	if (!cap1.isOpened() || !cap2.isOpened() || !cap3.isOpened()) {
		std::cout << "unable to open webcam" << std::endl;
	}

	// setting resolution | resizing. 
	cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	
	cap3.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap3.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

	FileStorage fs("master.yml", FileStorage::READ);

	if (!fs.isOpened()) {
		cerr << "Error: Could not open the file for reading!" << endl;
		return -1;
	}

	FileStorage transform_1("c1-mat.yml", FileStorage::READ);
	FileStorage transform_2("c2-mat.yml", FileStorage::READ);
	FileStorage transform_3("c3-mat.yml", FileStorage::READ);
	
	Mat mask_left_0 = imread("c1-view_blendmask_0.jpg");
	Mat mask_center_0 = imread("c2-view_blendmask_0.jpg");
	Mat mask_center_1 = imread("c2-view_blendmask_1.jpg");
	Mat mask_right_0 = imread("c3-view_blendmask_0.jpg");
	
	if (mask_left_0.empty()) {
		std::cerr << "error loading image! left" << std::endl;
		return -1;
	}

	if (mask_center_0.empty()) {
		std::cerr << "error loading image! center 0" << std::endl;
		return -1;
	}

	if (mask_center_1.empty()) {
		std::cerr << "error loading image! center 1" << std::endl;
		return -1;
	}

	if (mask_right_0.empty()) {
		std::cerr << "error loading image! right 0" << std::endl;
		return -1;
	}

	// Divide all pixel values by 255 to normalize them to [0, 1]
	Mat normalized_image_left, normalized_image_center_0, normalized_image_center_1, normalized_image_right;
	
	mask_left_0.convertTo(normalized_image_left, CV_32F, 1.0 / 255.0);
	mask_center_0.convertTo(normalized_image_center_0, CV_32F, 1.0 / 255.0);
	mask_center_1.convertTo(normalized_image_center_1, CV_32F, 1.0 / 255.0);
	mask_right_0.convertTo(normalized_image_right, CV_32F, 1.0 / 255.0);

	if (!transform_1.isOpened() || !transform_2.isOpened() || !transform_3.isOpened()){
		cerr << "Error: Could not open the file for writing!" << endl;
		return -1;
	}

	Mat t_1, t_2,t_3;
	transform_1["mat"] >> t_1;
	transform_2["mat"] >> t_2;
	transform_3["mat"] >> t_3;
	
	Mat img1, img2, img3;   
	Mat rotated1,rotated2,rotated3;
	Mat re1, re2, re3;
	Mat output;

	/* Variables to store angle of rotation */
	double a1 = 0;
	double a2 = 0;
	double a3 = 0;

	/* Variables to store scaling factor for rotation */
	double s1 = 0;
	double s2 = 0;
	double s3 = 0;	

	/* Translation coordinates of three images */
	int x1 = 0;
	int y1 = 0;
	int x2 = 0;
	int y2 = 0;
	int x3 = 0;
	int y3 = 0;

	/* Canvas width & Canvas Height */
	Mat canvas_size;
	fs["CANVAS SIZE"] >> canvas_size;
	int canvas_width = canvas_size.at<int>(0);
	int canvas_height = canvas_size.at<int>(1);

	cout << "Canvas Width " << canvas_width << endl;
	cout << "Canvas Height " << canvas_height << endl;
	cout << endl;
	/* Rotation angle */
	fs["img0_rotate_angle"] >> a1;
	fs["img1_rotate_angle"] >> a2;
	fs["img2_rotate_angle"] >> a3;

	cout << "Img 1 Angle: " << a1 << endl;	
	cout << "Img 2 Angle: " << a2 << endl;	
	cout << "Img 3 Angle: " << a3 << endl;	
	cout << endl;
	/* Scaling factor for rotation */
	fs["img0_scale"] >> s1;
	fs["img1_scale"] >> s2;
	fs["img2_scale"] >> s3;

	cout << "Img 1 Scale factor: " << s1 << endl;	
	cout << "Img 2 Scale factor: " << s2 << endl;	
	cout << "Img 3 Scale factor: " << s3 << endl;	
	cout << endl;
	/* Translated coordinates */
	Mat offset1, offset2, offset3;

	fs["img0offset"] >> offset1;
	x1 = offset1.at<int>(0);
	y1 = offset1.at<int>(1);
	cout << "img 1 offset: x " << x1 << " y " << y1 << endl; 

	fs["img1offset"] >> offset2;
	x2 = offset2.at<int>(0);
	y2 = offset2.at<int>(1);

	cout << "img 2 offset: x " << x2 << " y " << y2 << endl; 

	fs["img2offset"] >> offset3;
	x3 = offset3.at<int>(0);
	y3 = offset3.at<int>(1);

	cout << "img 3 offset: x " << x3 << " y " << y3 << endl; 
	cout << endl;
	/* Center of rotation */
	Mat center_r_1, center_r_2, center_r_3;
	fs["img0_rotate_center"] >> center_r_1;
	fs["img1_rotate_center"] >> center_r_2;
	fs["img2_rotate_center"] >> center_r_3;

	int t1 = center_r_1.at<int>(0);
	int t2 = center_r_1.at<int>(1);
	cout << "Img 1 Center of Rotation: (" << t1 << "," << t2 << ")" <<  endl;
	
	t1 = center_r_2.at<int>(0);
	t2 = center_r_2.at<int>(1);
	cout << "Img 2 Center of Rotation: (" << t1 << "," << t2 << ")" <<  endl;

	t1 = center_r_3.at<int>(0);
	t2 = center_r_3.at<int>(1);
	cout << "Img 3 Center of Rotation: (" << t1 << "," << t2 << ")" <<  endl;

	Mat nimg1, nimg2, nimg3;
	while(1){
		char key = (char)waitKey(1);

                if (key == 27) {
                        break;
                }


		cap1 >> nimg1;
		cap2 >> nimg2;
		cap3 >> nimg3;

		//Mat nimg1 = imread(argv[1]);
		//Mat nimg2 = imread(argv[3]);
		//Mat nimg3 = imread(argv[5]);
		//imshow("raw",nimg3);		
		
		if (nimg1.empty()) {
			cerr << "Error: Could not open one or more images! 1" << endl;
			return -1;
		}

		if (nimg2.empty()) {
			cerr << "Error: Could not open one or more images! 2" << endl;
			return -1;
		
		}

		if (nimg3.empty()) {
			cerr << "Error: Could not open one or more images! 3" << endl;
			return -1;
		}

		warpPerspective(nimg1, img1, t_1, Size(640,480));
		warpPerspective(nimg2, img2, t_2, Size(640,480));
		warpPerspective(nimg3, img3, t_3, Size(640,480));

		//	imshow("one", img1);
		//	imshow("1one", img2);
		//	imshow("2one", img3);

		//cout << "Rotation coordinates : Img 1 " << center_r_1 << endl;
		//cout << "Rotation angle : Img 2 " << center_r_2 << endl;
		//cout << "Rotation angle : Img 3 " << center_r_3 << endl;

		rotated1 = rotateImage(img1, a1, s1, center_r_1); 
		rotated2 = rotateImage(img2, a2, s2, center_r_2); 
		rotated3 = rotateImage(img3, a3, s3, center_r_3); 

		//imshow("one", rotated1);
		//imshow("two", rotated2);
		//imshow("three", rotated3);

		output = blendImages(rotated1, rotated2, rotated3, x1, y1, x2, y2, x3, y3, canvas_width, canvas_height,
				normalized_image_left,normalized_image_center_0,normalized_image_center_1,normalized_image_right);

		imshow("canvas.jpg", output);
		//waitKey(1);
	}

	imwrite("output.jpg", output);

	return 0;
}


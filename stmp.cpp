#include <opencv2/opencv.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	if (argc < 3) {
		cout << "Usage: " << argv[0] << " img1 img2" << endl;
		return -1;
	}

	// Load input images
	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);
	if (img1.empty() || img2.empty()) {
		cout << "Error: cannot load images" << endl;
		return -1;
	}

	// Approximate intrinsic matrix
	float scale = 1000.f;
	Mat K = Mat::eye(3, 3, CV_32F);
	K.at<float>(0,0) = scale;
	K.at<float>(1,1) = scale;
	K.at<float>(0,2) = img1.cols / 2.0f;
	K.at<float>(1,2) = img1.rows / 2.0f;

	// Identity rotation
	Mat R = Mat::eye(3, 3, CV_32F);

	// Create Cylindrical warper
	Ptr<WarperCreator> warper_creator = makePtr<CylindricalWarper>();
	Ptr<detail::RotationWarper> warper = warper_creator->create(scale);

	// Warp both images
	Point tl1, tl2;
	Mat warped1 = warper->warp(img1, K, R, INTER_LINEAR, BORDER_REFLECT, &tl1);
	Mat warped2 = warper->warp(img2, K, R, INTER_LINEAR, BORDER_REFLECT, &tl2);

	// Convert to CV_16SC3 (required by Blender)
	Mat warped1_16s, warped2_16s;
	warped1.convertTo(warped1_16s, CV_16SC3);
	warped2.convertTo(warped2_16s, CV_16SC3);

	// Masks (full valid)
	Mat mask1(warped1.size(), CV_8U, Scalar(255));
	Mat mask2(warped2.size(), CV_8U, Scalar(255));

	// Feather blender
	Ptr<detail::FeatherBlender> blender = makePtr<detail::FeatherBlender>();
	blender->setSharpness(0.02f);

	// Prepare ROI for blender
	Rect dst_roi = detail::resultRoi(tl1, warped1.size(),
			tl2, warped2.size());
	blender->prepare(dst_roi);

	// Feed images
	blender->feed(warped1_16s, mask1, tl1);
	blender->feed(warped2_16s, mask2, tl2);

	// Blend result
	Mat result_s, result_mask;
	blender->blend(result_s, result_mask);

	// Convert back for display
	Mat result;
	result_s.convertTo(result, CV_8UC3);

	imshow("Cylindrical Warped Panorama", result);
	waitKey(0);

	return 0;
}


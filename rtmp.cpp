#include <opencv2/opencv.hpp>
#include <opencv2/shape/shape_transformer.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
	// Load image
	Mat img = imread("im1.jpg");
	if (img.empty()) return -1;

	int h = img.rows, w = img.cols;

	// Build source mesh points
	vector<Point2f> srcPts;
	vector<Point2f> dstPts;
	int num_x = 10, num_y = 10;

	for (int i = 0; i < num_y; i++) {
		for (int j = 0; j < num_x; j++) {
			float x = j * (w / (float)(num_x - 1));
			float y = i * (h / (float)(num_y - 1));
			srcPts.push_back(Point2f(x, y));

			// Apply some warp (sinusoidal in X-direction as demo)
			float newX = x + 20 * sin(y * CV_PI / h);
			dstPts.push_back(Point2f(newX, y));
		}
	}

	// Convert to OpenCV format
	vector<DMatch> matches;
	for (size_t i = 0; i < srcPts.size(); i++) {
		matches.push_back(DMatch((int)i, (int)i, 0));
	}

	Ptr<ThinPlateSplineShapeTransformer> tps = cv::createThinPlateSplineShapeTransformer();
	Mat srcMat(srcPts.size(), 1, CV_32FC2, &srcPts[0]);
	Mat dstMat(dstPts.size(), 1, CV_32FC2, &dstPts[0]);

	tps->estimateTransformation(dstMat, srcMat, matches);

	Mat warped;
	tps->warpImage(img, warped);

	imshow("Mesh Warped", warped);
	waitKey(0);
	return 0;
}


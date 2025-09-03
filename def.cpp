// g++ -std=c++17 -O2 stitch_fg_bg.cpp -o stitch `pkg-config --cflags --libs opencv4`
// Usage: ./stitch [cam0] [cam1]
// Example: ./stitch 0 1   OR   ./stitch video1.mp4 video2.mp4

#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>

using namespace cv;
using namespace cv::detail;
using std::vector;
int blend_type = Blender::MULTI_BAND;
// ---- Helpers ----

// Compute projected corners of an image under H
static void projectCorners(const Size &sz, const Mat &H, std::array<Point2f,4> &out) {
	vector<Point2f> c = { Point2f(0,0), Point2f((float)sz.width,0),
		Point2f((float)sz.width,(float)sz.height), Point2f(0,(float)sz.height) };
	perspectiveTransform(c, c, H);
	for (int i = 0; i < 4; ++i) out[i] = c[i];
}

static Rect2f bboxOf(const std::array<Point2f,4> &c) {
	float minx = c[0].x, miny = c[0].y, maxx = c[0].x, maxy = c[0].y;
	for (int i = 1; i < 4; ++i) {
		minx = std::min(minx, c[i].x); miny = std::min(miny, c[i].y);
		maxx = std::max(maxx, c[i].x); maxy = std::max(maxy, c[i].y);
	}
	return Rect2f(Point2f(minx, miny), Point2f(maxx, maxy));
}

static Mat translationH(double tx, double ty) {
	Mat T = Mat::eye(3,3,CV_64F);
	T.at<double>(0,2) = tx; T.at<double>(1,2) = ty;
	return T;
}

static vector<DMatch> goodMatches(const vector<vector<DMatch>> &knn12,
		const vector<vector<DMatch>> &knn21,
		float ratio=0.75f) {
	vector<DMatch> m12;
	for (const auto &kv : knn12) {
		if (kv.size() < 2) continue;
		if (kv[0].distance < ratio * kv[1].distance) m12.push_back(kv[0]);
	}
	std::map<int,int> best21;
	for (const auto &kv : knn21) if (!kv.empty()) best21[kv[0].trainIdx] = kv[0].queryIdx;
	vector<DMatch> out;
	out.reserve(m12.size());
	for (const auto &m : m12) {
		auto it = best21.find(m.trainIdx);
		if (it != best21.end() && it->second == m.queryIdx) out.push_back(m);
	}
	return out;
}

int main(int argc, char** argv) {
	// ---- Open inputs ----
	VideoCapture cap0, cap1;
	if (argc >= 3) {
		// Try opening as numbers (camera indices)
	//	bool ok0 = cap0.open(std::string(argv[1]).size() == 1 && isdigit(argv[1][0]) ? std::stoi(argv[1]) : argv[1]);
	//	bool ok1 = cap1.open(std::string(argv[2]).size() == 1 && isdigit(argv[2][0]) ? std::stoi(argv[2]) : argv[2]);
		cap0.open(argv[1]);
	        cap1.open(argv[2]);	
		//if (!ok0 || !ok1) {
		//	std::cerr << "Error: cannot open inputs.\n";
		//	return 1;
		//}
	} else {
		cap0.open(0);
		cap1.open(1);
	}
	if (!cap0.isOpened() || !cap1.isOpened()) {
		std::cerr << "Error: cannot open inputs.\n";
		return 1;
	}

	// ---- Background subtractors ----
	Ptr<BackgroundSubtractor> bg0 = createBackgroundSubtractorMOG2(500, 16.0, true);
	Ptr<BackgroundSubtractor> bg1 = createBackgroundSubtractorMOG2(500, 16.0, true);

	// ---- Feature detector/descriptor + matcher ----
	Ptr<Feature2D> orb = ORB::create(3000);
	BFMatcher matcher(NORM_HAMMING, false);

	// ---- Seam finder / blender / exposure ----
	Ptr<SeamFinder> seamFinder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND, false);
	const int NUM_BANDS = 5;

	Mat f0, f1;

	namedWindow("cam0", WINDOW_NORMAL);
	namedWindow("cam1", WINDOW_NORMAL);
	namedWindow("stitched", WINDOW_NORMAL);

	while (true) {
		if (!cap0.read(f0) || !cap1.read(f1)) break;

		// ---- Background masks (invert to keep background) ----
		Mat fg0, fg1, bgMask0, bgMask1;
		bg0->apply(f0, fg0, -1);
		bg1->apply(f1, fg1, -1);

		threshold(fg0, fg0, 200, 255, THRESH_BINARY);
		threshold(fg1, fg1, 200, 255, THRESH_BINARY);
		bitwise_not(fg0, bgMask0);
		bitwise_not(fg1, bgMask1);

		// ---- Feature detection on background only ----
		vector<KeyPoint> k0, k1;
		Mat d0, d1;
		orb->detectAndCompute(f0, bgMask0, k0, d0);
		orb->detectAndCompute(f1, bgMask1, k1, d1);

		if (d0.empty() || d1.empty()) {
			imshow("cam0", f0); imshow("cam1", f1);
			if ((waitKey(1) & 0xFF) == 27) break;
			continue;
		}

		// ---- Matching + Homography ----
		vector<vector<DMatch>> knn01, knn10;
		matcher.knnMatch(d0, d1, knn01, 2);
		matcher.knnMatch(d1, d0, knn10, 2);
		vector<DMatch> matches = goodMatches(knn01, knn10, 0.75f);

		if (matches.size() < 12) {
			imshow("cam0", f0); imshow("cam1", f1);
			if ((waitKey(1) & 0xFF) == 27) break;
			continue;
		}

		vector<Point2f> p0, p1_pts;
		for (const auto &m : matches) {
			p0.push_back(k0[m.queryIdx].pt);
			p1_pts.push_back(k1[m.trainIdx].pt);
		}

		Mat H10 = findHomography(p1_pts, p0, RANSAC, 3.0);
		if (H10.empty()) continue;

		// ---- Build panorama geometry ----
		std::array<Point2f,4> c1proj;
		projectCorners(f1.size(), H10, c1proj);
		std::array<Point2f,4> c0 = { Point2f(0,0), Point2f((float)f0.cols,0),
			Point2f((float)f0.cols,(float)f0.rows), Point2f(0,(float)f0.rows) };
		Rect2f box1 = bboxOf(c1proj);
		Rect2f box0 = bboxOf(c0);
		float minx = std::min(0.f, box1.x);
		float miny = std::min(0.f, box1.y);
		float maxx = std::max(box0.br().x, box1.br().x);
		float maxy = std::max(box0.br().y, box1.br().y);

		int panoW = (int)std::ceil(maxx - minx);
		int panoH = (int)std::ceil(maxy - miny);

		Mat T = translationH(-minx, -miny);
		Mat H10t = T * H10;

		// ---- Warp images into panorama ----
		Mat f1_warp, m1_warp;
		warpPerspective(f1, f1_warp, H10t, Size(panoW, panoH), INTER_LINEAR, BORDER_CONSTANT);
		warpPerspective(bgMask1, m1_warp, H10t, Size(panoW, panoH), INTER_NEAREST, BORDER_CONSTANT);

		Mat f0_canvas(panoH, panoW, f0.type(), Scalar::all(0));
		Mat m0_canvas(panoH, panoW, CV_8U, Scalar::all(0));
		Rect r0((int)std::round(-minx), (int)std::round(-miny), f0.cols, f0.rows);
		f0.copyTo(f0_canvas(r0));
		Mat tmpMask0(f0.rows, f0.cols, CV_8U, Scalar::all(255));
		tmpMask0 = min(tmpMask0, bgMask0);
		tmpMask0.copyTo(m0_canvas(r0));

		// ---- Prepare for seam finding ----
		vector<Point> corners = { r0.tl(), Point(0,0) };
		vector<UMat> imgsU(2), masksU(2);
		f0_canvas.convertTo(imgsU[0], CV_16S);
		f1_warp.convertTo(imgsU[1], CV_16S);
		m0_canvas.copyTo(masksU[0]);
		m1_warp.copyTo(masksU[1]);

		seamFinder->find(imgsU, corners, masksU);

		// ---- Exposure compensation (corrected block) ----
		vector<UMat> imgsForComp(2), masksForComp(2);
		f0_canvas.copyTo(imgsForComp[0]);
		f1_warp.copyTo(imgsForComp[1]);
		m0_canvas.copyTo(masksForComp[0]);
		m1_warp.copyTo(masksForComp[1]);

		compensator->feed(corners, imgsForComp, masksForComp);
		compensator->apply(0, corners[0], imgsForComp[0], masksForComp[0]);
		compensator->apply(1, corners[1], imgsForComp[1], masksForComp[1]);

		// ---- Multi-band blending ----
		//blender->setNumBands(NUM_BANDS);
		// Cast to MultiBandBlender before setting bands
		if (blend_type == Blender::MULTI_BAND) {
			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
			if (mb) {
				mb->setNumBands(NUM_BANDS);
			}
		}

		blender->prepare(corners, vector<Size>{ imgsForComp[0].size(), imgsForComp[1].size() });

		//blender->prepare(corners, vector<Size>{ imgsForComp[0].size(), imgsForComp[1].size() });

		for (int i = 0; i < 2; i++) {
			Mat img16;
			imgsForComp[i].convertTo(img16, CV_16S);
			blender->feed(img16, masksForComp[i], corners[i]);
		}

		Mat pano16, panoMask;
		blender->blend(pano16, panoMask);
		Mat pano;
		pano16.convertTo(pano, CV_8U);

		// ---- Show ----
		imshow("cam0", f0);
		imshow("cam1", f1);
		imshow("stitched", pano16);

		int key = waitKey(1) & 0xFF;
		if (key == 27) break;
	}

	return 0;
}


#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
using namespace cv;
using namespace cv::detail;

int main() {
	VideoCapture cap1("/home/devashree-katarkar/Downloads/left.mkv"), cap2("/home/devashree-katarkar/Downloads/right.mkv");
	if (!cap1.isOpened() || !cap2.isOpened()) return -1;

	// -------------------------------
	// Phase 1: Initialization (first frames)
	// -------------------------------
	Mat frame1, frame2;
	cap1 >> frame1;
	cap2 >> frame2;
	if (frame1.empty() || frame2.empty()) return -1;

	std::vector<Mat> frames = {frame1, frame2};

	// === Feature detection ===
	Ptr<FeaturesFinder> finder = makePtr<OrbFeaturesFinder>();
	std::vector<ImageFeatures> features(frames.size());
	for (int i = 0; i < frames.size(); ++i) {
		computeImageFeatures(finder, frames[i], features[i]);
		features[i].img_idx = i;
	}

	// === Matching ===
	std::vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher = makePtr<BestOf2NearestMatcher>(false, 0.3);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	// === Camera Estimation ===
	std::vector<CameraParams> cameras;
	HomographyBasedEstimator estimator;
	estimator(features, pairwise_matches, cameras);

	Ptr<BundleAdjusterBase> adjuster = makePtr<BundleAdjusterRay>();
	adjuster->setConfThresh(1);
	(*adjuster)(features, pairwise_matches, cameras);

	// Wave correction
	std::vector<cv::detail::CameraParams> cameras_clone = cameras;
	std::vector<cv::detail::CameraParams> cameras_corrected = cameras;
	std::vector<cv::detail::CameraParams> tmp = cameras;
	std::vector<cv::Mat> rmats;
	for (auto &cam : cameras) rmats.push_back(cam.R);
	waveCorrect(rmats, WAVE_CORRECT_HORIZ);
	for (size_t i = 0; i < cameras.size(); ++i) cameras[i].R = rmats[i];

	// Warper
	Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));

	// Exposure compensator & seam finder
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	Ptr<SeamFinder> seam_finder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);

	// Prepare seam masks
	std::vector<Point> corners(frames.size());
	std::vector<Size> sizes(frames.size());
	std::vector<UMat> masks_warped(frames.size());

	for (int i = 0; i < frames.size(); ++i) {
		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		corners[i] = warper->warp(frames[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, frames[i]);
		sizes[i] = frames[i].size();
		masks_warped[i].create(sizes[i], CV_8U);
		masks_warped[i].setTo(Scalar::all(255));
	}
	seam_finder->find(frames, corners, masks_warped);

	// === VideoWriter (initialized after knowing panorama size) ===
	Mat test_pano;
	Blender::createDefault(Blender::MULTI_BAND, false)->blend(test_pano, noArray());
	VideoWriter writer;
	writer.open("stitched_output.mkv", VideoWriter::fourcc('X','2','6','4'), 30, test_pano.size());

	// -------------------------------
	// Phase 2: Per-frame stitching loop
	// -------------------------------
	while (true) {
		cap1 >> frame1;
		cap2 >> frame2;
		if (frame1.empty() || frame2.empty()) break;

		std::vector<Mat> curr_frames = {frame1, frame2};

		// Warp & compensate
		std::vector<Point> curr_corners(frames.size());
		std::vector<UMat> images_warped(frames.size());
		for (int i = 0; i < frames.size(); ++i) {
			Mat K;
			cameras[i].K().convertTo(K, CV_32F);
			curr_corners[i] = warper->warp(curr_frames[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		}

		compensator->feed(curr_corners, images_warped, masks_warped);

		// Blend
		Ptr<Blender> blender = Blender::createDefault(Blender::MULTI_BAND, false);
		blender->prepare(curr_corners, sizes);
		for (int i = 0; i < frames.size(); ++i) {
			blender->feed(images_warped[i], masks_warped[i], curr_corners[i]);
		}
		Mat result, result_mask;
		blender->blend(result, result_mask);

		// Convert & write
		Mat result_8u;
		if (result.depth() != CV_8U)
			result.convertTo(result_8u, CV_8U);
		else
			result_8u = result;

		writer.write(result_8u);

		imshow("Stitched", result_8u);
		if (waitKey(30) == 27) break; // ESC to quit
	}

	return 0;
}


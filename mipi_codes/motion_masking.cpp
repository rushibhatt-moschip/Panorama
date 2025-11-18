//perfect clean code + dynamic seam finding experiments
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <string>
#include <omp.h>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <chrono>
#include <thread>
#include <cmath>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;
// Default command line args
vector<String> img_names;
bool preview = false;
static int num_images = 2;
bool try_cuda = false;
double work_megapix = 0.2;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;

int crop=0;
int resize_win=0;
int left_x_cor=0;
int left_y_cor=0;
int left_width=640;
int left_height=400;
int right_x_cor=0;
int right_y_cor=0;
int right_width=640;
int right_height=400;

#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.3f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
//int blend_type = Blender::FEATHER;
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 0.5f;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;
Ptr<WarperCreator> warper_creator;


cv::Mat prev_gray;
cv::Mat prev_grayy;
cv::Mat save;
int t=1;
int i=1;
int ru=1;
int verif=1;
int now=0;
cv::Mat ru_mask;
cv::Mat first_frame;
cv::Mat computeMotionMask(const cv::Mat& frame)
{
	cv::Mat gray, diff, motion_mask;
	// convert to gray
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	// if first frame → initialize
	if (prev_gray.empty()) {
		gray.copyTo(prev_gray);
		motion_mask = cv::Mat::zeros(gray.size(), CV_8U);
		return motion_mask;
	}

	// frame differencing
	cv::absdiff(gray, prev_gray, diff);
	// binary threshold
	cv::threshold(diff, motion_mask, 25, 255, cv::THRESH_BINARY);

	// noise cleaning
	//	cv::GaussianBlur(motion_mask, motion_mask, cv::Size(5,5), 0);
	//	cv::dilate(motion_mask, motion_mask, cv::Mat(), cv::Point(-1,-1), 2);

	// remember this frame for next iteration
	//gray.copyTo(prev_gray);

	// (optional) restrict to right side region
	int right_start = frame.cols / 2;
	cv::Rect right_region(right_start, 0, frame.cols - right_start, frame.rows);
	cv::Mat right_only = cv::Mat::zeros(frame.size(), CV_8U);
	motion_mask(right_region).copyTo(right_only(right_region));
	return right_only;  // return mask of moving pixels (255=motion)
}

cv::Mat computeMotionMaskk(const cv::Mat& frame)
	//std::pair<cv::Mat, cv::Mat> computeMotionMaskk(const cv::Mat& frame)

{
	cv::Mat gray, diff, motion_mask;
	// convert to gray
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

	// if first frame → initialize
	if (prev_grayy.empty()) {
		gray.copyTo(prev_grayy);
		//cv::imshow("saved frame is",prev_grayy);
		//		cv::waitKey(0);

		motion_mask = cv::Mat::ones(gray.size(), CV_8U)*255;
		return motion_mask;
	}

	// frame differencing
	cv::absdiff(gray, prev_grayy, diff);

	// binary threshold
	cv::threshold(diff, motion_mask, 60, 255, cv::THRESH_BINARY);

	cv::bitwise_not(motion_mask, motion_mask); // invert mask (static=255, moving=0)

	// noise cleaning
	//	cv::GaussianBlur(motion_mask, motion_mask, cv::Size(5,5), 0);
	//	cv::dilate(motion_mask, motion_mask, cv::Mat(), cv::Point(-1,-1), 2);

	// remember this frame for next iteration
	//gray.copyTo(prev_gray);

	// (optional) restrict to right side region
	int left_end = frame.cols / 2;
	cv::Rect left_region(0, 0, left_end, frame.rows);
	cv::Mat left_only = cv::Mat::ones(frame.size(), CV_8U)*255;
	motion_mask(left_region).copyTo(left_only(left_region));
	//return {left_only, prev_grayy}; // <-- return both masks
	return left_only;  // return mask of moving pixels (255=motion)
}

vector<MatchesInfo> pairwise_matches;
Ptr<FeaturesMatcher> matcher;
Ptr<Estimator> estimator;
vector<CameraParams> cameras;
Ptr<detail::BundleAdjusterBase> adjuster;
vector<double> focals;
float warped_image_scale;
Ptr<SeamFinder> seam_finder;
Ptr<ExposureCompensator> compensator; 

using namespace std;
using namespace std::chrono;

// Helper lambda to compute & round duration in seconds
auto duration_s = [](auto start, auto end, int decimals=3) {
	double sec = duration<double>(end - start).count();
	double factor = pow(10.0, decimals);
	return round(sec * factor) / factor;
};

// --- ADD THIS FUNCTION OUTSIDE OF main() ---

Rect findLargestContourBoundingBox(const Mat& mask) {
	// Finds the largest continuous white area in the mask.
	vector<vector<Point>> contours;
	findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	double max_area = 0;
	int largest_contour_idx = -1;
	for (int i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > max_area) {
			max_area = area;
			largest_contour_idx = i;
		}
	}
	return (largest_contour_idx != -1) ? boundingRect(contours[largest_contour_idx]) : Rect();
}


cv::Rect adaptive_crop_panorama(const Mat& stitched_image) {
	const int SAFETY_BUFFER = 2; 
	const double CLEAN_THRESHOLD = 0.95; // 95% white or more

	if (stitched_image.empty()) return Rect();

	// 1. Create Binary Mask
	Mat gray_image, binary_mask;
	cvtColor(stitched_image, gray_image, COLOR_BGR2GRAY);
	threshold(gray_image, binary_mask, 1, 255, THRESH_BINARY);

	Rect bounding_box = findLargestContourBoundingBox(binary_mask);
	if (bounding_box.width == 0) return Rect(); // Return original if no content found

	// Initializing crop lines
	int top_crop_y = bounding_box.y;
	int bottom_crop_y = bounding_box.y + bounding_box.height;
	int left_crop_x = bounding_box.x;
	int right_crop_x = bounding_box.x + bounding_box.width;

	// A. Vertical Crop (Top/Bottom)
	for(int r = bounding_box.y; r < bounding_box.y + bounding_box.height / 2; ++r) {
		int white_pixels = countNonZero(binary_mask.row(r));
		//if (white_pixels > bounding_box.width * CLEAN_THRESHOLD) { //if (white_pixels > bounding_box.width * 0.95)
		if (white_pixels > bounding_box.width * 0.9) { 
			top_crop_y = r;
			break;
		}
		top_crop_y = r + 1;
	}

	for (int r = (bounding_box.y + bounding_box.height) - 1; r > bounding_box.y + bounding_box.height / 2; --r) {
		int white_pixels = countNonZero(binary_mask.row(r));
		//if (white_pixels > bounding_box.width * CLEAN_THRESHOLD) { 
		if (white_pixels > bounding_box.width * 0.8) { 
			bottom_crop_y = r;
			break;
		}
		bottom_crop_y = r - 1;
	}
	//cropping box (vertical)
	bounding_box.y = top_crop_y + SAFETY_BUFFER;
	bounding_box.height = bottom_crop_y - top_crop_y - SAFETY_BUFFER;
	bounding_box &= Rect(0, 0, stitched_image.cols, stitched_image.rows);

	// B. Horizontal Crop (Left/Right) - **Using the optimized method for speed**
	Mat col_sums;
	// Reduce the mask along the row dimension (axis 0, sum each column)
	reduce(binary_mask, col_sums, 0, REDUCE_SUM, CV_32S); 
	Mat white_pixel_counts = col_sums / 255; 
	const int required_count = static_cast<int>(bounding_box.height * CLEAN_THRESHOLD);
	// Find Left Crop Line
	for (int c = bounding_box.x; c < bounding_box.x + bounding_box.width / 2; ++c) {
		int white_pixels = white_pixel_counts.at<int>(0, c); 
		if (white_pixels > required_count) {
			left_crop_x = c;
			break;
		}
		left_crop_x = c + 1;
	}

	// Find Right Crop Line
	for (int c = (bounding_box.x + bounding_box.width) - 1; c > bounding_box.x + bounding_box.width / 2; --c) {
		int white_pixels = white_pixel_counts.at<int>(0, c); 
		if (white_pixels > required_count) {
			right_crop_x = c;
			break;
		}
		right_crop_x = c - 1;
	}
	//cropping box (horizontal)
	bounding_box.x = left_crop_x + SAFETY_BUFFER ;
	bounding_box.width = right_crop_x - left_crop_x - SAFETY_BUFFER ;
	//safety
	bounding_box &= Rect(0, 0, stitched_image.cols, stitched_image.rows);
	//cv::imshow("final mask",binary_mask(bounding_box));
	//waitKey(0);
	//cout << "rushi this are the results " << stitched_image.cols << " rows is    "<< stitched_image.rows << endl ;
	return bounding_box;
	}


	Mat auto_white_balance(const Mat& frame, int small_size = 50, double eps = 1e-6, double* gain_b_out = nullptr, double* gain_r_out = nullptr) {
		// Downscale to small_size x small_size
		Mat small;
		resize(frame, small, Size(small_size, small_size), 0, 0, INTER_NEAREST);

		// Compute mean per channel (B, G, R)
		Scalar avg = mean(small);  // avg[0]=B, avg[1]=G, avg[2]=R

		double avg_b = avg[0];
		double avg_g = avg[1];
		double avg_r = avg[2];

		// Compute gains relative to green
		double gain_b = avg_g / (avg_b + eps);
		double gain_r = avg_g / (avg_r + eps);

		if (gain_b_out) *gain_b_out = gain_b;
		if (gain_r_out) *gain_r_out = gain_r;

		// Convert to float for scaling
		Mat frame_f;
		frame.convertTo(frame_f, CV_32F);

		// Apply gains
		vector<Mat> channels;
		split(frame_f, channels);
		channels[0] *= gain_b;  // Blue
		channels[2] *= gain_r;  // Red
		merge(channels, frame_f);

		// Clip back to 8-bit
		Mat frame_out;
		frame_f.convertTo(frame_out, CV_8U, 1.0, 0.0);

		return frame_out;
	}

	static int parseCmdArgs(int argc, char** argv)
	{
		if (argc == 1)
		{
			cout << "Press 'q' to exit" << endl;
			cout<<" run command >  ./main --conf_thresh 0.5"<<endl;
			return -1;
		}
		for (int i = 1; i < argc; ++i)
		{
			if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
			{
				cout << "Press 'q' to exit" << endl;
				i++;
				cout<<" run command >  ./main --conf_thresh 0.5"<<endl;
				return -1;
			}
			else if (string(argv[i]) == "--conf_thresh")
			{
				conf_thresh = static_cast<float>(atof(argv[i + 1]));
				i++;
			}

			else if (string(argv[i]) == "--crop")
			{
				cout << "I am crop" << endl;
				crop=1;
				//	i++;
			}
			else if (string(argv[i]) == "--crop_input")
			{
				cout << "I am input crop" << endl;
				cout << "Rect (x_corr, y_corr, width, height)" << endl; 
				cout << "start with left cam input Rect (0, 0, 640, 400)" << endl; 
				left_x_cor= atoi(argv[++i]);
				left_y_cor= atoi(argv[++i]);
				left_width= atoi(argv[++i]);
				left_height=atoi(argv[++i]);

				right_x_cor= atoi(argv[++i]);
				right_y_cor= atoi(argv[++i]);
				right_width= atoi(argv[++i]);
				right_height=atoi(argv[++i]);

			}

			else if (string(argv[i]) == "--resize")
			{
				cout << "I am resize" << endl;
				resize_win=1;
				//	i++;
			}
		}
		return 0;
	}

	int main(int argc, char* argv[])
	{
		int w1,w2,w3,w4=0;


		cout << "Press 'q' to exit" << endl;
		cv::Mat xmap, ymap, xmap1, ymap1;
		cv::Rect roi, roi1;
		Rect safe_crop_box;

		int fg = 1; 
		int bg = 1; 
		int sg = 1;

		omp_set_num_threads(4); 
		int retval = parseCmdArgs(argc, argv);
		if (retval)
			return retval;

		VideoCapture cap1(1, CAP_V4L2);
		VideoCapture cap2(0, CAP_V4L2);

		if (!cap1.isOpened() || !cap2.isOpened()){
			cerr << "Error: Could not open video files" << endl;
			return -1;
		}

		cap1.set(CAP_PROP_CONVERT_RGB,0); 
		cap2.set(CAP_PROP_CONVERT_RGB,0); 

		//Work scale value and flag are primarily used for resizing the frame for feature point detection
		//Seam scale value and flag are used for downscaling the image for faster processing.

		double seam_work_aspect    = 1;
		double compose_work_aspect = 1;

		double work_scale = 1, seam_scale = 1, compose_scale = 1;
		bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

		Mat frame1, frame2;

		vector<UMat> masks_warped(num_images);
		vector<UMat> images_warped(num_images);
		vector<Size> sizes(num_images);
		vector<UMat> masks(num_images);
		vector<UMat> images_warped_f(num_images);
		vector<Point> corners(num_images);
		Ptr<RotationWarper> warper; 

		Size dst_sz;
		float blend_width;
		MultiBandBlender* mb;
		FeatherBlender* fb;

		//Size target_size(1920, 1200);
		Size target_size(640, 400);
		int count = 0;
		int flag = 0;
		//Using ORB finder
		Ptr<Feature2D> finder;
		finder = ORB::create();

		Mat_<float> K;

		if (resize_win)
			namedWindow("Stitched Video", WINDOW_NORMAL);
		else
			cv::namedWindow("Stitched Video", cv::WINDOW_AUTOSIZE);
		//setWindowProperty("Stitched Video", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);

		Ptr<Blender> saver;

		while(1){	
			cap1 >> frame1;
			cap2 >> frame2;


			flag+=1;
			if (flag == 1) { 
				std::system("v4l2-ctl -d /dev/video1 -c analogue_gain=800"); 
				//std::system("v4l2-ctl -d /dev/video0 --all"); 
				// Flush 20 frames so new settings apply 
				for (int i = 0; i < 20; i++) { 
					cap1  >> frame1; 
					cap2  >> frame2; 
				} 
			} 
			resize(frame1, frame1, target_size, 0, 0,INTER_LINEAR);
			resize(frame2, frame2, target_size, 0, 0,INTER_LINEAR);

			// cropping

			Rect cropRegion1(left_x_cor, left_y_cor, left_width, left_height);
			Rect cropRegion2(right_x_cor, right_y_cor, right_width, right_height);

			// Crop the image (extract the region of interest)
			frame1 = frame1(cropRegion1);//left cam
			frame2 = frame2(cropRegion2);//right cam
			// cropping ends

			//cv::imwrite("right_cam.jpg",frame2);
			cv::convertScaleAbs(frame1, frame1, 0.25, 0);
			cv::convertScaleAbs(frame2, frame2, 0.25, 0);
			cv::cvtColor(frame1, frame1, cv::COLOR_BayerGR2BGR);
			cv::cvtColor(frame2, frame2, cv::COLOR_BayerGR2BGR);

			if (frame1.empty() || frame2.empty()) return 0;

			std::vector<Mat> frames = {frame1, frame2};

			Mat full_img, img;
			vector<ImageFeatures> features(num_images);
			vector<Mat> images(num_images);
			vector<Size> full_img_sizes(num_images);
			for (int i = 0; (i < num_images); ++i) {
				if (count != 0)
					continue; // Only process first frames

				// Create local copies for thread safety
				Mat full_img = frames[i];  // each thread gets its own copy
				Mat img;
				Size full_img_size = full_img.size();

				if (full_img.empty()) {
					{
						LOGLN("Can't open image " << img_names[i]);
					}
					continue;
				}
				if (!is_work_scale_set) {
					work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
					is_work_scale_set = true;
				}

				// Resize image using local scale
				resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);

				if (!is_seam_scale_set) {
					seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
					seam_work_aspect = seam_scale / work_scale;
					is_seam_scale_set = true;
				}

				// Compute features
				computeImageFeatures(finder, img, features[i]);
				features[i].img_idx = i;

				// Resize again for seam
				resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
				images[i] = img.clone();

				// Save the size safely
				full_img_sizes[i] = full_img_size;
			}
			if (count != 0) {
				for (int i = 0; i < num_images; ++i) {
					Mat seam_img;
					resize(frames[i], seam_img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
					images[i] = seam_img.clone();
				}
			}

			if(fg == 1){
				matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
				(*matcher)(features, pairwise_matches);
				matcher->collectGarbage();
				estimator = makePtr<HomographyBasedEstimator>();

				if (!(*estimator)(features, pairwise_matches, cameras))
				{
					cout << "Homography estimation failed.\n";
					return -1;
				}

				for (size_t i = 0; i < cameras.size(); ++i)
				{
					Mat R;
					cameras[i].R.convertTo(R, CV_32F);
					cameras[i].R = R;
				}

				adjuster = makePtr<detail::BundleAdjusterRay>();
				adjuster->setConfThresh(conf_thresh);

				Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
				if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
				if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
				if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
				if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
				if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;

				adjuster->setRefinementMask(refine_mask);

				if (!(*adjuster)(features, pairwise_matches, cameras))
				{
					cout << "Camera parameters adjusting failed.\n";
					cout<< "Please stay far from camera" << endl;
					return -1;
				}

				// Find median focal length
				for (size_t i = 0; i < cameras.size(); ++i)
				{
					focals.push_back(cameras[i].focal);
				}

				sort(focals.begin(), focals.end());

				if (focals.size() % 2 == 1)
					warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
				else
					warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

				//wave correction - horizontal
				vector<Mat> rmats;
				for (size_t i = 0; i < cameras.size(); ++i)
					rmats.push_back(cameras[i].R.clone());
				waveCorrect(rmats, wave_correct);
				for (size_t i = 0; i < cameras.size(); ++i)
					cameras[i].R = rmats[i];

				for (int i = 0; i < num_images; ++i)
				{
					masks[i].create(images[i].size(), CV_8U);
					masks[i].setTo(Scalar::all(255));
				}

				//For warping frames onto one another spericalwarper is required to give wide panoramic view
				//Also coordinates used are sperical coordinates

				//warper_creator = makePtr<cv::PlaneWarper>();
				warper_creator = makePtr<cv::SphericalWarper>();
				//warper_creator = makePtr<cv::CylindricalWarper>();
				if (!warper_creator)
				{
					cout << "Can't create the following warper '" << warp_type << "'\n";
					return 1;
				}
				seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
				//seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_GRAD);
				if (!seam_finder)
				{
					//					seam_finder->setCostType(GraphCutSeamFinderBase::COST_COLOR_GRAD);

					cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
					return 1;
				}
				warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
				compensator = ExposureCompensator::createDefault(expos_comp_type);

				if (dynamic_cast<GainCompensator*>(compensator.get()))
				{
					GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
					gcompensator->setNrFeeds(expos_comp_nr_feeds);
				}

				if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
				{
					ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
					ccompensator->setNrFeeds(expos_comp_nr_feeds);
				}

				if (dynamic_cast<BlocksCompensator*>(compensator.get()))
				{
					BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
					bcompensator->setNrFeeds(expos_comp_nr_feeds);
					bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
					bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
				}

				fg = 0;
			}
			//line 655 ,633
			if(ru){

				for (int i = 0; i < num_images; ++i)
				{
					//if (count == 0){

					//cout<<"this should run" <<endl;
					masks[i].setTo(Scalar::all(255));
					cameras[i].K().convertTo(K, CV_32F);
					float swa = (float)seam_work_aspect;
					K(0,0) *= swa; K(0,2) *= swa;
					K(1,1) *= swa; K(1,2) *= swa;

					corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
					sizes[i] = images_warped[i].size();
					warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
					//}
				}

				for (int i = 0; i < num_images; ++i)
					images_warped[i].convertTo(images_warped_f[i], CV_32F);

				// Normalize corners so minimum x,y becomes 0 (avoids negative ROI)
				//if (count==0)
				//{
				compensator->feed(corners, images_warped, masks_warped);
				seam_finder->find(images_warped_f, corners, masks_warped);
				for (auto &mask : masks_warped) {
					cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
				}
				ru=0;
			}	/*			for (size_t i = 0; i < masks_warped.size(); ++i)
							{
							cv::imshow("Seam mask " + std::to_string(i), images_warped[i]);
							}*/

			//				cv::imshow("new",masks_warped);
			//}
			int a  = 0;

			Mat img_warped, img_warped_s;
			Mat dilated_mask, seam_mask, mask, mask_warped;
			Ptr<Blender> blender;
			Ptr<Timelapser> timelapser;
			for (int img_idx = 0; img_idx < num_images; ++img_idx)
			{
				t+=1;
				full_img = frames[a++];
				//cv::imshow("img",full_img);
				//cv::waitKey(0);
				if (!is_compose_scale_set)
				{
					if (compose_megapix > 0)
						compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
					is_compose_scale_set = true;
					// Compute relative scales
					compose_work_aspect = compose_scale / work_scale;

					// Update warped image scale
					warped_image_scale *= static_cast<float>(compose_work_aspect);
					warper = warper_creator->create(warped_image_scale);

					// Update corners and sizes
					for (int i = 0; i < num_images; ++i)
					{
						// Update intrinsics
						cameras[i].focal *= compose_work_aspect;
						cameras[i].ppx *= compose_work_aspect;
						cameras[i].ppy *= compose_work_aspect;

						// Update corner and size
						Size sz = full_img_sizes[i];
						if (std::abs(compose_scale - 1) > 1e-1)
						{
							sz.width = cvRound(full_img_sizes[i].width * compose_scale);
							sz.height = cvRound(full_img_sizes[i].height * compose_scale);
						}
						Mat K;
						cameras[i].K().convertTo(K, CV_32F);
						Rect roi = warper->warpRoi(sz, K, cameras[i].R);
						corners[i] = roi.tl();
						sizes[i] = roi.size();
					}
				}

				if (abs(compose_scale - 1) > 1e-1)
					resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
				else
					img = full_img;

				Size img_size = img.size();
				Mat K;
				cameras[img_idx].K().convertTo(K, CV_32F);
				if (count==0)
					roi= warper->buildMaps(img.size(), K, cameras[img_idx].R, xmap, ymap);
				cv::remap(img, img_warped, xmap, ymap, cv::INTER_LINEAR, cv::BORDER_REFLECT);
				// Warp the current image mask
				mask.create(img_size, CV_8U);
				mask.setTo(Scalar::all(255));
				if (count==0)
					roi1 = warper->buildMaps(img_size, K, cameras[img_idx].R, xmap1, ymap1);
				cv::remap(mask, mask_warped, xmap1, ymap1, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

				// Compensate exposure
				compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
				img_warped.convertTo(img_warped_s, CV_16S);

				// Compensate exposure
				full_img.release();
				mask.release();
				dilate(masks_warped[img_idx], dilated_mask, Mat());
				resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
				mask_warped = seam_mask & mask_warped;

				// Compensate exposure
				cv::Rect dst_roi ;
				if (!blender && !timelapse)
				{
					blender = Blender::createDefault(blend_type, try_cuda);
					dst_sz = resultRoi(corners, sizes).size();
					blend_width =sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
					cout<<"blend_width is" <<blend_width;
					mb = dynamic_cast<MultiBandBlender*>(blender.get());
					mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
					//mb->setNumBands(1);
					//blender->prepare(corners, sizes);

					dst_roi = cv::detail::resultRoi(corners, sizes);

					// Expand ROI slightly to guarantee fit for all warped images
					const int safety_margin = 64; // can tune between 32–128
					dst_roi.x -= safety_margin;
					dst_roi.y -= safety_margin;
					dst_roi.width  += safety_margin * 2;
					dst_roi.height += safety_margin * 2;
					

					blender->prepare(dst_roi);

					/*std::cout << "ROI from resultRoi: x=" << dst_roi.x
					  << " y=" << dst_roi.y
					  << " width=" << dst_roi.width
					  << " height=" << dst_roi.height
					  << std::endl;*/
				}


				if(t%2==1&&now==1){//&&count!=0){
				//	cv::imshow("untouched mask", mask_warped);
					//	cv::imshow("init image",img_warped);
					cv::Mat motion_mask = computeMotionMask(img_warped);
					//cv::imshow("motion mask is ", motion_mask);
					Mat masked_img,res_temp;
					//bitwise_or(motion_mask, motion_mask, masked_img,mask=mask_warped);
					bitwise_or(motion_mask, mask_warped, masked_img);
				//	cv::imshow("modified maskkk real right cam", masked_img);
					mask_warped=masked_img;
					
					bitwise_and(img_warped,img_warped,res_temp,mask_warped);
					cv::imshow("temp_result_right_cam",res_temp);


				}

				/*
				   if (count==0)
				   ru_mask=masked_img;
				   bitwise_or(masked_img,ru_mask, masked_img);
				   mask_warped=masked_img;
				   ru_mask=masked_img;
				   cv::imshow("mask",mask_warped);
				   }
				   */

				if (t%2 == 0&&count!=0){
					//		Mat my_mask(mask_warped.rows, mask_warped.cols, CV_8UC1, Scalar(255));
					//		Rect leftRegion(0, 0, mask_warped.cols/2, mask_warped.rows);
					//		my_mask(leftRegion) = Scalar(0);
					//		bitwise_and(my_mask, mask_warped, mask_warped);
					//		cv::imshow("leftcam",mask_warped);

					//auto [motion_mask, diff] = computeMotionMaskk(frame);
					//cv::bitwise_not(motion_mask, motion_mask); // invert mask (static=255, moving=0)
					if (count==1)
						first_frame=img_warped;
					//cv::imshow("input",img_warped);
					//cv::waitKey(0);	
					Mat masked_img,mask_temp,te,mask_copy;

					cv::Mat motion_maskk = computeMotionMaskk(img_warped);
				//	cv::imshow("motion mask is ", motion_maskk);

					cv::bitwise_not(mask_warped, mask_copy);	
				//	cv::imshow("mask_warped",mask_warped);
					bitwise_and(motion_maskk,mask_copy,te);
				//	cv::imshow("mask_copy + motion mask &",te);
					cv::bitwise_not(te, mask_temp);	
				//	cv::imshow("mask_copy + motion mask & but inverted",mask_temp);

					cv::Mat resi,resi_temp;
					bitwise_and(first_frame,first_frame,resi_temp,mask_temp);
					//bitwise_and(img_warped,img_warped,resi_temp,mask_temp);
					cv::imshow("temp_result_left_cam",resi_temp);

					Mat half_changing,half_right;
// Define the right half of first_frame as the target region
					half_right = resi_temp(Rect(first_frame.cols / 2, 0,first_frame.cols - first_frame.cols / 2,first_frame.rows));
					half_changing = img_warped(Rect(first_frame.cols / 2, 0,first_frame.cols - first_frame.cols / 2,first_frame.rows));
					half_changing.copyTo(half_right);



					//half_right = resi_temp(Rect(first_frame.cols / 2, 0,first_frame.cols - first_frame.cols / 2,first_frame.rows));
					//half_changing = img_warped(Rect(first_frame.cols / 2, 0,first_frame.cols - first_frame.cols / 2,first_frame.rows));
					//half_changing.copyTo(half_right);


					cv::imshow("half part is is ",half_right);

					img_warped_s.convertTo(img_warped_s, CV_8U);  // simple scale, might clip
					img_warped_s=resi_temp;
					mask_warped=mask_temp;
					cv::imshow("img_warped_sgoing to blender is ",img_warped_s);
					cv::imshow("mask warped going to blender is ",mask_warped);

					img_warped_s.convertTo(img_warped_s, CV_16S);

				}

				if (img_warped.empty() || mask_warped.empty() || img_warped_s.empty())  {
					cout<< "Warped image/mask is empty " << endl;
					cout<< "Please dont move on the overlapping region and run the code again" << endl;
					return -1;
				}
				/*w1=img_warped_s.cols + img_warped_s.rows;
				  if (count ==0 && img_idx == 0 )
				  w2=w1;
				  if (w2 < w1)
				  cout << "------ \n\n\n\n Break here \n\n\n " << endl ; 
				  */

				w1 = corners[img_idx].y + img_warped_s.rows  ;
				w3 = corners[img_idx].y ; 
				if (count ==0 && img_idx == 0 ){
					w2=w1; 
					w4=w3; 
					w4+=6; }

				//if (w2 < w1)
				/*if(dst_roi.width>2000||dst_roi.height>1100)	
				  {
				  cout << "------ \n\n\n\n Break here \n\n\n " << w4 <<"  "<< w3 << " "<< img_idx << endl ; 
				  }

				//the first image that comes is of left camera verify by cv::imshow("img",full_img); at the start of this block
				cout << "--- Blender Feed Diagnostics (Frame: " << count << ", Image: " << img_idx << ") ---" << endl;
				cout << "Warped Image Size (Cols x Rows): " << img_warped_s.cols << " x " << img_warped_s.rows << endl;
				cout << "Warped Mask Size (Cols x Rows): " << mask_warped.cols << " x " << mask_warped.rows << endl;
				cout << "Corner (X, Y) Position: (" << corners[img_idx].x << ", " << corners[img_idx].y << ")" << endl;
				cout << "------------------------------------------------------------------" << endl;
				*/
				blender->feed(img_warped_s, mask_warped, corners[img_idx]);
				img_warped.release();
			}

			Mat result, result_mask;
			blender->blend(result, result_mask);

			Mat result_8u;
			result.convertTo(result_8u, CV_8U);  // simple scale, might clip

			double gain_b, gain_r;
			result_8u = auto_white_balance(result_8u, 50, 1e-6, &gain_b, &gain_r);

			cout<< "ALMOSSSTT DONE " << endl;
			Rect rroi = resultRoi(corners, sizes); // valid stitched region

			if (crop){
				if (count == 0)
					safe_crop_box = adaptive_crop_panorama(result_8u);
				result_8u = result_8u(safe_crop_box);
			}
			if (resize_win){
				if (crop)
					resizeWindow("Stitched Video", safe_crop_box.width*2, safe_crop_box.height*2);
				else
					resizeWindow("Stitched Video", rroi.width+rroi.width, rroi.height+rroi.height);
			}
			imshow("Stitched Video", result_8u);
			int key = waitKey(1);
			count++;
			// this block is to reset the stitching and do again
			//if (count/100 == verif)
			if (key == 'r'){
				verif+=1;
				cout << "frame is  "<< count << endl;
				cout << "  " << endl;
				ru=1;}

			if (key == 'n'){
				count=0;
				fg=1;
				is_compose_scale_set=false;
				cout << "calculate again" << endl;
			}

			if (key == 'u'){
				now=1;}

			if (key == 'i'){
				i+=1;}

			if (key == 'q' || key == 'l')  // press 'q' or 'Q' to quit
			{
				cout << "Exiting" << endl;
				break;   // exit the loop
			}
		}
		return 0;
	}

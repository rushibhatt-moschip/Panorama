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
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;
Ptr<WarperCreator> warper_creator;


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
			cout<<" run command >  ./main --conf_thresh 0.5"<<endl;
			return -1;
		}
		else if (string(argv[i]) == "--conf_thresh")
		{
			conf_thresh = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
	}
	return 0;
}

int main(int argc, char* argv[])
{
	cout << "Press 'q' to exit" << endl;
	cv::Mat xmap, ymap, xmap1, ymap1;
	cv::Rect roi, roi1;

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

	//Informs opencv to not perform color conversion by default
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

	Size target_size(640, 400);
	int count = 0;
	int flag = 0;
	//Using ORB finder
	Ptr<Feature2D> finder;
	finder = ORB::create();

	Mat_<float> K;

	while(1){	
		cap1 >> frame1;
		cap2 >> frame2;

		flag+=1;

		//Brightness Compensator
		if (flag == 1) { 
			std::system("v4l2-ctl -d /dev/video0 -c analogue_gain=800"); 
			//std::system("v4l2-ctl -d /dev/video0 --all"); 
			// Flush 20 frames so new settings apply 
			for (int i = 0; i < 20; i++) { 
				cap1  >> frame1; 
				cap2  >> frame2; 
			} 
		} 
		
		resize(frame1, frame1, target_size, 0, 0,INTER_LINEAR);
		resize(frame2, frame2, target_size, 0, 0,INTER_LINEAR);
		
		//Conversion from 10-bit Bayer input to 8-bit BGR
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

		/* Matcher object initialization 
		 * 		|
		 * Homography object initialization
		 * 		|
		 * Camera parameter estimation
		 * 		|
		 * Bundle Adjustment
		 * 		|
		 * Mask creation
		 * 		|
		 * Seam Finding, warper selection, compensator obj init
		 *
		*/

		if(fg == 1){
			matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
			(*matcher)(features, pairwise_matches);
			matcher->collectGarbage();
			estimator = makePtr<HomographyBasedEstimator>();
			
			//estimating camera matrix parameter -> intrinsic + extrinsic parameters
			if (!(*estimator)(features, pairwise_matches, cameras))
			{
				cout << "Homography estimation failed.\n";
				return -1;
			}
			
			//matrix parameter conversion from double to float
			for (size_t i = 0; i < cameras.size(); ++i)
			{
				Mat R;
				cameras[i].R.convertTo(R, CV_32F);
				cameras[i].R = R;
			}

			//optimize cam matrix value to reduce geometric error
			//raybundleadjustment technique is used
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

			//creating masks for all images
			for (int i = 0; i < num_images; ++i)
			{
				masks[i].create(images[i].size(), CV_8U);
				masks[i].setTo(Scalar::all(255));
			}

			//For warping frames onto one another spericalwarper is required to give wide panoramic view
			//Also coordinates used are sperical coordinates

			warper_creator = makePtr<cv::SphericalWarper>();
			//warper_creator = makePtr<cv::CylindricalWarper>();
			if (!warper_creator)
			{
				cout << "Can't create the following warper '" << warp_type << "'\n";
				return 1;
			}
			
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
			if (!seam_finder)
			{
				cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
				return 1;
			}
			//exposure compensator init
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

		//warping
		for (int i = 0; i < num_images; ++i)
		{

			if (count == 0){

				masks[i].setTo(Scalar::all(255));
				cameras[i].K().convertTo(K, CV_32F);
				float swa = (float)seam_work_aspect;
				K(0,0) *= swa; K(0,2) *= swa;
				K(1,1) *= swa; K(1,2) *= swa;

				corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
				sizes[i] = images_warped[i].size();
				warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

			}
		}

		for (int i = 0; i < num_images; ++i)
			images_warped[i].convertTo(images_warped_f[i], CV_32F);

		// Normalize corners so minimum x,y becomes 0 (avoids negative ROI)
		if (count==0)
		{
			compensator->feed(corners, images_warped, masks_warped);
			seam_finder->find(images_warped_f, corners, masks_warped);
		}
		int a  = 0;

		Mat img_warped, img_warped_s;
		Mat dilated_mask, seam_mask, mask, mask_warped;
		Ptr<Blender> blender;
		Ptr<Timelapser> timelapser;

		for (int img_idx = 0; img_idx < num_images; ++img_idx)
		{
			full_img = frames[a++];
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
			if (!blender && !timelapse)
			{
				blender = Blender::createDefault(blend_type, try_cuda);
				dst_sz = resultRoi(corners, sizes).size();
				blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
				mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
				blender->prepare(corners, sizes);
			}

			if (img_warped.empty() || mask_warped.empty()) {
				cout<< "Warped image/mask is empty " << endl;
				cout<< "Please dont move on the overlapping region and run the code again" << endl;
				return -1;
			}

			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
			img_warped.release();
		}

		Mat result, result_mask;
		blender->blend(result, result_mask);

		Mat result_8u;
		result.convertTo(result_8u, CV_8U);  // simple scale, might clip

		double gain_b, gain_r;
		result_8u = auto_white_balance(result_8u, 50, 1e-6, &gain_b, &gain_r);

		count++;
		imshow("Stitched Video", result_8u);
		int key = waitKey(1);
		if (key == 'q' || key == 'Q')  // press 'q' or 'Q' to quit
		{
			cout << "Exiting on user request..." << endl;
			break;   // exit the loop
		}
	}
	return 0;
}

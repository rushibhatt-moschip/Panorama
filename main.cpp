#include <iostream>
#include <fstream>
#include <string>
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
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg << std::endl
#define LOGLN(msg) std::cout << msg << std::endl
using namespace std;
using namespace cv;
using namespace cv::detail;

Mat img_warped, img_warped_s;
Mat dilated_mask, seam_mask, mask, mask_warped;
Ptr<Blender> blender;
Ptr<Timelapser> timelapser;
double compose_seam_aspect = 1;
double compose_work_aspect = 1;
static int num_images = 2;
Mat full_img, img;
vector<ImageFeatures> features(num_images);
vector<Mat> images(num_images);
vector<Size> full_img_sizes(num_images);
double seam_work_aspect = 1;
static Ptr<SeamFinder> seam_finder;
static Ptr<WarperCreator> warper_creator;
vector<MatchesInfo> pairwise_matches;
Ptr<FeaturesMatcher> matcher;
Ptr<Feature2D> finder;
float warped_image_scale;
vector<double> focals;
Ptr<detail::BundleAdjusterBase> adjuster;
Ptr<Estimator> estimator;
vector<CameraParams> cameras;
vector<String> vid_names;
// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
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
static int parseCmdArgs(int argc, char** argv)
{
	if (argc == 1)
	{
		//printUsage(argv);
		return -1;
	}
	for (int i = 1; i < argc; ++i)
	{
		if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
		{
			//printUsage(argv);
			return -1;
		}
		else if (string(argv[i]) == "--preview")
		{
			preview = true;
		}
		else if (string(argv[i]) == "--try_cuda")
		{
			if (string(argv[i + 1]) == "no")
				try_cuda = false;
			else if (string(argv[i + 1]) == "yes")
				try_cuda = true;
			else
			{
				cout << "Bad --try_cuda flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--work_megapix")
		{
			work_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--seam_megapix")
		{
			seam_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--compose_megapix")
		{
			compose_megapix = atof(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--result")
		{
			result_name = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--features")
		{
			features_type = argv[i + 1];
			if (string(features_type) == "orb")
				match_conf = 0.3f;
			i++;
		}
		else if (string(argv[i]) == "--matcher")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				matcher_type = argv[i + 1];
			else
			{
				cout << "Bad --matcher flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--estimator")
		{
			if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
				estimator_type = argv[i + 1];
			else
			{
				cout << "Bad --estimator flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--match_conf")
		{
			match_conf = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--conf_thresh")
		{
			conf_thresh = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--ba")
		{
			ba_cost_func = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--ba_refine_mask")
		{
			ba_refine_mask = argv[i + 1];
			if (ba_refine_mask.size() != 5)
			{
				cout << "Incorrect refinement mask length.\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--wave_correct")
		{
			if (string(argv[i + 1]) == "no")
				do_wave_correct = false;
			else if (string(argv[i + 1]) == "horiz")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_HORIZ;
			}
			else if (string(argv[i + 1]) == "vert")
			{
				do_wave_correct = true;
				wave_correct = detail::WAVE_CORRECT_VERT;
			}
			else
			{
				cout << "Bad --wave_correct flag value\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--save_graph")
		{
			save_graph = true;
			save_graph_to = argv[i + 1];
			i++;
		}
		else if (string(argv[i]) == "--warp")
		{
			warp_type = string(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--expos_comp")
		{
			if (string(argv[i + 1]) == "no")
				expos_comp_type = ExposureCompensator::NO;
			else if (string(argv[i + 1]) == "gain")
				expos_comp_type = ExposureCompensator::GAIN;
			else if (string(argv[i + 1]) == "gain_blocks")
				expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
			else if (string(argv[i + 1]) == "channels")
				expos_comp_type = ExposureCompensator::CHANNELS;
			else if (string(argv[i + 1]) == "channels_blocks")
				expos_comp_type = ExposureCompensator::CHANNELS_BLOCKS;
			else
			{
				cout << "Bad exposure compensation method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--expos_comp_nr_feeds")
		{
			expos_comp_nr_feeds = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--expos_comp_nr_filtering")
		{
			expos_comp_nr_filtering = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--expos_comp_block_size")
		{
			expos_comp_block_size = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--seam")
		{
			if (string(argv[i + 1]) == "no" ||
					string(argv[i + 1]) == "voronoi" ||
					string(argv[i + 1]) == "gc_color" ||
					string(argv[i + 1]) == "gc_colorgrad" ||
					string(argv[i + 1]) == "dp_color" ||
					string(argv[i + 1]) == "dp_colorgrad")
				seam_find_type = argv[i + 1];
			else
			{
				cout << "Bad seam finding method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--blend")
		{
			if (string(argv[i + 1]) == "no")
				blend_type = Blender::NO;
			else if (string(argv[i + 1]) == "feather")
				blend_type = Blender::FEATHER;
			else if (string(argv[i + 1]) == "multiband")
				blend_type = Blender::MULTI_BAND;
			else
			{
				cout << "Bad blending method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--timelapse")
		{
			timelapse = true;
			if (string(argv[i + 1]) == "as_is")
				timelapse_type = Timelapser::AS_IS;
			else if (string(argv[i + 1]) == "crop")
				timelapse_type = Timelapser::CROP;
			else
			{
				cout << "Bad timelapse method\n";
				return -1;
			}
			i++;
		}
		else if (string(argv[i]) == "--rangewidth")
		{
			range_width = atoi(argv[i + 1]);
			i++;
		}
		else if (string(argv[i]) == "--blend_strength")
		{
			blend_strength = static_cast<float>(atof(argv[i + 1]));
			i++;
		}
		else if (string(argv[i]) == "--output")
		{
			result_name = argv[i + 1];
			i++;
		}
		else if(string(argv[i]) == "--video1")
		{

			cout << "Video 1 read" << endl;
			vid_names.push_back(argv[i+1]);
			i++;

		}
		else if(string(argv[i]) == "--video2")
		{

			cout << "Video 2 read" << endl;
			vid_names.push_back(argv[i+1]);
			i++;
		}else
			img_names.push_back(argv[i]);
	}
	if (preview)
	{
		compose_megapix = 0.6;
	}
	return 0;
}

int main(int argc, char* argv[])
{
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif
#if 0
	cv::setBreakOnError(true);
#endif
	int retval = parseCmdArgs(argc, argv);
	if (retval)
		return retval;

	// Open videos
	VideoCapture cap1(vid_names[0]);
	VideoCapture cap2(vid_names[1]);

	if (!cap1.isOpened() || !cap2.isOpened()) {
		cerr << "Error: Could not open video files" << endl;
		return -1;
	}
	vector<Mat> img1;
	vector<Mat> img2;

	Mat frame;

	// Extract frames from Video 1
	while (true) {
		if (!cap1.read(frame)) break;   // End of video
		img1.push_back(frame.clone()); // Store a copy of frame
	}
	cout << "Extracted " << img1.size() << " frames from " << endl;

	// Extract frames from Video 2
	while (true) {
		if (!cap2.read(frame)) break;
		img2.push_back(frame.clone());
	}
	cout << "Extracted " << img2.size() << " frames from " << endl;

	int num_mat = min(img1.size(),img2.size());

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	LOGLN("Finding features...");

#if ENABLE_LOG
	int64 t = getTickCount();
#endif		

	if (features_type == "orb")
	{
		finder = ORB::create();
	}
	else if (features_type == "akaze")
	{
		finder = AKAZE::create();
	}
#ifdef HAVE_OPENCV_XFEATURES2D
	else if (features_type == "surf")
	{
		finder = xfeatures2d::SURF::create();
	}
#endif
	else if (features_type == "sift")
	{
		finder = SIFT::create();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}
		


	for (int i = 0; i < num_images; ++i)
	{

		if(i == 0){
			full_img = img1[0];
		}else{
			full_img = img2[0];
		}

		full_img_sizes[i] = full_img.size();
		if (full_img.empty())
		{
			//LOGLN("Can't open image " << img_names[i]);
			return -1;
		}

		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}
		computeImageFeatures(finder, img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());
		resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();
	}

	full_img.release();
	img.release();
	LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOG("Pairwise matching");

#if ENABLE_LOG
	t = getTickCount();
#endif

	if (matcher_type == "affine")
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	else if (range_width==-1)
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	else
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);

	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
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
		LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}

	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}

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
		return -1;
	}
	// Find median focal length
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}
	sort(focals.begin(), focals.end());
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}
	LOGLN("Warping images (auxiliary)... ");

#if ENABLE_LOG
	t = getTickCount();
#endif

	for(int k = 0; k < num_mat; k++){

		vector<Point> corners(num_images);
		vector<UMat> masks_warped(num_images);
		vector<UMat> images_warped(num_images);
		vector<Size> sizes(num_images);
		vector<UMat> masks(num_images);

		cout << "#### seam scale " << seam_scale << endl;
		for(int i = 0; i < num_images && k != 0; i++){	

			if(i == 0)
				full_img = img1[k];
			else
				full_img = img2[k];

			resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
			images[i] = img.clone();
		}

		// Prepare images masks
		for (int i = 0; i < num_images; ++i)
		{
			masks[i].create(images[i].size(), CV_8U);
			masks[i].setTo(Scalar::all(255));
		}

		// Warp images and their masks
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarper>();
		else if (warp_type == "affine")
			warper_creator = makePtr<cv::AffineWarper>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarper>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarper>();
		else if (warp_type == "fisheye")
			warper_creator = makePtr<cv::FisheyeWarper>();
		else if (warp_type == "stereographic")
			warper_creator = makePtr<cv::StereographicWarper>();
		else if (warp_type == "compressedPlaneA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlaneA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniA2B1")
			warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniA1.5B1")
			warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniPortraitA2B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniPortraitA1.5B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "mercator")
			warper_creator = makePtr<cv::MercatorWarper>();
		else if (warp_type == "transverseMercator")
			warper_creator = makePtr<cv::TransverseMercatorWarper>();

		if (!warper_creator)
		{
			cout << "Can't create the following warper '" << warp_type << "'\n";
			return 1;
		}

		static Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
		
		for (int i = 0; i < num_images; ++i)
		{
			Mat_<float> K;
			cameras[i].K().convertTo(K, CV_32F);
			float swa = (float)seam_work_aspect;
			K(0,0) *= swa; K(0,2) *= swa;
			K(1,1) *= swa; K(1,2) *= swa;
			corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			sizes[i] = images_warped[i].size();
			warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
		}

		vector<UMat> images_warped_f(num_images);
		for (int i = 0; i < num_images; ++i)
			images_warped[i].convertTo(images_warped_f[i], CV_32F);
		LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		LOGLN("Compensating exposure...");
#if ENABLE_LOG
		t = getTickCount();
#endif
		static Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
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

		compensator->feed(corners, images_warped, masks_warped);
		LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		LOGLN("Finding seams...");
#if ENABLE_LOG
		t = getTickCount();
#endif
		if (seam_find_type == "no")
			seam_finder = makePtr<detail::NoSeamFinder>();
		else if (seam_find_type == "voronoi")
			seam_finder = makePtr<detail::VoronoiSeamFinder>();
		else if (seam_find_type == "gc_color")
		{
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
		}
		else if (seam_find_type == "gc_colorgrad")
		{
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		}
		else if (seam_find_type == "dp_color")
			seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
		else if (seam_find_type == "dp_colorgrad")
			seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
		if (!seam_finder)
		{
			cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
			return 1;
		}

		seam_finder->find(images_warped_f, corners, masks_warped);
		LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		// Release unused memory
		images.clear();
		images_warped.clear();
		images_warped_f.clear();
		masks.clear();
		
		LOGLN("Compositing...");
#if ENABLE_LOG
		t = getTickCount();
#endif
		for (int img_idx = 0; img_idx < num_images && k == 0; ++img_idx)
		{
			LOGLN("Compositing image #" << indices[img_idx]+1);
			// Read image and resize it if necessary

			if(img_idx <= 0){
				full_img = img1[k];
			}else{
				full_img = img2[k];
			}

			if (!is_compose_scale_set)
			{
				if (compose_megapix > 0)
					compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
				is_compose_scale_set = true;
				// Compute relative scales
			//	compose_seam_aspect = compose_scale / seam_scale;
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
			cout << " comp " << endl;
			full_img.release();
			Size img_size = img.size();
			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);
			// Warp the current image
			warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
			// Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(Scalar::all(255));
			warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
			// Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
			img_warped.convertTo(img_warped_s, CV_16S);
			img_warped.release();
			img.release();
			mask.release();
			dilate(masks_warped[img_idx], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
			mask_warped = seam_mask & mask_warped;
			cout << " bef blend " << endl;
			if(k == 0) {
				if (!blender && !timelapse)
				{
					blender = Blender::createDefault(blend_type, try_cuda);
					Size dst_sz = resultRoi(corners, sizes).size();
					float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
					if (blend_width < 1.f)
						blender = Blender::createDefault(Blender::NO, try_cuda);
					else if (blend_type == Blender::MULTI_BAND)
					{
						MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
						mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
						LOGLN("Multi-band blender, number of bands: " << mb->numBands());
					}
					else if (blend_type == Blender::FEATHER)
					{
						FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
						fb->setSharpness(1.f/blend_width);
						LOGLN("Feather blender, sharpness: " << fb->sharpness());
					}
					blender->prepare(corners, sizes);
				}
				else if (!timelapser && timelapse)
				{
					timelapser = Timelapser::createDefault(timelapse_type);
					timelapser->initialize(corners, sizes);
				}
				// Blend the current image
				blender->feed(img_warped_s, mask_warped, corners[img_idx]);
			}
		}
		if (!timelapse)
		{
			Mat result, result_mask;
			blender->blend(result, result_mask);
			LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
			imwrite(result_name, result);
		//	imshow(result_name, result);
		//	waitKey(30);
		}
	LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
	}
	return 0;
}

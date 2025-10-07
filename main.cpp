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
//int blend_type;
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
		else
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
	int fg = 1; 
	int bg = 1; 
	int sg = 1;

	omp_set_num_threads(4); 
	int retval = parseCmdArgs(argc, argv);
	if (retval)
		return retval;

	//int nn = (ceil(log(blend_width)/log(2.)) - 1.);
	//cout << "nn val" << nn << endl;
	/* Set the video Node for webcam
	 * set its properties ,i.e., Width, Height, Frame rate and color format
	 */
	//VideoCapture cap1("/home/devashree-katarkar/camera1_output_first.mkv");
	//VideoCapture cap2("/home/devashree-katarkar/camera2_output_first.mkv");

	VideoCapture cap1(2, CAP_V4L2);
	VideoCapture cap2(0, CAP_V4L2);
	//std::cout << "Backend used: " << cap1.getBackendName() << std::endl;
	//std::cout << "Backend used: " << cap2.getBackendName() << std::endl;
	//return 0;
	cap1.set(CAP_PROP_FRAME_WIDTH, 640);
	cap1.set(CAP_PROP_FRAME_HEIGHT, 480);
	cap1.set(CAP_PROP_FPS, 30);
	cap1.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
	//cap1.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('Y','U','Y','V'));

	cap2.set(CAP_PROP_FRAME_WIDTH, 640);
	cap2.set(CAP_PROP_FRAME_HEIGHT, 480);
	cap2.set(CAP_PROP_FPS,30);
	cap2.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
	//cap2.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('Y','U','Y','V'));

	//cap1.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));
	//cap2.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G'));

	if (!cap1.isOpened() || !cap2.isOpened()){
		cerr << "Error: Could not open video files" << endl;
		return -1;
	}

	//sleep(5);
	//Work scale value and flag are primarily used for resizing the frame for feature point detection
	//Seam scale value and flag are used for downscaling the image for faster processing.

	double seam_work_aspect    = 1;
	double compose_work_aspect = 1;

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	VideoWriter writer;
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

	Size target_size(320, 240);
	int count = 0;

	//Using SIFT finder
	Ptr<Feature2D> finder;
	finder = SIFT::create();

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	//sleep(5);

	Mat_<float> K;

	while(1){

		cap1 >> frame1;
		cap2 >> frame2;
/*
		int targetWidth  = 960;  // Change this to your desired width
		int targetHeight = 300; // Change this to your desired height

		// Define the cropping region (for example, starting from (100, 50) and cropping to (500, 350))
		int cropX = 0;
		int cropY = 300;
		int cropWidth = targetWidth;  // You can adjust the width of the cropped area
		int cropHeight = targetHeight; // You can adjust the height of the cropped area

		// Define the region of interest (ROI) based on the target resolution
		Rect cropRegion(cropX, cropY, cropWidth, cropHeight);

		// Crop the image (extract the region of interest)
		frame1 = frame1(cropRegion);
		frame2 = frame2(cropRegion);
*/
		//resize(frame1, frame1, target_size, 0, 0,INTER_LINEAR);
		//resize(frame2, frame2, target_size, 0, 0,INTER_LINEAR);

		if (frame1.empty() || frame2.empty()) return 0;

		vector<Mat> frames = {frame1, frame2};

		for (int i = 0; i < num_images && (count == 0); ++i)
		{

			full_img = frames[i];
			full_img_sizes[i] = full_img.size();

			if (full_img.empty())
			{
				LOGLN("Can't open image " << img_names[i]);
				return -1;
			}

			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}

			//resize(full_img, img, target_size, 0, 0, INTER_LINEAR_EXACT);
			//Work scale resizing is done for feature detection and matching
			resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);

			if (!is_seam_scale_set)
			{
				seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
				seam_work_aspect = seam_scale / work_scale;
				is_seam_scale_set = true;
			}

			computeImageFeatures(finder, img, features[i]);
			features[i].img_idx = i;
			//LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());
			//Seam scale resizing is for seam_find 
			resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
			//resize(full_img, img, target_size, 0, 0, INTER_LINEAR_EXACT);
			images[i] = img.clone();
		}

		for (int i = 0; i < num_images && ((count) != 0); ++i)
		{
			Mat seam_img;
			resize(frames[i], seam_img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
			images[i] = seam_img.clone();
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

			matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
			(*matcher)(features, pairwise_matches);

			//frees internal memory after matching
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


			for (int i = 0; i < num_images; ++i)
			{
				masks[i].create(images[i].size(), CV_8U);
				masks[i].setTo(Scalar::all(255));
			}

			//For warping frames onto one another spericalwarper is required to give wide panoramic view
			//Also coordinates used are sperical coordinates

			warper_creator = makePtr<cv::SphericalWarper>();
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

		compensator->feed(corners, images_warped, masks_warped);
		seam_finder->find(images_warped_f, corners, masks_warped);

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

				//Compute relative scales
				//Compose_seam_aspect = compose_scale / seam_scale;
				compose_work_aspect = compose_scale / work_scale;

				//Update warped image scale
				warped_image_scale *= static_cast<float>(compose_work_aspect);
				warper = warper_creator->create(warped_image_scale);

				//Update corners and sizes
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

			full_img.release();
			Size img_size = img.size();
			Mat K;
			cameras[img_idx].K().convertTo(K, CV_32F);

			//Warp the current image
			warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

			//Warp the current image mask
			mask.create(img_size, CV_8U);
			mask.setTo(Scalar::all(255));
			warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

			//Compensate exposure
			compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
			img_warped.convertTo(img_warped_s, CV_16S);

			//Compensate exposure
			img_warped.release();
			full_img.release();
			mask.release();
			dilate(masks_warped[img_idx], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
			mask_warped = seam_mask & mask_warped;

			//Compensate exposure
			if (!blender && !timelapse)
			{
				blender = Blender::createDefault(blend_type, try_cuda);
				dst_sz = resultRoi(corners, sizes).size();
				blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;

				mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
				//LOGLN("Multi-band blender, number of bands: " << mb->numBands());

				blender->prepare(corners, sizes);
			}

			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		}

		if (sg){
			writer.open("output.avi",
					cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), // MJPEG codec
					30,                                       // FPS
					dst_sz);                                  // Size of the video frames
			if (!writer.isOpened()) {
				std::cerr << "Error: Could not open the output video file!" << std::endl;
				return -1;
			}
			sg = 0;
		}

		Mat result, result_mask;
		blender->blend(result, result_mask);
		imwrite(result_name, result);
		Mat result_8u;

		if (result.depth() != CV_8U)
			result.convertTo(result_8u, CV_8U);  // simple scale, might clip
		else
			result_8u = result;

		count++;
		writer.write(result_8u);
		imshow("Stitched Video", result_8u);
		
		//waitKey(1);   
		if (waitKey(1) == 27) { // ESC key
			cout << "ESC pressed. Exiting..." << endl;
			break;
		}

	}

	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	writer.release();
	return 0;
}

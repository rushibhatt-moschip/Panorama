#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {

	if (argc < 5) {
		cerr << "Usage: " << argv[0]
			<< " --v1 left_video.mp4 --v2 right_video.mp4 --output stitched_output.mp4" << endl;
		return EXIT_FAILURE;
	}

	string leftVideo, rightVideo, outputVideo;
	for (int i = 1; i < argc; ++i) {
		string arg = argv[i];
		if (arg == "--v1" && i + 1 < argc) leftVideo = argv[++i];
		else if (arg == "--v2" && i + 1 < argc) rightVideo = argv[++i];
		else if (arg == "--output" && i + 1 < argc) outputVideo = argv[++i];
	}

	if (leftVideo.empty() || rightVideo.empty() || outputVideo.empty()) {
		cerr << "Invalid arguments!" << endl;
		return EXIT_FAILURE;
	}

	VideoCapture cap1(leftVideo), cap2(rightVideo);
	if (!cap1.isOpened() || !cap2.isOpened()) {
		cerr << "Error: Can't open input videos!" << endl;
		return EXIT_FAILURE;
	}

	// Read first frame from each video
	Mat frame1, frame2;
	while(true){
		
		cap1 >> frame1;
		cap2 >> frame2;

		if (frame1.empty() || frame2.empty()) {
			cerr << "Error: Can't read initial frames!" << endl;
			return EXIT_FAILURE;
		}

		vector<Mat>imgs;

		Rect rect(0, 0, frame1.cols / 2, frame1.rows);
		imgs.push_back(frame1(rect).clone());
		rect.x = frame1.cols / 3;
		imgs.push_back(frame1(rect).clone());
		rect.x = frame1.cols / 2;
		imgs.push_back(frame1(rect).clone());

		Rect rect1(0, 0, frame2.cols / 2, frame1.rows);
		imgs.push_back(frame2(rect1).clone());
		rect1.x = frame2.cols / 3;
		imgs.push_back(frame2(rect1).clone());
		rect1.x = frame2.cols / 2;
		imgs.push_back(frame2(rect1).clone());

		Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::SCANS);
		Mat pano;
		Stitcher::Status status = stitcher->stitch(imgs, pano);

		if (status != Stitcher::OK) {
			cerr << "Initial stitching failed, code = " << int(status) << endl;
			return EXIT_FAILURE;
		}
	//	cout << "enter" << endl;
		imshow("final_out", pano);
		waitKey(30);
		//cout << "Initial stitching successful." << endl;
	}
/*
	// Prepare output video writer
	double fps = cap1.get(CAP_PROP_FPS);
	if (fps <= 0) fps = 30; // Fallback if FPS is unknown
	VideoWriter writer(outputVideo,
			VideoWriter::fourcc('M','J','P','G'),
			fps,
			pano.size());
	if (!writer.isOpened()) {
		cerr << "Error: Can't open output video for writing!" << endl;
		return EXIT_FAILURE;
	}

	// Rewind videos to first frame
	cap1.set(CAP_PROP_POS_FRAMES, 0);
	cap2.set(CAP_PROP_POS_FRAMES, 0);

	// Frame-by-frame stitching
	while (true) {
		cap1 >> frame1;
		cap2 >> frame2;
		if (frame1.empty() || frame2.empty()) break;

		vector<Mat> frames = { frame1, frame2 };
		// Use composePanorama if supported, else stitch directly
		status = stitcher->composePanorama(frames, pano);
		if (status != Stitcher::OK) {
			cerr << "Stitching failed at frame: status = " << int(status) << endl;
			break;
		}

		writer.write(pano);
		imshow("Stitched Video", pano);
		if (waitKey(1) == 27) break; // Press ESC to stop early
	}
*/
	cap1.release();
	cap2.release();
//	writer.release();
	destroyAllWindows();

	cout << "Video stitching completed. Output saved at " << outputVideo << endl;
	return EXIT_SUCCESS;
}


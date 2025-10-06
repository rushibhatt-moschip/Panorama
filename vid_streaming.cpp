#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Open two video nodes
    VideoCapture cap1(1, CAP_V4L2);  // /dev/video0
    VideoCapture cap2(4, CAP_V4L2);  // /dev/video1

    if (!cap1.isOpened() || !cap2.isOpened()) {
        cerr << "Error: Could not open video devices" << endl;
        return -1;
    }

    // Optional: set resolution and FPS
    cap1.set(CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap1.set(CAP_PROP_FPS, 30);
    cap1.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G')); 

    cap2.set(CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(CAP_PROP_FRAME_HEIGHT, 480);
    cap2.set(CAP_PROP_FPS, 30);
    cap2.set(cv::CAP_PROP_FOURCC, VideoWriter::fourcc('M','J','P','G')); 
    Mat frame1, frame2;

    while (true) {
        cap1 >> frame1;
        cap2 >> frame2;

        if (frame1.empty() || frame2.empty()) {
            cerr << "Error: Empty frame grabbed" << endl;
            break;
        }

        imshow("Camera 1", frame1);
        imshow("Camera 2", frame2);

        // Exit when 'q' is pressed
        if (waitKey(1) == 'q') break;
    }

    cap1.release();
    cap2.release();
    destroyAllWindows();

    return 0;
}


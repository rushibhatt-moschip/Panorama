#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>


int main(int argc, char* argv[]) {

    
    cv::VideoCapture cap1(0, cv::CAP_V4L2);
    cv::VideoCapture cap2(2, cv::CAP_V4L2);


    // setting resolution | resizing. 
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

     // setting resolution | resizing. 
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, 480);


    if (!cap1.isOpened()) {
        std::cout << "unable to open webcam_1" << std::endl;
    }

    if (!cap2.isOpened()) {
        std::cout << "unable to open webcam_2" << std::endl;
    }


    cv::Mat frame_1,frame_2;

    while (1) {
        cap1.read(frame_1);
        cap2.read(frame_2);


        if (frame_1.empty()) {
            std::cout << "couldnot read frame 1" << std::endl;

        }

        if (frame_2.empty()) {
            std::cout << "couldnot read frame 2" << std::endl;

        }

        
        cv::imshow ("frame 1", frame_1);
        cv::imshow ("frame 2", frame_2);

        if (cv::waitKey(1) == 'q') {
            break;
        }

    }

    cap1.release();
    cap2.release();
    cv::destroyAllWindows();
    return 0;
}
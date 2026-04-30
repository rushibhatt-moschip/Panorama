/** 
calibrate bird eye view. 

*/

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>

#define STEP_SIZE 1


int main (int argc, char* argv[]) {

    if (argc < 2 || argc > 4) {
        printf("Invalid number of arguments. This_script image_path yml_save_path \n");
        exit(0);
    }
    
    int save_mat = 0;
    if(argc >= 3 ) {
        save_mat = 1;
    }

    cv::Mat image = cv::imread(argv[1]);
    cv::resize(image, image, cv::Size(image.cols , image.rows));

    int img_h, img_w; 
    img_h = image.rows;
    img_w = image.cols;

    std::cout << "image height : " << image.rows << "| image width : " << image.cols << std::endl;
    

    int h_point, v_point;
    h_point = 0;
    v_point = 0;

    std::vector <cv::Point2f> src_points = {cv::Point2f (0, v_point), cv::Point2f (img_w, v_point), 
                                            cv::Point2f ( img_w, img_h), cv::Point2f (0, img_h)};


    std::vector <cv::Point2f> dst_points = {cv::Point2f (0, 0), cv::Point2f (img_w, 0), 
                                            cv::Point2f ( img_w, img_h), cv::Point2f (0, img_h )};

    cv::Mat matrix = cv::getPerspectiveTransform (src_points, dst_points);
    cv::Mat transform_img;

    cv::warpPerspective (image, transform_img, matrix, cv::Size(img_w, img_h));
    std::cout << "trasnform image height : " << transform_img.rows << "| transform  image width : " << transform_img.cols << std::endl;

    int calculate_transform = 0;
    int key_pressed;
    while (true) {

        cv::imshow("transformed_img", transform_img);
        cv::imshow("original_img", image);

        key_pressed = cv::waitKey(1);
        if (key_pressed == 113 || key_pressed == 119 || key_pressed == 97 || key_pressed == 100 || key_pressed == 115) { // 'q', 'w', 's', 'a', 'd'
            if (key_pressed == 113) {
                break;
            }
            else if (key_pressed == 119) { // 'w'
                v_point += STEP_SIZE;
                src_points[0] = cv::Point2f (0, v_point);
                src_points[1] = cv::Point2f (img_w, v_point);
                calculate_transform = 1;
            }
            else if (key_pressed == 115){ // 's' 
                v_point -= STEP_SIZE;
                src_points[0] = cv::Point2f (0, v_point);
                src_points[1] = cv::Point2f (img_w, v_point);
                calculate_transform=1;
            }
            else if (key_pressed == 97) {  // 'a'
                h_point -= STEP_SIZE;
                dst_points[2] = cv::Point2f (img_w - h_point, img_h);
                dst_points[3] = cv::Point2f (0 + h_point, img_h );
                calculate_transform=1;
            }
            else if (key_pressed == 100)  { // 'd'
                h_point += STEP_SIZE;
                dst_points[2] = cv::Point2f (img_w - h_point, img_h);
                dst_points[3] = cv::Point2f (0 + h_point, img_h );
                calculate_transform=1;
            }
        }

        if (calculate_transform) {
            matrix = cv::getPerspectiveTransform (src_points, dst_points);
            cv::warpPerspective (image, transform_img, matrix, cv::Size(img_w, img_h));

            printf("src_points : (%.2f, %.2f) \t (%.2f, %.2f) \t (%.2f, %.2f) \t (%.2f, %.2f)\n", src_points[0].x, src_points[0].y, src_points[1].x, src_points[1].y, src_points[2].x, src_points[2].y, src_points[3].x, src_points[3].y);
            printf("dst_points : (%.2f, %.2f) \t (%.2f, %.2f) \t (%.2f, %.2f) \t (%.2f, %.2f)\n", dst_points[0].x, dst_points[0].y, dst_points[1].x, dst_points[1].y, dst_points[2].x, dst_points[2].y, dst_points[3].x, dst_points[3].y);
            
            calculate_transform = 0;
        }
    }
    std::cout << "=================================================================================================" << std::endl;

    if (save_mat) {
        // writing the Matrix M in /tmp/transform_matrix.yml
        std::cout << "Saving transform matrix to : " << argv[2] << std::endl;
        cv::FileStorage fs (argv[2], cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cout << "enable to write to file :: facing problem " << std::endl;
        }
        else {
            fs << "mat" << matrix;
            fs.release();
            std::cout << "Wrote Matrix to file : " << argv[2] << std::endl;
        }
    }
    if (argc == 4) {
        std::cout << "dumping img : " << argv[3] << std::endl;
        cv::imwrite(argv[3], transform_img);
    }
    return 0;
}
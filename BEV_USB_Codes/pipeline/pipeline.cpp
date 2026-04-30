/** 
 * cpp pipeline. 
 * takes an image | img1|img2|img3 | matrices | yml file. and gives the birdeye_view. 
 */

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>
#include <vector>

class parse_info {

public:
    int total_images;
    std::vector<std::string> naming_convertion{".jpg","_pT.yml"};

    std::string folder_path;                    // main folder path 
    std::vector <std::string> img_paths;        // image path. { not requireed in video script. there will be no images. }
    std::vector <std::string> pTransform_paths; // {perspective transform matrics path}
    std::string master_yml_path;                // master yml path 
    std::vector <cv::Mat> perspective_transform_matrix;

    // master yml info part 
    cv::FileStorage fs;
    std::vector <cv::Mat> rotation_center, offset;
    std::vector <int> rotation_angle;
    std::vector <float> scale;
    cv::Mat canvas_size;


    parse_info (std::string f_path, int images) {
        folder_path = f_path; total_images = images; 
        generate_paths();
        read_perspective_matrices();
        read_info_yml();
    }

    void generate_paths();
    void read_perspective_matrices();
    void read_info_yml();
};

/**
    populates the path of images, according to naming convertion. 
 */
void parse_info::generate_paths() {

    // generate the images path 
    std::string img_path;
    for (int i = 0; i < total_images;  i++) {
        img_path = folder_path + std::string("img_") + std::to_string(i) + naming_convertion[0];
        img_paths.push_back(img_path);
    }

    // generate the perspective transform part. 
    for (int i = 0; i < total_images;  i++) {
        img_path = folder_path + std::string("img_") + std::to_string(i) + naming_convertion[1];
        pTransform_paths.push_back(img_path);
    }

    master_yml_path = folder_path + std::string("master.yml");
}

// reads the perspective transform matrix. 
void parse_info::read_perspective_matrices() {

    cv::FileStorage fs_tmp;
    bool val;
    cv::Mat tmp;
    
    for (int i = 0; i < pTransform_paths.size(); i++) {

        val = fs_tmp.open (pTransform_paths[i] , cv::FileStorage::READ);
        if (val == 0) {
            std::cout << "enable to read matrix from file : " << pTransform_paths[i] << std::endl;
            std::cout << "Terminating program " << std::endl;
            exit(0);
        }
        fs_tmp["mat"] >> tmp;
        perspective_transform_matrix.push_back(tmp);
        fs_tmp.release();
        tmp.release();
    
    }
}

void parse_info :: read_info_yml() {
    if(fs.open(master_yml_path, cv::FileStorage::READ) == 0) {
        std::cout << "not able to open info yml : " << master_yml_path << std::endl;
        std::cout << "Terminating the program " << std::endl;
        exit(0);
    }

    // parameters reading. 
    std::vector<std::string> keys{"_rotate_center", "_rotate_angle", "_scale", "offset"};
    std::string tmp_key;
    
    cv::Mat tmp_mat;
    float tmp_int;
    for (int i = 0; i < total_images; i++) {
        
        // reading rotate_center
        tmp_key = "img" + std::to_string(i) + keys[0];
        fs[tmp_key] >> tmp_mat;
        rotation_center.push_back(tmp_mat); 
        tmp_mat.release();

        // reading rotation_center
        tmp_key = "img" + std::to_string(i) + keys[1];
        fs[tmp_key] >> tmp_int;
        rotation_angle.push_back(tmp_int);
        
        // reading the scale 
        tmp_key = "img" + std::to_string(i) + keys[2];
        fs[tmp_key] >> tmp_int;
        scale.push_back(tmp_int);

        // reading offset : 
        tmp_key = "img" + std::to_string(i) + keys[3];
        fs[tmp_key] >> tmp_mat;
        offset.push_back(tmp_mat); 
        tmp_mat.release(); 
    }
    fs["CANVAS SIZE"] >> canvas_size;
    fs.release();
 
}

using namespace std;

int main(int argc, char* argv[]) {

    if (argc < 3) {
        printf("Invalid number of arguments \n : this script | folder path | number of images \n");
        exit(0);
    }
    int total_images = atoi(argv[2]);
    parse_info parser(argv[1], total_images);

    // images read and converting into top view. 
    std::vector <cv::Mat> top_images;
    cv::Mat tmp;
    
    for(int i = 0; i < total_images; i++) {
        tmp = cv::imread(parser.img_paths[i]);
        cv::warpPerspective (tmp, tmp, parser.perspective_transform_matrix[i], cv::Size(tmp.cols, tmp.rows));
        top_images.push_back (tmp);

        tmp.release();
    }

    // adding the matrix part. 
    cv::Mat rot_mat, roi_0, roi_1, tmp_canvas; 
    
    vector<cv::Mat> translated_canvas{cv::Mat(cv::Size(1280, 720), top_images[0].type(), cv::Scalar(0, 0, 0)), \
    cv::Mat(cv::Size(1280, 720), top_images[0].type(), cv::Scalar(0, 0, 0)),\
    cv::Mat(cv::Size(1280, 720), top_images[0].type(), cv::Scalar(0, 0, 0))};

    cv::Mat final_canvas(cv::Size(1280, 720), top_images[0].type(), cv::Scalar(0, 0, 0));

    // hardcoding blending mask pART : TO INTEGRATE IN PARSE_INFO. 
    /*cv::Mat img0_mask00 = cv::imread("/home/dipesh/projects/Demo/WebCam_POC/pipeline/data/c2-view_blendmask_0.jpg");
    cv::Mat img0_mask01 = cv::imread("/home/dipesh/projects/Demo/WebCam_POC/pipeline/data/c2-view_blendmask_1.jpg");
    cv::Mat img1_mask00 = cv::imread("/home/dipesh/projects/Demo/WebCam_POC/pipeline/data/c1-view_blendmask_0.jpg");
    cv::Mat img2_mask00 = cv::imread("/home/dipesh/projects/Demo/WebCam_POC/pipeline/data/c3-view_blendmask_0.jpg");
    vector <cv::Mat> 
*/
    cv::Point2f point2f_tmp;
    int off_x, off_y;

    // rotate -> translate
    for(int i = 0; i < total_images; i++) {
        
        // rotation and scale part 
        cv::Point2f c (parser.rotation_center[i].at<int>(0,0), parser.rotation_center[i].at<int>(0,1));
        rot_mat = cv::getRotationMatrix2D(c, parser.rotation_angle[i], parser.scale[i]);
        cv::warpAffine(top_images[i], tmp_canvas, rot_mat, cv::Size(1280,720));

        off_x = parser.offset[i].at<int>(0,0);
        off_y = parser.offset[i].at<int>(0,1);

        roi_0 = tmp_canvas(cv::Rect (0, 0, 1280 - off_x, 720 - off_y));
        roi_1 = translated_canvas[i](cv::Rect (off_x, off_y, 1280 - off_x, 720- off_y));
        roi_0.copyTo(roi_1);

        roi_0.release();
        roi_1.release();
        tmp_canvas.release();
        rot_mat.release();

    }
for (int i = 0; i < total_images; i++) {
    cv::add(final_canvas, translated_canvas[i], final_canvas);
}
    cv::imshow("final canvas", final_canvas);
    cv::waitKey(0);
    
    return 0;
}

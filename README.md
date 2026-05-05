* Note: This directory contains code for video stitching using USB cameras and MIPI cameras.

## Contents of the directory
1. Usage    		 - Binary that prints the command line args which can be passed to the video stitching algorithm
2. opencv_example.cpp    - Code adapted from the OpenCV GitHub repository (`detailed_stitching.cpp`), demonstrating image stitching functionality.
3. vid_streaming.cpp     - Simple live streaming program that captures USB camera feed and displays it.
4. mipi_codes  		 - Contains codes for video stitching using mipi cameras
	* latest_updates.cpp     - Fully functional code with additional features like crop,resize.
	* perfect_clean_code.cpp - Well-commented and optimized code
	* one_time_init.cpp      - Un-optimized understandable code.
5. resultant_imgs  	- Contains example image and stitched output results.
6. usb_codes  		- Contains code for video stitching using USB cameras (feature matching is performed only once)
7. BEV_USB_Codes  	- Contains codes for bird eye view stitching using webcams. Refer to the README file in this directory for more details.
8. docs  		- Contains presentation slides explaining the approach and implementation.

## Compilation steps

```bash
g++ mipi_codes/latest_updates.cpp -o play `pkg-config --cflags --libs opencv4`
```
```bash
g++ opencv_example.cpp -o image_stitch `pkg-config --cflags --libs opencv4`
```
## Example pipeline

```bash
./play --conf_thresh 0.5
```
```bash
./image_stitch --conf_thresh 0.5 resultant_imgs/im1.jpg resultant_imgs/im2.jpg
```

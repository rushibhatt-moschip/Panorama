* Note: This directory contains codes for video stitchig foe USB cameras and Web-cameras as well.

## Contents of the directory
1. Usage    		 - Binary which prints the commnad line args which can be passed to the video stitching algorithm
2. opencv_example.cpp    - Code copied from github repo of opencv named detailed_stitching.cpp, Used for image stitching.
3. vid_streaming.cpp     - Simply live streaming camera capture to display code.
4. mipi_codes  		 - contains codes of video stitching using mipi cameras
	* perfect_clean_code.cpp - Commented and optimized code   
	* latest_updates.cpp     - Uncommented and contains additional features like crop,resize & assertion temporary solved
	* one_time_init.cpp      - Un-optimized understandable code.
5. resultant_imgs  	- contains stitched images.
6. usb_codes  		- contains code for usb camera video stitching (feature point matching happens only once)
7. BEV_USB_Codes  	- contains codes for bird eye view stitching using webcams.Go to this directory and see readme for more information
8. docs  		- contains ppt that explains the process and approach.

## Compilation steps

```bash
g++ mipi_codes/latest_updates.cpp -o play `pkg-config --cflags --libs opencv4`
g++ opencv_example.cpp -o play `pkg-config --cflags --libs opencv4`
```
## Example pipeline

```bash
./play --conf_thresh 0.5
./play --conf_thresh 0.5 resultant_imgs/im1.jpg resultant_imgs/im2.jpg
```







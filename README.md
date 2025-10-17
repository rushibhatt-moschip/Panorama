***Note: This directory contains codes for video stitchig foe USB cameras and Web-cameras as well.

## Contents of the directory
1. Usage    		- Binary which prints the commnad line args which can be passed to the video stitching algorithm
2. perf     		- Cpp binary for running 2 webcams and implementing image_stitching_detailed 
3. main     		- Binary for video stitching code for 2 usb webcams  
4. main.cpp 		- Video stitching code for 2 usb webcams 
5. vid_streaming.cpp    - streaming camera-to-diaplay code.
6. mipi_codes  		- contains codes of video stitching using mipi cameras
* perfect_clean_code.cpp - Commented and optimized code   
* latest_updates.cpp     - Uncommented and contains additional features like crop,resize & assertion temporary solved
* one_time_init.cpp      - Un-optimized understandable code.
7. resultant_imgs  	- contains stitched images.
8. usb_codes  		- contains code for usb camera video stitching (feature point matching happens only once)

## Compilation steps

g++ main.cpp -o main `pkg-config --cflags --libs opencv4`

### Note: If using openmp compiler directives 

g++ -fopenmp main.cpp -o main `pkg-config --cflags --libs opencv4`

## Example pipeline

./main --conf_thresh 0.5








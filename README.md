***Note: This directory contains codes for video stitchig foe USB cameras and Web-cameras as well.

## Contents of the directory
1. Usage    		- Binary which prints the commnad line args which can be passed to the video stitching algorithm
2. perf     		- Cpp binary for running 2 webcams and implementing image_stitching_detailed 
3. main     		- Binary for video stitching code for 2 usb webcams  
4. main.cpp 		- Video stitching code for 2 usb webcams 
5. vid_streaming.cpp    - streaming camera-to-diaplay code.

## Compilation steps

g++ -fopenmp main.cpp -o main `pkg-config --cflags --libs opencv4`

***Note: If using openmp compiler directives 

## g++ -fopenmp main.cpp -o main `pkg-config --cflags --libs opencv4`

## Example pipeline

./main --conf_thresh 0.5








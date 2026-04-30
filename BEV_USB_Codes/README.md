* Note: This directory contains codes for Bird Eye View video stitching for USB cameras on linux PC.

## Codebase guideline

To just display feed of two cameras you can run this: Scripts_stuff/perspective_transform/video_stream_2.cpp
To calibrate parameters for perspective transform (to get top view): Scripts_stuff/perspective_transform/calibrate_birdeye.cpp
To manually stitch the images and save its rotation matrix: Scripts_stuff/calibrate_all.py
To Create mask for seamless transition at the stitched region: Scripts_stuff/calibrate_blending.py
To run the combined code for live Bird eye view output: CPP/final_image_script.cpp
Just a sample program which uses the image data from  this repo and gives a sample image output: pipeline/pipeline.cpp

## Compilation steps
```bash
g++ <code name> -o play `pkg-config --cflags --libs opencv4`
```
## Process to calibrate and run the final code
1> Do perspective transform by choosing initial and final points visually untill best illusion of top view is found.
2> Manually overlap the two images and then translate and rotate until it looks perfectly stitched.
3> Create fading mask on the overlap region ffor both the images such that brightness looks uniform over whole image.
4> Once all things are done and saved in data now you can run the final script for starting live video source and getting a Bird Eye View output.

## Quick example
 
To simply run the sample code compile it with: g++ pipeline/pipeline.cpp -o play `pkg-config --cflags --libs opencv4`
and then run it with: ./play pipeline/data/ 3

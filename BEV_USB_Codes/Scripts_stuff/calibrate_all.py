HELP = '''
    Script or program to calibrate the images overlap. can calibrate 2 images at a time. out of 2, one is considered stationary, reference. 

    usage : this script <images path seperated with commas> <yml info save path> <image_2_side {default : right }
        first image path should be one with that is constant. 

    images it takes is after bird eyed view images.

    controls : 
        moving image : 
            q -> left | w -> up | e -> down | r -> right 
        scaling image : 
            '-' -> scale down 
            '+' -> scale up 
        rotation : 
            'n' -> angle - 
            'm' -> angle +
        rotation center movement : 
            u -> left | i -> up | o -> down | p -> right 
    
'''


import numpy as np 
import cv2
import sys 
import argparse
import os 

HEIGHT = 720
WIDTH = 1280

STEP = int(1)
SCALE_STEP = float(0.01)
ANGLE_STEP = int(1)


if len(sys.argv) < 3:
    print("insufficent number of argument provided. \n==================USAGE======================\n{}".format(HELP))
    exit(0)

i_paths = sys.argv[1]
yml_save_path = sys.argv[2]
all_image_paths = i_paths.split(',')

# checking the path of file
for i in all_image_paths:
    if os.path.exists(i) == False:
        print("file : {} does not exists".format(i))
        exit(0)

image_list = []
for img_path in all_image_paths:
    _img = cv2.imread (img_path)
    if type(_img) == type(None):
        print("cannot read image : {}".format(img_path))
        exit(0)
    image_list.append(_img)

# image being loaded successfully.

# resizing the images to 1280x720
canvas_array = np.zeros((len(image_list), HEIGHT, WIDTH, 3)).astype(np.uint8)

for i, image in enumerate(image_list):
    # incase image dim is more, resize it
    if image.shape[0] > HEIGHT or image.shape[1] > WIDTH:
        image = cv2.resize(image, (HEIGHT, WIDTH))

    canvas_array[i,0:image.shape[0], 0:image.shape[1], :] = image.copy()

total_images = canvas_array.shape[0]
offsets_array = np.zeros((canvas_array.shape[0], 2)).astype(np.int32)
scale_array = np.ones(canvas_array.shape[0]).astype(np.float32)
rotation_array = np.zeros(canvas_array.shape[0]).astype(np.float32)
rotation_center_array = np.zeros((canvas_array.shape[0], 2)).astype(np.int32)

rotated_canvas_array = np.zeros((len(image_list), HEIGHT, WIDTH, 3)).astype(np.uint8)

for i in range(rotated_canvas_array.shape[0]):
    print(rotation_center_array[i].tolist())
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center_array[i].tolist(), rotation_array[i], scale_array[i])
    rotated_image = cv2.warpAffine(image_list[i].copy(), rotation_matrix, (WIDTH, HEIGHT))
    rotated_canvas_array[i,:,:,:] = rotated_image


selected_image = 0

print_info = True
while True:
    
    key_pressed = cv2.waitKey(1)

    if key_pressed == 27:
        break

    if key_pressed == ord('1') or key_pressed == ord('2'):
        if key_pressed == ord('1'):
            if selected_image == (total_images - 1):
                selected_image = 0
            else:
                selected_image += 1
        elif key_pressed == ord('2'):
            if selected_image == (0):
                selected_image = (total_images - 1)
            else:
                selected_image -= 1
        print("======> image_selected : {}".format(selected_image))
        print_info = True

    # related to movement of selected image 
    if key_pressed == ord('w') or key_pressed == ord('q') or key_pressed == ord('e') or key_pressed == ord('r'):
        if key_pressed == ord('w'):
            offsets_array[selected_image, 1] -= STEP
        elif key_pressed == ord('e') and offsets_array[selected_image, 1] < HEIGHT:
            offsets_array[selected_image, 1] += STEP
        elif key_pressed == ord('q') and offsets_array[selected_image, 0] > 0:
            offsets_array[selected_image, 0] -= STEP
        elif key_pressed == ord('r') and offsets_array[selected_image, 0] < WIDTH:
            offsets_array[selected_image, 0] += STEP
        print_info = True
    
    # related to movement of center of rotation of selected image 
    if key_pressed == ord('u') or key_pressed == ord('i') or key_pressed == ord('o') or key_pressed == ord('p'):
        if key_pressed == ord('u') and rotation_center_array[selected_image, 0] > 0:
            rotation_center_array[selected_image, 0] -= STEP
        elif key_pressed == ord('i') and rotation_center_array[selected_image, 1] > 0:
            rotation_center_array[selected_image, 1] -= STEP
        elif key_pressed == ord('o') and rotation_center_array[selected_image, 1] < image_list[selected_image].shape[0]:
            rotation_center_array[selected_image, 1] += STEP
        elif key_pressed == ord('p') and rotation_center_array[selected_image, 0] < image_list[selected_image].shape[1]:
            rotation_center_array[selected_image, 0] += STEP
        print_info = True
    
    # related to rotation and scale of images. { image 1 cannot be rotated}
    if key_pressed == ord('n') or key_pressed == ord('m') or key_pressed == ord('-') or key_pressed == ord('='):
        if True: #and selected_image != 0:
            if key_pressed == ord('n'):
                rotation_array[selected_image] += ANGLE_STEP
            elif key_pressed == ord('m'):
                rotation_array[selected_image] -= ANGLE_STEP

        if key_pressed == ord('='):
            scale_array[selected_image] += SCALE_STEP
        elif key_pressed == ord('-'):
            scale_array[selected_image] -= SCALE_STEP
        print_info  = True

    if print_info:
        print_info = False
        print("offsets : (x,y) : ({}, {})".format(offsets_array[selected_image, 0], offsets_array[selected_image, 1]))
        print("rotation centrer offsets : x,y : ({}, {})".format(rotation_center_array[selected_image, 0], rotation_center_array[selected_image, 1]))
        print("rotation angle : {}".format(rotation_array[selected_image]))
        print("scale : {}".format(scale_array[selected_image]))


    # calculation translation and rotation array. 
    translation_img_array = np.zeros((len(image_list), HEIGHT, WIDTH, 3)).astype(np.uint8)
    for i in range(translation_img_array.shape[0]):
        
        # rotation part
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center_array[i].tolist(), rotation_array[i], scale_array[i])
        rotated_image = cv2.warpAffine(canvas_array[i].copy(), rotation_matrix, (WIDTH, HEIGHT))
        # translation part 
        if offsets_array[i, 1] < 0:
            translation_img_array[i, 0: HEIGHT + offsets_array[i, 1]  , offsets_array[i, 0] :, : ]  = rotated_image[ abs(offsets_array[i, 1]) : HEIGHT, 0: WIDTH - offsets_array [i, 0]]

        elif offsets_array[i, 0] < 0:
            translation_img_array[i, offsets_array[i, 1] : 0, WIDTH + offsets_array[i, 0] :, : ]  = rotated_image[ 0 : HEIGHT - offsets_array[i, 1], abs(offsets_array[i, 0]): WIDTH + offsets_array [i, 0]]
        else:    
            translation_img_array[i, offsets_array[i, 1] : , offsets_array[i, 0] :, : ]  = rotated_image[ 0 : HEIGHT - offsets_array[i, 1], 0: WIDTH - offsets_array [i, 0]]
        
    

    final_canvas = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)
    for i in range(rotated_canvas_array.shape[0]):
        final_canvas = cv2.add(final_canvas, translation_img_array[i])
     
        

        
    
    ## showing the selected image and center selected 
    circle_s_img = canvas_array[selected_image].copy()
    if True:# and selected_image != 0:
        cv2.circle(circle_s_img, rotation_center_array[selected_image], 10, (255,0,0), 1)
    cv2.imshow("center calibrate", circle_s_img)
    cv2.imshow("final canvas", final_canvas)


# saving  to yml path 
fs = cv2.FileStorage(yml_save_path, cv2.FILE_STORAGE_WRITE)

'''
offsets_array = np.zeros((canvas_array.shape[0], 2)).astype(np.int32)
scale_array = np.ones(canvas_array.shape[0]).astype(np.float32)
rotation_array = np.zeros(canvas_array.shape[0]).astype(np.float32)
rotation_center_array = np.zeros((canvas_array.shape[0], 2)).astype(np.int32)
'''

# writing the canvas size: 
fs.write("CANVAS SIZE", np.array([WIDTH, HEIGHT]))
for i in range(canvas_array.shape[0]):
    img_off_string = "img{}".format(i)
    fs.write(img_off_string + "_rotate_center", rotation_center_array[i])
    fs.write(img_off_string + "_rotate_angle", rotation_array[i])
    fs.write(img_off_string + "_scale", scale_array[i])
    fs.write(img_off_string + "offset", offsets_array[i])
fs.release()

cv2.destroyAllWindows()     


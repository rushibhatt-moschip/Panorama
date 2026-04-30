''' 
Calibrate the blending onto an image. 
    script format : arg_1 = image_path 



    CONTROLS: 
        Quit                    : Esc
        translation             : 'q' : left | 'w' : up | 'e' : down | 'r' : right 
        change point            : space 
        clean blending          : 'c'
        get out of clean blend  : Esc
        save blending mask      : 's'

'''


import numpy as np 
import cv2
import sys 
import argparse
import os
from collections import OrderedDict
import math

PIXEL_STEP = 5
alpha = 1e5

def savee(blend_num):
     
    i=blend_num
    #for i in range(blends_array.shape[0]):
    image_canvases[img_indx] = cv2.multiply(image_canvases[img_indx].astype(np.float32), blends_array[i]).astype(np.uint8)
        # saving the blend array 
        # blends_array = np.ones((no_of_blends, HEIGHT, WIDTH, 3)).astype(np.float32)
    p = image_name_list[img_indx].split('.')[0]
    #for i in range(blends_array.shape[0]):
    p_ = os.path.join(save_path, p + "_blendmask_{}.jpg".format(i))
    cv2.imwrite(p_, (blends_array[i] * 255).astype(np.uint8))
    print("saved to path : {}".format(p_))


def parse_yml(yml_path, img_list_len):
    fs = cv2.FileStorage (yml_path, cv2.FILE_STORAGE_READ)
    # reading the attributes 
    attr = dict() 

    canvas_size = fs.getNode('CANVAS SIZE').mat()[:,0].tolist()
    attr['canvas_size'] = canvas_size

    for i in range(img_list_len):
        x = "img{}".format(i)
        for s in ["_rotate_center", "_rotate_angle", "_scale", "offset"]:
            y = x + s
            if s in ["_rotate_center", "offset"]:
                attr[y] = fs.getNode(y).mat()[:,0].tolist()
                
            else:
                attr[y] = fs.getNode(y).real()
    return attr


def parse_name():

    # images_path, yml_path, save_folder. 
    if len(sys.argv) < 3:
        print("give image path as argument argument : images_path, yml_path, save_folder\n")
        exit(0)
    
    # reading image into a list 
    image_paths = sys.argv[1].split(',')
    image_list = []
    image_name_list = []
    for each_path in image_paths:
        
        if os.path.exists(each_path) == False:
            print("invalid path : {}".format(each_path))
            exit(0)
        img = cv2.imread(each_path)
        if type(img) == type(None):
            print("could not read image from path : {} ".format(each_path))
            exit(0)
        image_list.append(img)
        image_name_list.append(each_path.split('/')[-1])

    if os.path.exists(sys.argv[2]) == False:
        print("invalid YML path used : {}".format(sys.argv[2]))
        exit(0)

    yml_dict = parse_yml (sys.argv[2], len(image_list))
    
    # checking save directory path 
    if os.path.isdir(sys.argv[3]) == False:
        os.makedirs(sys.argv[3], exists_ok=True)
    

    return image_list, image_name_list, yml_dict, sys.argv[3]

def generate_canvases(image_array, yml_data):
    canvases = np.zeros(( image_array.shape[0], HEIGHT, WIDTH, image_array.shape[-1])).astype(np.uint8)
    for i in range(canvases.shape[0]):

        strx = "img{}".format(i)
        # rotation 
        rotate_center = yml_data[strx + "_rotate_center"]
        rotate_angle = yml_data[strx + "_rotate_angle"]
        scale = yml_data[strx + "_scale"]
        
        # rotate image 
        rot_matrix = cv2.getRotationMatrix2D(rotate_center, rotate_angle, scale)
        rotated_image = cv2.warpAffine(image_array[i].copy(), rot_matrix, yml_data['canvas_size'])

        # translation 
        offset = yml_data[strx + "offset"]
        canvases[i, offset[1] : , offset[0] : , : ] = rotated_image[0: HEIGHT - offset[1], 0 : WIDTH - offset[0], :].copy()

    return canvases

def blend_selection () :
    print("Enter the number of blends")
    print("\n=============================================================================\n")
    choice = int(input())
    return choice

def do_sloped_blend(blend_points, blend_canvas):
      
    m32 = (blend_points [3, 1] - blend_points  [2, 1]) / ((blend_points [3, 0] - blend_points[ 2, 0]))
    b32 = blend_points[ 3, 1] - m32 * blend_points[ 3,0]

    m01 = (blend_points[ 1, 1] - blend_points[ 0, 1]) / (blend_points[ 1, 0] - blend_points[ 0, 0])
    b01 = blend_points[ 1, 1] - m01 * blend_points[ 1,0]

    m02 = (blend_points[ 2, 1] - blend_points[ 0, 1]) / (blend_points[ 2, 0] - blend_points[ 0, 0])
    b02 = blend_points[ 2, 1] - m02 * blend_points[ 2,0]

    for y in range(blend_canvas.shape[0]):
        for x in range(blend_canvas.shape[1]):

            # m01 part  
            if math.isinf(m01):
                if x < blend_points[0, 0]:
                    blend_canvas[y, x, :] = [1, 1, 1]
            if (m01 > 0 and b01 < 0) or (m01 < 0 and b01 > 0):
                if (m01) > 0 and y > m01 * x + b01:
                    blend_canvas[y, x, :] = [1, 1, 1]
                elif (m01) < 0 and y < m01 * x + b01:
                    blend_canvas[y, x, :] = [1, 1, 1]
            elif (m01 > 0 and b02 > 0) or (m01 < 0 and b02 < 0):
                if (m01) > 0 and y < m01 * x + b01:
                    blend_canvas[y, x, :] = [1, 1, 1]
                elif (m01) < 0 and y > m01 * x + b01:
                    blend_canvas[y, x, :] = [1, 1, 1]
        
            # m02 part 
            if math.isinf(m02):
                if x < blend_points[0, 0]:
                    blend_canvas[y, x, :] = [1, 1, 1]
            if (m02 > 0 and b02 < 0) or (m02 < 0 and b02 > 0):
                if (m02) > 0 and y < m02 * x + b02:
                    blend_canvas[y, x, :] = [1, 1, 1]
                elif (m02) < 0 and y < m02 * x + b02:
                    blend_canvas[y, x, :] = [1, 1, 1]
            elif (m02 > 0 and b02 > 0) or (m02 < 0 and b02 < 0):
                if (m02) > 0 and y < m02 * x + b02:
                    blend_canvas[y, x, :] = [1, 1, 1]
                elif (m02) < 0 and y > m02 * x + b02:
                    blend_canvas[y, x, :] = [1, 1, 1]

            # m03 part. 
            if math.isinf(m32):
                if x < blend_points[3, 0]:
                    blend_canvas[y, x, :] = [1, 1, 1]
            if (m32 > 0 and b32 < 0) or (m32 < 0 and b32 > 0):
                if (m32) > 0 and y < m32 * x + b32:
                    blend_canvas[y, x, :] = [1, 1, 1]
                elif (m32) < 0 and y > m32 * x + b32:
                    blend_canvas[y, x, :] = [1, 1, 1]
            elif (m32 > 0 and b32 > 0) or (m32 < 0 and b32 < 0):
                if (m32) > 0 and y > m32 * x + b32:
                    blend_canvas[y, x, :] = [1, 1, 1]
                elif (m32) < 0 and y < m32 * x + b32:
                    blend_canvas[y, x, :] = [1, 1, 1]


    return blend_canvas


if __name__ == "__main__":
    image_list, image_name_list, yml_data, save_path = parse_name()
    WIDTH, HEIGHT = yml_data['canvas_size'] 
    print(HEIGHT, WIDTH)
    print("image_names : {}".format(image_name_list))
    image_array = np.array(image_list)
    image_canvases = generate_canvases (image_array, yml_data)

    # defining x1, x2, x3, x4
    points = np.array([[ 10, 10], [10, HEIGHT - 10], 
        [100, 10], [100, HEIGHT - 10]])

    img_without_blend = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)
    for each_canvas in image_canvases:
        img_without_blend = cv2.add (img_without_blend, each_canvas)

    circle_selected = 0
    clean  = False 
    add_blend_region = True

    blends_dict = OrderedDict()
    blend_coordinates = OrderedDict()
    for name in image_name_list:
        blends_dict[name] = list()
        blend_coordinates[name] = list()

    key_of_selected = None
    indx_selected = None
    
    for img_indx in range (image_canvases.shape[0]):

        print("===========================================")
        print("generating image blending regions for image : {}".format(image_name_list[img_indx]))
        no_of_blends = blend_selection()
        if no_of_blends == -1:
            break

        blends_array = np.ones((no_of_blends, HEIGHT, WIDTH, 3)).astype(np.float32)

        blend_points = np.array([[[ 10, 10], [10, HEIGHT - 10], [100, 10], [100, HEIGHT - 10]]])
        blend_points = np.tile (blend_points, (no_of_blends,1,1))
        blend_selected = 0
        
        c_change = True
        clean = False
        flip = False
        while True:

            ########################## keys and command define part #################################
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break

            if key == ord('n'):
                blend_selected = (blend_selected + 1) if blend_selected < (no_of_blends - 1) else 0
                print("blend_selected : {}".format(blend_selected))
            if key == ord('f'):     # for flipping. 
                flip = not flip

            if key == 32:
                circle_selected = (circle_selected + 1) if circle_selected < 3 else 0
            
            if key == ord ('q') or key == ord('w') or key == ord('e') or key == ord('r'):
                if key == ord('q') and blend_points[blend_selected, circle_selected, 0] > 0:
                    blend_points[blend_selected, circle_selected, 0] -= PIXEL_STEP
                elif key == ord('w') and blend_points[blend_selected, circle_selected, 1] > 0:
                    blend_points[blend_selected, circle_selected, 1] -= PIXEL_STEP
                elif key == ord('e') and blend_points[blend_selected, circle_selected, 1] < HEIGHT:
                    blend_points[blend_selected, circle_selected, 1] += PIXEL_STEP
                elif key == ord('r') and blend_points[blend_selected, circle_selected, 0] < WIDTH:
                    blend_points[blend_selected, circle_selected, 0] += PIXEL_STEP
                print("coordinate change : {}".format(blend_points[blend_selected].tolist()))
                c_change = True
                # blend_points [blend_selected, circle_selected, x/y]
            
            if key == ord('c'):
                clean = True


            #################################### final drawing part #################################
            x_min = np.min(blend_points[blend_selected, :2, 0])
            x_max = np.max(blend_points[blend_selected, 2:, 0])
            if c_change:
                print("x_min : {} | x_max : {}".format(x_min, x_max))
                c_change = False

            # generating the blended area 
            if flip:
                blended_area = np.linspace(1,0, x_max - x_min)
            else:
                blended_area = np.linspace(0,1, x_max - x_min)
            blended_area = np.tile( blended_area, (3, HEIGHT, 1) )
            blended_area = np.moveaxis(blended_area, 0, -1)

            tmp_canvas = np.ones ((HEIGHT, WIDTH, 3)).astype(np.float32)
            tmp_canvas[ :, x_min : x_max] = blended_area


            if clean:
                tmp_canvas = do_sloped_blend (blend_points[blend_selected], tmp_canvas)

            blends_array[blend_selected] = tmp_canvas.copy()

            ################################## calculation ##########################################
            
            # multiplying blends with corresponding image. 
            tmp_image_canvases = image_canvases.copy()
            for i in range(blends_array.shape[0]):
                tmp_image_canvases[img_indx] = cv2.multiply(tmp_image_canvases[img_indx].astype(np.float32), blends_array[i]).astype(np.uint8)
            
            final_canvas = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)

            img_without_blend = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)
            for i, each_canvas in enumerate(tmp_image_canvases):
                final_canvas = cv2.add(each_canvas, final_canvas)
                img_without_blend = cv2.add(image_canvases[i], img_without_blend)

            # drawing the lines 
            cv2.line (final_canvas, blend_points[blend_selected, 0], blend_points[blend_selected, 1], (0, 0, 255), 1)
            cv2.line (final_canvas, blend_points[blend_selected, 2], blend_points[blend_selected, 3], (0, 0, 255), 1)

            # drawing the circles
            for i in range(4):
                if i == circle_selected:
                    cv2.circle (final_canvas, blend_points[blend_selected, i], 5, (255, 255, 0), 3)
                    cv2.circle (img_without_blend, blend_points[blend_selected, i], 5, (255, 255, 0), 3)
                else:
                    cv2.circle (final_canvas, blend_points[blend_selected, i], 5, (0, 255, 0), 2)
                    cv2.circle (img_without_blend, blend_points[blend_selected, i], 5, (0, 255, 0), 2)
            ######################################################################################
            # display the canvas
            x = cv2.resize(img_without_blend, None, fx=0.5, fy=0.5)
            final_canvas = cv2.resize(final_canvas, None, fx=0.5, fy=0.5)

            cv2.imshow("canvas", final_canvas)
            cv2.imshow("ref", x)
            if clean:
                print("press s to save or esc to make changes")
                save_key=cv2.waitKey(0)
                if save_key == 27:
                    print("not saving")
                    clean=False
                if save_key == 115:
                    savee(blend_selected)
                    print("press n for next blend or esc for next img ")
                    clean = False


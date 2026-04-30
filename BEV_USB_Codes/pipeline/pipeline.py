import numpy as np
import cv2 
import os 
import sys 
from collections import OrderedDict

''' 
    this script : <images seperated by comma> <translation_matrix> 
'''

IMAGE = False
VIDEO = True

def parse_images():

    image_path_list = sys.argv[1].split(',')
    images = []
    for each_path in image_path_list:
        if os.path.exists(each_path) == False:
            print("invalid image path : {}".format(each_path))
            exit(0)
        images.append(cv2.imread(each_path))
    return images

# sys.argv[2]
def parse_perspective_mat():
    if (len(sys.argv) < 3):
        print("invalid inputs. perspective mat not found")
        exit(0)
    print(sys.argv[2])
    mat_path_list = sys.argv[2].split(',')
    p_mats = []
    for each_mat_path in mat_path_list:
        if os.path.exists(each_mat_path) == False:
            print("invalid perspective tranform path : {}".format(each_mat_path))
            exit(0)
        fs = cv2.FileStorage(each_mat_path, cv2.FILE_STORAGE_READ)
        x = fs.getNode('mat').mat()
        p_mats.append(x)
        fs.release()
    return np.array(p_mats)

# sys.argv[3]
def parse_yml(no_of_images):

    if os.path.exists(sys.argv[3]) == False:
        print("invalid yml path : {}".format(sys.argv[2]))
        exit(0)
    fs = cv2.FileStorage(sys.argv[3], cv2.FILE_STORAGE_READ)
    yml_data = OrderedDict()
    
    yml_keys_list = ["_rotate_center", "_rotate_angle", "_scale", "offset"]
    yml_data['canvas_size'] = fs.getNode('CANVAS SIZE').mat()[:,0].tolist()

    for i in range(no_of_images):
        for each_key in yml_keys_list:
            k = "img{}".format(i) + each_key
            if each_key in ["_rotate_center", "offset"]:
                yml_data[k] = fs.getNode(k).mat()[:,0].tolist()
            else:
                yml_data[k] = fs.getNode(k).real()
    return yml_data

# sys.argv[4]
def parse_mask():
    masks = []
    if (len(sys.argv) == 5):
        mask_paths = sys.argv[4].split(',')

    else:
        return []

    for i in range(len(mask_paths)):
        if os.path.exists(mask_paths[i]) == False:
            print("invalid mask path : {} ".format(mask_paths[i]))
            exit(0)
        masks.append(cv2.imread(mask_paths[i]))
    print("mask len : {}".format(len(masks)))
    return masks

def translate_img (images_list, yml_data, perspective_mat):

    # print(yml_data['canvas_size'])
    canvases = np.zeros((len(images_list), yml_data['canvas_size'][1], yml_data['canvas_size'][0], 3)).astype(np.uint8)
    WIDTH, HEIGHT = yml_data['canvas_size']

    for i in range(canvases.shape[0]):
        strx = "img{}".format(i)
        # rotation 
        rotate_center = yml_data[strx + "_rotate_center"]
        rotate_angle = yml_data[strx + "_rotate_angle"]
        scale = yml_data[strx + "_scale"]

        # rotate image
        dst_img = cv2.warpPerspective(images_list[i].copy(), perspective_mat[i],( images_list[i].shape[1], images_list[i].shape[0])) 
        rot_matrix = cv2.getRotationMatrix2D(rotate_center, rotate_angle, scale)
        rotated_image = cv2.warpAffine(dst_img, rot_matrix, yml_data['canvas_size'])

        # translation 
        offset = yml_data[strx + "offset"]
        canvases[i, offset[1] : , offset[0] : , : ] = rotated_image[0: HEIGHT - offset[1], 0 : WIDTH - offset[0], :].copy()
    return canvases


if __name__ == "__main__":
    
    if IMAGE:
        image_list = parse_images()
    
    elif VIDEO:
    
        cap_0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap_1 = cv2.VideoCapture(2, cv2.CAP_V4L2)

        cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

        cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 


    mask_list = parse_mask()
    perspective_mat = parse_perspective_mat()
    if IMAGE:
        yml_data = parse_yml(len(image_list))
    elif VIDEO:
        yml_data = parse_yml(2)
    WIDTH, HEIGHT = yml_data['canvas_size']
    
    
    if IMAGE:
        translated_img = translate_img(image_list, yml_data, perspective_mat)
        yml_data = parse_yml(len(image_list))
        if len(mask_list) > 0:
            images_with_mask = np.zeros((len(image_list), yml_data['canvas_size'][1], yml_data['canvas_size'][0], 3)).astype(np.uint8)

            for i in range(images_with_mask.shape[0]):
                images_with_mask[i] = cv2.multiply(translated_img[i].astype(np.float32), mask_list[i].astype(np.float32) / 255).astype(np.int8).copy()
        else:
            images_with_mask = translated_img.copy()

        f = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)
        for i in range(images_with_mask.shape[0]):
            f = cv2.add(f, images_with_mask[i])
        cv2.imshow("final", f)
        cv2.waitKey(0)    
    
    elif VIDEO:
        while True:

            frame_list = []
            ret0, frame_0 = cap_0.read()
            ret1, frame_1 = cap_1.read()
            
            if not ret0:
                print("problem in reading frame 0")
                exit(0)
            if not ret1:
                print("problem in reading frame 0")
                exit(0)
            frame_list.append(frame_0)
            frame_list.append(frame_1)

            # translating the frames 
            translated_img = translate_img(frame_list, yml_data, perspective_mat)
            
            if len(mask_list) > 0:
                frames_with_mask = np.zeros((len(frame_list), yml_data['canvas_size'][1], yml_data['canvas_size'][0], 3)).astype(np.uint8)

                for i in range(frames_with_mask.shape[0]):
                    frames_with_mask[i] = cv2.multiply(translated_img[i].astype(np.float32), mask_list[i].astype(np.float32) / 255).astype(np.int8).copy()
            else:
                frames_with_mask = translated_img.copy()
                        
            f = np.zeros((HEIGHT, WIDTH, 3)).astype(np.uint8)
            for i in range(translated_img.shape[0]):
                f = cv2.add(f, frames_with_mask[i])
                cv2.imshow("frame {} masked".format(i), cv2.resize(translated_img[i], None, fx=0.5, fy=0.5))
            cv2.imshow("final", f)
            if (cv2.waitKey(1) == ord('q')):
                break
            




    


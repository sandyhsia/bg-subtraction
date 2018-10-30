import numpy as np
import scipy
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def create_data_matrix_from_video(clip, fps=5, scale=50):
    init = np.vstack([clip.get_frame(i / float(fps)).flatten() 
                      for i in range(fps * int(clip.duration))]).T

    scaled = np.vstack([scipy.misc.imresize(rgb2gray(clip.get_frame(i / float(fps))).astype(int), scale).flatten() 
                      for i in range(fps * int(clip.duration))]).T
    return init, scaled


def getFrameID(counter):
    
    if counter <  10:
        return "00000"+str(counter)
    
    elif counter <100:
        return "0000"+str(counter)
    
    elif counter <1000:
        return "000"+str(counter)
    
    elif counter <10000:
        return "00"+str(counter)
    
    elif counter <100000:
        return "0"+str(counter)
    
    else:
        return str(counter) 


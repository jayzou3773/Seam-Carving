from SeamCarver import SeamCarver
import os
import cv2
import numpy as np

if __name__ == '__main__':
    input_filename="before.png"
    output_filename="after.png"
    img=cv2.imread(input_filename).astype(np.float64)
    height,width=img.shape[:2]
    
    sc = SeamCarver(input_filename, height, width/2)
    sc.seam_carving()
    sc.save_result(output_filename)

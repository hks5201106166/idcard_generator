import numpy as np
import os
import cv2
import shutil
path_save='/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_570w/data_570w/'

image_dirs= os.listdir(path_save)
while 1:
    print(len(image_dirs))
    # print(len(images))
    # for image in images:
    #     im=cv2.imread(path_save+image_d+'/'+image)
    #     cv2.imshow('ttt',im)
    #     cv2.waitKey(500)
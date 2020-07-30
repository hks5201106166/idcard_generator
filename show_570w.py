import numpy as np
import os
import cv2
import shutil
path_save = '/home/ubuntu/hks/ocr/idcard_generator_project/datas/data_train_fake/train_text_lines/'

image_dirs= os.listdir(path_save)
for image_d in image_dirs:
    images=os.listdir(path_save+image_d)

    if len(images)!=40:
        print('111111111111111111111111111111111111111111111111111111111')
        shutil.rmtree(path_save+image_d)
        print(images)
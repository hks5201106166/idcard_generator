import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import csv
import shutil
path='/home/ubuntu/hks/ocr/idcard_generator_project/idcard_pix2pix/data_train/'
dirs=os.listdir(path)
labels=list(csv.reader(open('/home/ubuntu/hks/ocr/idcard_generator_project/idcard_generator/labels/train.csv')))
labels=[l[0] for l in labels]
for d in dirs:
    if d not in labels:
        shutil.rmtree(path+d)

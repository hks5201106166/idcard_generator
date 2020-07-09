#-*-coding:utf-8-*-
import os
import csv
val_images_name=list(csv.reader(open('/home/simple/mydemo/ocr_project/segment/data/idcards_train/val.csv')))
val_images=set([image[0] for image in val_images_name])
images_name=os.listdir('/home/simple/mydemo/ocr_project/segment/data/test_results/data_split_clear')
for image in images_name:
    im=image.split('_')[0]
    if im in val_images:
        os.remove('/home/simple/mydemo/ocr_project/segment/data/test_results/data_split_clear/'+image)
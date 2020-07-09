#-*-coding:utf-8-*-
import os
import cv2
import csv
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2
#这里使用的Python 3
import numpy as np
import cv2
kernel_sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])
kernel_sharpen_2 = np.array([
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]])
kernel_sharpen_3 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]]) / 8.0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
# output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
# output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)



back1=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/image_match/back1.jpg')
back1=cv2.resize(back1,dsize=(449,283))
back=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/image_match/back.jpg')
back=cv2.resize(back,dsize=(449,283))
front=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/image_match/front.jpg')
front=cv2.resize(front,dsize=(449,283))
front=cv2.filter2D(front,-1,kernel_sharpen_3)
front1=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/image_match/front1.jpg')
front1=cv2.resize(front1,dsize=(449,283))
front1=cv2.filter2D(front1,-1,kernel_sharpen_3)
def sift_kp(image):
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SURF.create()
    kp,des = sift.detectAndCompute(image,None)
    kp_image = cv2.drawKeypoints(gray_image,kp,None)
    return kp_image,kp,des
def get_good_match(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
def siftImageAlignment(img1,img2):
   _,kp1,des1 = sift_kp(img1)
   _,kp2,des2 = sift_kp(img2)
   goodMatch = get_good_match(des1,des2)
   if len(goodMatch) > 4:
       ptsA= np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
       ransacReprojThreshold = 4
       H, status =cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold);
       imgOut = cv2.warpPerspective(img2, H, (img1.shape[1],img1.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
   return imgOut,H,status
imagoutfont,H,status=siftImageAlignment(front,front1)
imagoutback,H,status=siftImageAlignment(back,back1)
cv2.imshow('imagoutfont',imagoutfont)
cv2.imshow('imageback',imagoutback)
cv2.imshow('front',front)
cv2.imshow('front1',front1)
cv2.imshow('back',back)
cv2.imshow('back1',back1)
cv2.waitKey(0)
# H, status = cv2.findHomography(ptsA,ptsB,cv2.RANSAC,ransacReprojThreshold)
#其中H为求得的单应性矩阵矩阵
#status则返回一个列表来表征匹配成功的特征点。
#ptsA,ptsB为关键点
#cv2.RANSAC, ransacReprojThreshold这两个参数与RANSAC有关

path='/home/simple/mydemo/ocr_project/idcard_generator_project/test_results/data_split'
images_name=os.listdir(path)
kernel_sharpen_1 = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]])
kernel_sharpen_2 = np.array([
    [1, 1, 1],
    [1, -7, 1],
    [1, 1, 1]])
kernel_sharpen_3 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [-1, -1, -1, -1, -1]]) / 8.0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# # 限制对比度的自适应阈值均衡化


for image_name in images_name:
    image=cv2.imread(os.path.join(path,image_name),0)
    image=cv2.resize(image,dsize=(449,283))
    flag=image_name.split('_')[-1]
    if flag=='1.jpg':
        cv2.imshow('font',image)

    else:


        output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
        output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
        output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
        # 显示锐化效果
        cv2.imshow('Original Image', image)
        cv2.imshow('sharpen_1 Image', output_1)
        cv2.imshow('sharpen_2 Image', output_2)
        cv2.imshow('sharpen_3 Image', output_3)
    cv2.waitKey(300)
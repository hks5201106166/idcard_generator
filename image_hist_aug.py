#-*-coding:utf-8-*-
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
# img = cv.imread("/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/145b818e9698dbf4e858404b235144c6163.jpg", 0)
# # -*- coding: utf-8 -*-
# """
# Created on Sat Aug 25 14:35:33 2018
# @author: Miracle
# """
#
# import cv2
# import numpy as np

# 加载图像
image = cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/0a7caef3c8324cfc902175b267009295_1.jpg')
# 自定义卷积核
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
# 卷积
output_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
# 显示锐化效果
cv2.imshow('Original Image', image)
cv2.imshow('sharpen_1 Image', output_1)
cv2.imshow('sharpen_2 Image', output_2)
cv2.imshow('sharpen_3 Image', output_3)
# 停顿
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
# cv2.imshow('hks',dst)
# cv2.imshow('img',img)
# cv2.waitKey(0)
#img = cv.resize(img, None, fx=0.5, fy=0.5)
# 创建CLAHE对象
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# # 限制对比度的自适应阈值均衡化
# dst = clahe.apply(img)
# # 使用全局直方图均衡化
# equa = cv.equalizeHist(img)
# # 分别显示原图，CLAHE，HE
# cv.imshow("img", img)
# cv.imshow("dst", dst)
# cv.imshow("equa", equa)
# cv.waitKey()
# def calcGrayHist(I):
#     # 计算灰度直方图
#     h, w = I.shape[:2]
#     grayHist = np.zeros([256], np.uint64)
#     for i in range(h):
#         for j in range(w):
#             grayHist[I[i][j]] += 1
#     return grayHist
#
# def equalHist(img):
#     # 灰度图像矩阵的高、宽
#     h, w = img.shape
#     # 第一步：计算灰度直方图
#     grayHist = calcGrayHist(img)
#     # 第二步：计算累加灰度直方图
#     zeroCumuMoment = np.zeros([256], np.uint32)
#     for p in range(256):
#         if p == 0:
#             zeroCumuMoment[p] = grayHist[0]
#         else:
#             zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
#     # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
#     outPut_q = np.zeros([256], np.uint8)
#     cofficient = 256.0 / (h * w)
#     for p in range(256):
#         q = cofficient * float(zeroCumuMoment[p]) - 1
#         if q >= 0:
#             outPut_q[p] = math.floor(q)
#         else:
#             outPut_q[p] = 0
#     # 第四步：得到直方图均衡化后的图像
#     equalHistImage = np.zeros(img.shape, np.uint8)
#     for i in range(h):
#         for j in range(w):
#             equalHistImage[i][j] = outPut_q[img[i][j]]
#     return equalHistImage
#
#
# img = cv.imread("/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/0a7caef3c8324cfc902175b267009295_1.jpg", 0)
# # 使用自己写的函数实现
# equa = equalHist(img)
# # grayHist(img, equa)
# # 使用OpenCV提供的直方图均衡化函数实现
# # equa = cv.equalizeHist(img)
# cv.imshow("img", img)
# cv.imshow("equa", equa)
# cv.waitKey(0)
# img = cv.imread("/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/0a7caef3c8324cfc902175b267009295_1.jpg", 0)
# # 图像归一化
# fi = img / 255.0
# # 伽马变换
# gamma = 0.9
# out = np.power(fi, gamma)
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
img = cv.imread("/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/145b818e9698dbf4e858404b235144c6163.jpg", 0)
# 计算原图中出现的最小灰度级和最大灰度级
# 使用函数计算
Imin, Imax = cv.minMaxLoc(img)[:2]
# 使用numpy计算
# Imax = np.max(img)
# Imin = np.min(img)
Omin, Omax = 0, 255
# 计算a和b的值
a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * img + b
out = out.astype(np.uint8)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()








# # 绘制直方图函数
# def grayHist(img):
#     h, w = img.shape[:2]
#     pixelSequence = img.reshape([h * w, ])
#     numberBins = 256
#     histogram, bins, patch = plt.hist(pixelSequence, numberBins,
#                                       facecolor='black', histtype='bar')
#     plt.xlabel("gray label")
#     plt.ylabel("number of pixels")
#     plt.axis([0, 255, 0, np.max(histogram)])
#     plt.show()
#
#
# img = cv.imread("/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/0a7caef3c8324cfc902175b267009295_1.jpg", 0)
# out = 2.0 * img
# # 进行数据截断，大于255的值截断为255
# out[out > 255] = 255
# # 数据类型转换
# out = np.around(out)
# out = out.astype(np.uint8)
# # 分别绘制处理前后的直方图
# # grayHist(img)
# # grayHist(out)
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
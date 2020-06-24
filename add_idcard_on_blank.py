#-*-coding:utf-8-*-
import cv2
import numpy as np
def rotate_bound(image, angle):
    """

    :param image: 原图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(image, M, (nW, nH))
    # perform the actual rotation and return the image
    return img
tt=['background','up_obverse','up_reverse','drown_obverse','drown_reverse']
image=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/background_正反面_.jpg')
front=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/0a7caef3c8324cfc902175b267009295_1.jpg')
back=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/0a13a2e882514ddf87790beb337c9bf6_0.jpg')


flag_front_flip=np.random.randint(0,2)
angle_front=np.random.randint(0,6)
if flag_front_flip==0:
    front = rotate_bound(front, angle_front)
    mask_front = cv2.threshold(front, 5, 1, type=cv2.THRESH_BINARY)[1]
    mask_front_label = mask_front * 1
else:
    front=cv2.flip(front,-1)
    front=rotate_bound(front,angle_front)
    mask_front=cv2.threshold(front,5,1,type=cv2.THRESH_BINARY)[1]
    mask_front_label=mask_front*2


angle_back=np.random.randint(0,6)
flag_back_flip=np.random.randint(0,2)
if flag_back_flip==0:
    back = rotate_bound(back, angle_back)
    mask_back = cv2.threshold(back, 2, 255, type=cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary=mask_back[:,:,0]
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask_back=np.uint8(np.stack([binary,binary,binary],axis=2)/255)

    mask_back_label = mask_back * 3
else:
    back=cv2.flip(back,-1)
    back=rotate_bound(back,angle_back)
    mask_back=cv2.threshold(back,2,255,type=cv2.THRESH_BINARY)[1]
    binary = mask_back[:, :, 0]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭操作
    mask_back = np.uint8(np.stack([binary, binary, binary], axis=2)/255)
    mask_back_label = mask_back * 4

# cv2.imshow('hks',back)
# cv2.imshow('mask',mask_back*60)
# cv2.waitKey(0)
mask=np.zeros(shape=(image.shape[0],image.shape[1]),dtype=np.uint8)

h,w,c=front.shape
mask_roi_font=mask[150:(150+h),250:(250+w)]
mask_roi_font=mask_roi_font+mask_front_label[:,:,0]
mask[150:(150+h),250:(250+w)]=mask_roi_font


image_roi_font=image[150:(150+h),250:(250+w),:]
image_roi_font=image_roi_font*(1-mask_front)
image_with_logo_font=cv2.addWeighted(image_roi_font,1,front,1,0)
image[150:(150+h),250:(250+w),:]=image_with_logo_font



h_back,w_back,c_back=back.shape
mask_roi_back=mask[550:(550+h_back),200:(200+w_back)]
mask_roi_back=mask_roi_back+mask_back_label[:,:,0]
mask[550:(550+h_back),200:(200+w_back)]=mask_roi_back


image_roi_back=image[550:(550+h_back),200:(200+w_back),:]

# kernel_back = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
# binary_mask=mask_back
# binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_ERODE, kernel_back)

image_roi_back=image_roi_back*(1-mask_back)
image_with_logo_back=cv2.addWeighted(image_roi_back,1,back,1,0)
image[550:(550+h_back),200:(200+w_back),:]=image_with_logo_back
# cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/blend.jpg',image)
# back=rotate_bound(back,5)
# mask_back=cv2.threshold(back,5,255,type=cv2.THRESH_BINARY)
tt=(1-mask_front)
cv2.imshow('image',image)
cv2.imshow('hks',back)
cv2.imshow('mask',mask*100)
cv2.waitKey(0)










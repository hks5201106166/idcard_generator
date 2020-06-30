#-*-coding:utf-8-*-
import os
import cv2
import numpy as np
path='/home/simple/mydemo/ocr_project/idcard_generator_project/template/'
path_gen='/home/simple/mydemo/ocr_project/idcard_generator_project/'
files=os.listdir(os.path.join(path,'fuzhiwuxiao_mask'))
#files_gen=os.listdir(os.path.join(path_gen,'generator_datas1'))
files_gen=os.listdir(os.path.join(path_gen,'generator_datas2'))
#files_gen.extend(files_gen2)
roi_mask=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/背面.jpg',0)
roi_mask[roi_mask<=150]=0
roi_mask[roi_mask>150]=255
x,y=roi_mask.shape
point_select=[]
for i in range(x):
    for j in range(y):
        if roi_mask[i,j]==255:
            point_select.append([i,j])

for ttt,file_gen in enumerate(files_gen):
    print(ttt)
    image_gen=cv2.imread(os.path.join(path_gen+'generator_datas2',file_gen),0)
    cv2.imshow('hks',image_gen)
    cv2.waitKey(0)
    files_back=file_gen.split('_')[-1]
    if files_back=='1.jpg':
        for i in range(4):
            l=len(files)
            index=np.random.randint(0,l)

            image_gen_copy=image_gen.copy()
            mask=cv2.imread(os.path.join(path+'fuzhiwuxiao_mask',files[index]),0)
            mask[mask>150]=255
            mask[mask<=150]=0
            image=cv2.imread(os.path.join(path+'fuzhiwuxiao_template',files[index]),0)
            template=cv2.bitwise_and(image, image, mask=mask)


            h_image_gen,w_image_gen=image_gen.shape
            point_x, point_y= h_image_gen,w_image_gen
            h, w = template.shape
            while point_x+h>=h_image_gen or point_y+w>=w_image_gen:

                point_x,point_y=np.random.randint(0,h_image_gen),np.random.randint(0,w_image_gen)


            rect = image_gen_copy[point_x:(h+point_x),point_y:(w+point_y)]
            rect1=cv2.bitwise_and(rect, rect, mask=mask)
            mask =cv2.bitwise_not(mask)
            rect2=cv2.bitwise_and(rect,rect,mask=mask)
            image_with_logo=cv2.addWeighted(rect1,0.3,template,0.3,0)
            add_logo=image_with_logo+rect2

            image_gen_copy_logo = image_gen_copy.copy()
            image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo
            train_data = np.hstack((image_gen_copy, image_gen_copy_logo))
            image_gen_copy[point_x:(h+point_x),point_y:(w+point_y)]=add_logo
            cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/gen_data_with_logo/'+file_gen.split('.jpg')[0]+'_'+str(i)+'.jpg',train_data)
    else:
        for i in range(4):
            l = len(files)
            index = np.random.randint(0, l)
            image_gen_copy = image_gen.copy()
            mask = cv2.imread(os.path.join(path + 'fuzhiwuxiao_mask', files[index]), 0)
            mask[mask > 150] = 255
            mask[mask <= 150] = 0
            image = cv2.imread(os.path.join(path + 'fuzhiwuxiao_template', files[index]), 0)
            template = cv2.bitwise_and(image, image, mask=mask)

            h_image_gen, w_image_gen = image_gen.shape
            point_x, point_y = h_image_gen, w_image_gen
            h, w = template.shape
            while point_x + h >= h_image_gen or point_y + w >= w_image_gen:
                l_point=len(point_select)
                l_index=np.random.randint(0,l_point)
                point_x, point_y = point_select[l_index]

            rect = image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)]
            rect1 = cv2.bitwise_and(rect, rect, mask=mask)
            mask = cv2.bitwise_not(mask)
            rect2 = cv2.bitwise_and(rect, rect, mask=mask)
            image_with_logo = cv2.addWeighted(rect1, 0.3, template, 0.3, 0)
            add_logo = image_with_logo + rect2
            image_gen_copy_logo=image_gen_copy.copy()
            image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo
            train_data = np.hstack((image_gen_copy, image_gen_copy_logo))

            # cv2.imshow('template', template)
            # cv2.imshow('logo',image_with_logo)
            # cv2.imshow('image_gen',image_gen_copy)
            # #cv2.imshow('rect',rect2)
            # # cv2.imshow('image', train_data)
            # cv2.waitKey(300000)
            cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/gen_data_with_logo/' +
                        file_gen.split('.jpg')[0] + '_'+str(i) + '.jpg', train_data)
            del image_gen_copy_logo

        #del image_gen_copy
        # cv2.imshow('template', template)
        # cv2.imshow('logo',image_with_logo)
        # cv2.imshow('image_gen',image_gen_copy)
        # #cv2.imshow('rect',rect2)
        # cv2.imshow('image', train_data)
        # cv2.waitKey(3000)



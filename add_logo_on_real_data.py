#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
import os
import cv2
import numpy as np
import cv2
import imgaug.augmenters as iaa
aug_brightness = iaa.MultiplyBrightness((0.7, 1.2))
aug_gaussian =iaa.GaussianBlur((0, 1.0))
path='/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/template/'
path_gen='/home/simple/mydemo/ocr_project/idcard_generator_project/data/idcard_pix2pix/'
files=os.listdir(os.path.join(path,'fuzhiwuxiao_mask'))
#files_gen=os.listdir(os.path.join(path_gen,'generator_datas1'))
files_gen=os.listdir(os.path.join(path_gen,'data_split_clear'))
#files_gen.extend(files_gen2)
roi_mask=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/背面.jpg',0)
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
    name=file_gen.split('.jpg')[0].split('_')[0]
    image_gen=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/data/idcard_pix2pix/data_train/'+name+'/'+file_gen,0)
    image_gen=cv2.resize(image_gen,dsize=(449,283))
    logo_mask=cv2.imread('/home/simple/mydemo/ocr_project/idcard_generator_project/data/idcard_pix2pix/data_train/'+name+'/'+'mask-'+file_gen,0)
    logo_mask=cv2.resize(logo_mask,dsize=(449,283))

    kernel=np.ones((10,10),np.uint8)
    logo_mask=cv2.erode(logo_mask,kernel=np.ones((5,5),np.uint8))
    logo_mask=cv2.dilate(logo_mask,kernel=np.ones((10,10),np.uint8))
    contours, hierarchy = cv2.findContours(logo_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ls = []
    for contour in contours:
        l = cv2.contourArea(contour)
        ls.append(l)
    index_max = np.argmax(ls)
    x,y,w,h=cv2.boundingRect(contours[index_max])
    logo_mask=np.zeros(shape=image_gen.shape,dtype=np.uint8)
    logo_mask[y:y+h,x:x+w]=255
    ind_mask_real=list(np.argwhere(logo_mask.flatten()==255)[:,0])
    image_gen[y:y+h,x:x+w]=255
   # cv2.rectangle(image_gen,(x,y),(x+w,y+h),(0,255,0),2)
    #rect = cv2.minAreaRect(contours[index_max])
    # cv2.imshow('hks',image_gen)
    # cv2.imshow('iii',cv2.imread(path_gen+'data_split_clear/'+file_gen,0))
    # cv2.imshow('mask',logo_mask)
    # cv2.waitKey(0)
    files_back=file_gen.split('_')[-1]

    if files_back=='0.jpg':
        for i in range(20):
            l=len(files)
            index=np.random.randint(0,l)

            image_gen_copy=image_gen.copy()
            mask=cv2.imread(os.path.join(path+'fuzhiwuxiao_mask',files[index]),0)
            mask[mask>150]=255
            mask[mask<=150]=0

            image=cv2.imread(os.path.join(path+'fuzhiwuxiao_template',files[index]),0).astype('float')
            image=cv2.blur(image,ksize=(5,5))
            image=image+20


            image[image>255]=255
            image=image.astype('uint8')
            template=cv2.bitwise_and(image, image, mask=mask)
            # template+=50
            cv2.imshow('tttdsggt', image)
            #cv2.waitKey(0)


            h_image_gen,w_image_gen=image_gen.shape

            h, w = template.shape
            inter=1
            while inter>0:
                point_x, point_y = h_image_gen, w_image_gen
                while point_x+h>=h_image_gen or point_y+w>=w_image_gen:
                    point_x,point_y=np.random.randint(0,h_image_gen),np.random.randint(0,w_image_gen)
                logo_mask_fake = np.zeros(shape=image_gen.shape, dtype=np.uint8)
                logo_mask_fake[point_x:point_x+h, point_y:point_y+w] = 255
                ind_mask_fake = list(np.argwhere(logo_mask_fake.flatten() == 255)[:, 0])
                inter=len(list(set(ind_mask_real)&set(ind_mask_fake)))






            rect = image_gen_copy[point_x:(h+point_x),point_y:(w+point_y)]
            rect1=cv2.bitwise_and(rect, rect, mask=mask)
            mask =cv2.bitwise_not(mask)
            rect2=cv2.bitwise_and(rect,rect,mask=mask)
            image_with_logo=cv2.addWeighted(rect1,0.3,template,0.3,0)
            #image_with_logo=cv2.blur(image_with_logo,ksize=(3,3) )
            #cv2.imshow('tttdsat',image_with_logo)
            #cv2.waitKey(0)
            add_logo=image_with_logo+rect2

            image_gen_copy_logo = image_gen_copy.copy()
            image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo
            image_gen_copy_logo = np.stack([image_gen_copy_logo, image_gen_copy_logo, image_gen_copy_logo], axis=2)
            aug_brightness_deterministic = aug_brightness.to_deterministic()
            aug_gaussian_deterministic = aug_gaussian.to_deterministic()
            image_gen_copy_logo = aug_brightness_deterministic.augment_images(images=[image_gen_copy_logo])[0]
            image_gen_copy_logo = aug_gaussian_deterministic.augment_images(images=[image_gen_copy_logo])[0]

            image_gen_copy = np.stack([image_gen_copy, image_gen_copy, image_gen_copy], axis=2)
            image_gen_copy = aug_brightness_deterministic.augment_images(images=[image_gen_copy])[0]
            image_gen_copy = aug_gaussian_deterministic.augment_images(images=[image_gen_copy])[0]

            train_data = np.hstack((image_gen_copy, image_gen_copy_logo))
            cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/data/idcard_pix2pix/data_with_fakelogo/'  + 'logozero'+str(i)+'-' + file_gen,train_data)
            # image_gen_copy[point_x:(h+point_x),point_y:(w+point_y)]=add_logo
            # cv2.imshow('tttt',train_data)
            # cv2.waitKey(100)




            # image_gen_copy=seq_nologo(image=np.stack([image_gen_copy,image_gen_copy,image_gen_copy],axis=2))


            # cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/idcard_generator/images/'+file_gen,image_gen_copy)
    else:
        for i in range(20):
            l = len(files)
            index = np.random.randint(0, l)
            image_gen_copy = image_gen.copy()
            mask = cv2.imread(os.path.join(path + 'fuzhiwuxiao_mask', files[index]), 0)
            mask[mask > 150] = 255
            mask[mask <= 150] = 0
            image = cv2.imread(os.path.join(path + 'fuzhiwuxiao_template', files[index]), 0)
            image = cv2.blur(image, ksize=(5, 5))
            image = image + 50
            template = cv2.bitwise_and(image, image, mask=mask)

            h_image_gen, w_image_gen = image_gen.shape
            point_x, point_y = h_image_gen, w_image_gen
            h, w = template.shape


            inter = 1
            nums=0
            while inter > 0:
                point_x, point_y = h_image_gen, w_image_gen
                while point_x + h >= h_image_gen or point_y + w >= w_image_gen:
                    l_point = len(point_select)
                    l_index = np.random.randint(0, l_point)
                    point_x, point_y = point_select[l_index]
                logo_mask_fake = np.zeros(shape=image_gen.shape, dtype=np.uint8)
                logo_mask_fake[point_x:point_x + h, point_y:point_y + w] = 255
                ind_mask_fake = list(np.argwhere(logo_mask_fake.flatten() == 255)[:, 0])
                inter = len(list(set(ind_mask_real) & set(ind_mask_fake)))
                nums+=1
                if nums==300:
                    break
            if nums<300:
                rect = image_gen_copy[point_x:(h + point_x), point_y:(w + point_y)]
                rect1 = cv2.bitwise_and(rect, rect, mask=mask)
                mask = cv2.bitwise_not(mask)
                rect2 = cv2.bitwise_and(rect, rect, mask=mask)
                image_with_logo = cv2.addWeighted(rect1, 0.3, template, 0.3, 0)
                add_logo = image_with_logo + rect2
                image_gen_copy_logo=image_gen_copy.copy()
                image_gen_copy_logo[point_x:(h + point_x), point_y:(w + point_y)] = add_logo


                image_gen_copy_logo = np.stack([image_gen_copy_logo, image_gen_copy_logo, image_gen_copy_logo], axis=2)
                aug_brightness_deterministic = aug_brightness.to_deterministic()
                aug_gaussian_deterministic = aug_gaussian.to_deterministic()
                image_gen_copy_logo = aug_brightness_deterministic.augment_images(images=[image_gen_copy_logo])[0]
                image_gen_copy_logo = aug_gaussian_deterministic.augment_images(images=[image_gen_copy_logo])[0]

                image_gen_copy = np.stack([image_gen_copy, image_gen_copy, image_gen_copy], axis=2)
                image_gen_copy = aug_brightness_deterministic.augment_images(images=[image_gen_copy])[0]
                image_gen_copy = aug_gaussian_deterministic.augment_images(images=[image_gen_copy])[0]

                train_data = np.hstack((image_gen_copy, image_gen_copy_logo))
                cv2.imwrite(
                    '/home/simple/mydemo/ocr_project/idcard_generator_project/data/idcard_pix2pix/data_with_fakelogo/'  + 'logoone' + str(
                        i) + '-' + file_gen, train_data)

                #train_data = np.hstack((image_gen_copy, image_gen_copy_logo))
                # cv2.imshow('template', train_data)
                # cv2.waitKey(100)
            # cv2.imshow('logo',image_with_logo)
            # cv2.imshow('image_gen',image_gen_copy)
            # #cv2.imshow('rect',rect2)
            # # cv2.imshow('image', train_data)
            # cv2.waitKey(300)
            # cv2.imwrite('/home/simple/mydemo/ocr_project/idcard_generator_project/gen_data_with_logo/' +
            #             file_gen.split('.jpg')[0] + '_'+str(i) + '.jpg', train_data)
            # del image_gen_copy_logo

        # del image_gen_copy
        # cv2.imshow('template', template)
        # cv2.imshow('logo',image_with_logo)
        # cv2.imshow('image_gen',image_gen_copy)
        # #cv2.imshow('rect',rect2)
        # cv2.imshow('image', train_data)
        # cv2.waitKey(3000)



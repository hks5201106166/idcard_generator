#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
import cv2
import numpy as np
import math
import os
import pandas as pd


def __blend_imgquadbox(image1, image2, bbox, label, image_coordinate=None, scale=1.5):
    assert label.shape[0] == bbox.shape[0]
    h_image1, w_image1, _ = image1.shape
    h_image2, w_image2, _ = image2.shape
    if h_image1 <= h_image2 or w_image1 <= w_image2:
        image1 = cv2.resize(image1, dsize=(int(w_image2 * scale), int(h_image2 * scale)))
    h_image1, w_image1, _ = image1.shape
    center_x, center_y = w_image1 / 2, h_image1 / 2
    length_x = np.int0(center_x - w_image2 / 2)
    length_y = np.int0(center_y - h_image2 / 2)
    bbox[:, :, 0] += length_x
    bbox[:, :, 1] += length_y
    if image_coordinate is not None:
        image_coordinate[:, 0] += length_x
        image_coordinate[:, 1] += length_y
        cv2.fillPoly(image1, [image_coordinate], (0, 0, 0))
    image1[length_y:length_y + h_image2, length_x:length_x + w_image2] = cv2.add(
        image1[length_y:length_y + h_image2, length_x:length_x + w_image2], image2)
    return image1, bbox, label


def _resize_imgquadbox(image, bbox, label, image_coordinate=None, dsize=(800, 600)):
    assert label.shape[0] == bbox.shape[0]
    h, w, _ = image.shape
    ratio_h, ratio_w = h / dsize[1], w / dsize[0]
    bbox[:, :4, 0] = bbox[:, :4, 0] // ratio_w
    bbox[:, :4, 1] = bbox[:, :4, 1] // ratio_h
    img = cv2.resize(image, dsize)
    if image_coordinate is None:
        return img, bbox
    image_coordinate[:, 0] = image_coordinate[:, 0] // ratio_w
    image_coordinate[:, 1] = image_coordinate[:, 1] // ratio_h
    return img, bbox, label, image_coordinate


def _rotate_imgquadbox(image, bbox, label, angle):
    assert label.shape[0] == bbox.shape[0]
    h, w, _ = image.shape
    cX, cY = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    M[0, 2] += (nw / 2) - cX
    M[1, 2] += (nh / 2) - cY

    image_x1 = np.array([0, w, w, 0])
    image_y1 = np.array([h, h, 0, 0])  # h - [0, 0, h, h]
    x1 = bbox[:, :4, 0]
    y1 = h - bbox[:, :4, 1]
    x2 = cX
    y2 = h - cY

    # 由于图片坐标y值与欧几里得坐标正好相反，所以这里sin值默认取相反数（对比论文里的公式自己动手算一下就明白了）
    image_x = (image_x1 - x2) * math.cos(math.pi / 180.0 * -angle) - (image_y1 - y2) * math.sin(
        math.pi / 180.0 * -angle) + x2
    image_y = (image_x1 - x2) * math.sin(math.pi / 180.0 * -angle) + (image_y1 - y2) * math.cos(
        math.pi / 180.0 * -angle) + y2
    x = (x1 - x2) * math.cos(math.pi / 180.0 * -angle) - (y1 - y2) * math.sin(math.pi / 180.0 * -angle) + x2
    y = (x1 - x2) * math.sin(math.pi / 180.0 * -angle) + (y1 - y2) * math.cos(math.pi / 180.0 * -angle) + y2
    image_x = np.int0(image_x + (nw - w) / 2).flatten()
    image_y = np.int0(h - image_y + (nh - h) / 2).flatten()
    x = np.int0(x + (nw - w) / 2).flatten()
    y = np.int0(h - y + (nh - h) / 2).flatten()

    image_coordinate = np.array([[x, y] for x, y in zip(image_x, image_y)]).reshape(-1, 2)
    bbox = np.array([[x, y] for x, y in zip(x, y)]).reshape(bbox.shape)
    image = cv2.warpAffine(image, M, (nw, nh))
    return image, bbox, label, image_coordinate


def blend2img(img1, img2, bbox, label, angle):
    """
    :param img1: dst img
    :param img2: roi img
    :param bbox: quadbox in roi img(4 point represent one rectangle box in roi img)
    :param label: each label math one quadbox
    :param angle: clockwise rotate angle
    :return:
    """
    if isinstance(img1, np.ndarray):
        img1 = img1
    elif isinstance(img1, str):
        img1 = cv2.imread(img1)
    else:
        raise Exception("img must be ndarray or str(img path)", img1)

    if isinstance(img2, np.ndarray):
        img2 = img2
    elif isinstance(img2, str):
        img2 = cv2.imread(img2)
    else:
        raise Exception("img must be ndarray or str(img path)", img2)
    assert bbox.ndim == 3
    assert bbox.shape[0] == label.shape[0]
    assert label.shape[1] == 1
    image, bbox, label = __blend_imgquadbox(img1, *_resize_imgquadbox(*_rotate_imgquadbox(img2, bbox, label, angle)))
    return image, bbox, label


if __name__ == '__main__':

    imglist = os.listdir("zzzmydata/bbbox")
    expresslist = os.listdir("zzzmydata/express")
    orgincsvdata = pd.read_csv("zzzmydata/annotation.csv")

    xml_list = []
    for img in imglist:
        img1 = cv2.imread("zzzmydata/bbbox/{}".format(img))
        for ep in expresslist:
            img2 = cv2.imread("zzzmydata/express/{}".format(ep))
            bbox = np.array(orgincsvdata[orgincsvdata['filename'] == ep].loc[:,
                            ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']].values)
            newbbox = bbox.reshape(bbox.shape[0], -1, 2)
            label = np.array(orgincsvdata[orgincsvdata['filename'] == ep].loc[:, ['class']].values)
            for angle in [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]:
                lastimage, lastbbox, lastlabel = blend2img(img1.copy(), img2.copy(), newbbox.copy(), label.copy(),
                                                           angle)
                resultbbox = lastbbox.reshape(bbox.shape[0], 8)
                for i in range(lastlabel.shape[0]):
                    value = ("{}_{}_{}".format(angle, ep.split('.')[0], img), resultbbox[i][0], resultbbox[i][1],
                             resultbbox[i][2], resultbbox[i][3], resultbbox[i][4], resultbbox[i][5], resultbbox[i][6],
                             resultbbox[i][7], lastlabel[i][0], angle)
                    xml_list.append(value)
                cv2.imwrite("newbarcodedata/train/{}_{}_{}".format(angle, ep.split('.')[0], img), lastimage)

    column_name = ['filename', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'class', 'angle']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_df.to_csv(("newbarcodedata/train_ann.csv"), index=None)
    print('Successfully make mydataset.')
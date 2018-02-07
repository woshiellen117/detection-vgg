import cv2
import xlrd
import xlwt
from numpy import array
import sys
import argparse
import skimage.io as io
import tensorflow as tf


def load_photo(coll,num):
    h,w = coll[num].shape[:2]
    a,c = 0,0
    for i in range(100,w-1):
        if coll[num][300,i+1]>250:
            a=i
            break
    if a == 0:
        a = w
    for i in range(100,h - 1):
        if coll[num][i, 300] > 250:
            c = i
            break
    if c == 0:
        c = h
    crop_img = coll[num][0: c, 0: a]
    # image = cv2.resize(crop_img,(IMAGE_SIZE,IMAGE_SIZE),interpolation=cv2.INTER_CUBIC)
    # image = image.reshape(IMAGE_SIZE*IMAGE_SIZE)
    # image = (image-PIXEL_DEPTH/2)/PIXEL_DEPTH
    # image=image.reshape(IMAGE_SIZE,IMAGE_SIZE)
    # print(num)
    # cv2.namedWindow('image', 0)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # print(a)
    # print(c)
    return crop_img


def load_label(defect):
    labels = []
    for i in range(len(defect)):
        if defect[i] == '':
            labels.append(1)
            #无效图像
        else:
            labels.append(0)
    labels = array(labels)
    return labels


def main(_):
    # file_path = "/Volumes/TOSHIBA EXT/final/try"
    file_path = "D:\\final\\thumb"
    out_file_name = "D:\\final\\useful_data\\"
    png = file_path + '/*.png'
    # xml_path = "/Volumes/TOSHIBA EXT/final/final.xlsx"
    xml_path = "D:\\final\\final.xlsx"
    labels_xml = xlrd.open_workbook(xml_path)
    labels_table = labels_xml.sheets()[0]
    defect = labels_table.col_values(3)[1:]
    labels = load_label(defect)
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet("sheet1")
    coll = io.ImageCollection(png)
    for n in range(19):
        worksheet.write(0, n, labels_table.row_values(0)[n])
    m=1
    for i in range(16950):
        print("i=",i,"m=",m,"n=",n)
        if labels[i]!=1:
            image=load_photo(coll, i)
            cv2.imwrite(out_file_name+str(i)+".png", image)
            for n in range(19):
                worksheet.write(m, n, labels_table.row_values(i+1)[n])
            m = m+1
    workbook.save('D:\\final\\useful_data\\useful_data.xlsx')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
#coding=utf-8
#__author__ = 'lg 2018-8-9'

import copy
import cv2
import os
import numpy as np
import time
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


PRINT = False    #True
      
def main(input_imagespath,output_image,output_orig_image):
    #check path
    if not os.path.exists(input_imagespath):
        print ("input_imagespath is not exist!!!")
        return 
    if not os.path.exists(output_image):
        os.makedirs(output_image)

    files= os.listdir(input_imagespath)
    
    time_all = 0
    group_num = 0
    for file in files:
        if file.endswith('.jpg'):
            continue
        else:
            input_imagespath_new = os.path.join(input_imagespath,file)
            files_new = os.listdir(input_imagespath_new)
            score_lst_all = []
            i=0
            num = len(files_new)

            if PRINT:
                print "group=", file
                
            savePath_lst_all = []
            for file_new in files_new:
                if file_new.endswith('.jpg'):
                    print (i,"/", num)
                    imgPath = os.path.join(input_imagespath_new,file_new)
                    output_image_new = os.path.join(output_image,file)             
                    '''if not os.path.exists(output_image_new):
                        os.makedirs(output_image_new)'''
                    #savePath = os.path.join(output_image_new,file_new)
                    savePath = os.path.join(output_image,file_new)                       
                    
                    start = time.time()
                    image = cv2.imread(imgPath)
                    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #imageVar = cv2.Laplacian(img2gray, cv2.CV_64F, ksize=7).var()
                    
                    x = cv2.Sobel(img2gray,cv2.CV_16S,1,0)
                    y = cv2.Sobel(img2gray,cv2.CV_16S,0,1)
                    absX = cv2.convertScaleAbs(x)   # 转回uint8
                    absY = cv2.convertScaleAbs(y)
                    imageVar = cv2.addWeighted(absX,0.5,absY,0.5,0).var()
                    end = time.time()
                    time_all = time_all + (end-start)
                    group_num = group_num + 1

                    print imageVar
                    font= cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image, str(round(imageVar,2)) + "       " + file + "_" + str(i), (50,50), font, 2,(0,0,255),2)

                    ss = file+ "_" + str(i)+".jpg"
                    cv2.imwrite(os.path.join(output_image,ss), image)
                    i = i + 1
                   
    print time_all,group_num,time_all/group_num
if __name__ == '__main__':
    input_imagespath = '/opt/ligang/Detectron/test_xml/test10'
    replot_input_imagespath = '/opt/ligang/Detectron/test_xml/test10' #img_replot
    output_image     = '/opt/ligang/Detectron/test_xml/out10'
    output_orig_image     = '/opt/ligang/Detectron/test_xml/out_orig'
    main(input_imagespath,output_image,output_orig_image)
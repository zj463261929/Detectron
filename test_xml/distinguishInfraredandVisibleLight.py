#coding=utf-8
import os
import random
from os import path
import shutil
import cv2


orig_img_dir = "/opt/ligang/Detectron/test_xml/errorimg/" 
save_infrared_dir = "/opt/ligang/Detectron/test_xml/img11/" 
save_light_dir = "/opt/ligang/Detectron/test_xml/img2/" 
if not os.path.exists(save_infrared_dir):
    os.mkdir(save_infrared_dir)
if not os.path.exists(save_light_dir):
    os.mkdir(save_light_dir)
	
files = [x for x in os.listdir(orig_img_dir) if path.isfile(orig_img_dir+os.sep+x) and x.endswith('.jpg')]

num_light = 0
num_infrared = 0
for i in xrange(len(files)): 
    file=files[i]
    basename = os.path.splitext(file)[0]
    s = orig_img_dir + basename+".jpg"
    if os.path.exists(s):  
        img = cv2.imread(s)
        if img is not None:
            if img.shape[0]> 1000: 
                shutil.copy(orig_img_dir+basename+".jpg" , save_light_dir+basename+".jpg" )
                num_light = num_light + 1
            else:
                shutil.copy(orig_img_dir+basename+".jpg" , save_infrared_dir+basename+".jpg" )
                num_infrared = num_infrared +1
	else:
		print basename
		
print len(files), num_infrared,num_light
	
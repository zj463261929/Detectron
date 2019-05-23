#coding=utf-8
import os
import random
from os import path
import shutil

orig_xml_dir = "/opt/ligang/Detectron/test_xml/xml1/" 
orig_img_dir = "/opt/ligang/Detectron/test_xml/img1/" 
orig_img_dir_temp = "/opt/ligang/Detectron/test_xml/out1/" 
save_xml_dir = "/opt/ligang/Detectron/test_xml/xml_filter/" 
save_img_dir = "/opt/ligang/Detectron/test_xml/img_filter/" 
if not os.path.exists(save_xml_dir):
	os.mkdir(save_xml_dir)
if not os.path.exists(save_img_dir):
	os.mkdir(save_img_dir)
	
files = [x for x in os.listdir(orig_img_dir_temp) if path.isfile(orig_img_dir_temp+os.sep+x) and x.endswith('.jpg')]

lst = []
for i in xrange(len(files)): 
	file=files[i]
	basename = os.path.splitext(file)[0]
	if os.path.exists(orig_xml_dir + basename+".xml"):  #os.path.exists(orig_img_dir + basename+".jpg"):
		shutil.copy(orig_xml_dir+basename+".xml" , save_xml_dir+basename+".xml" )
		shutil.copy(orig_img_dir+basename+".jpg" , save_img_dir+basename+".jpg" )
		lst.append(basename)
	else:
		print basename
		
print len(lst),len(list(set(lst)))
	
#coding=utf-8
import os
import random
from os import path
import shutil

isOnlyStatisticsName = True #True  	#True:已知图片文件夹，生成txt；
								#False:已知txt，生成train.txt、test.txt。
if 1:#isOnlyStatisticsName: 
	trainval = open("/opt/zhangjing/Detectron/test/other.txt",'w+')
	#test = open("/opt/zhangjing/caffe/caffe/data/actions_new/benchmark_test1.txt",'w+')
	interval = 10
	orig_image_dir= "/opt/zhangjing/Detectron/test/aug/" #"/opt/zhangjing/caffe/data/actions_new/benchmark/pictures_1080_ratio_new_new_ratio_ratio/"
	files = [x for x in os.listdir(orig_image_dir) if path.isfile(orig_image_dir+os.sep+x) and x.endswith('.jpg')]
	random.shuffle(files)
	#print files
	print len(files),len(list(set(files)))  

	lst = []
	for i in xrange(len(files)): 
		file=files[i]
		#print file     
		basename = os.path.splitext(file)[0]
		if 1:#os.path.exists("/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/Annotations_benchmark_night150/" + basename+".xml"): 
			result = basename	# + ".jpg" + " " + basename + ".xml"
			lst.append(result)
			trainval.write(result + "\n")
		else:
			print basename

	print len(lst),len(list(set(lst)))

elif 0:
	train_num = 0
	test_num = 0
	folder_path = "/opt/zhangjing/caffe/caffe/data/illbuild/"
	trainval = open("/opt/zhangjing/caffe/caffe/data/illbuild/train.txt",'w+')
	test = open("/opt/zhangjing/caffe/caffe/data/illbuild/val.txt",'w+')
	
	with open(folder_path + "all.txt", 'r') as ann_file:
		lines = ann_file.readlines()
		random.shuffle(lines)
		
		print len(lines),len(list(set(lines)))

		for i in xrange(len(lines)):
			l = lines[i]
			lst = l.strip().split()
			if len(lst)>1:
				if lst[0].endswith('.jpg') and lst[1].endswith('.xml'):
					if os.path.exists(folder_path + lst[0]) and os.path.exists(folder_path + lst[1]):
						if i% 10 < 8:
							trainval.write(l)
							train_num = train_num + 1
						else:
							test.write(l)
							test_num = test_num + 1
					else:
						print ("not exist: ", lst)
	print ("train num: ", train_num)
	print ("test num: ", test_num)
						
	trainval.close()
	test.close()
			
else:
	'''trainval = open("/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/img_aug.txt",'w+')
	#test = open("/opt/zhangjing/caffe/caffe/data/actions_new/benchmark_test1.txt",'w+')
	interval = 10
	orig_img_dir = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/img_aug/"
	orig_xml_dir = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/xml_aug/" 
	
	save_img_dir = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/img_aug_res/"
	save_xml_dir = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/xml_aug_res/" 
	if not os.path.exists(save_xml_dir):
		os.mkdir(save_xml_dir)
	if not os.path.exists(save_img_dir):
		os.mkdir(save_img_dir)
	
	files = [x for x in os.listdir(orig_img_dir) if path.isfile(orig_img_dir+os.sep+x) and x.endswith('.jpg')]
	#random.shuffle(files)
	#print files
	print len(files),len(list(set(files)))  

	ann_file = open("/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt", 'r')
	lines = ann_file.readlines()
	lines_new = []
	for l in lines:
		lst = l.strip().split()
		lines_new.append(lst[0])
	ann_file.close()	
		
	lst = []
	for i in xrange(len(files)): 
		file=files[i]
		#print file     
		basename = os.path.splitext(file)[0]
		if os.path.exists(orig_xml_dir + basename+".xml"): 
			result = basename	# + ".jpg" + " " + basename + ".xml"
			#
			lst1 = basename.strip().split("_")
			lst2 = lst1[:3]
			s = "_".join(lst2)
			#20180921_1C1B0D228AF1_00042_contrastAndBright0 20180921_1C1B0D228AF1_00042
			print basename, s, len(s), len(lines_new[0]),type(s), type(lines_new[0])
			if s in lines_new:
				lst.append(basename)
			
				trainval.write(basename + "\n")
				
				shutil.copy(orig_xml_dir+basename+".xml" , save_xml_dir+basename+".xml" )
				shutil.copy(orig_img_dir+basename+".jpg" , save_img_dir+basename+".jpg" )
		else:
			print basename

	print len(lst),len(list(set(lst)))'''
	
	trainval = open("/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt",'w+')
	ann_file1 = open("/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets/Main/train1.txt", 'r')
	lines1 = ann_file1.readlines()

	ann_file2 = open("/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/aug/img_aug.txt", 'r')
	lines2 = ann_file2.readlines()
	
	lines = lines1 + lines2
	random.shuffle(lines)
	for i in xrange(len(lines)): 
		file=lines[i]
		basename = os.path.splitext(file)[0]
		trainval.write(basename)




#coding=utf-8
#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2	# NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time
import datetime
import xlwt

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from core.config import merge_cfg_from_list
from core.test_engine import run_inference
import utils.c2
import utils.logging

utils.c2.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def parse_args():
	parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
	parser.add_argument(
		'--cfg',
		dest='cfg_file',
		help='optional config file',
		default=None,
		type=str
	)
	parser.add_argument(
		'--wait',
		dest='wait',
		help='wait until net file exists',
		default=True,
		type=bool
	)
	parser.add_argument(
		'--vis', dest='vis', help='visualize detections', action='store_true'
	)
	parser.add_argument(
		'--multi-gpu-testing',
		dest='multi_gpu_testing',
		help='using cfg.NUM_GPUS for inference',
		action='store_true'
	)
	parser.add_argument(
		'--range',
		dest='range',
		help='start (inclusive) and end (exclusive) indices',
		default=None,
		type=int,
		nargs=2
	)
	parser.add_argument(
		'opts',
		help='See lib/core/config.py for all options',
		default=None,
		nargs=argparse.REMAINDER
	)

	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)
	return parser.parse_args()

'''
test_result: 根据测试结果打印的信息，统计每个类别的AP、耗时、AP、AP50、AP75、APs、APm、APl
input：
	modelName：模型名称，如model_iter24999.pkl
	logPath：测试结果写入的文件名称，比如"log/oil/test.txt"
	dataType：字符串，表示该数据是训练集还是验证集(train 或 val)
'''
def test_result(logPath,classNum,fw):

	totalTime = 0
	modelLst = []
	dataNameLst = []
	classAPLst_temp = []
	classAPLst = []
	APLst = []
	AP50Lst= []
	AP75Lst= []
	APsLst= []
	APmLst= []
	APlLst= []
	recallLst = []
	#需保存的数据是：模型、每个类别的AP、耗时、AP、AP50、AP75、APs、APm、APl
	with open(logPath, 'r') as ann_file:
		lines = ann_file.readlines()
		for ll in lines:
			l = ll.strip()
			#total time
			'''f l.startswith("INFO test_engine.py: 162: Total inference time:"):
				lst = l.strip().split(":")
				totalTime = lst[-1]'''
			#存放模型序号
			if l.startswith("INFO net.py:  57: Loading weights from: "):
				lst = l.strip().split(":")
				l2 = lst[-1]
				lst2 = l2.strip().split("/")
				l3 = lst2[-1]
				l4 = l3[10:len(l3)-4]
				modelLst.append( l4 )
			
			#存放数据集名称
			if l.startswith("INFO test_engine.py: 322: Wrote detections to:"):
				lst = l.strip().split(":")
				l2 = lst[-1]
				lst2 = l2.strip().split("/")
				l3 = lst2[-3]
				dataNameLst.append( str(l3) )#'\t' tab
			
			#每个类别的AP
			if l.startswith("INFO json_dataset_evaluator.py: 231: "):
				lst = l.strip().split(":")
				ll = lst[-1]
				#print (ll)
				classAPLst_temp.append(ll)

			#recall
			if l.startswith("INFO cocoeval.py: 472:  Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				recallLst.append(ll)
				
			#AP
			if l.startswith("INFO cocoeval.py: 472:  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				APLst.append(ll)
			
			#ap50
			if l.startswith("INFO cocoeval.py: 472:  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				AP50Lst.append(ll)
				
			#AP75
			if l.startswith("INFO cocoeval.py: 472:  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				AP75Lst.append(ll)
				
			#APs
			if l.startswith("INFO cocoeval.py: 472:  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				APsLst.append(ll)
				
			#APm
			if l.startswith("INFO cocoeval.py: 472:  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				APmLst.append(ll)
				
			#APl
			if l.startswith("INFO cocoeval.py: 472:  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = "):
				lst = l.strip().split(" = ")
				ll = lst[-1]
				APlLst.append(ll)
	
	l = len(classAPLst_temp)

	if l%classNum!=0:
		fw.write("class num error!\n")  
		return
	
	lst = list(set(dataNameLst))
	n = len(lst)
	for i in range(int(l/classNum)):
		classAPLst.append(classAPLst_temp[i*classNum:(i+1)*classNum])
	''' 
	print (len(modelLst))
	#print (modelLst)
	print (len(dataNameLst))
	print (len(classAPLst))
	print (len(APLst))
	print (len(AP50Lst))
	#print (APLst)
	#print (AP50Lst)
	print (len(AP75Lst))
	print (len(APsLst))
	print (len(APmLst))
	print (len(APlLst))
	print (len(recallLst))'''#

	return modelLst,dataNameLst,classAPLst,recallLst,APLst,AP50Lst,AP75Lst,APsLst,APmLst,APlLst


def process_result(model_lst_all,dataset_lst_all,classAP_lst_all,recall_lst_all,ap_lst_all,ap50_lst_all,ap75_lst_all,aps_lst_all,apm_lst_all,apl_lst_all):
	
	if not ( len(model_lst_all)==len(dataset_lst_all)==len(recall_lst_all)==len(classAP_lst_all)==len(ap_lst_all)==len(ap50_lst_all)==len(ap75_lst_all)==len(aps_lst_all)==len(apm_lst_all)==len(apl_lst_all)): 
		fw.write("data error!")
		return
	
	for i in range(len(ap_lst_all)):
		fw.write("\n\n")
		fw.write("model_iter: " + model_lst_all[i]+"\n") #[,]
		fw.write("datasets: " + dataset_lst_all[i]+"\n") #[,]
		fw.write("recall: " + recall_lst_all[i]+"\n")    #[,]
		fw.write("class AP: " + " ".join(classAP_lst_all[i])+"\n") #[[],[]]
		s = ap_lst_all[i] + " " + ap50_lst_all[i] + " " +ap75_lst_all[i]+ " " +aps_lst_all[i]+ " " +apm_lst_all[i]+ " " +apl_lst_all[i]
		fw.write("AP,AP50,AP75,APs,APm,APl: " + s+"\n") #[[],[]]


def write_execl(modelNumLst,model_lst_all,dataset_lst_all,recall_lst_all,ap_lst_all,ap50_lst_all,datasetName):

	model_lst_all_temp = list(set(model_lst_all))

	modelNum = len(model_lst_all_temp)
	#print (len(modelNumLst),modelNum)
	if len(modelNumLst)!=modelNum:
		sheet.write(0, 0, 'model num error!')
		return
	row = 2 + modelNum

	dataset_lst_all_temp = list(set(dataset_lst_all))
	datasetNum = len(dataset_lst_all_temp)
	#print (len(datasetName),datasetNum)
	
	if len(datasetName)!=datasetNum:
		sheet.write_merge(0, 1, 'dataset num error!')
		return
	col = 1 + datasetNum*3
	
	style = xlwt.XFStyle() # 创建一个样式对象，初始化样式
	style0 = xlwt.XFStyle() # 创建一个样式对象，初始化样式
	al = xlwt.Alignment() 
	al.horz = 0x02      # 设置水平居中
	al.vert = 0x01      # 设置垂直居中
	font = xlwt.Font()
	font.bold = True    #字体加粗
    
	style.alignment = al
	style0.alignment = al
	style0.font = font

	#写表头
	sheet.write_merge(0, 1, 0, 0, 'iter num',style0) #row1,row2,col1,col2
	for i in range(modelNum):
		sheet.write(i+2, 0, modelNumLst[i],style0) 
	for i in range(datasetNum):	
		sheet.write_merge(0, 0, i*3+1,(i+1)*3-1+1, datasetName[i],style0) 
		sheet.write(1, i*3+0+1, "ap",style0) 
		sheet.write(1, i*3+1+1, "ap50",style0) 
		sheet.write(1, i*3+2+1, "recall",style0) 
		
	#写数据
	for j in range(modelNum):
		for i in range(datasetNum):	
			t = j*datasetNum + i
			sheet.write(j+2, i*3+0+1, float(ap_lst_all[t]),style) 
			sheet.write(j+2, i*3+1+1, float(ap50_lst_all[t]),style) 
			sheet.write(j+2, i*3+2+1, float(str(recall_lst_all[t])),style) 
					
if __name__ == '__main__':
	isTxt= False #True
	workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
	args = parse_args()
	logName = "testAP.log"
	if not isTxt:
		logger = utils.logging.setup_logging(logName)
		logger.info('Called with args:')
		logger.info(args)#'''

	if args.cfg_file is not None:
		merge_cfg_from_file(args.cfg_file)
	if args.opts is not None:
		merge_cfg_from_list(args.opts)
	assert_and_infer_cfg()
	if not isTxt:
		logger.info('Testing with config:')
		logger.info(pprint.pformat(cfg))

		while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
			logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
			print('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
			time.sleep(10)#''''''
	
	l = cfg.TEST.WEIGHTS
	lst = l.strip().split("/")
	modelPath = "/".join(lst[:len(lst)-1]) + "/"

	print ( modelPath )
	files = [x for x in os.listdir(modelPath) if os.path.isfile(modelPath+x) and x.startswith('model_iter')]
	modelNumLst = []
	for l in files:
		ll = l[10:len(l)-4]
		modelNumLst.append( int(ll) )
	modelNumLst.sort()
	
	#cfg.TEST.DATASETS
	l = str(cfg.TEST.DATASETS)   #('coco_2007_trainval_infrared','coco_2007_val_infrared','coco_2007_benchmark_230','coco_2007_benchmark_608',) 
	l2 = l.strip()
	l3 = l2.lstrip('(')
	l4 = l3.rstrip(')')
	datasetName1 = l4.split(",")  #print (datasetName[0], type(str(datasetName[0])))
	datasetName = []
	for i in range(len(datasetName1)):
		l = datasetName1[i]
		if len(l)>1:
			l2 = l.strip(" ' ")
			datasetName.append(l2)
	
	for ll in modelNumLst:
		l = "model_iter" + str(ll) + ".pkl"
		if not isTxt:
			run_inference(
				modelPath+l,
				ind_range=args.range,
				multi_gpu_testing=args.multi_gpu_testing,
				check_expected_results=True,
				)#''''''
	
	model_lst_all = []
	dataset_lst_all = []
	recall_lst_all = []
	classAP_lst_all = [] #每个类的
	ap_lst_all = []  # AP,AP50,AP75,APs,APm,APl
	ap50_lst_all = []
	ap75_lst_all = []
	aps_lst_all = []
	apm_lst_all = []
	apl_lst_all = []
    
	today = datetime.date.today()
	testResult_Path = str( today )+ "_result.txt"
	fw = open(testResult_Path, 'a')
	classNum = cfg.MODEL.NUM_CLASSES-1
	model_lst_all,dataset_lst_all,classAP_lst_all,recall_lst_all,ap_lst_all,ap50_lst_all,ap75_lst_all,aps_lst_all,apm_lst_all,apl_lst_all= test_result(logName,classNum,fw)
	'''print ('\n')
	print (model_lst_all)
	print (dataset_lst_all)
	print (recall_lst_all)
	print ("classAP: ", classAP_lst_all)
	print ("ap: ", ap_lst_all)
	print ('\n') #'''
	
		
	process_result(model_lst_all,dataset_lst_all,classAP_lst_all,recall_lst_all,ap_lst_all,ap50_lst_all,ap75_lst_all,aps_lst_all,apm_lst_all,apl_lst_all)
	
	fw.close()
	workbook = xlwt.Workbook() 
	today = datetime.date.today()
	testResult_Path = str( today )+ "_result.xls"  
	sheet = workbook.add_sheet(str(today))
	write_execl(modelNumLst,model_lst_all,dataset_lst_all,recall_lst_all,ap_lst_all,ap50_lst_all,datasetName)
	
	#保存文件
	workbook.save(testResult_Path)

	#os.remove( "testzj.log" )
		
		

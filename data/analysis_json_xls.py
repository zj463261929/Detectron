#coding=utf-8
import os
import random
from os import path
import json
import numpy as np
import datetime
import xlwt
from xlwt.Style import *

class_name_lst = ["suv", "forklift", "digger", "car", "bus", "tanker", "person", "minitruck", "minibus", "truckbig", "trucksmall", "tricycle", "bicycle"]  #改
train_json = "/gds/train-data/oilstealing/oil_vehicle_person_10cls/train_65892.json"  #改
val_json = "/gds/train-data/oilstealing/oil_vehicle_person_10cls/val_65892.json"    #改


def getbboxInfo(jsonPath):
    with open(jsonPath, "r") as f:
        ann_data = json.load(f)
        images = ann_data["images"]
        print ("images num = ", len(images))
        annotations = ann_data["annotations"]
        all_num = len(annotations)+0.0
        print ("all box num = ", len(annotations))
        
        def get_bboxArea(annotations, class_index):
            lst = []
            for ann in annotations:
                image_id = ann["image_id"]		#"i
                bbox = ann["bbox"]
                area = bbox[3]*bbox[2]
                category_id = ann["category_id"]
                if class_index == category_id:
                    lst.append( area )
                '''if area<=30*30:
                    print image_id,bbox[0],bbox[1],bbox[2],bbox[3]'''
            return lst		
        
        num = 0    
        class_lst = []
        for n in range(len(class_name_lst)):
            area_lst = get_bboxArea(annotations, n+1)
            print (class_name_lst[n], len(area_lst), round(len(area_lst)/all_num,3)*100 )#, len(area_lst)/(len(annotations)+0.0) )	
            
            lst_temp = []
            num = 0
            for area in area_lst:
                if area>30*30:
                    num = num + 1
            print ("area >30*30 num=", num)
            lst_temp.append(num) 
            
            num = 0
            for area in area_lst:
                if area>40*40:
                    num = num + 1
            print ("area >40*40 num=", num)
            lst_temp.append(num) 
            
            num = 0
            for area in area_lst:
                if area>50*50:
                    num = num + 1
            print ("area >50*50 num=", num)
            lst_temp.append(num) 
            
            num = 0
            for area in area_lst:
                if area>60*60:
                    num = num + 1
            print ("area >60*60 num=", num)
            lst_temp.append(num) 
            lst_temp.append(len(area_lst))
            lst_temp.append(round(len(area_lst)/all_num,3)*100)
            
            class_lst.append(lst_temp)
    return len(images), len(annotations), class_lst

def write_execl(class_name_lst,trainImgNum, trainBBoxNum, trainBBoxAreaLst,valImgNum, valBBoxNum, valBBoxAreaLst,sheet):

    if len(class_name_lst)!=len(trainBBoxAreaLst)!=len(valBBoxAreaLst):
        sheet.write(0, 0, 'class num error!')
        return

    style = xlwt.XFStyle() # 创建一个样式对象，初始化样式, #居中
    style0 = xlwt.XFStyle() # 创建一个样式对象，初始化样式, #居中、加粗
    style1 = xlwt.XFStyle() # 创建一个样式对象，初始化样式, #左对齐、加粗
    style2 = xlwt.XFStyle() # 创建一个样式对象，初始化样式, #左对齐
    
    al = xlwt.Alignment()
    al0 = xlwt.Alignment()
    al.horz = 0x02      # 设置水平居中
    al.vert = 0x01      # 设置垂直居中
    
    al0.horz = xlwt.Alignment.HORZ_LEFT      # 设置水平居中xlwt.Alignment.HORZ_LEFT  Alignment.HORZ_CENTER
    al0.vert = 0x01      # 设置垂直居中 Alignment.VERT_CENTER
    
    font = xlwt.Font()
    font.bold = True    #字体加粗
    
    style.alignment = al  #居中
    style0.alignment = al  
    style0.font = font
    style1.alignment = al0  
    style1.font = font
    style2.alignment = al0  

    #写表头
    sheet.write_merge(0, 1, 0, 0, 'index',style0) #row1,row2,col1,col2
    sheet.write_merge(0, 1, 1, 1, 'class',style0) #row1,row2,col1,col2
    
    s = "train("+str(trainImgNum)+"image,"+str(trainBBoxNum) + "bbox)"
    sheet.write_merge(0, 0, 2, 7, s,style0) #row1,row2,col1,col2
    s = "val("+str(valImgNum)+"image,"+str(valBBoxNum) + "bbox)"
    sheet.write_merge(0, 0, 8, 13, s,style0) #row1,row2,col1,col2
    
    lst = ['>30^2',	'>40^2','>50^2','>60^2','all','ratio']
    n = len(lst)
    for j in range(2):
        for i in range(n):
            sheet.write(1, j*n+i+2, lst[i],style1)
    
    classNum=len(class_name_lst)  
    for j in range(classNum):
        sheet.write(2+j, 0, j+1,style0) 
        sheet.write(2+j, 1, class_name_lst[j],style1) 
        
    #写训练集的数据
    for j in range(classNum):
        for i in range(n):
            sheet.write(2+j, i+2, trainBBoxAreaLst[j][i],style2)  

    #写验证集的数据   
    for j in range(classNum):
        for i in range(n):
            sheet.write(2+j, n+i+2, valBBoxAreaLst[j][i],style2) 
    
    
trainImgNum, trainBBoxNum, trainBBoxAreaLst = getbboxInfo(train_json)
valImgNum, valBBoxNum, valBBoxAreaLst = getbboxInfo(val_json)
today = datetime.date.today()
workbook = xlwt.Workbook() 
today = datetime.date.today()
testResult_Path = str( today )+ "_bboxArea.xls"  
sheet = workbook.add_sheet(str(today))    
write_execl(class_name_lst,trainImgNum, trainBBoxNum, trainBBoxAreaLst,valImgNum, valBBoxNum, valBBoxAreaLst,sheet)

#保存文件
workbook.save(testResult_Path)
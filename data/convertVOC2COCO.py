#coding=utf-8

import os
import xml.etree.ElementTree as ET
import xmltodict
import json
from xml.dom import minidom
from collections import OrderedDict

#存放xml的根目录文件夹
#rootDir = "./OBJECT_DETECTION/Annotations"      
#rootDir = '/opt/yushan/data/VOCdevkit/VOC2007/Annotations'
rootDir='/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/Annotations' 
save_image_dir="/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages/" #pictures
save_xml_dir='/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/Annotations/' #Annotations

#处理XML文件的索引的txt文件
#trainFile = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/benchmark_night150.txt"
trainFile = '/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt'

#输出json文件的路径
outjson = "/opt/zhangjing/Detectron/data/oil_vehicle_person_10cls/train_76638.json"

#attrDict = {"images":[{"file_name":[],"height":[], "width":[],"id":[]}], "type":"instances", "annotations":[], "categories":[]}

attrDict = dict()

#检测的类型都是instances
attrDict["type"] = "instances"

#检测对象的类别，VOC数据一共有20类，不包括背景，从1开始。
'''attrDict["categories"]=[
    {"supercategory":"none","id":1,"name":"aeroplane"},
    {"supercategory":"none","id":2,"name":"bicycle"},
    {"supercategory":"none","id":3,"name":"bird"},
    {"supercategory":"none","id":4,"name":"boat"},
    {"supercategory":"none","id":5,"name":"bottle"},
    {"supercategory":"none","id":6,"name":"bus"},
    {"supercategory":"none","id":7,"name":"car"},
    {"supercategory":"none","id":8,"name":"cat"},
    {"supercategory":"none","id":9,"name":"chair"},
    {"supercategory":"none","id":10,"name":"cow"},
    {"supercategory":"none","id":11,"name":"diningtable"},
    {"supercategory":"none","id":12,"name":"dog"},
    {"supercategory":"none","id":13,"name":"horse"},
    {"supercategory":"none","id":14,"name":"motorbike"},
    {"supercategory":"none","id":15,"name":"person"},
    {"supercategory":"none","id":16,"name":"pottedplant"},
    {"supercategory":"none","id":17,"name":"sheep"},
    {"supercategory":"none","id":18,"name":"sofa"},
    {"supercategory":"none","id":19,"name":"train"},
    {"supercategory":"none","id":20,"name":"tvmonitor"}]'''

    
#新检测对象的类别，一共有4类，不包括背景，id从1开始。
attrDict["categories"]=[
    {"supercategory":"none","id":1,"name":"suv"},
    {"supercategory":"none","id":2,"name":"forklift"},
    {"supercategory":"none","id":3,"name":"digger"},
    {"supercategory":"none","id":4,"name":"car"},
    {"supercategory":"none","id":5,"name":"bus"},
    {"supercategory":"none","id":6,"name":"tanker"},
    {"supercategory":"none","id":7,"name":"person"},
    {"supercategory":"none","id":8,"name":"minitruck"},
    {"supercategory":"none","id":9,"name":"minibus"},
    {"supercategory":"none","id":10,"name":"truckbig"},
    {"supercategory":"none","id":11,"name":"trucksmall"},
    {"supercategory":"none","id":12,"name":"tricycle"},
    {"supercategory":"none","id":13,"name":"bicycle"}]
'''    
attrDict["categories"]=[
    {"supercategory":"none","id":1,"name":"person"},
    {"supercategory":"none","id":2,"name":"car"}]'''


#class_name_lst = ["autotruck", "forklift", "digger", "car", "bus", "tanker", "person", "minitruck", "minibus"]     
images = list()
annotations = list()

def protectXY(x,w):
    x_new = x
    if x<1:
        x_new=1
    elif x>w-1:
        x_new=w-1
    return x_new

#遍历所有的categories，存储一个目标相关的annotation信息
def createAnnoLst(obj, img_id_temp, id1,w,h):
    bValidObject=False
    for value in attrDict["categories"]:
        annotation = dict() 

        #obj名称和某个category的名称相同，则存储
        if str(obj['name']) == value["name"]: 
            annotation["iscrowd"] = 0
            annotation["image_id"] = img_id_temp
            
            #print int(obj["bndbox"]["xmin"]),int(obj["bndbox"]["ymin"]),int(obj["bndbox"]["xmax"]),int(obj["bndbox"]["ymax"])
            x1 = int(obj["bndbox"]["xmin"]) - 1
            y1 = int(obj["bndbox"]["ymin"]) - 1
            x1 = protectXY(x1,w)
            y1 = protectXY(y1,h)
            
            x2_temp = int(obj["bndbox"]["xmax"])
            y2_temp = int(obj["bndbox"]["ymax"])
            x2_temp = protectXY(x2_temp,w)
            y2_temp = protectXY(y2_temp,h)
            
            #print x1,y1,x2_temp,y2_temp
            x2 = x2_temp - x1
            y2 = y2_temp - y1
            
            annotation["bbox"] = [x1, y1, x2, y2]
            annotation["area"] = int(x2 * y2)              
            annotation["segmentation"] = [[x1,y1, x1, y1+y2, x1+x2,y1+y2, x1+x2,y1]]
            annotation["category_id"] = value["id"] #类别id

            annotation["ignore"] = 0
            annotation["id"] = id1 
            id1 +=1
            
            annotations.append(annotation)
            bValidObject=True
        
    return id1,bValidObject
    
def generateVOC2Json(rootDir,testXMLFiles): 
    id1 = 1
    for root, dirs, files in os.walk(rootDir):
        for file in testXMLFiles:
            if file in files:
                #拼接xml文件的全路径
                annotation_path = os.path.abspath(os.path.join(root, file))
                #print(annotation_path)
                
                #tree = ET.parse(annotation_path)#.getroot()
                image = dict()
                
                #将xml文件转换成字典结构
                doc = xmltodict.parse(open(annotation_path).read())
                im_id1=file.split('.xml')[0]
                #print doc['annotation']['filename']
                image['file_name'] = str(im_id1 + ".jpg") #str(doc['annotation']['filename'])                   
                image['height'] = int(doc['annotation']['size']['height'])                 
                image['width'] = int(doc['annotation']['size']['width']) 
                #image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
                image['id'] = str(im_id1)
                
                
                #id1 = 1
                if 'object' in doc['annotation']:                     
                    img_id_temp = im_id1 #str(doc['annotation']['filename']).split('.jpg')[0]                         
                    #如果XML文件中有多个目标，则doc['annotation']['object']类型是列表
                    #如果XML文件中只有一个目标，则doc['annotation']['object']类型是字典
                    HasValidObject=False
                    if type(doc['annotation']['object']) == type(list()):
                        for obj in doc['annotation']['object']:                              
                            id1,ValidObject = createAnnoLst(obj,  img_id_temp, id1,image['width'],image['height'])
                            if ValidObject:
                                HasValidObject=True                                
                    else:
                        obj = doc['annotation']['object']
                        id1,HasValidObject = createAnnoLst(obj, img_id_temp, id1,image['width'],image['height'])

                    if HasValidObject:
                        images.append(image) 
                    else:
                        print "invalid object:"+img_id_temp
                else:
                    print "not object:"+im_id1
                    
    attrDict["images"] = images       
    attrDict["annotations"] = annotations
    
    #attrDict中保存了全部的信息
    jsonString = json.dumps(attrDict)
    with open(outjson, "w") as f:
        f.write(jsonString)
    
    print("img num:{}".format(len(images)))
    print("annotations num:{}".format(len(annotations)))
    
trainXMLFiles = list()
with open(trainFile, "rb") as f:
    for line in f:
        fileName = line.strip()
        
        '''if os.path.exists(save_image_dir+fileName+".jpg") and os.path.exists(save_image_dir+fileName+"_gray.jpg") and os.path.exists(save_xml_dir+fileName+".xml") and os.path.exists(save_xml_dir+fileName+"_gray.xml"):
            trainXMLFiles.append(fileName + ".xml")
            trainXMLFiles.append(fileName + "_gray.xml")
        elif os.path.exists(save_image_dir+fileName+".jpg") and os.path.exists(save_xml_dir+fileName+".xml"):
            trainXMLFiles.append(fileName + ".xml")'''

        if os.path.exists(save_image_dir+fileName+".jpg") and os.path.exists(save_xml_dir+fileName+".xml"):
            trainXMLFiles.append(fileName + ".xml")
        else:
            #print(fileName)
            print (save_image_dir+fileName+".jpg")
			

generateVOC2Json(rootDir, trainXMLFiles)

#coding=utf-8
#__author__ = 'zj 2019-5-5'

import copy
import cv2
import os
import numpy as np
import time

PRINT = False #True
class Model():
    
    def __init__(self,iou_th,ave_th,score_th):
        self.iou_th = iou_th
        self.ave_th = ave_th
        self.score_th = score_th
        
    def filterBBox(self,img_all,cls_lst_all,score_lst_all,xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all,iou_th,ave_th,score_th):
        if 1==len(img_all):
            static_lst_res1 = []
            for i in range(len(cls_lst_all)):
                lst = []
                for j in range( len(cls_lst_all[i]) ):
                    lst.append(0)
                static_lst_res1.append(lst)
            return static_lst_res1
   
        x1_lst = []
        y1_lst = []
        x2_lst = []
        y2_lst = []
        s_lst = []
        c_lst = []
        imgIndex_lst = []   #存放每个bbox属于哪个图片的
        for i in range(len(score_lst_all)): #image num
            for j in range(len(score_lst_all[i])): #bbox num
                x1_lst.append(xmin_lst_all[i][j])
                y1_lst.append(ymin_lst_all[i][j])
                x2_lst.append(xmax_lst_all[i][j])
                y2_lst.append(ymax_lst_all[i][j])
                s_lst.append(float(score_lst_all[i][j]))
                c_lst.append(cls_lst_all[i][j])
                imgIndex_lst.append(i)

        rows = len(x1_lst)
        iou_arr = np.zeros((rows,rows),dtype=int)
        iou_arr_temp = np.zeros((rows,rows),dtype=float)
        
        #计算相邻图像的差的绝对值
        img_all_abs = []
        if len(img_all)>1:
            for i in range(1,len(img_all)):
                img = cv2.absdiff(img_all[i-1],img_all[i])
                img_all_abs.append(img)

        lst_group = []
        for row in range(rows): # all bbox num
            for col in range(rows): 
                if row<col:  # top 
                    xmin1 = x1_lst[row]
                    ymin1 = y1_lst[row]
                    xmax1 = x2_lst[row]
                    ymax1 = y2_lst[row]
                    
                    xmin2 = x1_lst[col]
                    ymin2 = y1_lst[col]
                    xmax2 = x2_lst[col]
                    ymax2 = y2_lst[col]
                    o = self.calcIOU(xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2)
                    iou_arr_temp[row][col] = o
                    if o>iou_th and (imgIndex_lst[row]!=imgIndex_lst[col]) and (c_lst[row]==c_lst[col]):   ########### 0.9
                        iou_arr[row][col] = 1
                        lst_group.append([row,col])
                        
        iou_arr_index = np.zeros((rows,1),dtype=int)
        ##################################
        if PRINT:
            print "score_th=",score_th
            print "iou:"
            print iou_arr_temp                
            print "iou > ", iou_th,":"
            print iou_arr                
            print "lst_group: ", lst_group

        ################################################
        for row in range(rows): # all bbox num
            for col in range(rows):  
                if row<col:  # top 
                    if 1==iou_arr[row][col] and (0==iou_arr_index[row][0] or 0==iou_arr_index[col][0]):
                        iou_arr_index[row][0] = row+1
                        iou_arr_index[col][0] = row+1
                        for i in range(rows):
                            if 1==iou_arr[i][col] or 1==iou_arr[row][i] or 1==iou_arr[i][row] or 1==iou_arr[col][i]:
                                iou_arr_index[i][0] =  row+1
        if PRINT:                           
            print "iou_arr_index:" ,iou_arr_index
        #################################################
        mv_arr_index = np.zeros((rows,1),dtype=int)
        for i in range(rows):
            if s_lst[i]>score_th:
                continue
            
            if iou_arr_index[i][0]<1:  #处理时序图像中没有交集的bbox的索引
                bboxOK_index = i
                x1 = x1_lst[bboxOK_index]
                y1 = y1_lst[bboxOK_index]
                x2 = x2_lst[bboxOK_index]
                y2 = y2_lst[bboxOK_index]
                img = img_all[0][y1:y2,x1:x2]
                
                v = self.calcROIMaxAve(img_all_abs, x1,y1,x2,y2)
                if v<ave_th:   #移除该误检
                    mv_arr_index[i] = 1
                    
                if PRINT:
                    print "IOU<",iou_th,":"  
                    print "ave=", v, ave_th
                     
        if PRINT:                
            print "v<th and iou<th:", mv_arr_index
        
        #处理IOU>th的bbox
        lst_temp = []
        for ii in range(rows):
            if iou_arr_index[ii]>0:
                lst_temp.append(iou_arr_index[ii][0])
        lst_temp = list(set(lst_temp))
        lst_temp.sort()  #
        
        for j in range(len(lst_temp)):
            lst = []                #存放IOU>th的索引
            for ii in range(rows):
                if lst_temp[j]==iou_arr_index[ii][0]:
                    lst.append(ii)
            if PRINT:
                print lst_temp[j], ":", lst   #4:[0,2,3,4](索引)
                    
            if len(lst)<2:
                continue
            else:
                x11 = x1_lst[lst[0]]
                y11 = y1_lst[lst[0]]
                x12 = x2_lst[lst[0]]
                y12 = y2_lst[lst[0]]
                for iii in range(1,len(lst)):
                    x21 = x1_lst[lst[iii]]
                    y21 = y1_lst[lst[iii]]
                    x22 = x2_lst[lst[iii]]
                    y22 = y2_lst[lst[iii]]
                    x11,y11,x12,y12 = self.calcUnion(x11,y11,x12,y12,x21,y21,x22,y22)
                    
                if PRINT:
                    print "IOU>",iou_th,":"
                
                v = self.calcROIMaxAve(img_all_abs, x11,y11,x12,y12)
                if PRINT:            
                    print "ave=", v,ave_th
          
                if v<ave_th:   #移除该误检
                    isOK = 0
                    for j1 in range(len(lst)):
                        if s_lst[lst[j1]]>score_th:
                            isOK=1
                            continue
                            
                    if not isOK: 
                        for j1 in range(len(lst)):  #一组bbox只要有一个的置信度>th，就不过滤
                                mv_arr_index[lst[j1]] = 1

        if PRINT:            
            print "mv:",mv_arr_index
        
        '''x1_lst_res = []
        y1_lst_res = []
        x2_lst_res = []
        y2_lst_res = []
        s_lst_res = []
        c_lst_res = []'''
        static_lst_res = []
        n = 0
        for i in range(len(score_lst_all)): #image num ####################
            '''x1_lst1 = []
            y1_lst1 = []
            x2_lst1 = []
            y2_lst1 = []
            s_lst1 = []
            c_lst1 = []'''
            static_lst = []
            for j in range(len(score_lst_all[i])): #bbox num
                '''if 1!=mv_arr_index[n]:
                    x1_lst1.append(xmin_lst_all[i][j])
                    y1_lst1.append(ymin_lst_all[i][j])
                    x2_lst1.append(xmax_lst_all[i][j])
                    y2_lst1.append(ymax_lst_all[i][j])
                    s_lst1.append(score_lst_all[i][j])
                    c_lst1.append(cls_lst_all[i][j])'''
                static_lst.append( int(mv_arr_index[n]) )
                n = n + 1
            '''x1_lst_res.append(x1_lst1)
            y1_lst_res.append(y1_lst1)
            x2_lst_res.append(x2_lst1)
            y2_lst_res.append(y2_lst1)
            s_lst_res.append(s_lst1)
            c_lst_res.append(c_lst1)'''
            static_lst_res.append(static_lst)
            
        return static_lst_res #c_lst_res,s_lst_res,x1_lst_res,y1_lst_res,x2_lst_res,y2_lst_res 

    def calcIOU(self,xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2): #计算IOU
        xx1 = np.maximum(xmin1, xmin2)
        yy1 = np.maximum(ymin1, ymin2)
        xx2 = np.minimum(xmax1, xmax2)
        yy2 = np.minimum(ymax1, ymax2)
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h        
        o = inter / ((xmax1-xmin1+1)*(ymax1-ymin1+1) + (xmax2-xmin2+1)*(ymax2-ymin2+1)- inter)
        return o
    
    def calcUnion(self,xmin1,ymin1,xmax1,ymax1,xmin2,ymin2,xmax2,ymax2):  #计算2个矩形的外接矩形
        xx1 = np.minimum(xmin1, xmin2)
        yy1 = np.minimum(ymin1, ymin2)
        xx2 = np.maximum(xmax1, xmax2)
        yy2 = np.maximum(ymax1, ymax2)
        return xx1,yy1,xx2,yy2
    
    def calcROIMaxAve(self,img_all_abs, x1,y1,x2,y2): #计算时序相邻图像差绝对值的均值的最大值
        w = x2-x1+1
        h = y2-y1+1
        n = len(img_all_abs)
        if n<1:
            return None
        else:
            lst = []
            for i in range(n):
                rImg = img_all_abs[i][y1:y2,x1:x2]
                s = np.sum(rImg)
                ave0 = s/(w*h)
                lst.append(ave0)
            if PRINT:
                print lst
                
            return max(lst)
    
    def judgeImgSizeIsOK(self,img_lst): #判断一组图片大小是否一致
        num = len(img_lst)
        w = 0
        h = 0
        isOK = True
        if num > 0:
            w = img_lst[0].shape[1]
            h = img_lst[0].shape[0]
            
        for i in range(1,num):
            w_temp = img_lst[i].shape[1]
            h_temp = img_lst[i].shape[0]
            if w_temp!=w or h_temp!=h:
                isOK = False
                
        return isOK
        
    def drawResult(self, img,cls_lst,score_lst,xmin_lst,ymin_lst,xmax_lst,ymax_lst,color=0,pixel=2):
        for i in range(len(score_lst)):
            xmin = xmin_lst[i]
            ymin = ymin_lst[i]
            xmax = xmax_lst[i]
            ymax = ymax_lst[i]
            strname = cls_lst[i]
            strscore = score_lst[i]
            font= cv2.FONT_HERSHEY_SIMPLEX
            
            if 0==color:
                cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,0,255))
                #cv2.putText(img, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin), font, 1,(0,0,255),2)
                cv2.putText(img, strname + str(strscore) + '('+str(xmin)+','+ str(ymin)+','+str(xmax)+','+str(ymax)+')', (xmin,ymin), font, 0.5,(0,0,255),1)
            else:
                cv2.rectangle(img, (xmin-pixel,ymin-pixel), (xmax+pixel,ymax+pixel),(0,255,0))
                #cv2.putText(img, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin), font, 1,(0,255,0),2)
                cv2.putText(img, strname+str(strscore) + '('+str(xmin)+','+ str(ymin)+','+str(xmax)+','+str(ymax)+')', (xmin,ymin), font, 0.5,(0,255,0),1)
            #cv2.imwrite(savePath, img)
        return img
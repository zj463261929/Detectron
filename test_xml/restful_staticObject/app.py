#coding=utf-8
#__author__ = 'zj 2019-5-5'

import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from model import Model
from flask import request,Response
from scipy import misc
import json
import urllib,urllib2
import cv2
import os
import time
from model import Model
app = Flask(__name__)

@app.route('/user', methods=['POST'])
def info():
    # logger add
    
    formatter = logging.Formatter("[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/oilsteal_seq.log', maxBytes=2000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    ip = request.remote_addr
    info_str = 'IP:' + ip
    logger.info(info_str)
    #info_str = 'model_path:' + cfg_path + '-' + weights_path
    #logger.info(info_str)
    #imagepath = request.form.getlist('data')
    #imagepath = request.form.get("data",type=str,default=None)
    start_time = time.time()
    js = request.get_json()
    #return json init
    out_json = {"data":[]}
    
    #js is not dict return []
    info_str = ''
    imgPath_all = []
    img_all = []
    cls_lst_all = []
    score_lst_all = []
    xmin_lst_all = []
    ymin_lst_all = []
    xmax_lst_all = []
    ymax_lst_all = []
    ave_th_input = 10
    iou_th_input = 0.7
    score_th_input = 1.0

    if isinstance(js,dict) and js.has_key('data') :
        all_lst_temp = js.get('data',None)

        for j in range(len(all_lst_temp)):
            all_lst1 = all_lst_temp[j]
            keys_lst = all_lst1.keys()
            for index_key in range( len(keys_lst) ):
                if keys_lst[index_key]=="imgPath":
                    imgPath_all.append( all_lst1["imgPath"] )
                if keys_lst[index_key]=="cls":
                    cls_lst_all.append( all_lst1["cls"] )
                if keys_lst[index_key]=="score":
                    score_lst_all.append( all_lst1["score"] )
                if keys_lst[index_key]=="xmin":
                    xmin_lst_all.append( all_lst1["xmin"] )
                if keys_lst[index_key]=="ymin":
                    ymin_lst_all.append( all_lst1["ymin"] )
                if keys_lst[index_key]=="xmax":
                    xmax_lst_all.append( all_lst1["xmax"] )
                if keys_lst[index_key]=="ymax":
                    ymax_lst_all.append( all_lst1["ymax"] )
          
        all_lst_temp1 = js.get('param',None)
        keys_lst1 = all_lst_temp1.keys()
        for index_key in range(len(keys_lst1)):
            if keys_lst1[index_key]=="scoreth":
                score_th_input = all_lst_temp1["scoreth"]
            if keys_lst1[index_key]=="aveth":
                ave_th_input = all_lst_temp1["aveth"]
            if keys_lst1[index_key]=="iouth":
                iou_th_input = all_lst_temp1["iouth"] 

        info_str = 'images path:' + ','.join(imgPath_all)
        logger.info(info_str)

        info_str = 'iou_ave_score(th):' + str(iou_th_input)+"、"+str(ave_th_input)+"、"+str(score_th_input)
        logger.info(info_str)

    else:
        logger.warning('post has not data or data-key!!!')
        end_time = time.time() - start_time
        logger.info('process time:{}'.format(end_time))
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
        
    for i in range(len(imgPath_all)):   
        imagepath = imgPath_all[i]
        if (imagepath.startswith("https://") or imagepath.startswith("http://") or imagepath.startswith("file://")):
            imagefile = urllib.urlopen(imagepath)
            status=imagefile.code
            # url
            if(status==200): 
                image_data = imagefile.read()
                image_name = os.path.basename(imagepath)
                #new_imagepath = filepath+"/"+image_name
                new_imagepath = image_name
                with open(new_imagepath, 'wb') as code:
                    code.write(image_data)
                #img_np = misc.imread(new_imagepath)
                img_np = cv2.imread(new_imagepath,1)  #read image by cv2 ,the same as /tool/test_net.py
                if img_np is None:
                    logger.warning('the images is NONE!!!')
                    end_time = time.time() - start_time
                    logger.info('filterBBox time:{}'.format(end_time))
                    logger.removeHandler(handler)
                    handler.close()
                    return json.dumps(out_json)
            else:
                logger.warning('the image is not download on internet!!!')
                end_time = time.time() - start_time
                logger.info('filterBBox time:{}'.format(end_time))
                logger.removeHandler(handler)
                handler.close()
                return json.dumps(out_json)
        # path 
        else:
            if not os.path.exists(imagepath):
                logger.warning('the image is not exists!!!')
                end_time = time.time() - start_time
                logger.info('filterBBox time:{}'.format(end_time))
                logger.removeHandler(handler)
                handler.close()
                return json.dumps(out_json)
            else:
                #img_np = misc.imread(imagepath)
                start_readimage_time = time.time()
                img_np = cv2.imread(imagepath,1)  #read image by cv2 ,the same as /tool/test_net.py
                end_readimage_time = time.time() - start_readimage_time
                logger.info('readimage time:{}'.format(end_readimage_time))
                if img_np is None:
                    logger.warning('the images is NONE!!!')
                    end_time = time.time() - start_time
                    logger.info('filterBBox time:{}'.format(end_time))
                    
                    logger.removeHandler(handler)
                    handler.close()
                    return json.dumps(out_json)
                    
        img_all.append( img_np )
        
    if mm.judgeImgSizeIsOK(img_all) is False:
        logger.info('Image size is error!')
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
        
    #判断输入的信息是否匹配        
    if not (len(img_all)==len(cls_lst_all)==len(score_lst_all)==len(xmin_lst_all)==len(ymin_lst_all)==len(xmax_lst_all)==len(ymax_lst_all)):
        lst1 = [str(len(img_all)),str(len(cls_lst_all)),str(len(score_lst_all)),str(len(xmin_lst_all)),str(len(ymin_lst_all)),str(len(xmax_lst_all)),str(len(ymax_lst_all))]
        
        logger.info('len(img),len(cls),len(score),len(xmin),len(ymin),len(xmax),len(ymax):{}'.format(",".join(lst1)))
        logger.removeHandler(handler)
        handler.close()
        return json.dumps(out_json)
        
    for i in range(len(cls_lst_all)):  
        if not (len(cls_lst_all[i])==len(score_lst_all[i])==len(xmin_lst_all[i])==len(ymin_lst_all[i])==len(xmax_lst_all[i])==len(ymax_lst_all[i])):
            lst1 = [str(len(cls_lst_all[i])),str(len(score_lst_all[i])),str(len(xmin_lst_all[i])),str(len(ymin_lst_all[i])),str(len(xmax_lst_all[i])),str(len(ymax_lst_all[i]))]
            s = ','.join(lst1)
            logger.info('imageIndex,len(cls),len(score),len(xmin),len(ymin),len(xmax),len(ymax):{}'.format(','.join(lst1)))
            logger.removeHandler(handler)
            handler.close()
            return json.dumps(out_json)

    static_lst_res = mm.filterBBox(img_all,cls_lst_all,score_lst_all,xmin_lst_all,ymin_lst_all,xmax_lst_all,ymax_lst_all,iou_th_input,ave_th_input,score_th_input) #1:静止，0：动

    #打包成json输出
    data_lst = []
    for img_index in range(len(imgPath_all)):
        singleImgInfo_lst = {"imgPath":imgPath_all[img_index],"cls":cls_lst_all[img_index],"statics":static_lst_res[img_index],"score":score_lst_all[img_index],"xmin":xmin_lst_all[img_index],"ymin":ymin_lst_all[img_index],"xmax":xmax_lst_all[img_index],"ymax":ymax_lst_all[img_index]}
        data_lst.append(singleImgInfo_lst)   
    
    if len(static_lst_res) > 0:
        logger.info('the images filterBBox completed!!!')
        lst_temp = []
        for i in range(len(static_lst_res)):
            lst = []
            s = ""
            for j in range(len(static_lst_res[i])):
                lst.append( str(static_lst_res[i][j]) )
            s = "[" + ",".join(lst) + "]"
            lst_temp.append(s)
        ss = "[" + ",".join(lst_temp) + "]"
        logger.info("static index:{}".format(ss))
        out_json = {"data":data_lst}
    else:
        logger.warning('the images has not right bbox!!!')
    end_time = time.time() - start_time
    logger.info('filterBBox time:{}'.format(end_time))
    logger.removeHandler(handler)
    handler.close()
    return json.dumps(out_json)
    
if __name__ == '__main__':

    if not os.path.exists('./log'):
        os.makedirs('./log')
    
    logger = logging.getLogger('oilsteal_seq')    #set root level , default is WRAINING
    logger.setLevel(logging.DEBUG)
    
    '''
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s - %(module)s - %(funcName)s:%(lineno)d} - %(message)s")
    handler = RotatingFileHandler('./log/oilsteal.log', maxBytes=10000000, backupCount=10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)     #ok  start root log
    #app.logger.addHandler(handler)  #ok  start private log
    '''
    iou_th=0.7
    ave_th=35
    score_th=1.0

    mm = Model(iou_th,ave_th,score_th)

    #app.run(host="0.0.0.0",port=8080,debug=False)   #threaded=True
    app.run(host="0.0.0.0",port=8080,debug=False)
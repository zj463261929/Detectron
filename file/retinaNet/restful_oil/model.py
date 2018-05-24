#coding=utf-8

import numpy as np
import cv2 
import json
from caffe2.python import workspace

from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import core.test_engine as infer_engine

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# Restful服务，可参考tools/ infer_simple.py来写调用过程。

class Model():
    
    def __init__(self,cfg_path,weights_path):
        self.gpu_id = 0
        self.thresh = 0.3
        merge_cfg_from_file(cfg_path)
        self.model = infer_engine.initialize_model_from_cfg(weights_path,self.gpu_id)
        self.dummy_coco_dataset = dummy_datasets.get_steal_oil_class8_dataset()	#类别名称
        print ("model is ok")

    def predict(self,im):
        out_json = {"data":[]}
        
        with c2_utils.NamedCudaScope(self.gpu_id):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, im, None, None
            )
            
        #get box classes
        if isinstance(cls_boxes, list):
            boxes, segms, keypoints, classes = self.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.thresh:
            return json.dumps(out_json)
        #get score
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        #class_str_list = []
        data_list = []
        for i in sorted_inds:
            
            bbox = boxes[i, :4]
            score = boxes[i, -1]
            if score < self.thresh:
                continue
            #get class-str
            class_str = self.get_class_string(classes[i], score, self.dummy_coco_dataset)
            #class_str_list.append(class_str)
            
            single_data = {"class":class_str,"score":float('%.2f' % score),"box":{"xmin":int(bbox[0]),"ymin":int(bbox[1]),"xmax":int(bbox[2]),"ymax":int(bbox[3])}}
            data_list.append(single_data)
        #construcrion - json
        out_json["data"] = data_list
        
        return json.dumps(out_json)
        
    def convert_from_cls_format(self,cls_boxes, cls_segms, cls_keyps):
        """Convert from the class boxes/segms/keyps format generated by the testing
        code.
        """
        box_list = [b for b in cls_boxes if len(b) > 0]
        if len(box_list) > 0:
            boxes = np.concatenate(box_list)
        else:
            boxes = None
        if cls_segms is not None:
            segms = [s for slist in cls_segms for s in slist]
        else:
            segms = None
        if cls_keyps is not None:
            keyps = [k for klist in cls_keyps for k in klist]
        else:
            keyps = None
        classes = []
        for j in range(len(cls_boxes)):
            classes += [j] * len(cls_boxes[j])
        return boxes, segms, keyps, classes
        
    def get_class_string(self,class_index, score, dataset):
        class_text = dataset.classes[class_index] if dataset is not None else \
            'id{:d}'.format(class_index)
        #return class_text + ' {:0.2f}'.format(score).lstrip('0')
        return class_text
    
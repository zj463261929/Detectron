# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os


# Path to data dir
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },   
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/coco_stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },    
    'voc_2007_val': {
        IM_DIR:
            '/gds/train-data/oilstealing/data/illbuild/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/data/illbuild/voc_illbuild_val_new.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/data/illbuild/VOCdevkit2007/'
    },     
    'voc_2007_train_new': {
        IM_DIR:
            '/opt/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_aug',
        ANN_FN:
            '/opt/oil_vehicle_person_10cls/voc_oil_train_color_and_gray_new.json',
        DEVKIT_DIR:
            '/opt/oil_vehicle_person_10cls/VOCdevkit2007/'
    }, 
    'voc_2007_train_infrared': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/train_75673.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
	'coco_2007_val_infrared': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/val_75673.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
	'coco_2007_trainval_infrared': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/trainval_75673.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'voc_2007_train1': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/img_all_tower_brickspile',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/train_tower_brickspile.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_trainval1': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/img_all_tower_brickspile',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/train_tower_brickspile.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_val1': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/img_all_tower_brickspile',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/val_tower_brickspile.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
	'voc_2007_train': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/train_104832.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },	
    'coco_2007_val': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/val_color_clear.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_trainval': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/trainval_color_clear.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
	'coco_2007_benchmark_50': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages_benchmark',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/benchmark_50.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'coco_2007_benchmark_day_269': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_benchmark_c14_day_269',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/benchmark_day_269.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_benchmark_day200': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_benchmark_day200',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/benchmark_day200.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_benchmark_night150': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_benchmark_night150',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/benchmark_night150.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_benchmark_night_129': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_benchmark_c14_night_129',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/benchmark_night_129.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2015_benchmark_night_new': {
        IM_DIR:
            '/opt/Detectron/data/JPEGImages_benchmark_night_new',
        ANN_FN:
            '/opt/Detectron/data/benchmark_night_new.json'       
    }, 
    'coco_2015_benchmark100_oil': {
        IM_DIR:
            '/opt/Detectron/data/JPEGImages_benchmark_new',
        ANN_FN:
            '/opt/Detectron/data/benchmark_new.json'       
    },
    'coco_2015_benchmark100_oil_gray': {
        IM_DIR:
            '/opt/Detectron/data/JPEGImages_benchmark_new_gray',
        ANN_FN:
            '/opt/Detectron/data/benchmark_new.json'       
    },
    'coco_2015_yushan_cls12': {
        IM_DIR:
            '/gds/train-data/oilstealing/data/illbuild_class12/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/data/illbuild_class12/voc_illbuild_train.json'       
    }
    ,
	'coco_2007_benchmark_50': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages_benchmark',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/benchmark_50.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },	
    'coco_2007_move_infrared': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages_move',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/test_1176.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
	'coco_2007_benchmark_608': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages_benchmark_608',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/benchmark_608.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'coco_2007_benchmark_230': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007/VOC2007/JPEGImages_benchmark_230',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/benchmark_230.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'voc_2007_train_infrared_separate': {
        IM_DIR:
            '/gds/train-data/oilstealing/sequence_oilinfrared/data/oil_infrared_all/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/sequence_oilinfrared/data/oil_infrared_all/train_separate.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'coco_2007_trainval_infrared_separate': {
        IM_DIR:
            '/gds/train-data/oilstealing/sequence_oilinfrared/data/oil_infrared_all/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/sequence_oilinfrared/data/oil_infrared_all/trainval_separate.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'coco_2007_val_infrared_separate': {
        IM_DIR:
            '/gds/train-data/oilstealing/sequence_oilinfrared/data/oil_infrared_all/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/sequence_oilinfrared/data/oil_infrared_all/val_separate.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls_infrared/VOCdevkit2007'
    },
    'voc_2007_train_1cls': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/train_77607_1cls.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },	
    'coco_2007_trainval_1cls': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/trainval_77607_1cls.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },
    'coco_2007_val_1cls': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/val_77607_1cls.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    },    
    'coco_2007_train_14cls_crop': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_crop',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/train_crop_8972.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    }, 
    'coco_2007_val_14cls_crop': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_crop',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/val_crop_996.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    }
    ,    
    'voc_2007_train_jinjiang': {
        IM_DIR:
            '/gds/train-data/hainan/jinjiang/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/hainan/jinjiang/train_100238.json',
        DEVKIT_DIR:
            '/gds/train-data/hainan/jinjiang/VOCdevkit2007'
    },    
    'coco_2007_trainval_jinjiang': {
        IM_DIR:
            '/gds/train-data/hainan/jinjiang/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/hainan/jinjiang/trainval_100238.json',
        DEVKIT_DIR:
            '/gds/train-data/hainan/jinjiang/VOCdevkit2007'
    },    
    'coco_2007_val_jinjiang': {
        IM_DIR:
            '/gds/train-data/hainan/jinjiang/VOCdevkit2007/VOC2007/JPEGImages',
        ANN_FN:
            '/gds/train-data/hainan/jinjiang/val_100238.json',
        DEVKIT_DIR:
            '/gds/train-data/hainan/jinjiang/VOCdevkit2007'
    } ,    
    'coco_2007_val_jiuquan': {
        IM_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007/VOC2007/JPEGImages_jiuquan',
        ANN_FN:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/jiuquan_2420.json',
        DEVKIT_DIR:
            '/gds/train-data/oilstealing/oil_vehicle_person_10cls/VOCdevkit2007'
    } 
}

''''coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
        'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/coco_test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    }, 
    
    '''
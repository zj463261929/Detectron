#coding=utf-8
import requests
import time
import json
import base64
import os


''' #输入
{"data":[
        {"info":[
            [
             {"imgPath": "image path"},
             {"cls":"autotruck","forklift"},
             {"score":"1.00","0.98"},
             {"xmin"：51.3,21.3},
             {"ymin"：151.3,30.3},
             {"xmax"：151.3,51.3},
             {"ymax"：251.3,151.3}
            ],

            [
             {"imgPath": "image path"},
             {"cls":"autotruck","forklift"},
             {"score":"1.00","0.98"},
             {"xmin"：51.3,21.3},
             {"ymin"：151.3,30.3},
             {"xmax"：151.3,51.3},
             {"ymax"：251.3,151.3}
            ]
        ]},
 {"param":[{"scoreth":1.0},{"aveth":35},{"iouth":0.7}]}
 ]
}
'''

''' #输出
{"data":[
        [{"imgPath": "image path"},
         {"cls":"autotruck",
           "score":"1.00",
           "static":1,
           "bbox":{"xmin"：51.3，
                    "ymin"：151.3，
                    "xmax"：151.3，
                    "ymax"：251.3}
          },
          {"cls":"forklift",
            "score":"0.98",
            "static":0,
            "bbox":{"xmin"：21.3,
                    "ymin"：30.3,
                    "xmax"：51.3,
                    "ymax"：151.3}
           },
        ],
        
        [{"imgPath": "image path"},
         {"cls":"autotruck",
          "score":"1.00",
          "static":1,
          "bbox":{"xmin"：51.3,
                  "ymin"：151.3,
                  "xmax"：151.3,
                  "ymax"：251.3}
         },
         {"cls":"forklift",
          "score":"0.98",
          "static":0,
          "bbox":{"xmin"：21.3,
                  "ymin"：30.3,
                  "xmax"：51.3,
                  "ymax"：151.3}
         },
        ],
        
        ]
}
'''

num = 1
alltime = 0

i = 0
start = time.time()
for i in xrange(num):
    start = time.clock()
    s = requests
    '''
    img_lst = ["/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083614001.jpg","/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083615001.jpg","/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083617001.jpg","/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083618001.jpg"]
    xmin_lst = [ [188,404],[404,189],[404,189],[403] ]
    ymin_lst = [ [211,276],[276,211],[277,211],[278] ]
    xmax_lst = [ [200,413],[413,200],[413,200],[412] ]
    ymax_lst = [ [242,297],[298,241],[298,242],[300] ]
    score_lst = [ [0.57,0.47],[0.64,0.39],[0.5,0.34],[0.63] ]
    cls_lst = [ ["person","person"],["person","person"],["person","person"],["person"] ]'''
    
    '''
    cls_lst = [["person", "person"],["person", "person"],["person"],["person", "person"]]
    img_lst =  ["/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083615001.jpg","/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083614001.jpg",
    "/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083618001.jpg",
    "/opt/yangguan/zhangjing/Detectron/test_xml/test1/42/20190122083617001.jpg"]
    score_lst = [["0.64", "0.39"], ["0.57", "0.47"],["0.63"],["0.5", "0.34"]]
    xmax_lst = [[413, 200],[200, 413],[412],[413, 200]]
    xmin_lst =  [[404, 189],[188, 404],[403],[404, 189]]
    ymax_lst = [[298, 241],[242, 297],[300],[298, 241]]
    ymin_lst = [[276, 211],[211, 276],[278],[277, 211]]'''
    
    cls_lst = [["digger", "suv"],["digger", "suv"],["digger","suv"],["digger", "suv"]]
    img_lst =  ["/opt/yangguan/zhangjing/Detectron/test_xml/test2/17/20190506053523595.jpg","/opt/yangguan/zhangjing/Detectron/test_xml/test2/17/20190506053524884.jpg",
    "/opt/yangguan/zhangjing/Detectron/test_xml/test2/17/20190506053525628.jpg",
    "/opt/yangguan/zhangjing/Detectron/test_xml/test2/17/20190506053526137.jpg"]
    score_lst = [["1.0", "0.95"], ["0.99", "0.85"],["0.98","0.89"],["0.99", "0.56"]]
    xmin_lst =  [[889, 1622],[891, 1709],[886,1753],[889,1789]]
    ymin_lst = [[670,366],[669,374],[666,377],[671,385]]
    xmax_lst = [[1431,1857],[1427,1914],[1430,1917],[1429,1917]]
    ymax_lst = [[996,475],[994,481],[991,476],[996,487]]
    

                   
    for i in range(num):
        data_lst = []
        for img_index in range(len(img_lst)):
            singleImgInfo_lst = {"imgPath":img_lst[img_index],"cls":cls_lst[img_index],"score":score_lst[img_index],"xmin":xmin_lst[img_index],"ymin":ymin_lst[img_index],"xmax":xmax_lst[img_index],"ymax":ymax_lst[img_index]}

            data_lst.append(singleImgInfo_lst)   
        data = {"data":data_lst,"param":{"scoreth":1.0,"aveth":35.0,"iouth":0.7}}
        print data
        #print data_lst
        
        my_json_data = json.dumps(data)
        headers = {'Content-Type': 'application/json'}
        #r = s.post('http://192.168.15.100:9528/user', headers=headers,data = my_json_data,)
        r = s.post('http://192.168.15.100:9828/user', headers=headers,data = my_json_data,)
        
        #print type(r)
        #print (r)
        #print type(r.json())
        
        print (r.json())
        #print (i)
        i = i+1
        #add plot
        '''
        img = cv2.imread(os.path.join(imagepath,file))
        data= {}
        data = r.json()
        datalist = []
        datalist = data['data']
        print(len(datalist))
        for j in xrange(len(datalist)):
            singledata = {}
            boxdict = {}
            singledata = datalist[j]
            boxdict = singledata['bbox']
            xmin = boxdict['xmin']
            ymin = boxdict['ymin']
            xmax = boxdict['xmax']
            ymax = boxdict['ymax']
            cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(0,255,0))
            
            font= cv2.FONT_HERSHEY_SIMPLEX
            strname = singledata['cls']
            strscore = singledata['score']
            #print (type(strscore))
            print (strscore)
            cv2.putText(img, strname + str(strscore) + '(' + str(xmax - xmin) + ',' + str(ymax - ymin) + ')', (xmin,ymin-10), font, 1,(0,0,255),2)
        print(os.path.join(imagepath_out,file))
        cv2.imwrite(os.path.join(imagepath_out,file), img)'''
#end = time.time() - start
#print end


'''
################################################################
############################# curl #############################
curl -X POST 'http://192.168.200.213:8081/user' -d '{"data":"/opt/ligang/Detectron/test_xml/test2/1/20190506090506410.jpg"}' -H 'Content-Type: application/json'


curl -X POST 'http://192.168.200.213:9527/user' -d '{"data":"https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1526895699811&di=5ce6acbcfe8f1d93fe65d3ae8eb3287d&imgtype=0&src=http%3A%2F%2Fimg1.fblife.com%2Fattachments1%2Fday_130616%2F20130616_e4c0b7ad123ca263d1fcCnkYLFk97ynn.jpg.thumb.jpg"}' -H 'Content-Type: application/json'
'''
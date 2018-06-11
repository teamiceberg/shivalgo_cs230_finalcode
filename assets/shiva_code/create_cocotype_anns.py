""" Code Copyright Shiva Badruswamy, Stanford University. Written by Shiva Badruswamy
to generate MS-COCO type mask annotations
Code creates JSON files for inpout similar to MS COCO's input JSON files
"""

import json
import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageColor
import pandas as pd
import codecs
import time
from skimage.io import imread


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from pycocotools import cocostuffhelper as cocohelp

import zipfile
import urllib.request
import shutil
import skimage.io


dataset_dir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D"
annfiledir = dataset_dir+"/samples/stanford2D/annotations/"
semimgdir1 = dataset_dir+"/samples/stanford2D/s2DDataset/area_3/data/semantic/"
semimgdir2 = dataset_dir+"/samples/stanford2D/s2DDataset/area_5a/data/semantic/"
semimgdir3 = dataset_dir+"/samples/stanford2D/s2DDataset/area_1/data/semantic/"
sem_ann_file = annfiledir+"semantic_labels.json"

sem_labels = json.load(open(sem_ann_file))
labelmap=np.array(sem_labels)
l_count = len(sem_labels)
class_dict = {"clutter":0,"ceiling":1,"floor":2,"wall":3,"column":4,"beam":5,"window":6,"door":7,"table":8,"chair":9,"bookcase":10,"sofa":11,"board":12}
class_list = [{"name":"ceiling","id":1},{"name":"floor","id":2},{"name":"wall","id":3},{"name":"column","id":4},
                        {"name":"beam","id":5},{"name":"window","id":6},{"name":"door","id":7},{"name":"table","id":8},
                        {"name":"chair","id":9},{"name":"bookcase","id":10},{"name":"sofa","id":11},{"name":"board","id":12}]

def parse_label(label):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split("_")
    res['instance_class'] = clazz
    res['instance_num'] = int(instance_num)
    res['room_type'] = room_type
    res['room_num'] = int(room_num)
    res['area_num'] = int(area_num)
    return res


def gen_labelmap(im_name):
    file_path1 = semimgdir1+im_name+"_domain_semantic.png"
    file_path2 = semimgdir2+im_name+"_domain_semantic.png"
    file_path3 = semimgdir3+im_name+"_domain_semantic.png"
    if os.path.isfile(file_path1):
        img = imread(file_path1)
        w, h = img.shape[0], img.shape[1]
    elif os.path.isfile(file_path2):
        img = imread(file_path1)
        w, h = img.shape[0], img.shape[1]
    elif os.path.isfile(file_path3):
        img = imread(file_path1)
        w, h = img.shape[0], img.shape[1]
    else:
        raise FileNotFoundError

    conv256 = np.array([[256*256],
              [256],
              [1]])
    indexedImage = np.dot(img,conv256)
    labmap=labelmap[indexedImage]
    for element in labmap:
        if element in range(0,l_count):
            ins_label = sem_labels[element]
            label_dict = parse_label(ins_label)
            class_name = label_dict['instance_class']
            class_id = class_dict[class_name]
            element = int(class_id)
    labmap=np.array(labmap)
    return labmap


if __name__ == '__main__':

    rgb_traindir = dataset_dir+"/samples/stanford2D/train_s2D"
    rgb_valdir = dataset_dir+"/samples/stanford2D/val_s2D"
    rgb_testdir = dataset_dir+"/samples/stanford2D/test_s2D"
    rgb_traindict = json.load(open(annfiledir+"train_s2D.json"))
    rgb_valdict = json.load(open(annfiledir+"val_s2D.json"))
    rgb_testdict = json.load(open(annfiledir+"test_s2D.json"))
    rgb_trainlist = rgb_traindict['children']
    rgb_vallist = rgb_valdict['children']
    rgb_testlist = rgb_testdict['children']
    len_train = len(rgb_trainlist)
    len_val = len(rgb_vallist)
    len_test = len(rgb_testlist)
    train_anns = {}
    val_anns = {}

    generate annotations data for train images
    train_anns = []
    data = {}
    t_ann_count = 0
    for i in range(0,1000):
        im_name = rgb_trainlist[i]['name']
        im_id = rgb_trainlist[i]['id']
        iend = im_name.find('_domain')
        im_name = im_name[0:iend]
        labmap = gen_labelmap(im_name) #generate label map
        t_ann = cocohelp.segmentationToCocoResult(labmap,int(im_id),stuffStartId=0)
        train_anns.append(t_ann)
        t_ann_count+=1
        print("train ann count:", t_ann_count)
    
    data["annotations"] = train_anns
    
    #create train_anns2.json
    with open('train_anns2.json','w') as outfile:
        json.dump(train_anns,outfile,sort_keys=True,indent=4)
    print("finished generating train_anns2.json")
    
    generate annotations data for val images
    val_anns = []
    data = {}
    v_ann_count = 0
    for i in range(0,200):
        im_name = rgb_vallist[i]['name']
        im_id = rgb_vallist[i]['id']
        iend = im_name.find('_domain')
        im_name = im_name[0:iend]
        labmap = gen_labelmap(im_name) # generate label map
        v_ann = cocohelp.segmentationToCocoResult(labmap, int(im_id),stuffStartId=0)
        val_anns.append(v_ann)
        v_ann_count+=1
        print("val ann count:", v_ann_count)
    
    data["annotations"] = val_anns
    
    #create train_anns.json
    with open('val_anns2.json','w') as outfile:
        json.dump(val_anns,outfile,sort_keys=True,indent=4)
    print("finished generating val_anns2.json")
 
    
    generate annotations data for test images
    test_anns = []
    data = {}
    tt_ann_count = 0
    for i in range(0,len_test):
        im_name = rgb_testlist[i]['name']
        im_id = rgb_testlist[i]['id']
        iend = im_name.find('_domain')
        im_name = im_name[0:iend]
        labmap = gen_labelmap(im_name) #generate label map
        tt_ann = cocohelp.segmentationToCocoResult(labmap,int(im_id),stuffStartId=0)
        test_anns.append(tt_ann)
        tt_ann_count+=1
        print("test ann count:", tt_ann_count)
    
    data["annotations"] = test_anns
    
    ## create test_anns2.json
    with open('test_anns2.json','w') as outfile:
        json.dump(data,outfile,sort_keys=True,indent=4)
    print("finished generating test_anns2.json")

    data["categories"] = class_list
    with open('test_anns2.json','a+') as infile:
        json.dump(data,infile,sort_keys=True,indent=4)
    

    





    

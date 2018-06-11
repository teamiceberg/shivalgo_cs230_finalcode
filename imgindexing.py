import json
import os
import sys

seg_file = json.load(open('test_s2D.json','r'))
f_list = seg_file["children"]
data = []
f_dict = {}

for i in range(0,len(f_list)):
    f_id = f_list[i]['id']
    f_name = f_list[i]['name']
    f_path = f_list[i]['path']
    f_dict[f_id]= {"name":f_name,"path":f_path}

with open('test_img.json', 'w') as outfile:
        json.dump(f_dict, outfile, indent=4)


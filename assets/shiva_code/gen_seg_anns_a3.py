import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import codecs
import json
import skimage.io

## specify all paths here - add a '/' at the end of dir names
rgbimgdir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/images/area_3/data/rgb/"
annfiledir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/annotations/"
rgbimgname_dict = json.load(open(annfiledir+"area_3.json"))
rgbimgname_list = rgbimgname_dict['children']
semimgdir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/images/area_3/data/semantic/"


#json file generation routine
def gen_img_json(rgbimgdir,rgbimgname,semimgdir,annfiledir,fcount):

    # reduce decimal place to 1 while printing
    np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})

    # image directory path
    dirpath = rgbimgdir
    img_name = rgbimgname
    iend = img_name.find('_domain')
    imname_clean = img_name[0:iend]
    

    # display image
    img = cv.imread(dirpath+img_name)
    # cv.imshow("Original image:"+img_name, img)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # extract edges
    imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgrayblur = cv.blur(imggray, (3, 3))
    imedges = cv.Canny(imgrayblur, 50, 75, apertureSize=3, L2gradient=True)
    # cv.imshow("Extracted edges", imedges)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # # applying closing function
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    # closed = cv.morphologyEx(imedges, cv.MORPH_CLOSE, kernel)
    # cv.imshow("Closed", closed)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # find contours
    img2,contours,hierarchy = cv.findContours(
        imedges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # cv.imshow("Extracted contours", img2)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # store polydp contours in JSON
    # file_path = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/annotations/contours/"
    # file_path = file_path+imname_clean+".json"
    # contours = np.asarray(contours)
    # contours = contours.tolist()


    # draw contours
    # for c in contours:
    #     cnt_img = cv.drawContours(img, [c], -1, (0, 255, 0), 2)
    # cnt_img = cv.drawContours(img, contours, -1, (0, 255, 0), 2)
    # cv.imshow("Contours applied on original image:"+img_name, cnt_img)
    # cv.waitKey()
    # cv.destroyAllWindows()


    ## Draw bounding boxes
    area_thresh = 20
    # color = (255,0,0)
    # for c in contours:
    #     rect = cv.boundingRect(c)
    #     area = cv.contourArea(c)
    #     x, y, w, h = rect
    #     if  area >= area_thresh:
    #         cv.rectangle(img, (x, y), (x+w, y+h),color, 2)
    #         cv.putText(img, 'Area:' + str(area),
    #                    (x+w+10, y+h), 0, 0.3, color)
    # cv.imshow("Ground Truth Bounding Boxes", img)
    # cv.waitKey()
    # cv.destroyAllWindows()

    # Draw Min area rectangle
    # color = (0,255,0)
    # for c in contours:
    #     rect = cv.minAreaRect(c)
    #     center, w_h, aor = rect
    #     area = w_h[0] * w_h[1]
    #     if area > 1 and aor in range(0,361,90):
    #         print("center:", center, "width:", w_h[0], "height:", w_h[1], " aor:", aor)
    #         box = np.int0(cv.boxPoints(rect))
    #         imbox = cv.drawContours(img,[box],0,color,2)
    # cv.imshow("Ground Truth: Close-fit Bounding Boxes", imbox)
    # cv.waitKey()
    # cv.destroyAllWindows()

    #draw polygons
    # color = (0,0,255)
    # for c in contours:
    #     epsilon = 0.005*cv.arcLength(c,True)
    #     polydp = cv.approxPolyDP(c,epsilon,True)
    #     impolydp = cv.drawContours(img,[polydp],0,color,2)
    # cv.imshow("Ground Truth: polygons", impolydp)
    # cv.waitKey()
    # cv.destroyAllWindows()


    def cenvalue(center,fname):
        """
        takes center of bounding box as input and returns the instance's class name and instance number
        """
        sem_ann_file = annfiledir+"semantic_labels.json"
        sem_labels = json.load(open(sem_ann_file))
        semdir_path = semimgdir
        file_path = semdir_path+fname+"_domain_semantic.png"
        image = skimage.io.imread(file_path)
        x = np.int32(center[0])
        y = np.int32(center[1])
        rgbval = image[x,y,:]
        idx = get_index(rgbval)
        print("filecount:",fcount,"||rgbval:",rgbval,"||semfileindex:",idx)
        
        l_count = len(sem_labels)
        if idx in range(1,l_count):
            ins_label = sem_labels[idx]
            label_dict = parse_label(ins_label)
            class_name = label_dict['instance_class']
            ins_num = label_dict['instance_num']
           
        else:
            class_name = "<UNK>"
            ins_num = 0
        
        return class_name, ins_num
            
       
        
    def get_index(color):
        ''' Parse a color as a base-256 number and returns the index
        Args:
            color: A 3-tuple in RGB-order where each element \in [0, 255]
        Returns:
            index: an int containing the index specified in 'color'
        '''
        return color[0] * 256 * 256 + color[1] * 256 + color[2]

        
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


    #save outputs to json file (1 file per image: Data has contour (X,Y), 
    #BOX[X,Y,W,H],  minBOX[CENTER, H,W, AOR])
    #CENTER of Box will be used to generate segment label; CENTER COLOR IN SEMANTICS FILE  = LABEL;
    #IF LABEL NOT FOUND, LABEL = CLUTTER
    #set area_thresh in Bounding Box generation code section

    clist = []
    for c in contours:
        rect1 = cv.boundingRect(c)
        area1 = cv.contourArea(c)
        x,y,w,h = rect1
        rect2 = cv.minAreaRect(c)
        center, wh, aor = rect2
        area2 = wh[0]*wh[1]
        if area2 >= area_thresh:
            class_name,ins_num = cenvalue(center,imname_clean)
            c = c.tolist()
            clist.append([{"contour":c},{"box":[x,y,w,h,area1]},{"min_box":[center,wh[0],wh[1],aor]}, 
                {"class_name":class_name},{"instance_num":ins_num}])
        
    #construct dictionary to output to JSON file
    file_path = annfiledir+"segments/"+imname_clean+".json"
    c = {"data":[{"file_id":imname_clean},{"Segments":clist}]} 
    json.dump(c, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    


#generate segment annotations file for all images 
#call gen_img_json for each image in rgb image directory
fcount = len(rgbimgname_list)
for f in range(0,fcount):
    rgbimgname = rgbimgname_list[f]['name']
    gen_img_json(rgbimgdir,rgbimgname,semimgdir,annfiledir,f)













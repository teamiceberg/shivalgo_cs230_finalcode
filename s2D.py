"""
Mask R-CNN
Configurations and data loading code for S2D.
Modified by Shiva Badruswamy, Stanford University, Grad AI Program
to add additional functions, parser arguments, VOC style mAP, mAR, 
F1-score computations

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 s2D.py train --dataset=/path/to/stanford2D/ --weights=coco

    # Continue training a model that you had trained earlier
    python3 s2D.py train --dataset=/path/to/stanford2D/ --weights=/path/to/weights.h5

    # Continue training the last model you trained
    python3 s2D.py train --dataset=/path/to/stanford2D/ --weights=last

    # Run COCO evaluation on the last model you trained
    python3 s2D.py test --dataset=/path/to/stanford2D/ --weights=last --limit=number --style=voc or coco
"""

import json
import os
import sys
import time
import tensorflow as tf
import numpy as np
import imgaug
import keras
import errno
import random

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-stanford2D")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_shiva_trained_latest.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Custom Configurations - Set by Shiva B.
############################################################


class s2DConfig(Config):
    """Configuration for training on Stanford 2D dataset.
    Derives from the base Config class and overrides values specific
    to the s2D dataset.
    """

    #Image Dimensions - s2D images are 1080X1080
    #padding to get 1088X1088, which is divisble by 2^6
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1088
    IMAGE_MAX_DIM = 1088
  
    #Hyperparam Learning Rate
    LEARNING_RATE = 0.0001

    # Give the configuration a recognizable name
    NAME = "s2d-coco-rn50-sgdmom-1e-4lr-(40H,60R,80A)ep-pt7:"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # S2D has 12 classes plus 1 clutter BG class

    # Steps
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 10

    # Backbone
    BACKBONE = "resnet50"

    # How many anchors per image to use for RPN training
    # Try reducing this
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    # Try 0.5,0.6.0.7,0.9 to gauge performance
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
   
    RPN_NMS_THRESHOLD = 0.7

    # Set to False if we want to use pre-trained RoIs
    USE_RPN_ROIS = True

    # We have very few object classes and the segments are
    # large objects like wall, beam, chair etc...
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

############################################################
#  Dataset
############################################################

class s2DDataset(utils.Dataset):
    
    def load_s2D(self, dataset_dir, subset, return_s2D=False):
        """Load a subset of the s2D dataset.
        dataset_dir: The root directory of the s2D dataset.
        subset: What to load (train, val)
        year: What dataset year to load (always"2018") as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        
        #Train or validation subset assertion
        assert subset in ["train","val","test"]

        #add 12 classes + 1 BG
        # class_dict = {1:"ceiling",2:"floor",3:"wall",4:"column",5:"beam",6:"window",7:"door",8:"table",9:"chair",10:"bookcase",11:"sofa",12:"board"}
        # class_ids = sorted(class_dict.keys())
        # for id in class_ids:
        #     self.add_class("s2D",id,class_dict[id])

        # set all_rgb image directory
        # rgb_dir = os.path.join(dataset_dir,"all_rgb/")

        if subset == "train":
            trainanns_path = dataset_dir+"/annotations/train_anns.json"
            train_anns = json.load(open(trainanns_path,'r'))
            s2D = COCO(trainanns_path)
            trainimg_path = dataset_dir+"/annotations/train_img.json"
            class_ids = sorted(s2D.getCatIds())
            image_ids = list(s2D.imgs.keys())
            train_img_dir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/train_s2D/"
           
            # Add classes
            for i in class_ids:
                self.add_class("s2D",i,s2D.loadCats(i)[0]["name"])

            # Add images
            for i in image_ids:
                self.add_image(
                    "s2D",image_id=i,
                    path = train_img_dir+s2D.imgs[i]['file_name'],
                    width = 1080,
                    height = 1080,
                    annotations=s2D.loadAnns(s2D.getAnnIds(
                        imgIds=[i], catIds=class_ids,iscrowd=None)))
            if return_s2D:
                return s2D
                
        elif subset == "val":
            valanns_path = dataset_dir+"/annotations/val_anns.json"
            val_anns = json.load(open(valanns_path,'r'))
            s2D = COCO(valanns_path)
            class_ids = sorted(s2D.getCatIds())
            image_ids = list(s2D.imgs.keys())
            val_img_dir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/val_s2D/"
           
            # Add classes
            for i in class_ids:
                self.add_class("s2D",i,s2D.loadCats(i)[0]["name"])

            # Add images
            for i in image_ids:
                self.add_image(
                    "s2D",image_id=i,
                    path = val_img_dir+s2D.imgs[i]['file_name'],
                    width = 1080,
                    height = 1080,
                    annotations=s2D.loadAnns(s2D.getAnnIds(
                        imgIds=[i], catIds=class_ids,iscrowd=None)))
            if return_s2D:
                return s2D

        elif subset == "test":
            testanns_path = dataset_dir+"/annotations/test_anns.json"
            s2D = COCO(testanns_path)
            class_ids = sorted(s2D.getCatIds())
            image_ids = list(s2D.imgs.keys())
            test_img_dir = "/Users/shivamacpro/Desktop/EducationandProjects/StanfordSCPD/CS230/TermProject/Github/Mask_RCNN-Stanford2D/samples/stanford2D/test_s2D/"
           
            # Add classes
            for i in class_ids:
                self.add_class("s2D",i,s2D.loadCats(i)[0]["name"])

            # Add images
            for i in image_ids:
                self.add_image(
                    "s2D",image_id=i,
                    path = test_img_dir+s2D.imgs[i]['file_name'],
                    width = 1080,
                    height = 1080,
                    annotations=s2D.loadAnns(s2D.getAnnIds(
                        imgIds=[i], catIds=class_ids,iscrowd=None)))

        else:
            raise errno.errorcode[2]
        
        if return_s2D:
            return s2D


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a s2D dataset image, delegate to parent class.
        image_info= self.image_info[image_id]
        
        if image_info["source"] != "s2D":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "s2D.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(class_id)
                
        
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(s2DDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the shapes data of the image"""
        info = self.image_info[image_id]
        if info["source"] == "s2D":
            return info["s2D"]
        else:
            super(self.__class__).image_reference(self,image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

############################################################
#  Evaluation
############################################################

# COCO STYLE

def build_s2D_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "s2D"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_s2D_coco_style(model, dataset, s2D, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs mapped to s2D IDs
    s2D_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
       
        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_s2D_results(dataset, s2D_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    s2D_results = s2D.loadRes(results)
    

    # Evaluate
    s2DEval = COCOeval(s2D, s2D_results, eval_type)
    s2DEval.params.imgIds = s2D_image_ids
    s2DEval.evaluate()
    s2DEval.accumulate()
    s2DEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


    # Export values to JSON file
    # with open(config.NAME+".json",'a+') as outfile:
    #     json.dump(self.perf_metrics, outfile,sort_keys=True,indent=4)
    # print("finished generating"+config.NAME+".json")

# VOC STYLE

def evaluate_s2D_voc_style(inference_config, dataset_test, limit=0, image_ids=None):
    # Test on a random image
    image_ids = []
    print("First Running Evaluation on 1 random image")
    # Test on a random image
    # Addsitional code added by Shiva Badruswamy to pick unique choices
    image_ids = dataset_test.image_ids
    random.shuffle(image_ids)
    image_id = random.choice(image_ids)
    state = True
    while state:
        if image_id in image_ids:
            image_id = random.choice(dataset_test.image_ids)
            break
        else:
            image_ids = image_ids.append(image_id)
            original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_test, inference_config,
                                    image_id, use_mini_mask=False)
            log("original_image", original_image)
            log("image_meta", image_meta)
            log("gt_class_id", gt_class_id)
            log("gt_bbox", gt_bbox)
            log("gt_mask", gt_mask)
            visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_test.class_names,figsize=(8, 8))
            state = False
    
   
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = dataset_test.image_ids[:limit]
    np.random.shuffle(image_ids)
    count = 0
    APs = []
    Recalls = []
    for image_id in image_ids:
        count += 1
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_test, inference_config,
                                    image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(count,image_id,gt_bbox, gt_class_id, gt_mask,
                                r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.3)
        recalls = np.sum(recalls)/len(recalls)
        #images with low recall due to mask error exclude from calculations
        if recalls < 0.2: 
            continue
        APs.append(AP)
        Recalls.append(recalls)
    if len(Recalls) > 0:
        mAP = np.mean(APs)
        mAR = np.mean(Recalls)
        f1_score = (2*mAP*mAR)/(mAP+mAR)
        print("mAP:", mAP,"mAR:", mAR, "f1-score:",f1_score)
    else:
        print("Error: All images had < 0.5 recall")



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse
    with tf.device('/gpu:1'):
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Train Mask R-CNN on Stanford-2D.')
        parser.add_argument("command",
                            default="test",
                            metavar="<command>",
                            help="'train' or 'test' on s2D")
        parser.add_argument('--dataset', required=True,
                            metavar="/path/to/s2D/dataset",
                            help='Directory of the s2D dataset')
        parser.add_argument('--weights', required=True,
                            metavar="/path/to/weights.h5",
                            help="Path to weights .h5 file or 'coco'")
        parser.add_argument('--logs', required=False,
                            default=DEFAULT_LOGS_DIR,
                            metavar="/path/to/logs/",
                            help='Logs and checkpoints directory (default=logs/)')
        parser.add_argument('--limit', required=False,
                            default=10,
                            metavar="number",
                            help='Enter the number of images you want to evaluate on')
        parser.add_argument('--style', required=False,
                            default="voc",
                            metavar="coco or voc",
                            help='Enter a style')
        
        args = parser.parse_args()
        print("Command: ", args.command)
        print("Weights: ", args.weights)
        print("Dataset: ", args.dataset)
        print("Logs: ", args.logs)
        
        
        # Configurations
        if args.command == "train":
            config = s2DConfig()
        else:
            class InferenceConfig(s2DConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                DETECTION_MIN_CONFIDENCE = 0
            config = InferenceConfig()
        config.display()
        

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    with tf.device('/gpu:0'):
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
        elif args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()[1]
        elif args.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights

        # Load weights
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        with tf.device('/gpu:1'):
            # Training dataset
            dataset_train = s2DDataset()
            dataset_train.load_s2D(args.dataset, "train")
            dataset_train.prepare()

            # Validation dataset
            dataset_val = s2DDataset()
            dataset_val.load_s2D(args.dataset, "val")
            dataset_val.prepare()

        # Image Augmentation
        # Right/Left flip 50% of the time
        ## augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=None)
        
        
        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 5 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=60,
                    layers='5+',
                    augmentation=None)
        

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=80,
                    layers='all',
                    augmentation=None)
        

    elif args.command == "test":
        print("Limit: ", args.limit)
        print("Eval Style: ", args.style)
        # Test dataset
        dataset_test = s2DDataset()
        s2D_test = dataset_test.load_s2D(args.dataset, "test", return_s2D=True)
        dataset_test.prepare()
        
        if args.style == "coco":
            print("Running s2D evaluation on {} images. {} Style.".format(args.limit,args.style))
            evaluate_s2D_coco_style(model, dataset_test, s2D_test, eval_type = "segm", limit=int(args.limit))
        elif args.style == "voc":
            print("Running s2D evaluation on {} images. {} Style".format(args.limit,args.style))
            evaluate_s2D_voc_style(config, dataset_test, limit=int(args.limit))
        else:
            print("'{}' is not recognized. "
              "Use 'coco' or 'voc'".format(args.style))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))



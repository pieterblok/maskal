# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-10-08 10:20:39

## Use a trained model to auto-annotate unlabelled images

## general libraries
import argparse
import numpy as np
import os
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

## detectron2-libraries 
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

## libraries for preparing the datasets
from active_learning.sampling import list_files, write_cvat_annotations, write_labelme_annotations


## function to visualize the output of Mask R-CNN
def visualize(img_np, classes, scores, masks, boxes, class_names):
    masks = masks.astype(dtype=np.uint8)
    font_scale = 0.6
    font_thickness = 1
    text_color = [0, 0, 0]

    if masks.any():
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        red_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        blue_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        green_mask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1]),dtype=np.uint8)
        all_masks = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],3),dtype=np.uint8) # BGR

        colors = [(0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 255), (255, 255, 255)]
        color_list = np.remainder(np.arange(len(class_names)), len(colors))
        imgcopy = img_np.copy()

        for i in range (maskstransposed.shape[-1]):
            x1, y1, x2, y2 = boxes[i, :]
            cv2.rectangle(imgcopy, (int(x1), int(y1)), (int(x2), int(y2)), colors[classes[i]], 1)

            _class = class_names[classes[i]]
            color = colors[color_list[classes[i]]]
            text_str = '%s: %.2f' % (_class, scores[i])

            font_face = cv2.FONT_HERSHEY_DUPLEX

            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            text_pt = (int(x1), int(y1) - 3)

            cv2.rectangle(imgcopy, (int(x1), int(y1)), (int(x1) + text_w, int(y1) - text_h - 4), colors[classes[i]], -1)
            cv2.putText(imgcopy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

            mask = maskstransposed[:,:,i]

            if colors.index(color) == 0: # green
                green_mask = cv2.add(green_mask,mask)
            elif colors.index(color) == 1: # blue
                blue_mask = cv2.add(blue_mask,mask)
            elif colors.index(color) == 2: # magenta
                blue_mask = cv2.add(blue_mask,mask)
                red_mask = cv2.add(red_mask,mask)
            elif colors.index(color) == 3: # red
                red_mask = cv2.add(red_mask,mask)
            elif colors.index(color) == 4: # yellow
                green_mask = cv2.add(green_mask,mask)
                red_mask = cv2.add(red_mask,mask)
            else: #white
                blue_mask = cv2.add(blue_mask,mask)
                green_mask = cv2.add(green_mask,mask)
                red_mask = cv2.add(red_mask,mask)

        all_masks[:,:,0] = blue_mask
        all_masks[:,:,1] = green_mask
        all_masks[:,:,2] = red_mask
        all_masks = np.multiply(all_masks,255).astype(np.uint8)

        img_mask = cv2.addWeighted(imgcopy,1,all_masks,0.5,0)
    else:
        img_mask = img_np

    return img_mask



if __name__ == "__main__":
    ## Initialize the args_parser and some variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='datasets/train', help='the folder with images that need to be auto-annotated')
    parser.add_argument('--network_config', type=str, default='COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml', help='specify the backbone of the CNN')
    parser.add_argument('--classes', nargs="+", default=[], help='a list with all the classes')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold for the Mask R-CNN inference')
    parser.add_argument('--nms_thres', type=float, default=0.2, help='non-maximum suppression threshold for the Mask R-CNN inference')
    parser.add_argument('--weights_file', type=str, default=[], help='weight-file (.pth)')
    parser.add_argument('--export_format', default='cvat', help='Choose either "labelme" or "cvat"')

    ## Load the args_parser and initialize some variables
    opt = parser.parse_args()
    print(opt)
    print()

    images, annotations = list_files(opt.img_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(opt.network_config))
    cfg.NUM_GPUS = 1
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(opt.classes)
    cfg.MODEL.WEIGHTS = opt.weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf_thres
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = opt.nms_thres
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    predictor = DefaultPredictor(cfg)

    for i in tqdm(range(len(images))):
        imgname = images[i]
        basename = os.path.basename(imgname)
        img = cv2.imread(os.path.join(opt.img_dir, imgname))
        height, width, _ = img.shape

        # Do the image inference and extract the outputs from Mask R-CNN
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()

        class_names = []
        for h in range(len(classes)):
            class_id = classes[h]
            class_name = opt.classes[class_id]
            class_names.append(class_name)

        img_vis = visualize(img, classes, scores, masks, boxes, opt.classes)

        if opt.export_format == "cvat":
            write_cvat_annotations(opt.img_dir, basename, class_names, masks, height, width)
        
        if opt.export_format == "labelme":
            writedata = write_labelme_annotations(basename, class_names, masks, height, width)
# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-06-04 13:49:37

## t-SNE visualization of the box-features of Mask R-CNN
## t-SNE is a technique for dimensionality reduction for the visualization of high-dimensional datasets
## https://lvdmaaten.github.io/tsne/

## general libraries
import sys
import argparse
import yaml
import time
import numpy as np
import torch
import os
import cv2
import csv
import random
import operator
from sklearn.manifold import TSNE
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import warnings
warnings.filterwarnings("ignore")

## detectron2-libraries 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

## libraries that are specific for the intermediate feature-output (for t-SNE visualization)
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.structures import ImageList
from detectron2.modeling.backbone import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper

## libraries that are specific for dropout inference (recall that without the baal-function the dropout isn't applied during inference)
from active_learning.strategies.dropout import FastRCNNConvFCHeadDropout
from active_learning.strategies.dropout import FastRCNNOutputLayersDropout
from active_learning.strategies.dropout import MaskRCNNConvUpsampleHeadDropout

## default matplotlib-plotting
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


def register_datasets(dataroot, traindir, valdir, testdir):
    register_coco_instances("train", {}, os.path.join(dataroot, "train.json"), traindir)
    register_coco_instances("val", {}, os.path.join(dataroot, "val.json"), valdir)
    register_coco_instances("test", {}, os.path.join(dataroot, "test.json"), testdir)

    train_metadata = MetadataCatalog.get("train")
    val_metadata = MetadataCatalog.get("val")
    test_metadata = MetadataCatalog.get("test")

    dataset_dicts_train = DatasetCatalog.get("train")
    dataset_dicts_val = DatasetCatalog.get("val")
    dataset_dicts_test = DatasetCatalog.get("test")

    return dataset_dicts_train, train_metadata, dataset_dicts_val, val_metadata, dataset_dicts_test, test_metadata


def model_inference(cfg, classes, weightsfolder, weightsfile, score_thres, nms_thres, gpu_num):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHeadDropout'
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsDropout'
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHeadDropout"
    cfg.NUM_GPUS = gpu_num
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.OUTPUT_DIR = weightsfolder
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightsfile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thres
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thres
    cfg.DATALOADER.NUM_WORKERS = 0
    
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    return model


def get_box_features(model, image):
    ## thanks to: https://stackoverflow.com/questions/62442039/detectron2-extract-region-features-at-a-threshold-for-object-detection
    with torch.no_grad():
        height, width = image.shape[:2]
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        
        images = model.preprocess_image(inputs)
        model = model.eval()
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)

        features_ = [features[f] for f in model.roi_heads.box_in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)
        
        feats = box_features[pred_inds] ## these are the in_features of the box_predictor for each of the instances
        
    return feats, pred_instances[0]
       

def make_mask_img(input, height, width, method):
    if method == "circle":
        cps_X = input[0]
        cps_Y = input[1]
        ds = input[2]

        masks = np.ones((len(cps_X),height,width)).astype(dtype=np.bool)

        for k in range(len(cps_X)):
            mask = np.zeros((height,width, 1),dtype = np.uint8)
            cp_X = cps_X[k]
            cp_Y = cps_Y[k]
            r = np.divide(ds[k], 2)

            mask = cv2.circle(mask, (int(cp_X),int(cp_Y)), int(r), [1], -1)
            mask = mask.transpose(2,0,1).astype(np.bool)
            masks[k,:,:] = np.multiply(masks[k,:,:],mask.reshape((height,width)))


    if method == "polylines":
        polylines = input
        masks = np.ones((len(polylines),height,width)).astype(dtype=np.bool)

        for k in range(len(polylines)):
            xy = polylines[k]
            mask = np.zeros((height,width, 1),dtype = np.uint8)

            for w in range(len(xy)):
                cur_xy = xy[w]
                poly = np.array(cur_xy).reshape(-1,1,2)
                mask = cv2.fillPoly(mask, np.int32([poly]), [1])

            mask = mask.transpose(2,0,1).astype(np.bool)
            masks[k,:,:] = np.multiply(masks[k,:,:],mask.reshape((height,width)))

    return masks


def calculate_iou(predictions, annotations):
    gtmasks = annotations.transpose(1,2,0).astype(np.uint8)
    detmasks = predictions.transpose(1,2,0).astype(np.uint8)

    gtmask_num = gtmasks.shape[-1]
    detmask_num = detmasks.shape[-1]

    IoU_matrix = np.zeros((detmask_num, gtmask_num)).astype(dtype=np.float32)

    for i in range (detmask_num):
        mask = detmasks[:,:,i]*255
        maskimg = np.expand_dims(mask, axis=2)

        for k in range (gtmask_num):
            gtmask = gtmasks[:,:,k]*255
            gtmaskimg = np.expand_dims(gtmask, axis=2)

            intersection_area = cv2.countNonZero(cv2.bitwise_and(maskimg,gtmaskimg))
            union_area = cv2.countNonZero(cv2.bitwise_or(maskimg,gtmaskimg))

            IoU = np.divide(intersection_area,union_area)
            IoU_matrix[i,k] = IoU

    return IoU_matrix


def obtain_iou_matrix(data, instances):
    classes_annot = []
    masks_poly = []
    
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy()

    for k in range(len(data["annotations"])):
        classes_annot.append(data["annotations"][k]['category_id'])
        masks_poly.append(data["annotations"][k]['segmentation'])

    masks_annot = make_mask_img(masks_poly, data['height'], data['width'], "polylines")
    iou_matrix = calculate_iou(masks, masks_annot)

    return iou_matrix, classes, classes_annot, scores


def get_tsne_features(data, classes, classes_annot, scores, iou_matrix, metadata, image_paths, labels, features_tsne, mode='gt'):
    if iou_matrix.size != 0:
        highest_row_values = np.max(iou_matrix, axis=1)
        highest_column_values = np.max(iou_matrix, axis=0)

        for hrv in range(len(highest_row_values)):
            highest_row_value = highest_row_values[hrv]
            if highest_row_value != 0.0:
                image_paths.append(data["file_name"])
                array_pos = np.where(iou_matrix == highest_row_value)
                det_idx = array_pos[0][0]
                gt_idx = array_pos[1][0]

                if mode == 'gt':
                    label = classes_annot[gt_idx]
                elif mode == 'det':
                    label = classes[det_idx]
                else:
                    label = classes_annot[gt_idx]

                labels.append(metadata.thing_classes[label])
                score = scores[det_idx]
                f_tsne = feats[det_idx]
                f_tsne = f_tsne.cpu().detach().numpy() 

                if features_tsne is None:
                    features_tsne = f_tsne
                else:
                    features_tsne = np.vstack([features_tsne, f_tsne])

    return image_paths, labels, features_tsne


## the functions below were extracted from: https://github.com/spmallick/learnopencv/tree/master/TSNE
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape
    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)
    image = cv2.resize(image, (image_width, image_height))
    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape
    center_x = int(image_centers_area_size * x) + offset
    center_y = int(image_centers_area_size * (1 - y)) + offset
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)
    br_x = tl_x + image_width
    br_y = tl_y + image_height
    return tl_x, tl_y, br_x, br_y


def visualize_tsne_points(tx, ty, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label in colors_per_class:
        indices = [i for i, l in enumerate(labels) if l == label]
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255
        ax.scatter(current_tx, current_ty, c=color, label=label)

    ax.legend(loc='best')
    plt.show()


def visualize_tsne(tsne, labels, plot_size=1000, max_image_size=100):
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)
    
    visualize_tsne_points(tx, ty, labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./active_learning/utils/tsne.yaml', help='yaml with the tsne parameters')
    args = parser.parse_args()

    try:
        with open(args.config, 'rb') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        sys.exit(f"Could not find configuration-file: {args.config}, closing application")

    print("Configuration:")
    for key, value in config.items():
        print(key, ':', value)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']
    gpu_num = len(config['cuda_visible_devices'])
    colors_per_class = dict(zip(config['classes'], config['colors']))

    ## initialize parameters
    features_tsne = None
    labels = []
    image_paths = []

    ## register the datasets 
    dataset_dicts_train, train_metadata, dataset_dicts_val, val_metadata, dataset_dicts_test, test_metadata = register_datasets(config['dataroot'], config['traindir'], config['valdir'], config['testdir'])

    ## prepare Mask R-CNN for inference
    cfg = get_cfg()
    model = model_inference(cfg, config['classes'], config['weightsfolder'], config['weightsfile'], config['score_thres'], config['nms_thres'], gpu_num)

    ## do the model inference and save the box-features
    for d in tqdm(random.sample(dataset_dicts_train, config['num_images'])):
        image = cv2.imread(d["file_name"])
        image_vis = image.copy()

        feats, outputs = get_box_features(model, image)
        instances = outputs["instances"].to("cpu") ## obtain the default-outputs from mask r-cnn

        if config['visualize']:
            visualizer = Visualizer(image_vis[:, :, ::-1], metadata=train_metadata, scale=0.8)
            vis = visualizer.draw_instance_predictions(instances)
            imshow(vis.get_image()[:, :, ::-1])

        iou_matrix, classes, classes_annot, scores = obtain_iou_matrix(d, instances)
        image_paths, labels, features_tsne = get_tsne_features(d, classes, classes_annot, scores, iou_matrix, train_metadata, image_paths, labels, features_tsne, mode=config['mode'])

    ## do the t-SNE visualization
    tsne = TSNE(n_components=2).fit_transform(features_tsne)
    visualize_tsne(tsne, labels)
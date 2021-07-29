# -*- coding: utf-8 -*-
# @Author: Pieter Blok
# @Date:   2021-05-25 11:11:53
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-07-29 11:00:44
## Determine the consistency of the uncertainty estimate as a function of the number of forward passes 

## general libraries
import argparse
import time
import torch
import yaml
import sys
import numpy as np
import os
import cv2
import pickle
from PIL import Image
import random
import warnings
import operator
from collections import OrderedDict
from tqdm import tqdm
import seaborn as sns
import pandas as pd
warnings.filterwarnings("ignore")

## detectron2-libraries 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

## libraries that are specific for monte-carlo dropout
from active_learning.strategies.dropout import FastRCNNConvFCHeadDropout
from active_learning.strategies.dropout import FastRCNNOutputLayersDropout
from active_learning.strategies.dropout import MaskRCNNConvUpsampleHeadDropout
from active_learning.sampling import prepare_initial_dataset, update_train_dataset
from active_learning.sampling.montecarlo_dropout import MonteCarloDropout, MonteCarloDropoutHead
from active_learning.sampling import observations
from detectron2.checkpoint import DetectionCheckpointer

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()

def imshow_pil(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()
       
## run on gpu 0
os.environ["CUDA_VISIBLE_DEVICES"]="0"

supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def list_files(rootdir):
    images = []

    if os.path.isdir(rootdir):
        for root, dirs, files in list(os.walk(rootdir)):
            for name in files:
                subdir = root.split(rootdir)
                all('' == s for s in subdir)
                
                if subdir[1].startswith('/'):
                    subdirname = subdir[1][1:]
                else:
                    subdirname = subdir[1]

                if name.lower().endswith(supported_cv2_formats):
                    if all('' == s for s in subdir):
                        images.append(name)
                    else:
                        images.append(os.path.join(subdirname, name))
    
        images.sort()

    return images


def calculate_max_entropy(classes):
    least_confident = np.divide(np.ones(len(classes)), len(classes)).astype(np.float32)
    probs = torch.from_numpy(least_confident)
    max_entropy = torch.distributions.Categorical(probs).entropy()
    return max_entropy


## this is a copy-paste of the uncertainty function of uncertainty.py, but with one more outputs for u_h of the observations
def uncertainty(observations, iterations, max_entropy, width, height, device, mode = 'min'):
    uncertainty_list = []
    mean_bboxs = []
    mean_masks = []
    
    for key, val in observations.items():
        softmaxes = [v['softmaxes'] for v in val]
        entropies = torch.stack([torch.distributions.Categorical(softmax).entropy() for softmax in softmaxes])
        
        ## first normalize the entropy-value with the maximum entropy (which is the least confident situation with equal softmaxes for all classes)
        entropies_norm = torch.stack([torch.divide(entropy, max_entropy.to(device)) for entropy in entropies])

        ## invert the normalized entropy-values so it can be properly used in the uncertainty calculation
        inv_entropies_norm = torch.stack([torch.subtract(torch.ones(1).to(device), entropy_norm) for entropy_norm in entropies_norm])

        mean_bbox = torch.mean(torch.stack([v['pred_boxes'].tensor for v in val]), axis=0)
        mean_mask = torch.mean(torch.stack([v['pred_masks'].flatten().type(torch.cuda.FloatTensor) for v in val]), axis=0)
        mean_mask[mean_mask < 0.25] = 0.0
        mean_mask = mean_mask.reshape(-1, width, height)

        mean_bbox_np = mean_bbox.detach().cpu().numpy()
        mean_mask_np = mean_mask.detach().cpu().numpy()
        mean_mask_np[mean_mask_np > 0] = 1
        mean_mask_np = mean_mask_np.astype(np.bool)
        # mean_mask_np_vis = np.multiply(mean_mask_np, 255).transpose(1,2,0)
        mean_bboxs.append(mean_bbox_np)
        mean_masks.append(mean_mask_np)

        mask_IOUs = []
        for v in val:
            current_mask = v['pred_masks']
            overlap = torch.logical_and(mean_mask, current_mask)
            union = torch.logical_or(mean_mask, current_mask)
            if union.sum() > 0:
                IOU = torch.divide(overlap.sum(), union.sum())
                mask_IOUs.append(IOU.unsqueeze(0))

        if len(mask_IOUs) > 0:
            mask_IOUs = torch.cat(mask_IOUs)
        else:
            mask_IOUs = torch.tensor([float('NaN')]).to(device)

        bbox_IOUs = []
        mean_bbox = mean_bbox.squeeze(0)
        boxAArea = torch.multiply((mean_bbox[2] - mean_bbox[0] + 1), (mean_bbox[3] - mean_bbox[1] + 1))
        for v in val:
            current_bbox = v['pred_boxes'].tensor.squeeze(0)
            xA = torch.max(mean_bbox[0], current_bbox[0])
            yA = torch.max(mean_bbox[1], current_bbox[1])
            xB = torch.min(mean_bbox[2], current_bbox[2])
            yB = torch.min(mean_bbox[3], current_bbox[3])
            interArea = torch.multiply(torch.max(torch.tensor(0).to(device), xB - xA + 1), torch.max(torch.tensor(0).to(device), yB - yA + 1))
            boxBArea = torch.multiply((current_bbox[2] - current_bbox[0] + 1), (current_bbox[3] - current_bbox[1] + 1))
            bbox_IOU = torch.divide(interArea, (boxAArea + boxBArea - interArea))
            bbox_IOUs.append(bbox_IOU.unsqueeze(0))

        if len(bbox_IOUs) > 0:
            bbox_IOUs = torch.cat(bbox_IOUs)
        else:
            bbox_IOUs = torch.tensor([float('NaN')]).to(device)        

        val_len = torch.tensor(len(val)).to(device)
        outputs_len = torch.tensor(iterations).to(device)

        u_sem = torch.clamp(torch.mean(inv_entropies_norm), min=0, max=1)
        
        u_spl_m = torch.clamp(torch.divide(mask_IOUs.sum(), val_len), min=0, max=1)
        u_spl_b = torch.clamp(torch.divide(bbox_IOUs.sum(), val_len), min=0, max=1)
        u_spl = torch.multiply(u_spl_m, u_spl_b)

        u_sem_spl = torch.multiply(u_sem, u_spl)
        
        try:
            u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
        except:
            u_n = 0.0

        u_h = torch.multiply(u_sem_spl, u_n)
        uncertainty_list.append(u_h.unsqueeze(0))

    if uncertainty_list:
        uncertainty_list = torch.cat(uncertainty_list)

        if mode == 'min':
            uncertainty = torch.min(uncertainty_list)
        elif mode == 'mean':
            uncertainty = torch.mean(uncertainty_list)
        elif mode == 'max':
            uncertainty = torch.max(uncertainty_list)
        else:
            uncertainty = torch.mean(uncertainty_list)
            
    else:
        uncertainty = torch.tensor([float('NaN')]).to(device)
        uncertainty_list = torch.tensor([float('NaN')]).to(device)

    return uncertainty.detach().cpu().numpy().squeeze(0), uncertainty_list.detach().cpu().numpy().tolist(), mean_bboxs, mean_masks
    

def calculate_iou(mask_to_check, current_masks):
    maskstransposed = mask_to_check.transpose(1,2,0)
    mask_to_check = maskstransposed.astype(np.uint8)

    IoUs = []

    for i in range (len(current_masks)):
        current_mask = current_masks[i].transpose(1,2,0)
        current_mask = current_mask.astype(np.uint8)

        try:
            intersection_area = cv2.countNonZero(cv2.bitwise_and(mask_to_check, current_mask))
            union_area = cv2.countNonZero(cv2.bitwise_or(mask_to_check, current_mask))

            IoU = np.divide(intersection_area,union_area)
            if not np.isnan(IoU):
                IoUs.append(IoU)
            else:
                IoUs.append(0.0)
        except:
            IoUs.append(0.0)

    return IoUs


def cluster_observation(reps_ranked):
    u_h = []
    fps = []
    masks_to_check = []
    for r in range(len(reps_ranked)):
        if len(masks_to_check) == 0:
            masks_to_check = reps_ranked[r][1]
            u_h.append(reps_ranked[r][0])
            fps.append(reps_ranked[r][2])
        else:
            cur_u_h = np.empty(len(u_h[0]))
            cur_u_h[:] = np.NaN
            for mc in range(len(masks_to_check)):
                mask_to_check = masks_to_check[mc]
                current_masks = reps_ranked[r][1]
                IoUs = calculate_iou(mask_to_check, current_masks)
                if not all(IoU == 0 for IoU in IoUs):
                    try:
                        match_id = np.argmax(IoUs)
                        cur_u_h[mc] = reps_ranked[r][0][match_id]
                    except:
                        pass
            u_h.append(cur_u_h.tolist())
            fps.append(reps_ranked[r][2])

    return u_h, fps


def visualize_obs(img, mean_bboxs, mean_masks, u_total):
    width, height = img.shape[:-1]
    img_vis = img.copy()
    
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.25
    font_thickness = 1
    thickness = 3
    text_color = [0, 0, 0]
    text_boxes_color = [255, 255, 255]
    obs_color = [255, 255, 255]
    
    if len(mean_masks) > 0:
        red_mask = np.zeros((width, height),dtype=np.uint8)
        blue_mask = np.zeros((width, height),dtype=np.uint8)
        green_mask = np.zeros((width, height),dtype=np.uint8)
        all_masks = np.zeros((width, height,3),dtype=np.uint8)

        for i in range(len(mean_masks)):
            boxes = mean_bboxs[i]
            
            mean_mask = mean_masks[i]
            maskstransposed = mean_mask.transpose(1,2,0)
            mask = maskstransposed.astype(np.uint8)

            x1, y1, x2, y2 = boxes[0]
            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), obs_color, thickness)

            text_str = "Observation {:d}".format(i+1)
            text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
            text_pt = (x1, y1 - 3)

            cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x1) + int(text_w), int(y1) - int(text_h) - 4), obs_color, -1)
            cv2.putText(img_vis, text_str, (int(text_pt[0]), int(text_pt[1])), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

            blue_mask = cv2.add(blue_mask,mask)
            green_mask = cv2.add(green_mask,mask)
            red_mask = cv2.add(red_mask,mask)

        all_masks[:,:,0] = blue_mask
        all_masks[:,:,1] = green_mask
        all_masks[:,:,2] = red_mask
        all_masks = np.multiply(all_masks, obs_color[0]).astype(np.uint8)

        strs = ['u_h']
        for j, (u_h) in enumerate(zip(u_total)):
            boxes = mean_bboxs[j]
            x1, y1, x2, y2 = boxes[0]

            values = []
            values.append([u_h])

            for v in range(len(values[0])):
                val = values[0][v]
                text_str = strs[v] + ": {:.2f}".format(val[0])

                text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                text_pt = (x2 + 5, y1 + (38 * (v+1))+10)

                cv2.rectangle(img_vis, (int(text_pt[0]), int(text_pt[1]) + 7), (int(text_pt[0]) + int(text_w), int(text_pt[1]) - int(text_h) - 4), text_boxes_color, -1)
                cv2.putText(img_vis, text_str, (int(text_pt[0]), int(text_pt[1])), font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)

        img_obs = cv2.addWeighted(img_vis, 1, all_masks, 0.5, 0)

    else:
        img_obs = img_vis

    return img_obs
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./active_learning/utils/uncertainty_forward_passes.yaml', help='yaml with the tsne parameters')
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

    ## prepare the network for inference
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = config['dropout_probability']
    cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = config['dropout_probability']
    cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHeadDropout'
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsDropout'
    cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHeadDropout"
    cfg.MODEL.ROI_HEADS.SOFTMAXES = True

    cfg.NUM_GPUS = gpu_num
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config['classes'])
    cfg.OUTPUT_DIR = config['weightsfolder']
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, config['weightsfile'])
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    cfg.DATALOADER.NUM_WORKERS = 0

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    device = cfg.MODEL.DEVICE
    max_entropy = calculate_max_entropy(config['classes'])
    images = list_files(config['dataroot'])
    check_direxcist(config['resultsfolder'])
    
    uncertainties = {}

    for i in tqdm(range(len(images))):
        filename = images[i]
        img = cv2.imread(os.path.join(config['dataroot'], filename))
        width, height = img.shape[:-1]
        fps = []

        for it in range(len(config['forward_passes'])):
            iter = config['forward_passes'][it]

            if config['dropout_method'] == 'head':
                predictor = MonteCarloDropoutHead(cfg, iter)
            else:
                predictor = MonteCarloDropout(cfg, iter, config['al_batch_size'])

            outputs = predictor(img)
            obs = observations(outputs, config['iou_thres'])
            img_uncertainty, uncertainty_obs, mean_bboxs, mean_masks = uncertainty(obs, iter, max_entropy, width, height, device, 'mean')
            fps.append([uncertainty_obs, mean_masks, iter])

            if config['visualize']:
                img_obs = visualize_obs(img, mean_bboxs, mean_masks, uncertainty_obs)
                img_obs = cv2.cvtColor(img_obs, cv2.COLOR_BGR2RGB)
                img_obs = Image.fromarray(img_obs)
                img_obs.save(os.path.join(config['resultsfolder'], '{:s}_fp_{:03d}_prob{:.2f}.jpg'.format(os.path.splitext(filename)[0], iter, config['dropout_probability'])), "JPEG")
                # imshow_pil(img_obs)

        lengths = [len(fps[r][0]) for r in range(len(fps))]
        ranked = np.argsort(lengths)
        largest_obs = ranked[::-1]
        reps_ranked = [fps[i] for i in largest_obs]
        u_h, fwps = cluster_observation(reps_ranked)

        if len(u_h) > 0:
            zipped_lists = zip(fwps, u_h)
            sorted_zipped_lists = sorted(zipped_lists)
            u_h_f = [element for _, element in sorted_zipped_lists]
            uncertainties[filename] = u_h_f

        pickle_name = os.path.join(config['resultsfolder'], 'uncertainty_metrics_prob_{:.2f}.pkl'.format(config['dropout_probability']))
        with open(pickle_name, 'wb') as handle:
            pickle.dump(uncertainties, handle, protocol=pickle.HIGHEST_PROTOCOL)
# -*- coding: utf-8 -*-
# @Author: Pieter Blok
# @Date:   2021-05-25 11:11:53
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-07-28 09:52:29
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
    
    for key, val in observations.items():
        softmaxes = [v['softmaxes'] for v in val]
        entropies = torch.stack([torch.distributions.Categorical(softmax).entropy() for softmax in softmaxes])
        
        ## first normalize the entropy-value with the maximum entropy (which is the least confident situation with equal softmaxes for all classes)
        entropies_norm = torch.stack([torch.divide(entropy, max_entropy.to(device)) for entropy in entropies])

        ## invert the normalized entropy-values so it can be properly used in the uncertainty calculation
        inv_entropies_norm = torch.stack([torch.subtract(torch.ones(1).to(device), entropy_norm) for entropy_norm in entropies_norm])

        mean_bbox = torch.mean(torch.stack([v['pred_boxes'].tensor for v in val]), axis=0)
        mean_mask = torch.mean(torch.stack([v['pred_masks'].flatten().type(torch.cuda.FloatTensor) for v in val]), axis=0)
        mean_mask[mean_mask <= 0.3] = 0.0
        mean_mask = mean_mask.reshape(-1, width, height)

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

    return uncertainty.detach().cpu().numpy().squeeze(0), uncertainty_list.detach().cpu().numpy().tolist()


def mean_bbox_mask(obs):
    mean_bboxs = []
    mean_masks = []
    for key, val in obs.items():
        mean_bbox = torch.mean(torch.stack([v['pred_boxes'].tensor for v in val]), axis=0)
        mean_mask = torch.mean(torch.stack([v['pred_masks'].flatten().type(torch.cuda.FloatTensor) for v in val]), axis=0)
        mean_mask[mean_mask <= 0.3] = 0.0
        mean_mask = mean_mask.reshape(-1, width, height)
        mean_bboxs.append(mean_bbox)
        mean_masks.append(mean_mask)
    return mean_bboxs, mean_masks
    

def calculate_iou(mask_to_check, current_masks):
    maskstransposed = mask_to_check.detach().cpu().numpy().transpose(1,2,0)
    mask_to_check = maskstransposed.astype(np.uint8)

    IoUs = []

    for i in range (len(current_masks)):
        current_mask = current_masks[i].detach().cpu().numpy().transpose(1,2,0)
        current_mask = current_mask.astype(np.uint8)

        try:
            intersection_area = cv2.countNonZero(cv2.bitwise_and(mask_to_check, current_mask))
            union_area = cv2.countNonZero(cv2.bitwise_or(mask_to_check, current_mask))

            IoU = np.divide(intersection_area,union_area)
            IoUs.append(IoU)
        except:
            IoUs.append(0.0)

    return IoUs


def cluster_observation(reps_ranked):
    u_h = []
    masks_to_check = []
    for r in range(len(reps_ranked)):
        if len(masks_to_check) == 0:
            masks_to_check = reps_ranked[r][1]
            u_h.append(reps_ranked[r][0])
        else:
            cur_u_h = np.empty(len(u_h[0]))
            cur_u_h[:] = np.NaN
            for mc in range(len(masks_to_check)):
                mask_to_check = masks_to_check[mc]
                current_masks = reps_ranked[r][1]
                IoUs = calculate_iou(mask_to_check, current_masks)
                try:
                    match_id = np.argmax(IoUs)
                    cur_u_h[match_id] = reps_ranked[r][0][match_id]
                except:
                    pass
            u_h.append(cur_u_h.tolist())

    return u_h
                

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

    for it in range(len(config['forward_passes'])):
        uncertainties = {}
        iter = config['forward_passes'][it]
        print("\n\r Number of forward passes: {:d}".format(iter))

        if config['dropout_method'] == 'head':
            predictor = MonteCarloDropoutHead(cfg, iter)
        else:
            predictor = MonteCarloDropout(cfg, iter, config['al_batch_size'])

        for i in tqdm(range(len(images))):
            filename = images[i]
            img = cv2.imread(os.path.join(config['dataroot'], filename))
            width, height = img.shape[:-1]
            unc = []
            reps = []

            for n in range(config['repetitions']):
                outputs = predictor(img)
                obs = observations(outputs, config['iou_thres'])
                img_uncertainty, uncertainty_obs = uncertainty(obs, iter, max_entropy, width, height, device, 'mean')
                unc.append(uncertainty_obs)
                mean_bboxs, mean_masks = mean_bbox_mask(obs)
                reps.append([uncertainty_obs, mean_masks])

            lengths = [len(reps[r][0]) for r in range(len(reps))]
            ranked = np.argsort(lengths)
            largest_obs = ranked[::-1]
            reps_ranked = [reps[i] for i in largest_obs]
            u_h = cluster_observation(reps_ranked)

            if len(u_h) > 0:
                uncertainties[filename] = u_h

        pickle_name = os.path.join(config['resultsfolder'], 'forward_passes_{:03d}_prob{:.2f}.pkl'.format(iter, config['dropout_probability']))
        with open(pickle_name, 'wb') as handle:
            pickle.dump(uncertainties, handle, protocol=pickle.HIGHEST_PROTOCOL)
# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-03-29 20:45:21

## Active learning with Mask R-CNN

## general libraries
import numpy as np
import os
import cv2
import csv
import random
import operator
from collections import OrderedDict
from tqdm import tqdm
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
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

## libraries that are specific for dropout training
from active_learning.strategies.dropout import FastRCNNConvFCHeadDropout
from active_learning.strategies.dropout import FastRCNNOutputLayersDropout
from active_learning.sampling import prepare_initial_dataset, update_train_dataset
from active_learning.sampling.montecarlo_dropout import MonteCarloDropout
from active_learning.sampling import observations
from active_learning.heuristics import uncertainty

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
       
## run on gpu 0
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder



def Train_Eval(dataroot, imgdir, classes, weightsfolder, resultsfolder, csv_name, init):
    ## CustomTrainer with evaluator
    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, ("bbox", "segm"), False, output_folder)

    if init:
        register_coco_instances("train", {}, os.path.join(dataroot, "train.json"), imgdir)
        register_coco_instances("val", {}, os.path.join(dataroot, "val.json"), imgdir)
        register_coco_instances("test", {}, os.path.join(dataroot, "test.json"), imgdir)

        train_metadata = MetadataCatalog.get("train")
        val_metadata = MetadataCatalog.get("val")
        test_metadata = MetadataCatalog.get("test")

        dataset_dicts_train = DatasetCatalog.get("train")
        dataset_dicts_val = DatasetCatalog.get("val")
        dataset_dicts_test = DatasetCatalog.get("test")
    else:
        DatasetCatalog.remove("train")
        register_coco_instances("train", {}, os.path.join(dataroot, "train.json"), imgdir)
        train_metadata = MetadataCatalog.get("train")
        dataset_dicts_train = DatasetCatalog.get("train")


    ## add dropout layers to the architecture of Mask R-CNN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHeadDropout'
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsDropout'
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False


    ## initialize the training parameters  
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.NUM_GPUS = 1
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.LR_POLICY = 'steps_with_decay'
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (1000, 3000, 4000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.OUTPUT_DIR = weightsfolder
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


    ## evaluation
    trainer.resume_or_load(resume=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    cfg.DATASETS.TEST = ("test",)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=resultsfolder)
    val_loader = build_detection_test_loader(cfg, "test")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    segm_strings = [c.replace(c, 'AP-' + c) for c in classes]
    segm_values = [round(eval_results['segm'][s], 1) for s in segm_strings]
    write_values = [len(dataset_dicts_train), round(eval_results['segm']['AP'], 1)] + segm_values

    with open(os.path.join(resultsfolder, csv_name), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(write_values)

    return cfg, dataset_dicts_train


## initialize some paths
weightsroot = "./weights"
resultsroot = "./results"
dataroot = "./datasets"
imgdir = "./datasets/rgb_annotated"

classes = ['broccoli', 'damaged', 'matured', 'cateye', 'headrot']
train_val_test_split = [0.8, 0.1, 0.1]
strategies = ['active_learning', 'random']

initial_datasize = 50
pool_size = 50
loops = 9

if os.path.exists(os.path.join(dataroot, "initial_train.txt")):
    os.remove(os.path.join(dataroot, "initial_train.txt"))

for strat in range(len(strategies)):
    strategy = strategies[strat]
    weightsfolder = os.path.join(weightsroot, strategy)
    resultsfolder = os.path.join(resultsroot, strategy)
    check_direxcist(weightsfolder)
    check_direxcist(resultsfolder)

    if not os.path.exists(os.path.join(dataroot, "initial_train.txt")):
        prepare_initial_dataset(dataroot, imgdir, classes, train_val_test_split, initial_datasize)
    else:
        initial_train_file = open(os.path.join(dataroot, "initial_train.txt"), "r")
        initial_train_names = initial_train_file.readlines()
        initial_train_names = [initial_train_names[i].rstrip('\n') for i in range(len(initial_train_names))]
        update_train_dataset(dataroot, imgdir, classes, initial_train_names)

    segm_strings = [c.replace(c, 'mAP-' + c) for c in classes]
    write_strings = ['train_size', 'mAP'] + segm_strings
    csv_name = strategy + '.csv'
    with open(os.path.join(resultsfolder, csv_name), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(write_strings)

    ## perform the training on the initial dataset
    if strat == 0:
        cfg, dataset_dicts_train = Train_Eval(dataroot, imgdir, classes, weightsfolder, resultsfolder, csv_name, init=True)
    else:
        cfg, dataset_dicts_train = Train_Eval(dataroot, imgdir, classes, weightsfolder, resultsfolder, csv_name, init=False)

    ## sampling of images
    for l in range(loops):    
        pool = {}

        train_names = [os.path.basename(dataset_dicts_train[i]['file_name']) for i in range(len(dataset_dicts_train))]
        train_file = open(os.path.join(dataroot, "train.txt"), "r")
        all_train_names = train_file.readlines()
        all_train_names = [all_train_names[i].rstrip('\n') for i in range(len(all_train_names))]
        pool_list = list(set(all_train_names) - set(train_names))

        if strategy == 'active_learning':
            iterations = 5
            batch_size = 5
            iou_thres = 0.5

            cfg.MODEL.ROI_HEADS.SOFTMAXES = True
            predictor = MonteCarloDropout(cfg, iterations, batch_size)
            device = cfg.MODEL.DEVICE

            if len(pool_list) > 0:
                ## find the images from the pool_list the algorithm is most uncertain about
                for d in tqdm(range(len(pool_list))):
                    filename = pool_list[d]
                    img = cv2.imread(os.path.join(imgdir, filename))
                    width, height = img.shape[:-1]
                    outputs = predictor(img)

                    obs = observations(outputs, iou_thres)
                    img_uncertainty = uncertainty(obs, iterations, width, height, device) ## reduce the iterations when facing a "CUDA out of memory" error

                    if len(pool) < pool_size:
                        pool[filename] = float(img_uncertainty)
                    else:
                        max_id, max_val = max(pool.items(), key=operator.itemgetter(1))
                        if float(img_uncertainty) < max_val:
                            del pool[max_id]
                            pool[filename] = float(img_uncertainty)

                sorted_pool = sorted(pool.items(), key=operator.itemgetter(1))
                pool = {}
                for k, v in sorted_pool:
                    pool[k] = v    

                ## update the training list and retrain the algorithm
                train_list = train_names + list(pool.keys())

                update_train_dataset(dataroot, imgdir, classes, train_list)
                cfg, dataset_dicts_train = Train_Eval(dataroot, imgdir, classes, weightsfolder, resultsfolder, csv_name, init=False)
                
            else:
                print("All images are used for the training, stopping the program...")

        if strategy == 'random':
            if len(pool_list) > 0:
                sample_list = random.choices(pool_list, k=pool_size)
                train_list = train_names + sample_list

                update_train_dataset(dataroot, imgdir, classes, train_list)
                cfg, dataset_dicts_train = Train_Eval(dataroot, imgdir, classes, weightsfolder, resultsfolder, csv_name, init=False)
            else:
                print("All images are used for the training, stopping the program...")

print("Finished...")
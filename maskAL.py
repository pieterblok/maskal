# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-05-11 16:24:51

## Active learning with Mask R-CNN

## general libraries
import sys
import argparse
import yaml
import numpy as np
import torch
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
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

## libraries that are specific for dropout training
from active_learning.strategies.dropout import FastRCNNConvFCHeadDropout
from active_learning.strategies.dropout import FastRCNNOutputLayersDropout
from active_learning.strategies.dropout import MaskRCNNConvUpsampleHeadDropout
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


def remove_initial_training_set(dataroot):
    if os.path.exists(os.path.join(dataroot, "initial_train.txt")):
        os.remove(os.path.join(dataroot, "initial_train.txt"))


def init_folders_and_files(weightsroot, resultsroot, classes, strategies):
    weightsfolders = []
    resultsfolders = []
    csv_names = []

    for strat in range(len(strategies)):
        strategy = strategies[strat]
        weightsfolder = os.path.join(weightsroot, strategy)
        check_direxcist(weightsfolder)
        weightsfolders.append(weightsfolder)
        
        resultsfolder = os.path.join(resultsroot, strategy)
        check_direxcist(resultsfolder)
        resultsfolders.append(resultsfolder)

        segm_strings = [c.replace(c, 'mAP-' + c) for c in classes]
        write_strings = ['train_size', 'mAP'] + segm_strings
        csv_name = strategy + '.csv'
        csv_names.append(csv_name)

        with open(os.path.join(resultsfolder, csv_name), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(write_strings)

    return weightsfolders, resultsfolders, csv_names


def write_train_files(train_names, writefolder, iteration):
    write_txt_name = "trainfiles_iteration{:03d}.txt".format(iteration+1)
    with open(os.path.join(writefolder, write_txt_name), 'w') as filehandle:
        for train_name in train_names:
            filehandle.write("{:s}\n".format(train_name))
    filehandle.close()


def calculate_max_entropy(classes):
    least_confident = np.divide(np.ones(len(classes)), len(classes)).astype(np.float32)
    probs = torch.from_numpy(least_confident)
    max_entropy = torch.distributions.Categorical(probs).entropy()
    return max_entropy
    

def Train_Eval(dataroot, imgdir, classes, weightsfolder, resultsfolder, csv_name, init):
    ## Hook to automatically save the best checkpoint
    class BestCheckpointer(HookBase):
        def __init__(self, eval_period, metric):
            self._period = eval_period
            self.metric = metric
            self.logger = setup_logger(name="d2.checkpointer.best")
            
        def store_best_model(self):
            metric = self.trainer.storage._latest_scalars

            try:
                current_value = metric[self.metric][0]
                try:
                    highest_value = metric['highest_value'][0]
                except:
                    highest_value = 0

                self.logger.info("current-value ({:s}): {:.2f}, highest-value ({:s}): {:.2f}".format(self.metric, current_value, self.metric, highest_value))

                if current_value > highest_value:
                    self.logger.info("saving best model...")
                    self.trainer.checkpointer.save("best_model")
                    self.trainer.storage.put_scalar('highest_value', current_value)
                    comm.synchronize()
            except:
                pass

        def after_step(self):
            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                self.store_best_model()
            self.trainer.storage.put_scalars(timetest=12)


    ## CustomTrainer with evaluator and automatic checkpoint-saver
    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, ("bbox", "segm"), False, output_folder)

        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, 'segm/AP'))
            return hooks


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
    cfg.MODEL.ROI_MASK_HEAD.NAME = 'MaskRCNNConvUpsampleHeadDropout'
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
    cfg.SOLVER.CHECKPOINT_PERIOD = (cfg.SOLVER.MAX_ITER+1)
    cfg.TEST.EVAL_PERIOD = 500
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
    cfg.OUTPUT_DIR = weightsfolder
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    ## evaluation
    trainer.resume_or_load(resume=True)

    try:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model.pth")
    except:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01
    cfg.DATASETS.TEST = ("test",)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=resultsfolder)
    val_loader = build_detection_test_loader(cfg, "test")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    segm_strings = [c.replace(c, 'AP-' + c) for c in classes]

    if len(classes) == 1:
        segm_values = [round(eval_results['segm']['AP'], 1) for s in segm_strings]
    else:
        segm_values = [round(eval_results['segm'][s], 1) for s in segm_strings]

    write_values = [len(dataset_dicts_train), round(eval_results['segm']['AP'], 1)] + segm_values

    with open(os.path.join(resultsfolder, csv_name), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(write_values)

    return cfg, dataset_dicts_train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./active_learning/config/maskAL.yaml', help='yaml with the training parameters')
    args = parser.parse_args()

    try:
        with open(args.config, 'rb') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        sys.exit(f"Could not find configuration-file {args.config}, closing application")

    print("Configuration:")
    for key, value in config.items():
        print(key, ':', value)

    weightsfolders, resultsfolders, csv_names = init_folders_and_files(config['weightsroot'], config['resultsroot'], config['classes'], config['strategies'])
    remove_initial_training_set(config['dataroot'])
    max_entropy = calculate_max_entropy(config['classes'])

    for strat in range(len(config['strategies'])):
        strategy = config['strategies'][strat]
        weightsfolder = weightsfolders[strat]
        resultsfolder = resultsfolders[strat]
        csv_name = csv_names[strat]

        if not os.path.exists(os.path.join(config['dataroot'], "initial_train.txt")):
            prepare_initial_dataset(config['dataroot'], config['imgdir'], config['classes'], config['train_val_test_split'], config['initial_datasize'])
        else:
            initial_train_file = open(os.path.join(config['dataroot'], "initial_train.txt"), "r")
            initial_train_names = initial_train_file.readlines()
            initial_train_names = [initial_train_names[i].rstrip('\n') for i in range(len(initial_train_names))]
            update_train_dataset(config['dataroot'], config['imgdir'], config['classes'], initial_train_names)

        ## perform the training on the initial dataset
        if strat == 0:
            cfg, dataset_dicts_train = Train_Eval(config['dataroot'], config['imgdir'], config['classes'], weightsfolder, resultsfolder, csv_name, init=True)
        else:
            cfg, dataset_dicts_train = Train_Eval(config['dataroot'], config['imgdir'], config['classes'], weightsfolder, resultsfolder, csv_name, init=False)

        ## sampling of images
        for l in range(config['loops']):    
            pool = {}

            train_names = [os.path.basename(dataset_dicts_train[i]['file_name']) for i in range(len(dataset_dicts_train))]
            write_train_files(train_names, resultsfolder, l)
            
            train_file = open(os.path.join(config['dataroot'], "train.txt"), "r")
            all_train_names = train_file.readlines()
            all_train_names = [all_train_names[i].rstrip('\n') for i in range(len(all_train_names))]
            pool_list = list(set(all_train_names) - set(train_names))

            if strategy == 'uncertainty':
                cfg.MODEL.ROI_HEADS.SOFTMAXES = True
                predictor = MonteCarloDropout(cfg, config['iterations'], config['batch_size'])
                device = cfg.MODEL.DEVICE

                if len(pool_list) > 0:
                    ## find the images from the pool_list the algorithm is most uncertain about
                    for d in tqdm(range(len(pool_list))):
                        filename = pool_list[d]
                        img = cv2.imread(os.path.join(config['imgdir'], filename))
                        width, height = img.shape[:-1]
                        outputs = predictor(img)

                        obs = observations(outputs, config['iou_thres'])
                        img_uncertainty = uncertainty(obs, config['iterations'], max_entropy, width, height, device) ## reduce the iterations when facing a "CUDA out of memory" error

                        if len(pool) < config['pool_size']:
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
                    update_train_dataset(config['dataroot'], config['imgdir'], config['classes'], train_list)
                    cfg, dataset_dicts_train = Train_Eval(config['dataroot'], config['imgdir'], config['classes'], weightsfolder, resultsfolder, csv_name, init=False)
                else:
                    print("All images are used for the training, stopping the program...")

            if strategy == 'certainty':
                cfg.MODEL.ROI_HEADS.SOFTMAXES = True
                predictor = MonteCarloDropout(cfg, config['iterations'], config['batch_size'])
                device = cfg.MODEL.DEVICE

                if len(pool_list) > 0:
                    ## find the images from the pool_list the algorithm is most uncertain about
                    for d in tqdm(range(len(pool_list))):
                        filename = pool_list[d]
                        img = cv2.imread(os.path.join(config['imgdir'], filename))
                        width, height = img.shape[:-1]
                        outputs = predictor(img)

                        obs = observations(outputs, config['iou_thres'])
                        img_uncertainty = uncertainty(obs, config['iterations'], max_entropy, width, height, device) ## reduce the iterations when facing a "CUDA out of memory" error

                        if len(pool) < config['pool_size']:
                            pool[filename] = float(img_uncertainty)
                        else:
                            min_id, min_val = min(pool.items(), key=operator.itemgetter(1))
                            if float(img_uncertainty) > min_val:
                                del pool[min_id]
                                pool[filename] = float(img_uncertainty)

                    sorted_pool = sorted(pool.items(), key=operator.itemgetter(1))
                    pool = {}
                    for k, v in sorted_pool:
                        pool[k] = v    

                    ## update the training list and retrain the algorithm
                    train_list = train_names + list(pool.keys())
                    update_train_dataset(config['dataroot'], config['imgdir'], config['classes'], train_list)
                    cfg, dataset_dicts_train = Train_Eval(config['dataroot'], config['imgdir'], config['classes'], weightsfolder, resultsfolder, csv_name, init=False)
                else:
                    print("All images are used for the training, stopping the program...")

            if strategy == 'random':
                if len(pool_list) > 0:
                    sample_list = random.choices(pool_list, k=config['pool_size'])
                    train_list = train_names + sample_list
                    update_train_dataset(config['dataroot'], config['imgdir'], config['classes'], train_list)
                    cfg, dataset_dicts_train = Train_Eval(config['dataroot'], config['imgdir'], config['classes'], weightsfolder, resultsfolder, csv_name, init=False)
                else:
                    print("All images are used for the training, stopping the program...")

    print("Finished...")
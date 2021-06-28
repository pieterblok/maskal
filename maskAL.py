# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-06-22 11:46:46

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
import logging
from shutil import copyfile
from collections import OrderedDict, Counter
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
from active_learning.sampling import prepare_initial_dataset, update_train_dataset, prepare_complete_dataset
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


## initialize the logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s \n'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
file_handler = logging.StreamHandler()
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)


def process_config_file(config, ints_to_lists):
    lengths = []

    for il in range(len(ints_to_lists)):
        int_to_list = ints_to_lists[il]
        config[int_to_list] = (config[int_to_list] if type(config[int_to_list]) is list else [config[int_to_list]])
        lengths.append(len(config[int_to_list])) 
    max_length = max(lengths)

    for il in range(len(ints_to_lists)):
        int_to_list = ints_to_lists[il]
        config[int_to_list] += [config[int_to_list][0]] * (max_length - len(config[int_to_list]))
        
    return config


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def init_folders_and_files(config):
    if not config['train_complete_trainset']:
        weightsfolders = []
        resultsfolders = []
        csv_names = []

        counts = Counter(config['strategies'])
        counts = list(counts.values())
        duplicates = any(x > 1 for x in counts)

        for s, (strategy, mode, dropout_probability, mcd_iterations, pool_size) in enumerate(zip(config['strategies'], config['mode'], config['dropout_probability'], config['iterations'], config['pool_size'])):
            if duplicates:
                folder_name = strategy + "_" + mode + "_" + "{:.2f}".format(dropout_probability) + "_" + str(mcd_iterations) + "_" + str(pool_size)
            else:
                folder_name = strategy

            weightsfolder = os.path.join(config['weightsroot'], config['experiment_name'], folder_name)            
            check_direxcist(weightsfolder)
            weightsfolders.append(weightsfolder)
            
            resultsfolder = os.path.join(config['resultsroot'], config['experiment_name'], folder_name)
            check_direxcist(resultsfolder)
            resultsfolders.append(resultsfolder)

            segm_strings = [c.replace(c, 'mAP-' + c) for c in config['classes']]
            write_strings = ['train_size', 'mAP'] + segm_strings
            csv_name = folder_name + '.csv'
            csv_names.append(csv_name)

            with open(os.path.join(resultsfolder, csv_name), 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(write_strings)
    else:
        folder_name = 'complete_trainset'
        weightsfolders = os.path.join(config['weightsroot'], folder_name)            
        check_direxcist(weightsfolders)
        
        resultsfolders = os.path.join(config['resultsroot'], folder_name)
        check_direxcist(resultsfolders)

        segm_strings = [c.replace(c, 'mAP-' + c) for c in config['classes']]
        write_strings = ['train_size', 'mAP'] + segm_strings
        csv_names = folder_name + '.csv'

        with open(os.path.join(resultsfolders, csv_names), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(write_strings)

    return weightsfolders, resultsfolders, csv_names


def remove_initial_training_set(dataroot):
    if os.path.exists(os.path.join(dataroot, "initial_train.txt")):
        os.remove(os.path.join(dataroot, "initial_train.txt"))


def calculate_max_entropy(classes):
    least_confident = np.divide(np.ones(len(classes)), len(classes)).astype(np.float32)
    probs = torch.from_numpy(least_confident)
    max_entropy = torch.distributions.Categorical(probs).entropy()
    return max_entropy


def get_train_names(dataset_dicts_train, traindir):
    train_names = []
    for i in range(len(dataset_dicts_train)):
        imgname = dataset_dicts_train[i]['file_name'].split(traindir)
        if imgname[1].startswith('/'):
            train_names.append(imgname[1][1:])
        else:
            train_names.append(imgname[1])
    return train_names


def get_initial_train_names(config):
    initial_train_file = open(os.path.join(config['dataroot'], "initial_train.txt"), "r")
    initial_train_names = initial_train_file.readlines()
    initial_train_names = [initial_train_names[idx].rstrip('\n') for idx in range(len(initial_train_names))]
    return initial_train_names


def create_pool_list(config, train_names):
    train_file = open(os.path.join(config['dataroot'], "train.txt"), "r")
    all_train_names = train_file.readlines()
    all_train_names = [all_train_names[idx].rstrip('\n') for idx in range(len(all_train_names))]
    pool_list = list(set(all_train_names) - set(train_names))
    return pool_list


def write_train_files(train_names, writefolder, iteration):
    write_txt_name = "trainfiles_iteration{:03d}.txt".format(iteration)
    with open(os.path.join(writefolder, write_txt_name), 'w') as filehandle:
        for train_name in train_names:
            filehandle.write("{:s}\n".format(train_name))
    filehandle.close()


def copy_initial_weight_file(read_folder, write_folder, iter):
    weight_file = "best_model_{:s}.pth".format(str(iter).zfill(3))
    check_direxcist(write_folder)
    if os.path.exists(os.path.join(read_folder, weight_file)):
        copyfile(os.path.join(read_folder, weight_file), os.path.join(write_folder, weight_file))
        

def copy_previous_weights(weights_folder, iteration):
    check_direxcist(weights_folder)
    previous_weights_file = os.path.join(weights_folder, "best_model_{:s}.pth".format(str(iteration-1).zfill(3)))
    next_weights_file = os.path.join(weights_folder, "best_model_{:s}.pth".format(str(iteration).zfill(3)))
    if os.path.isfile(previous_weights_file):
        copyfile(previous_weights_file, next_weights_file)
    

def Train_MaskRCNN(config, weightsfolder, gpu_num, iter, val_value, dropout_probability, init):    
    ## Hook to automatically save the best checkpoint
    class BestCheckpointer(HookBase):
        def __init__(self, iter, eval_period, val_value, metric):
            self.iter = iter
            self._period = eval_period
            self.val_value = val_value
            self.metric = metric
            self.logger = setup_logger(name="d2.checkpointer.best")
            
        def store_best_model(self):
            metric = self.trainer.storage._latest_scalars

            try:
                current_value = metric[self.metric][0]
                try:
                    highest_value = metric['highest_value'][0]
                except:
                    highest_value = self.val_value

                self.logger.info("current-value ({:s}): {:.2f}, highest-value ({:s}): {:.2f}".format(self.metric, current_value, self.metric, highest_value))

                if current_value > highest_value:
                    self.logger.info("saving best model...")
                    self.trainer.checkpointer.save("best_model_{:s}".format(str(iter).zfill(3)))
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
            hooks.insert(-1, BestCheckpointer(iter, cfg.TEST.EVAL_PERIOD, val_value, 'segm/AP'))
            return hooks


    if init:
        register_coco_instances("train", {}, os.path.join(config['dataroot'], "train.json"), config['traindir'])
        train_metadata = MetadataCatalog.get("train")
        dataset_dicts_train = DatasetCatalog.get("train")

        register_coco_instances("val", {}, os.path.join(config['dataroot'], "val.json"), config['valdir'])
        val_metadata = MetadataCatalog.get("val")
        dataset_dicts_val = DatasetCatalog.get("val")
    else:
        DatasetCatalog.remove("train")
        register_coco_instances("train", {}, os.path.join(config['dataroot'], "train.json"), config['traindir'])
        train_metadata = MetadataCatalog.get("train")
        dataset_dicts_train = DatasetCatalog.get("train")


    ## add dropout layers to the architecture of Mask R-CNN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = dropout_probability
    cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = dropout_probability
    cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHeadDropout'
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsDropout'
    cfg.MODEL.ROI_MASK_HEAD.NAME = 'MaskRCNNConvUpsampleHeadDropout'
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False
    cfg.OUTPUT_DIR = weightsfolder


    ## initialize the network weights, with an option to do the transfer-learning on previous models
    if config['transfer_learning_on_previous_models'] == True:
        if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter-1).zfill(3)))):
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter-1).zfill(3)))
        else:
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")


    ## initialize the training parameters  
    cfg.DATALOADER.SAMPLER_TRAIN = config['train_sampler']
    cfg.DATALOADER.REPEAT_THRESHOLD = config['repeat_threshold']
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.NUM_GPUS = gpu_num
    cfg.DATALOADER.NUM_WORKERS = config['num_workers']
    cfg.SOLVER.IMS_PER_BATCH = config['train_batch_size']
    cfg.SOLVER.WEIGHT_DECAY = config['weight_decay']
    cfg.SOLVER.LR_POLICY = config['learning_policy']
    cfg.SOLVER.BASE_LR = config['learning_rate']
    cfg.SOLVER.GAMMA = config['gamma']
    cfg.SOLVER.WARMUP_ITERS = config['warmup_iterations']
    cfg.SOLVER.MAX_ITER = config['max_iterations']
    cfg.SOLVER.STEPS = config['steps']
    cfg.SOLVER.CHECKPOINT_PERIOD = (cfg.SOLVER.MAX_ITER+1)
    cfg.TEST.EVAL_PERIOD = config['eval_period']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config['classes'])
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    try:
        val_value_output = trainer.storage._latest_scalars['highest_value'][0]
    except: 
        val_value_output = val_value

    return cfg, dataset_dicts_train, trainer, val_value_output


def Eval_MaskRCNN(cfg, config, dataset_dicts_train, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, iter, init):      
    if init:
        register_coco_instances("test", {}, os.path.join(config['dataroot'], "test.json"), config['testdir'])
        test_metadata = MetadataCatalog.get("test")
        dataset_dicts_test = DatasetCatalog.get("test")

    cfg.OUTPUT_DIR = weightsfolder
    trainer.resume_or_load(resume=True)

    try:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter).zfill(3)))
    except:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['confidence_threshold']   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    cfg.DATASETS.TEST = ("test",)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=resultsfolder)
    val_loader = build_detection_test_loader(cfg, "test")
    eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    segm_strings = [c.replace(c, 'AP-' + c) for c in config['classes']]

    if len(config['classes']) == 1:
        segm_values = [round(eval_results['segm']['AP'], 1) for s in segm_strings]
    else:
        segm_values = [round(eval_results['segm'][s], 1) for s in segm_strings]

    write_values = [len(dataset_dicts_train), round(eval_results['segm']['AP'], 1)] + segm_values

    with open(os.path.join(resultsfolder, csv_name), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(write_values)

    return cfg


def uncertainty_pooling(pool_list, pool_size, cfg, config, max_entropy, mcd_iterations, mode):
    pool = {}
    cfg.MODEL.ROI_HEADS.SOFTMAXES = True
    predictor = MonteCarloDropout(cfg, mcd_iterations, config['al_batch_size'])
    device = cfg.MODEL.DEVICE

    if len(pool_list) > 0:
        ## find the images from the pool_list the algorithm is most uncertain about
        for d in tqdm(range(len(pool_list))):
            filename = pool_list[d]
            img = cv2.imread(os.path.join(config['traindir'], filename))
            width, height = img.shape[:-1]
            outputs = predictor(img)

            obs = observations(outputs, config['iou_thres'])
            img_uncertainty = uncertainty(obs, mcd_iterations, max_entropy, width, height, device, mode) ## reduce the iterations when facing a "CUDA out of memory" error

            if not np.isnan(img_uncertainty):
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
    else:
        print("All images are used for the training, stopping the program...")

    return pool


def certainty_pooling(pool_list, pool_size, cfg, config, max_entropy, mcd_iterations, mode):
    pool = {}
    cfg.MODEL.ROI_HEADS.SOFTMAXES = True
    predictor = MonteCarloDropout(cfg, mcd_iterations, config['al_batch_size'])
    device = cfg.MODEL.DEVICE

    if len(pool_list) > 0:
        ## find the images from the pool_list the algorithm is most uncertain about
        for d in tqdm(range(len(pool_list))):
            filename = pool_list[d]
            img = cv2.imread(os.path.join(config['traindir'], filename))
            width, height = img.shape[:-1]
            outputs = predictor(img)

            obs = observations(outputs, config['iou_thres'])
            img_uncertainty = uncertainty(obs, mcd_iterations, max_entropy, width, height, device, mode) ## reduce the iterations when facing a "CUDA out of memory" error

            if not np.isnan(img_uncertainty):
                if len(pool) < pool_size:
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
    else:
        print("All images are used for the training, stopping the program...")

    return pool


def random_pooling(pool_list, pool_size, cfg, config, max_entropy, mcd_iterations, mode):
    pool = {}
    if len(pool_list) > 0:
        sample_list = random.sample(pool_list, k=pool_size)
        pool = {k:0.0 for k in sample_list}
    else:
        print("All images are used for the training, stopping the program...")

    return pool    


if __name__ == "__main__":
    logger.addHandler(file_handler)
    logger.info("Starting main-application")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./active_learning/config/maskAL.yaml', help='yaml with the training parameters')
    args = parser.parse_args()

    try:
        with open(args.config, 'rb') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error(f"Could not find configuration-file: {args.config}")
        sys.exit("Closing application")

    print("Configuration:")
    for key, value in config.items():
        print(key, ':', value)

    config = process_config_file(config, ['strategies', 'mode', 'pool_size', 'dropout_probability', 'iterations'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']
    gpu_num = len(config['cuda_visible_devices'])
    check_direxcist(config['dataroot'])

    if not config['train_complete_trainset']:
        weightsfolders, resultsfolders, csv_names = init_folders_and_files(config)
        remove_initial_training_set(config['dataroot'])
        prepare_initial_dataset(config['dataroot'], config['classes'], config['traindir'], config['valdir'], config['testdir'], config['initial_datasize'])
        
        if config['duplicate_initial_model']:
            ## train Mask R-CNN on the initial-dataset and duplicate the results on the other strategies
            cfg, dataset_dicts_train_init, trainer, val_value_init = Train_MaskRCNN(config, weightsfolders[0], gpu_num, 0, 0, config['dropout_probability'][0], init=True)

            ## do the evaluation with the same initial-weight-files
            for e, (weightsfolder, resultsfolder, csv_name) in enumerate(zip(weightsfolders, resultsfolders, csv_names)):
                if e == 0:
                    cfg_init = Eval_MaskRCNN(cfg, config, dataset_dicts_train_init, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, 0, init=True)
                else:
                    copy_initial_weight_file(weightsfolders[0], weightsfolder, 0)
                    cfg_init = Eval_MaskRCNN(cfg, config, dataset_dicts_train_init, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, 0, init=False)

                train_names = get_train_names(dataset_dicts_train_init, config['traindir'])
                write_train_files(train_names, resultsfolder, 0)

        max_entropy = calculate_max_entropy(config['classes'])

        ## active-learning strategies
        for i, (strategy, pool_size, mcd_iterations, mode, dropout_probability, weightsfolder, resultsfolder, csv_name) in enumerate(zip(config['strategies'], config['pool_size'], config['iterations'], config['mode'], config['dropout_probability'], weightsfolders, resultsfolders, csv_names)):
            if config['duplicate_initial_model']:
                ## load the initial_cfg to obtain the model-weights and image-names of the initial training
                cfg = cfg_init 
                train_names = get_train_names(dataset_dicts_train_init, config['traindir'])
                val_value = val_value_init
            else:
                ## train and evaluate Mask R-CNN on the initial dataset with different settings
                if i == 0:
                    cfg, dataset_dicts_train, trainer, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, 0, 0, dropout_probability, init=True)
                    cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, 0, init=True)
                else:
                    initial_train_names = get_initial_train_names(config)
                    update_train_dataset(config['dataroot'], config['traindir'], config['classes'], initial_train_names)
                    cfg, dataset_dicts_train, trainer, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, 0, 0, dropout_probability, init=False)
                    cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, 0, init=False)
                train_names = get_train_names(dataset_dicts_train, config['traindir'])
                write_train_files(train_names, resultsfolder, 0)
            
            
            ## do the iterative pooling
            for l in range(config['loops']):
                copy_previous_weights(weightsfolder, l+1)
                pool_list = create_pool_list(config, train_names)

                if strategy + '_pooling' in dir():
                    ## do the pooling (eval is a python-method that executes a function with a string-input)
                    pool = eval(strategy + '_pooling(pool_list, pool_size, cfg, config, max_entropy, mcd_iterations, mode)')

                    ## update the training list and retrain the algorithm
                    train_list = train_names + list(pool.keys())
                    update_train_dataset(config['dataroot'], config['traindir'], config['classes'], train_list)
                    cfg, dataset_dicts_train, trainer, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, l+1, val_value, dropout_probability, init=False)

                    ## evaluate and write the pooled image-names to a txt-file
                    cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, l+1, init=False)
                    train_names = get_train_names(dataset_dicts_train, config['traindir'])
                    write_train_files(train_names, resultsfolder, l+1)
                else:
                    logger.error(f"The {strategy}-strategy is not defined")
                    sys.exit("Closing application")


    else:
        ## train the complete train-pool to determine the base-line performance
        weightsfolder, resultsfolder, csv_name = init_folders_and_files(config)
        prepare_complete_dataset(config['dataroot'], config['classes'], config['traindir'], config['valdir'], config['testdir'])
        cfg, dataset_dicts_train, trainer, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, 0, 0, 0.5, init=True)
        cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, trainer, weightsfolder, resultsfolder, csv_name, gpu_num, 0, init=True)
        train_names = get_train_names(dataset_dicts_train, config['traindir'])
        write_train_files(train_names, resultsfolder, 0)

    logger.info("Active learning is finished!")
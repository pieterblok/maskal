# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-11-11 18:54:32

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
from itertools import chain
import pickle
from collections import OrderedDict, Counter
from tqdm import tqdm
from cerberus import Validator
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
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

## libraries that are specific for dropout training
from active_learning.strategies.dropout import FastRCNNConvFCHeadDropout, FastRCNNOutputLayersDropout, MaskRCNNConvUpsampleHeadDropout, Res5ROIHeadsDropout
from active_learning.sampling import observations, prepare_initial_dataset, prepare_initial_dataset_randomly, update_train_dataset, prepare_complete_dataset, calculate_repeat_threshold, calculate_iterations
from active_learning.sampling.montecarlo_dropout import MonteCarloDropout, MonteCarloDropoutHead
from active_learning.heuristics import uncertainty

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")
supported_annotation_formats = (".json", ".xml")


## initialize the logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s \n'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
file_handler = logging.StreamHandler()
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)


def check_config_file(config, config_filename, input_yaml):
    config_ok = True
    error_list = {}
    schema = {}
    
    try:
        with open(input_yaml, 'rb') as file:
            desired_inputs = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error(f"Could not find inputs.yaml file")
        sys.exit("Closing application")

    def check_network_config(field, value, error):
        if "mask_rcnn" not in value or not value.lower().endswith(".yaml"):
            error(field, "choose a Mask R-CNN config-file (.yaml) in the folder './configs'")

    def check_pretrained_weights(field, value, error):
        if not value.lower().endswith((".yaml", ".pth", ".pkl")):
            error(field, "load either the pretrained weights from the config-file (.yaml) or custom pretrained weights (.pth or .pkl)")

    threshold_list = ['confidence_threshold', 'nms_threshold', 'dropout_probability', 'iou_thres']
    for key, value in config.items():
        for key1, value1 in desired_inputs.items():
            if key == key1:
                if key == "network_config":
                    schema[key] = {'type': value1, 'check_with': check_network_config}
                elif key == "pretrained_weights":
                    schema[key] = {'type': value1, 'check_with': check_pretrained_weights}
                elif key == "repeat_factor_smallest_class":
                    schema[key] = {'type': value1, "min": 1}
                elif key == "learning_policy":
                    schema[key] = {'type': value1, 'allowed': ['steps_with_lrs', 'steps_with_decay', 'step', 'cosine_decay', 'exp_decay']}
                elif key == "train_sampler":
                    schema[key] = {'type': value1, 'allowed': ['TrainingSampler', 'RepeatFactorTrainingSampler']}
                elif key == "strategies":
                    schema[key] = {'type': value1, 'allowed': ['uncertainty', 'certainty', 'random']}
                elif key == "mode":
                    schema[key] = {'type': value1, 'allowed': ['mean', 'min']}
                elif key == "export_format":
                    schema[key] = {'type': value1, 'allowed': ['labelme', 'cvat', 'supervisely']}
                elif key in threshold_list:
                    schema[key] = {'type': value1, "min": 0, "max": 1}
                else:
                    schema[key] = {'type': value1}

    known_types = {
        'integer': int,
        'float': float,
        'string': str,
        'boolean': bool,
    }

    v = Validator(schema)
    for key, value in config.items():
        check = {}
        if isinstance(value, list):
            for key1, value1 in desired_inputs.items():
                if key == key1:
                    if not all(isinstance(v, known_types[value1[0]]) for v in value):
                        error_list.update({key: ["not all items are of type: " + str(value1[0])]})
                    if key == "strategies":
                        if not all(v in ['uncertainty', 'certainty', 'random'] for v in value):
                            error_list.update({key: ["choose 1 of these 3 options: 'uncertainty', 'certainty', 'random'"]})
                    if key == "mode":
                        if not all(v in ['mean', 'min'] for v in value):
                            error_list.update({key: ["choose 1 of these 2 options: 'mean', 'min'"]})
        else:
            check[key] = value
            if not v.validate(check):
                error_list.update(v.errors.copy())

    if error_list:
        config_ok = False
        print("")
        logger.error(f"Errors in the configuration-file: {config_filename}")
        for key, value in error_list.items():
            print(f"config['{key}']: {value}")
    
    return config_ok


def process_config_file(config, ints_to_lists):
    lengths = []

    nest = False
    if not config['equal_pool_size']:
        nest = True

    for il in range(len(ints_to_lists)):
        int_to_list = ints_to_lists[il]
        if nest:
            config[int_to_list] = [config[int_to_list]]
        else:
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
    weightsfolders = []
    resultsfolders = []
    csv_names = []

    counts = Counter(config['strategies'])
    counts = list(counts.values())
    duplicates = any(x > 1 for x in counts)
    hybrid_count = 0
        
    for s, (strategy, mode, dropout_probability, mcd_iterations, pool_size) in enumerate(zip(config['strategies'], config['mode'], config['dropout_probability'], config['mcd_iterations'], config['pool_size'])):
        if duplicates:
            if isinstance(pool_size, list):
                hybrid_count += 1
                pool_size = "hybrid{:02d}".format(hybrid_count)
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

    return weightsfolders, resultsfolders, csv_names


def remove_initial_training_set(dataroot):
    if os.path.exists(os.path.join(dataroot, "initial_train.txt")):
        os.remove(os.path.join(dataroot, "initial_train.txt"))


def store_initial_files(cfg, config, dataset_dicts_train_init, val_value_init, weightsfolders):
    for wf in range(len(weightsfolders)):
        with open(os.path.join(weightsfolders[wf], "cfg_init.yaml"), "w") as f1:
            f1.write(cfg.dump())
        with open(os.path.join(weightsfolders[wf], 'val_value_init.pkl'), 'wb') as f2:
            pickle.dump(val_value_init, f2)

    with open(os.path.join(config['dataroot'], 'dataset_dicts_train_init.pkl'), 'wb') as f3:
        pickle.dump(dataset_dicts_train_init, f3)


def load_initial_files(config, weightsfolders):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(weightsfolders[0], "cfg_init.yaml"))
    with open(os.path.join(weightsfolders[0], 'val_value_init.pkl'), 'rb') as f1:
        val_value_init = pickle.load(f1)
    with open(os.path.join(config['dataroot'], 'dataset_dicts_train_init.pkl'), 'rb') as f2:
        dataset_dicts_train_init = pickle.load(f2)
    return cfg, dataset_dicts_train_init, val_value_init
    

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


def write_train_files(train_names, writefolder, iteration, pool={}):
    write_txt_name = "trainfiles_iteration{:03d}.txt".format(iteration)
    with open(os.path.join(writefolder, write_txt_name), 'w') as filehandle:
        for train_name in train_names:
            if bool(pool) == True:
                written = False
                for name, val in pool.items():
                    if name == train_name:
                        filehandle.write("{:s}, {:.6f}\n".format(name, val))
                        written = True
                if written == False:
                    filehandle.write("{:s}, NaN\n".format(train_name))
            else:
                filehandle.write("{:s}, NaN\n".format(train_name))
    filehandle.close()


def move_initial_train_dir(initial_train_dir, traindir, export):
    if export == "images":
        fileext = supported_cv2_formats
    elif export == "annotations":
        fileext = supported_annotation_formats

    all_files = os.listdir(initial_train_dir)
    for cur_file in all_files:
        if cur_file.lower().endswith(fileext):
            copyfile(os.path.join(initial_train_dir, cur_file), os.path.join(traindir, cur_file))


def copy_initial_weight_file(read_folder, weightsfolders, iter):
    weight_file = "best_model_{:s}.pth".format(str(iter).zfill(3))
    for wf in range(1, len(weightsfolders)):
        write_folder = weightsfolders[wf]
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
    cfg.merge_from_file(model_zoo.get_config_file(config['network_config']))
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = dropout_probability
    cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY = dropout_probability

    if any(x in config['network_config'] for x in ["FPN", "DC5"]):
        cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHeadDropout'
        cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsDropout'
    elif any(x in config['network_config'] for x in ["C4"]):
        cfg.MODEL.ROI_HEADS.NAME = 'Res5ROIHeadsDropout'
        
    cfg.MODEL.ROI_MASK_HEAD.NAME = 'MaskRCNNConvUpsampleHeadDropout'
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False
    cfg.OUTPUT_DIR = weightsfolder


    ## initialize the network weights, with an option to do the transfer-learning on previous models
    if config['transfer_learning_on_previous_models'] == True:
        if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter-1).zfill(3)))):
            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter-1).zfill(3)))
        else:
            if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "model_final.pth")):
                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            else:
                if config['pretrained_weights'].lower().endswith(".yaml"):
                    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['pretrained_weights'])
                elif config['pretrained_weights'].lower().endswith((".pth", ".pkl")):
                    cfg.MODEL.WEIGHTS = config['pretrained_weights']
    else:
        if config['pretrained_weights'].lower().endswith(".yaml"):
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['pretrained_weights'])
        elif config['pretrained_weights'].lower().endswith((".pth", ".pkl")):
            cfg.MODEL.WEIGHTS = config['pretrained_weights']


    ## initialize the train-sampler
    cfg.DATALOADER.SAMPLER_TRAIN = config['train_sampler']
    if cfg.DATALOADER.SAMPLER_TRAIN == 'RepeatFactorTrainingSampler':
        repeat_threshold = calculate_repeat_threshold(config, dataset_dicts_train)
        cfg.DATALOADER.REPEAT_THRESHOLD = repeat_threshold
        
    max_iterations, steps = calculate_iterations(config, dataset_dicts_train)

    ## initialize the training parameters  
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
    cfg.SOLVER.MAX_ITER = max_iterations
    cfg.SOLVER.STEPS = steps

    if config['checkpoint_period'] == -1:
        cfg.SOLVER.CHECKPOINT_PERIOD = (cfg.SOLVER.MAX_ITER+1)
    else:
        cfg.SOLVER.CHECKPOINT_PERIOD = config['checkpoint_period']

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

    return cfg, dataset_dicts_train, val_value_output


def Eval_MaskRCNN(cfg, config, dataset_dicts_train, weightsfolder, resultsfolder, csv_name, iter, init):      
    if init:
        register_coco_instances("test", {}, os.path.join(config['dataroot'], "test.json"), config['testdir'])
        test_metadata = MetadataCatalog.get("test")
        dataset_dicts_test = DatasetCatalog.get("test")

    cfg.OUTPUT_DIR = weightsfolder

    if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter).zfill(3)))):
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter).zfill(3)))
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['confidence_threshold']   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    cfg.DATASETS.TEST = ("test",)

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", ("bbox", "segm"), False, output_dir=resultsfolder)
    val_loader = build_detection_test_loader(cfg, "test")
    eval_results = inference_on_dataset(model, val_loader, evaluator)
    
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
    predictor = MonteCarloDropoutHead(cfg, mcd_iterations)
    device = cfg.MODEL.DEVICE

    if len(pool_list) > 0:
        ## find the images from the pool_list the algorithm is most uncertain about
        for d in tqdm(range(len(pool_list))):
            filename = pool_list[d]
            if os.path.isfile(os.path.join(config['traindir'], filename)):
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
    predictor = MonteCarloDropoutHead(cfg, mcd_iterations)  
    device = cfg.MODEL.DEVICE

    if len(pool_list) > 0:
        ## find the images from the pool_list the algorithm is most uncertain about
        for d in tqdm(range(len(pool_list))):
            filename = pool_list[d]
            if os.path.isfile(os.path.join(config['traindir'], filename)):
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
    parser.add_argument('--config', type=str, default='maskAL.yaml', help='yaml with the training parameters')
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

    config_ok = check_config_file(config, args.config, 'types.yaml')
    if not config_ok: 
        sys.exit("Closing application")

    config = process_config_file(config, ['strategies', 'mode', 'equal_pool_size', 'pool_size', 'dropout_probability', 'mcd_iterations', 'loops'])
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']
    gpu_num = len(config['cuda_visible_devices'])
    check_direxcist(config['dataroot'])

    weightsfolders, resultsfolders, csv_names = init_folders_and_files(config)
    remove_initial_training_set(config['dataroot'])
    max_entropy = calculate_max_entropy(config['classes'])
    if config['use_initial_train_dir']:
        move_initial_train_dir(config['initial_train_dir'], config['traindir'], "images")
        prepare_initial_dataset(config)
        move_initial_train_dir(config['initial_train_dir'], config['traindir'], "annotations")
    else:
        prepare_initial_dataset_randomly(config)
        
    ## active-learning
    for i, (strategy, equal_pool_size, pool_size, mcd_iterations, mode, dropout_probability, loops, weightsfolder, resultsfolder, csv_name) in enumerate(zip(config['strategies'], config['equal_pool_size'], config['pool_size'], config['mcd_iterations'], config['mode'], config['dropout_probability'], config['loops'], weightsfolders, resultsfolders, csv_names)):
        ## train and evaluate Mask R-CNN on the initial dataset
        if i == 0:
            cfg, dataset_dicts_train, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, 0, 0, dropout_probability, init=True)
            cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, weightsfolder, resultsfolder, csv_name, 0, init=True)
        else:
            initial_train_names = get_initial_train_names(config)
            update_train_dataset(config, cfg, initial_train_names)
            cfg, dataset_dicts_train, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, 0, 0, dropout_probability, init=False)
            cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, weightsfolder, resultsfolder, csv_name, 0, init=False)
        train_names = get_train_names(dataset_dicts_train, config['traindir'])
        write_train_files(train_names, resultsfolder, 0)
        
        if not equal_pool_size:
            pool_size_list = list(chain.from_iterable([[pool_size[ll]] * loops[ll] for ll in range(len(loops))]))
            loops = sum(loops)
        
        ## do the iterative pooling
        for l in range(loops):
            copy_previous_weights(weightsfolder, l+1)
            pool_list = create_pool_list(config, train_names)

            if not equal_pool_size:
                pool_size = pool_size_list[l]

            if strategy + '_pooling' in dir():
                ## do the pooling (eval is a python-method that executes a function with a string-input)
                pool = eval(strategy + '_pooling(pool_list, pool_size, cfg, config, max_entropy, mcd_iterations, mode)')

                ## update the training list and retrain the algorithm
                train_list = train_names + list(pool.keys())
                update_train_dataset(config, cfg, train_list)
                cfg, dataset_dicts_train, val_value = Train_MaskRCNN(config, weightsfolder, gpu_num, l+1, val_value, dropout_probability, init=False)

                ## evaluate and write the pooled image-names to a txt-file
                cfg = Eval_MaskRCNN(cfg, config, dataset_dicts_train, weightsfolder, resultsfolder, csv_name, l+1, init=False)
                train_names = get_train_names(dataset_dicts_train, config['traindir'])
                write_train_files(train_names, resultsfolder, l+1, pool)
            else:
                logger.error(f"The {strategy}-strategy is not defined")
                sys.exit("Closing application")

    logger.info("Active learning is finished!")
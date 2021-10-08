# @Author: Pieter Blok
# @Date:   2021-03-25 15:03:44
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-10-08 09:14:31

# This function is inspired by the uncertainty_aware_dropout function:
# https://github.com/RovelMan/active-learning-framework/blob/master/al_framework/strategies/dropout.py

import numpy as np
import torch

def observations(outputs, iou_thres=0.5):
    """
    To cluster the segmentations for the different Monte-Carlo runs
    """
    observations = {}
    obs_id = 0

    for i in range(len(outputs)):
        sample = outputs[i]
        detections = len(sample['instances'])
        dets = sample['instances'].get_fields()
        
        for det in range(detections):
            if not observations:
                detection = {}
                for key, val in dets.items():
                    detection[key] = val[det]
                observations[obs_id] = [detection]

            else:
                addThis = None
                for group, ds, in observations.items():
                    for d in ds:
                        thisMask = dets['pred_masks'][det]
                        otherMask = d['pred_masks']
                        overlap = torch.logical_and(thisMask, otherMask)
                        union = torch.logical_or(thisMask, otherMask)
                        IOU = overlap.sum()/float(union.sum())
                        if IOU <= iou_thres:
                            break
                        else:
                            detection = {}
                            for key, val in dets.items():
                                detection[key] = val[det]
                            addThis = [group, detection]
                            break
                    if addThis:
                        break
                if addThis:
                    observations[addThis[0]].append(addThis[1])
                else:
                    obs_id += 1
                    detection = {}
                    for key, val in dets.items():
                        detection[key] = val[det]
                    observations[obs_id] = [detection]

    return observations
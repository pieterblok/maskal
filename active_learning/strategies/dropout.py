# @Author: Pieter Blok
# @Date:   2021-03-22 09:43:07
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2021-11-08 15:09:29

#!/usr/bin/env python

## based on: detectron2.modeling.roi_heads.box_head
## based on: detectron2.modeling.roi_heads.fast_rcnn

import torch
from torch import nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Union
from fvcore.nn import giou_loss, smooth_l1_loss
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable, get_cfg
from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple, get_norm
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

## these specific libraries are needed to alter the network architecture
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads, Res5ROIHeads
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference, _log_classification_stats
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY, BaseMaskRCNNHead


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHeadDropout(nn.Sequential):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="",
        dropout_probability: float = 0.5,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            dropout = nn.Dropout(p=dropout_probability)
            self.add_module("dropout{}".format(k + 1), dropout)
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
            "dropout_probability": cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY,
        }

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    @property
    @torch.jit.unused
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])



class FastRCNNOutputLayersDropout(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        softmaxes: bool = False,
        dropout_probability: float = 0.5,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.dropout1 = nn.Dropout(p=dropout_probability)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.dropout2 = nn.Dropout(p=dropout_probability)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.softmaxes = softmaxes


    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            "softmaxes"             : cfg.MODEL.ROI_HEADS.SOFTMAXES,
            "dropout_probability"   : cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY,
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        x = self.dropout1(x)
        scores = self.cls_score(x)
        x = self.dropout2(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.softmaxes,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)



@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsDropout(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape, box_predictor=FastRCNNOutputLayersDropout(cfg, 2048))
    


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsDropout(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape, box_predictor=FastRCNNOutputLayersDropout(cfg, 1024))



@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHeadDropout(BaseMaskRCNNHead, nn.Sequential):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, conv_dims, conv_norm="", 
            dropout_probability: float = 0.5, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim

        self.dropout1 = nn.Dropout(p=dropout_probability)

        self.deconv = ConvTranspose2d(
            cur_channels, conv_dims[-1], kernel_size=2, stride=2, padding=0
        )
        self.add_module("deconv_relu", nn.ReLU())
        cur_channels = conv_dims[-1]

        self.dropout2 = nn.Dropout(p=dropout_probability)

        self.predictor = Conv2d(cur_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
        )
        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        ret["dropout_probability"] = cfg.MODEL.ROI_MASK_HEAD.DROPOUT_PROBABILITY
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x
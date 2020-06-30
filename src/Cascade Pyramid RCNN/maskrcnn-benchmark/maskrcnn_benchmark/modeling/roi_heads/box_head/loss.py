# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss,sine_delta_mse_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_proposals(self, proposal, target):
        # print(target.get_field("rotations"), '===================================')
        # print(proposal,target,'++++')
        match_quality_matrix = boxlist_iou(target, proposal, type=0)  # calculate the iter/ area1  !!!!!

        # print(match_quality_matrix[match_quality_matrix>0],'++++++++++')
        # print(target.get_field('rotations'),'==========')
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # print(matched_idxs[matched_idxs>-1],'++++++++++')
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "rotations"])

        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        # print(target.get_field('rotations'),target.bbox,'=======================')
        matched_targets = target[matched_idxs.clamp(min=0)]
        # print(matched_targets.get_field('rotations').size(), '========matched===============')
        matched_targets.add_field("matched_idxs", matched_idxs)

        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        orien_targets = []
        # print(list(target.get_field("rotations") for target in targets), '===================================')
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets  OFFSETs!!!

            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            # for a,b,c in zip(matched_targets.bbox,proposals_per_image.bbox,regression_targets_per_image):
            #     print(a,'///\n',b,'///\n',c,'======================')
            # positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)
            # compute orientation targets=======================================

            # orien_targets_per_image = matched_targets.get_field("rotations") - proposals_per_image.get_field(
                # "rotations")
            orien_targets_per_image = matched_targets.get_field("rotations")   # shall not be orien offset due to axis-aligned proposals!

            # orien_targets_per_image = orien_targets_per_image[positive_inds]

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            orien_targets.append(orien_targets_per_image)
        # print(orien_targets, '===================================')
        return labels, regression_targets, orien_targets

    def subsample(self, proposals, targets):
        # for target in targets:
        #     print(target.get_field('rotations'), '==========')
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, orien_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        # print(orien_targets,'======================================')

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes============
        for labels_per_image, regression_targets_per_image, orien_targets_per_image, proposals_per_image in zip(
                labels, regression_targets, orien_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            # print(labels_per_image,'===')
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            # ===================================================================
            proposals_per_image.add_field("rotations", orien_targets_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        # print(proposals_per_image.get_field("rotations"),"===========================================")
        return proposals

    def __call__(self, class_logits, box_regression, box_orien):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # for target in box_orien:
        # print(box_orien, '==========')

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        orien_regression = cat(box_orien, dim=0)
        device = class_logits.device
        # print(class_logits.size(), '\n',box_regression.size(), '\n', orien_regression.size(),'\n=================================')

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        # print(labels, '==========================')
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        # =========================================================
        orien_targets = cat(
            [proposal.get_field("rotations") for proposal in proposals], dim=0
        )
        # print([proposal.get_field("regression_targets") for proposal in proposals],'===========')

        classification_loss = F.cross_entropy(class_logits, labels)
        # classification_loss = focal_loss_class(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        # print(labels)
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)
        # print(regression_targets[sampled_pos_inds_subset],'====================')
        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            # reduction='sum',
            size_average=False,
            beta=1,
        )

        # print(sampled_pos_inds_subset,'\n',map_inds)
        # orien_loss = smooth_l1_loss(
        #     orien_regression[sampled_pos_inds_subset],
        #     orien_targets[sampled_pos_inds_subset].type(torch.cuda.FloatTensor),
        #     size_average=False,
        #     beta=1.0 / 9,
        # )
        orien_loss = F.mse_loss(
            orien_regression[sampled_pos_inds_subset],
            orien_targets[sampled_pos_inds_subset].type(torch.cuda.FloatTensor),
            # orien_targets[sampled_pos_inds_subset],
            reduction='sum',
            # size_average=False,
            # beta=1,
        )
        # if torch.isnan(orien_loss) > 0:   # catch nan
        #     print(orien_targets[sampled_pos_inds_subset], '=========target===========')
        #     print(orien_regression[sampled_pos_inds_subset], '=========regression===========\n')
        #     print(sampled_pos_inds_subset,'===subsetidx')
        #     print(            box_regression[sampled_pos_inds_subset[:, None], map_inds],
        #     regression_targets[sampled_pos_inds_subset],'============box')
        #     print(orien_loss, '///', labels.numel(), '=============')
        box_loss = box_loss / labels.numel()
        orien_loss = orien_loss / labels.numel()

        return classification_loss, box_loss, orien_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    loss_evaluator = FastRCNNLossComputation(matcher, fg_bg_sampler, box_coder)

    return loss_evaluator


def focal_loss_class(predictions, targets, alpha=1, gamma=2):
    loss_class = torch.nn.NLLLoss()
    predictions = F.log_softmax(predictions, dim=1)  # log(Pt)
    # print(predictions,targets,'=====================')
    return alpha * loss_class(predictions.mul((1 - predictions.exp()).pow(gamma)), targets)

# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import datetime
from mmpose.registry import METRICS
from ..functional import oks_nms, soft_oks_nms
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Optional, Sequence, Union

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS
from ..functional import (keypoint_auc, keypoint_epe, keypoint_nme,
                          keypoint_pck_accuracy, keypoint_pck_accuracy_acupoint)


@METRICS.register_module()
class CocoMetric_PCK(BaseMetric):
    """PCK accuracy evaluation metric.
        Calculate the pose accuracy of Percentage of Correct Keypoints (PCK) for
        each individual keypoint and the averaged accuracy across all keypoints.
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the person bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.
        Note:
            - length of dataset: N
            - num_keypoints: K
            - number of keypoint dimensions: D (typically D = 2)
        Args:
            thr(float): Threshold of PCK calculation. Default: 0.05.
            norm_item (str | Sequence[str]): The item used for normalization.
                Valid items include 'bbox', 'head', 'torso', which correspond
                to 'PCK', 'PCKh' and 'tPCK' respectively. Default: ``'bbox'``.
            collect_device (str): Device name used for collecting results from
                different ranks during distributed training. Must be ``'cpu'`` or
                ``'gpu'``. Default: ``'cpu'``.
            prefix (str, optional): The prefix that will be added in the metric
                names to disambiguate homonymous metrics of different evaluators.
                If prefix is not provided in the argument, ``self.default_prefix``
                will be used instead. Default: ``None``.

        Examples:
            >>> from mmpose.evaluation.metrics import PCKAccuracy
            >>> import numpy as np
            >>> from mmengine.structures import InstanceData
            >>> num_keypoints = 15
            >>> keypoints = np.random.random((1, num_keypoints, 2)) * 10
            >>> gt_instances = InstanceData()
            >>> gt_instances.keypoints = keypoints
            >>> gt_instances.keypoints_visible = np.ones(
            ...     (1, num_keypoints, 1)).astype(bool)
            >>> gt_instances.bboxes = np.random.random((1, 4)) * 20
            >>> pred_instances = InstanceData()
            >>> pred_instances.keypoints = keypoints
            >>> data_sample = {
            ...     'gt_instances': gt_instances.to_dict(),
            ...     'pred_instances': pred_instances.to_dict(),
            ... }
            >>> data_samples = [data_sample]
            >>> data_batch = [{'inputs': None}]
            >>> pck_metric = PCKAccuracy(thr=0.5, norm_item='bbox')
            ...: UserWarning: The prefix is not set in metric class PCKAccuracy.
            >>> pck_metric.process(data_batch, data_samples)
            >>> pck_metric.evaluate(1)
            10/26 15:37:57 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...  # noqa
            {'PCK': 1.0}

        """

    def __init__(self,
                 thr: float = 0.0027,  # 0.05
                 norm_item: Union[str, Sequence[str]] = 'bbox',
                 collect_device: str = 'cpu',
                 ann_file: Optional[str] = None,
                 prefix: Optional[str] = None,
                 fix_image_shape=(4096, 2160)) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.thr = thr
        self.fix_image_shape = fix_image_shape
        self.results = []
        self.ann_file = ann_file
        self.norm_item = norm_item if isinstance(norm_item,
                                                 (tuple,
                                                  list)) else [norm_item]
        allow_normalized_items = ['bbox', 'head', 'torso']

        for item in self.norm_item:
            if item not in allow_normalized_items:
                raise KeyError(
                    f'The normalized item {item} is not supported by '
                    f"{self.__class__.__name__}. Should be one of 'bbox', "
                    f"'head', 'torso', but got {item}.")

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:
            # predicted keypoints coordinates, [1, K, D]
            pred_coords = data_sample['pred_instances']['keypoints']
            # ground truth data_info
            gt = data_sample['gt_instances']
            # ground truth keypoints coordinates, [1, K, D]
            gt_coords = gt['keypoints']
            # ground truth keypoints_visible, [1, K, 1]
            mask = gt['keypoints_visible'].astype(bool).reshape(1, -1)

            result = {
                'pred_coords': pred_coords,
                'gt_coords': gt_coords,
                'mask': mask,
            }

            # # 自定义获取参数的方法
            # pred_coords = data_sample.pred_instances.keypoints
            # # ground truth data_info
            # gt = data_sample.gt_instances
            # # ground truth keypoints coordinates, [1, K, D]
            # gt_coords = gt.keypoints
            # # ground truth keypoints_visible, [1, K, 1]
            # mask = gt.keypoints_visible.astype(bool).reshape(1, -1)
            #
            # result = {
            #     'pred_coords': pred_coords,
            #     'gt_coords': gt_coords,
            #     'mask': mask,
            # }

            if 'bbox' in self.norm_item:
                assert 'bboxes' in gt, 'The ground truth data info do not ' \
                                       'have the expected normalized_item ``"bbox"``.'
                # ground truth bboxes, [1, 4]
                bbox_size_ = np.max(gt['bboxes'][0][2:] - gt['bboxes'][0][:2])
                bbox_size = np.array([bbox_size_, bbox_size_]).reshape(-1, 2)
                result['bbox_size'] = bbox_size

            if 'head' in self.norm_item:
                assert 'head_size' in gt, 'The ground truth data info do ' \
                                          'not have the expected normalized_item ``"head_size"``.'
                # ground truth bboxes
                head_size_ = gt['head_size']
                head_size = np.array([head_size_, head_size_]).reshape(-1, 2)
                result['head_size'] = head_size

            if 'torso' in self.norm_item:
                # used in JhmdbDataset
                torso_size_ = np.linalg.norm(gt_coords[0][4] - gt_coords[0][5])
                if torso_size_ < 1:
                    torso_size_ = np.linalg.norm(pred_coords[0][4] -
                                                 pred_coords[0][5])
                    warnings.warn('Ground truth torso size < 1. '
                                  'Use torso size from predicted '
                                  'keypoint results instead.')
                torso_size = np.array([torso_size_,
                                       torso_size_]).reshape(-1, 2)
                result['torso_size'] = torso_size

            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
            The returned result dict may have the following keys:
                - 'PCK': The pck accuracy normalized by `bbox_size`.
                - 'PCKh': The pck accuracy normalized by `head_size`.
                - 'tPCK': The pck accuracy normalized by `torso_size`.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # pred_coords: [N, K, D]
        pred_coords = np.concatenate(
            [result['pred_coords'] for result in results])
        # gt_coords: [N, K, D]
        gt_coords = np.concatenate([result['gt_coords'] for result in results])
        # mask: [N, K]
        mask = np.concatenate([result['mask'] for result in results])

        metrics = dict()
        if 'bbox' in self.norm_item:
            norm_size_bbox = np.concatenate(
                [result['bbox_size'] for result in results])

            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"bbox_size"``)...')

            # acc, pck, cnt = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
            #                                       10.24, norm_size_bbox) #self.thr
            # metrics['Acc'] = acc
            # metrics['PCK'] = pck
            # metrics['Cnt'] = cnt

            acc_5mm, ave_acc_5mm, acc_10mm, ave_acc_10mm, cnt = keypoint_pck_accuracy_acupoint(pred_coords, gt_coords,
                                                                                               mask,
                                                                                               10.24, np.ones_like(
                    norm_size_bbox) * max(self.fix_image_shape))  # 0.0025
            acupoint_name = self.dataset_meta["CLASSES"][0]["keypoints"]
            # acupoint_name = ['SP10_L','SP10_R','ST41_L','ST41_R',
            #                  'CV4','CV6','CV8','CV10','CV12','BL40_L','BL40_R','GV14']

            # # 绘制5mm柱状图
            #
            # plt.bar(acupoint_name, acc_5mm,width=0.5)
            # for i in range(len(acupoint_name)):
            #     plt.text(acupoint_name[i], acc_5mm[i], f'{acc_5mm[i]:.2f}', ha='center', va='bottom')
            # plt.title('5mmPCK')
            # plt.xlabel('acupoint_name')
            # plt.ylabel('5mm_PCK')
            # plt.xticks(fontsize=6,rotation=60)
            # plt.savefig(rf'/home/hantang/usr/acupoint/mmpose/out_putdir/save_result_pic/{datetime.datetime.now()}5mm_PCK.jpg',dpi=1200)
            # plt.show()
            #
            # # 绘制10mm柱状图
            # plt.bar(acupoint_name, acc_10mm,width=0.5)
            # for i in range(len(acupoint_name)):
            #     plt.text(acupoint_name[i], acc_10mm[i], f'{acc_10mm[i]:.2f}', ha='center', va='bottom')
            # plt.title('10mmPCK')
            # plt.xlabel('acupoint_name')
            # plt.xticks(fontsize=6,rotation="vertical")
            #
            # plt.ylabel('10mm_PCK')
            # plt.savefig(rf'/home/hantang/usr/acupoint/mmpose/out_putdir/save_result_pic/{datetime.datetime.now()}10mm_PCK.jpg',dpi=1200)
            #
            # plt.show()
            #
            # x = ["ave_acc_5mm","ave_acc_10mm"]
            # y = [ave_acc_5mm,ave_acc_10mm]
            # plt.bar(x, y,width=0.5)
            # for i in range(len(x)):
            #     plt.text(x[i], y[i], f'{y[i]:.2f}', ha='center', va='bottom')
            # plt.title('5mm 10mm  ave_PCK')
            # plt.xlabel('name')
            # plt.ylabel('PCK')
            # plt.savefig(rf'/home/hantang/usr/acupoint/mmpose/out_putdir/save_result_pic/{datetime.datetime.now()}_5mm10mm_avePCK.jpg',dpi=1200)
            #
            # plt.show()


            acc_5mm = [f"{x}:{y}" for x, y in zip(acupoint_name, acc_5mm)]
            acc_10mm = [f"{x}:{y}" for x, y in zip(acupoint_name, acc_10mm)]
            metrics['Acc_5mm'] = acc_5mm
            metrics['PCK_5mm'] = ave_acc_5mm
            metrics['Acc_10mm'] = acc_10mm
            metrics['PCK_10mm'] = ave_acc_10mm
            metrics['Cnt'] = cnt

        if 'head' in self.norm_item:
            norm_size_head = np.concatenate(
                [result['head_size'] for result in results])

            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"head_size"``)...')

            _, pckh, _ = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                               self.thr, norm_size_head)
            metrics['PCKh'] = pckh

        if 'torso' in self.norm_item:
            norm_size_torso = np.concatenate(
                [result['torso_size'] for result in results])

            logger.info(f'Evaluating {self.__class__.__name__} '
                        f'(normalized by ``"torso_size"``)...')

            acc, tpck, ant = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                                   self.thr, norm_size_torso)
            acc, tpck, ant = keypoint_pck_accuracy(pred_coords, gt_coords, mask,
                                                   self.thr, norm_size_torso)
            metrics['tPCK'] = tpck
            metrics['Acc'] = acc
            metrics['Ant'] = ant

        return metrics

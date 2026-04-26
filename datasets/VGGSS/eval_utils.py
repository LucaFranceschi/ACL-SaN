import torch
import numpy as np
from sklearn import metrics as mt
from typing import List, Optional, Tuple, Dict
import copy


class Evaluator(object):
    def __init__(self) -> None:
        """
        Initialize the VGG-Sound Source (VGG-SS) Evaluator.

        Attributes:
            ciou (List[float]): Buffer of cIoU values.
            AUC (List[float]): Buffer of AUC values.
            N (int): Counter for the number of evaluations.
            metrics (List[str]): List of metric names.
        """
        super(Evaluator, self).__init__()
        self.std_metrics = {
            'cIoU': [],
            'metrics': {
                'AUC': None,
                'cIoU_ap50': None,
                'cIoU_hat': None
            }
        }
        self.silence_metrics = {
            'pIA': [],
            'metrics': {
                'AUC_N': None,
                'pIA_ap50': None,
                'pIA_hat': None
            }
        }
        self.noise_metrics = copy.deepcopy(self.silence_metrics)

        self.offscreen_metrics = {
            'cIoU': [],
            'pIA': [],
            'metrics': {
                'AUC': None,
                'cIoU_ap50': None,
                'cIoU_hat': None,
                'AUC_N': None,
                'pIA_ap50': None,
                'pIA_hat': None
            }
        }

    def evaluate_batch(self, heatmap: torch.Tensor, target: torch.Tensor, thr: Optional[float] = None, **kwargs) -> None:

        """
        Evaluate a batch of predictions against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground truth maps.
            thr (Optional[float]): Threshold for binary classification. If None, dynamically determined.

        Returns:
            None
        """
        self._evaluate_batch(heatmap, 'pos', thr, target)

        sil_heatmap = kwargs.get('silence_heatmap', None)
        if sil_heatmap != None:
            self._evaluate_batch(sil_heatmap, 'sil', thr, target)

        noise_heatmap = kwargs.get('noise_heatmap', None)
        if noise_heatmap != None:
            self._evaluate_batch(noise_heatmap, 'noi', thr, target)

        offscreen_heatmap = kwargs.get('offscreen_heatmap', None)
        if offscreen_heatmap != None:
            self._evaluate_batch(offscreen_heatmap, 'off', thr, target)


    def _evaluate_batch(self, heatmap, metric, thr_param, gt):
        for i in range(heatmap.size(0)):
            pred = heatmap[i].detach().cpu()
            target = gt[i].cpu()
            if thr_param is None:
                thr = np.sort(pred.flatten())[int(pred.shape[1] * pred.shape[2]) // 2]
            elif thr_param == 'adap':
                gt_nums = (target!=0).sum()
                if int(gt_nums) == 0:
                    gt_nums = int(target.shape[1] * target.shape[2]) // 2
                thr = np.sort(pred.flatten())[int(pred.shape[1] * pred.shape[2]) - int(gt_nums)] # adap
            else:
                thr = thr_param

            if metric in ('sil', 'noi'):
                self.cal_pIA(pred, metric, thr)
            elif metric == 'pos':
                self.cal_CIOU(pred, target, metric, thr)
            elif metric == 'off':
                self.cal_pIA(pred, metric, thr)
                self.cal_CIOU(pred, target, metric, thr)

    def cal_CIOU(self, infer: torch.Tensor, gtmap: torch.Tensor, metric, thres: float = 0.01):
        """
        Calculate cIoU (consensus Intersection over Union).

        Args:
            infer (torch.Tensor): Model prediction.
            gtmap (torch.Tensor): Ground truth map.
            thres (float): Threshold for binary classification.

        Returns:
            List[float]: List of cIoU values for each instance in the batch.
        """
        infer_map = torch.zeros_like(gtmap)
        infer_map[infer >= thres] = 1
        ciou = (infer_map * gtmap).sum(2).sum(1) / (gtmap.sum(2).sum(1) + (infer_map * (gtmap == 0)).sum(2).sum(1) + 1e-12)
        ciou = ciou.detach().cpu().float()

        if metric == 'pos':
            self.std_metrics['cIoU'].append(ciou)
        elif metric == 'off':
            self.offscreen_metrics['cIoU'].append(ciou)
        return

    def cal_pIA(self, infer: torch.Tensor, metric: str, thres: float = 0.01):
        '''
        Calculate the percentage of Image Area as described in:
            Juanola, Xavier, et al. "Learning from Silence and Noise for Visual Sound Source Localization."

        :param self: Description
        '''
        infer_map = torch.zeros_like(infer)
        infer_map[infer >= thres] = 1

        shape = infer_map.shape

        pIA = torch.sum(infer_map.detach().cpu(), dim=(1, 2)).float() / (shape[1] * shape[2])

        if metric == 'sil':
            self.silence_metrics['pIA'].append(pIA)
        elif metric == 'noi':
            self.noise_metrics['pIA'].append(pIA)
        elif metric == 'off':
            self.offscreen_metrics['pIA'].append(pIA)
        return

    def finalize_AUC(self):
        """
        Calculate the Area Under the Curve (AUC).

        Returns:
            float: AUC value.
        """
        for metric in [self.std_metrics, self.offscreen_metrics]:
            if len(metric['cIoU']) > 0:
                cious = [np.sum(np.array(metric['cIoU']) >= 0.05 * i) / len(metric['cIoU'])
                        for i in range(21)]
                thr = [0.05 * i for i in range(21)]
                auc = mt.auc(thr, cious)
                metric['metrics']['AUC'] = auc

        for metric in [self.silence_metrics, self.noise_metrics, self.offscreen_metrics]:
            if len(metric['pIA']) > 0:
                aucs = [np.sum(np.array(metric['pIA']) < 0.05 * i) / len(metric['pIA']) for i in range(21)]
                thr = [0.05 * i for i in range(21)]
                auc = mt.auc(thr, aucs)
                metric['metrics']['AUC_N'] = auc

    def finalize_AP50(self):
        """
        Calculate Average Precision (cIoU@0.5).

        Returns:
            float: cIoU@0.5 value.
        """
        for metric in [self.std_metrics, self.offscreen_metrics]:
            if len(metric['cIoU']) > 0:
                ap50 = np.mean(np.array(metric['cIoU']) >= 0.5)
                metric['metrics']['cIoU_ap50'] = ap50

        for metric in [self.silence_metrics, self.noise_metrics, self.offscreen_metrics]:
            if len(metric['pIA']) > 0:
                ap50 = np.mean(np.array(metric['pIA']) < 0.5)
                metric['metrics']['pIA_ap50'] = ap50

    def finalize_means(self):
        """
        Calculate mean cIoU.

        Returns:
            float: Mean cIoU value.
        """
        for metric in [self.std_metrics, self.offscreen_metrics]:
            if len(metric['cIoU']) > 0:
                ciou = np.mean(np.array(metric['cIoU']))
                metric['metrics']['cIoU_hat'] = ciou

        for metric in [self.silence_metrics, self.noise_metrics, self.offscreen_metrics]:
            if len(metric['pIA']) > 0:
                pia = np.mean(np.array(metric['pIA']))
                metric['metrics']['pIA_hat'] = pia

    def finalize(self):
        """
        Finalize and return evaluation metrics.

        Returns:
            Tuple[List[str], Dict[str, float]]: List of metric names and corresponding values.
        """
        self.finalize_AUC()
        self.finalize_AP50()
        self.finalize_means()
        return self.std_metrics['metrics'], self.silence_metrics['metrics'], self.noise_metrics['metrics'], self.offscreen_metrics['metrics']

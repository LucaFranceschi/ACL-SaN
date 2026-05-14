import torch
import numpy as np
from sklearn import metrics as mt
from typing import List, Optional, Tuple, Dict


class Evaluator(object):
    def __init__(self) -> None:
        """
        Initialize the VGG-Sound (VGGS) Evaluator.

        Attributes:
            PIA (List[float]): Buffer of Percentage of Image Area values.
            AUC_N (List[float]): Buffer of AUC_N values.
            N (int): Counter for the number of evaluations.
            metrics (List[str]): List of metric names.
        """
        super(Evaluator, self).__init__()
        self.std_metrics = {'pIA': [], 'metrics': {'AUC_N': None, 'pIA_ap50': None, 'pIA_hat': None}}
        self.silence_metrics = {'pIA': [], 'metrics': {'AUC_N': None, 'pIA_ap50': None, 'pIA_hat': None}}
        self.noise_metrics = {'pIA': [], 'metrics': {'AUC_N': None, 'pIA_ap50': None, 'pIA_hat': None}}

    def evaluate_batch(self, heatmap: torch.Tensor, thr: Optional[float] = None, **kwargs) -> None:
        """
        Evaluate a batch of predictions.

        Args:
            pred (torch.Tensor): Model predictions.
            thr (Optional[float]): Threshold for binary classification. If None, dynamically determined.

        Returns:
            None
        """

        self._evaluate_batch(heatmap, 'pos', thr)

        sil_heatmap = kwargs.get('silence_heatmap', None)
        if sil_heatmap != None:
            self._evaluate_batch(sil_heatmap, 'sil', thr)

        noise_heatmap = kwargs.get('noise_heatmap', None)
        if noise_heatmap != None:
            self._evaluate_batch(noise_heatmap, 'noise', thr)

    def _evaluate_batch(self, heatmap, metric, thr_param):
        for i in range(heatmap.size(0)):
            pred = heatmap[i].detach().cpu()
            if thr_param is None:
                thr = np.sort(pred.flatten())[int(pred.shape[1] * pred.shape[2]) // 2]
            elif thr_param == 'adap':
                raise NotImplementedError('This dataset does not have GT')
            else:
                thr = thr_param

            self.cal_pIA(pred, metric, thr)

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
        elif metric == 'noise':
            self.noise_metrics['pIA'].append(pIA)
        elif metric == 'pos':
            self.std_metrics['pIA'].append(pIA)
        return


    def finalize_AUC_N(self):
        """
        Calculate the Area Under the Curve for Negative audio samples (AUC_N).

        Returns:
            float: AUC value.
        """
        for metrics in [self.silence_metrics, self.noise_metrics, self.std_metrics]:
            if len(metrics['pIA']) > 0:
                aucs = [np.sum(np.array(metrics['pIA']) < 0.05 * i) / len(metrics['pIA']) for i in range(21)]
                thr = [0.05 * i for i in range(21)]
                auc = mt.auc(thr, aucs)
                metrics['metrics']['AUC_N'] = auc

    def finalize_AP50(self):
        """
        Calculate Average Precision (piA@0.5).

        Returns:
            float: pIA@0.5 value.
        """
        for metrics in [self.silence_metrics, self.noise_metrics, self.std_metrics]:
            if len(metrics['pIA']) > 0:
                ap50 = np.mean(np.array(metrics['pIA']) < 0.5)
                metrics['metrics']['pIA_ap50'] = ap50

    def finalize_pIA(self):
        """
        Calculate mean pIA.

        Returns:
            float: Mean pIA value.
        """
        for metrics in [self.silence_metrics, self.noise_metrics, self.std_metrics]:
            if len(metrics['pIA']) > 0:
                pIA_hat = np.mean(np.array(metrics['pIA']))
                metrics['metrics']['pIA_hat'] = pIA_hat

    def finalize(self):
        """
        Finalize and return evaluation metrics.

        Returns:
            Tuple[List[str], Dict[str, float]]: List of metric names and corresponding values.
        """
        self.finalize_AP50()
        self.finalize_AUC_N()
        self.finalize_pIA()
        return self.std_metrics, self.silence_metrics['metrics'], self.noise_metrics['metrics']

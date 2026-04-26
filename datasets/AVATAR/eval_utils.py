import torch
import numpy as np
from sklearn import metrics as mt
from typing import List, Tuple, Dict, Optional
import copy

class Evaluator(object):
    def __init__(self) -> None:
        """
        Initialize the AVATAR evaluator.

        Attributes:
            miou (List[float]): Buffer of mIoU values.
            F (List[float]): Buffer of F-measure values.
            N (int): Counter for the number of evaluations.
            metrics (List[str]): List of metric names.
        """
        super(Evaluator, self).__init__()
        self.N = 0
        self.std_metrics = {
            'mIoU': [],
            'F_values': [],
            'cIoU': [],
            'metrics': {
                'AUC': None,
                'cIoU_ap50': None,
                'cIoU_hat': None,
                'mIoU': None,
                'Fmeasure': None
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
            'mIoU': [],
            'F_values': [],
            'cIoU': [],
            'pIA': [],
            'metrics': {
                'AUC': None,
                'cIoU_ap50': None,
                'cIoU_hat': None,
                'mIoU': None,
                'Fmeasure': None,
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
            target (torch.Tensor): Ground truth.
            thr (List[float], optional): List of thresholds. If None, calculate threshold as median. Default is None.

        Notes:
            Updates metric buffers (self.mask_iou, self.Eval_Fmeasusre)
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
        thrs = []

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

            thrs.append(thr)

            if metric in ('sil', 'noi'):
                self.cal_pIA(pred, metric, thr)
            elif metric == 'pos':
                self.cal_CIOU(pred, target, metric, thr)
            elif metric == 'off':
                self.cal_pIA(pred, metric, thr)
                self.cal_CIOU(pred, target, metric, thr)

        if metric in ['pos', 'off']:
            infers, gts = heatmap.squeeze(1), gt.squeeze(1)
            self.mask_iou(infers, gts, metric, thrs)
            self.Eval_Fmeasure(infers, gts, metric)

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

    def mask_iou(self, preds: torch.Tensor, targets: torch.Tensor, metric, thrs: List[float], eps: float = 1e-7) -> float:
        """
        Calculate mask IoU.

        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth.
            thrs (List[float]): List of thresholds.
            eps (float, optional): Small epsilon to avoid division by zero. Default is 1e-7.

        Returns:
            float: mIoU value.
        """
        assert len(preds.shape) == 3 and preds.shape == targets.shape
        self.N += 1

        N = preds.size(0)
        miou = 0.0
        for i in range(N):
            pred = preds[i].unsqueeze(0)
            target = targets[i].unsqueeze(0)

            num_pixels = pred.size(-1) * pred.size(-2)
            no_obj_flag = (target.sum(2).sum(1) == 0)

            pred = (pred > thrs[i]).int()
            inter = (pred * target).sum(2).sum(1)
            union = torch.max(pred, target).sum(2).sum(1)

            inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
            inter[no_obj_flag] = inter_no_obj[no_obj_flag]
            union[no_obj_flag] = num_pixels
            miou += (torch.sum(inter / (union + eps))).squeeze()
        miou = miou / N

        if metric == 'pos':
            self.std_metrics['mIoU'].append(miou.detach().cpu())
        elif metric == 'off':
            self.offscreen_metrics['mIoU'].append(miou.detach().cpu())

        return miou

    @staticmethod
    def _eval_pr(y_pred: torch.Tensor, y: torch.Tensor, num: int, cuda_flag: bool = True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate precision and recall.

        Args:
            y_pred (torch.Tensor): Model predictions.
            y (torch.Tensor): Ground truth.
            num (int): Number of threshold values.
            cuda_flag (bool, optional): Whether to use CUDA. Default is True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Precision and recall values.
        """
        if cuda_flag:
            prec, recall = torch.zeros(num).to(y_pred.device), torch.zeros(num).to(y_pred.device)
            thlist = torch.linspace(0, 1 - 1e-10, num).to(y_pred.device)
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

        return prec, recall

    def Eval_Fmeasure(self, pred: torch.Tensor, gt: torch.Tensor, metric, pr_num: int = 255) -> float:
        """
        Evaluate F-measure.

        Args:
            pred (torch.Tensor): Model predictions.
            gt (torch.Tensor): Ground truth.
            pr_num (int, optional): Number of precision-recall values. Default is 255.

        Returns:
            float: F-measure value.

        Notes:
            Fix bug in official test code (Issue: Results vary depending on the batch number)
            The official code had an issue because it optimized precision-recall thresholds for each mini-batch.
        """
        N = pred.size(0)
        beta2 = 0.3
        avg_f, img_num = 0.0, 0
        score = torch.zeros(pr_num).to(pred.device)

        for img_id in range(N):
            # examples with totally black GTs are out of consideration
            if torch.sum(gt[img_id]) == 0.0:
                continue
            prec, recall = self._eval_pr(pred[img_id], gt[img_id], pr_num)
            f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
            f_score[f_score != f_score] = 0  # for Nan
            avg_f += f_score
            img_num += 1
            score = avg_f / img_num

            if metric == 'pos':
                self.std_metrics['F_values'].append(f_score.detach().cpu().numpy())
            elif metric == 'off':
                self.offscreen_metrics['F_values'].append(f_score.detach().cpu().numpy())

        return score.max().item()

    def finalize_mIoU(self) -> float:
        """
        Calculate the final mIoU value.

        Returns:
            float: Final mIoU value.
        """
        for metric in [self.std_metrics, self.offscreen_metrics]:
            if len(metric['mIoU']) > 0:
                miou = np.sum(np.array(metric['mIoU'])) / self.N
                metric['metrics']['mIoU'] = miou

    def finalize_Fmeasure(self) -> float:
        """
        Calculate the final F-measure value.

        Returns:
            float: Final F-measure value.

        Notes:
            Fix bug in official test code (Issue: Results vary depending on the batch number)
            The official code had an issue because it optimized precision-recall thresholds for each mini-batch
        """
        for metric in [self.std_metrics, self.offscreen_metrics]:
            if len(metric['F_values']) > 0:
                F = np.max(np.mean(metric['F_values'], axis=0))
                metric['metrics']['Fmeasure'] = F

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
        Finalize evaluation and return the results.

        Returns:
            Tuple[List[str], Dict[str, float]]: Tuple containing metric names and corresponding values.
        """
        self.finalize_mIoU()
        self.finalize_Fmeasure()
        self.finalize_AUC()
        self.finalize_AP50()
        self.finalize_means()

        print(self.offscreen_metrics)

        return self.std_metrics['metrics'], self.silence_metrics['metrics'], self.noise_metrics['metrics'], self.offscreen_metrics['metrics']

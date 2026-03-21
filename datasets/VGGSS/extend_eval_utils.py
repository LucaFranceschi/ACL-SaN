import os
import numpy as np
import torch
from sklearn import metrics as mt

import utils.util as util
import copy


class Evaluator(object):
    def __init__(self, iou_thrs=(0.5, ), default_conf_thr=0.5, pred_size=0.5, pred_thr=0.5,
                 results_dir='./results'):
        """
        Initialize the Extended VGGSS evaluator.

        Notes:
            Taking computation speed into consideration, it is set to output only the 'all' subset. (AP, Max-F1)
        """
        super(Evaluator, self).__init__()
        self.iou_thrs = iou_thrs
        self.default_conf_thr = default_conf_thr
        self.min_sizes = {'small': 0, 'medium': 32 ** 2, 'large': 96 ** 2, 'huge': 144 ** 2}
        self.max_sizes = {'small': 32 ** 2, 'medium': 96 ** 2, 'large': 144 ** 2, 'huge': 10000 ** 2}

        self.name_list = []
        self.bb_list = []
        self.confidence_list = []
        self.area_list = []

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

        self.results_dir = results_dir
        self.viz_save_dir = f"{results_dir}/viz_conf" + str(default_conf_thr) + "_predsize" + str(
            pred_size) + "_predthr" + str(pred_thr)
        self.results_save_dir = f"{results_dir}/results_conf" + str(default_conf_thr) + "_predsize" + str(
            pred_size) + "_predthr" + str(pred_thr)

    @staticmethod
    def calc_precision_recall(bb_list, ciou_list, confidence_list, confidence_thr, ciou_thr=0.5):
        assert len(bb_list) == len(ciou_list) == len(confidence_list)
        true_pos, false_pos, false_neg = 0, 0, 0
        for bb, ciou, confidence in zip(bb_list, ciou_list, confidence_list):
            if bb == 0:
                # no sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected
                    false_pos += 1
            else:
                # sounding objects in frame
                if confidence >= confidence_thr:
                    # sounding object detected...
                    if ciou >= ciou_thr:  # ...in correct place
                        true_pos += 1
                    else:  # ...in wrong place
                        false_pos += 1
                else:
                    # no sounding objects detected
                    false_neg += 1

        precision = 1. if true_pos + false_pos == 0 else true_pos / (true_pos + false_pos)
        recall = 1. if true_pos + false_neg == 0 else true_pos / (true_pos + false_neg)

        return precision, recall

    def calc_ap(self, bb_list_full, ciou_list_full, confidence_list_full, iou_thr=0.5):

        assert len(bb_list_full) == len(ciou_list_full) == len(confidence_list_full)

        # for visible objects
        # ss = [i for i, bb in enumerate(bb_list_full) if bb > 0]
        # bb_list = [bb_list_full[i] for i in ss]
        # ciou_list = [ciou_list_full[i] for i in ss]
        # confidence_list = [confidence_list_full[i] for i in ss]

        precision, recall, skip_thr = [], [], max(1, len(ciou_list_full) // 200)
        for thr in np.sort(np.array(confidence_list_full))[:-1][::-skip_thr]:
            p, r = self.calc_precision_recall(bb_list_full, ciou_list_full, confidence_list_full, thr, iou_thr)
            precision.append(p)
            recall.append(r)
        precision_max = [np.max(precision[i:]) for i in range(len(precision))]
        ap = sum([precision_max[i] * (recall[i + 1] - recall[i])
                  for i in range(len(precision_max) - 1)])
        return ap

    def cal_auc(self, bb_list, ciou_list):
        ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        ciou = [ciou_list[i] for i in ss]
        cious = [np.sum(np.array(ciou) >= 0.05 * i) / len(ciou)
                 for i in range(21)]
        thr = [0.05 * i for i in range(21)]
        auc = mt.auc(thr, cious)
        return auc

    def filter_subset(self, subset, name_list, area_list, bb_list, ciou_list, conf_list):
        if subset == 'visible':
            ss = [i for i, bb in enumerate(bb_list) if bb > 0]
        elif subset == 'non-visible/non-audible':
            ss = [i for i, bb in enumerate(bb_list) if bb == 0]
        elif subset == 'all':
            ss = [i for i, bb in enumerate(bb_list) if bb >= 0]
        else:
            ss = [i for i, sz in enumerate(area_list)
                  if self.min_sizes[subset] <= sz < self.max_sizes[subset] and bb_list[i] > 0]

        if len(ss) == 0:
            return [], [], [], [], []

        name = [name_list[i] for i in ss]
        area = [area_list[i] for i in ss]
        bbox = [bb_list[i] for i in ss]
        ciou = [ciou_list[i] for i in ss]
        conf = [conf_list[i] for i in ss]

        return name, area, bbox, ciou, conf

    def finalize_stats(self):

        for metric in [self.std_metrics]:
            cious = metric['cIoU']
            for iou_thr in self.iou_thrs:
                # for subset in ['all', 'visible']:
                for subset in ['all']:
                    _, _, bb_list, ciou_list, conf_list = self.filter_subset(subset, self.name_list, self.area_list,
                                                                            self.bb_list, cious,
                                                                            self.confidence_list)
                    subset_name = f'{subset}@{int(iou_thr * 100)}' if subset is not None else f'@{int(iou_thr * 100)}'
                    if len(ciou_list) == 0:
                        p, r, ap, f1, auc = np.nan, np.nan, np.nan, np.nan, np.nan
                    else:
                        p, r = self.calc_precision_recall(bb_list, ciou_list, conf_list, -1000, iou_thr)
                        ap = self.calc_ap(bb_list, ciou_list, conf_list, iou_thr)
                        auc = self.cal_auc(bb_list, ciou_list)

                        conf_thr = list(sorted(conf_list))[::max(1, len(conf_list) // 10)]
                        pr = [self.calc_precision_recall(bb_list, ciou_list, conf_list, thr, iou_thr) for thr in conf_thr]
                        f1 = [2 * r * p / (r + p) if r + p > 0 else 0. for p, r in pr]
                        if subset == 'all' and iou_thr == 0.5:
                            ef1 = max(f1)
                            eap = ap
                            metric['metrics']['ef1'] = ef1
                            metric['metrics']['eap'] = eap
                        if subset == 'visible' and iou_thr == 0.5:
                            eloc = self.precision_at_50(cious)
                            eauc = auc
                            metric['metrics']['eloc'] = eloc
                            metric['metrics']['eauc'] = eauc
                    metric['metrics'][f'Precision-{subset_name}'] = p
                    # metrics[f'Recall-{subset_name}'] = r
                    if np.isnan(f1).any():
                        metric['metrics'][f'F1-{subset_name}'] = f1
                    else:
                        _ef1 = ' '.join([f'{f * 100:.1f}' for f in f1])
                        metric['metrics'][f'F1-{subset_name}'] = max([float(num) for num in _ef1.split(' ')])
                    metric['metrics'][f'AP-{subset_name}'] = ap * 100
                    metric['metrics'][f'AUC-{subset_name}'] = auc

    def precision_at_50(self, cious):
        ss = [i for i, bb in enumerate(self.bb_list) if bb > 0]
        return np.mean(np.array([cious[i] for i in ss]) > 0.5)

    def precision_at_50_object(self, cious):
        max_num_obj = max(self.bb_list)
        for num_obj in range(1, max_num_obj + 1):
            ss = [i for i, bb in enumerate(self.bb_list) if bb == num_obj]
            precision = np.mean(np.array([cious[i] for i in ss]) > 0.5)
            print('\n' + f'num_obj:{num_obj}, precision:{precision}')

    def f1_at_50(self, cious):
        # conf_thr = np.array(self.confidence_list).mean()
        p, r = self.calc_precision_recall(self.bb_list, cious, self.confidence_list, self.default_conf_thr,
                                          0.5)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.

    def ap_at_50(self, cious):
        return self.calc_ap(self.bb_list, cious, self.confidence_list, 0.5)

    def update(self, bb, gt, conf, pred, pred_thr, name, metric):
        if isinstance(conf, torch.Tensor):
            conf = conf.detach().cpu().numpy()
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        # Compute binary prediction map
        infer = np.zeros_like(gt)
        infer[pred >= pred_thr] = 1

        # Compute ciou between prediction and ground truth
        ciou = np.sum(infer * gt) / (np.sum(gt) + np.sum(infer * (gt == 0)) + 1e-12)

        # Compute ground truth size
        area = gt.sum()

        # Save
        if metric == 'std':
            self.std_metrics['cIoU'].append(ciou)
            # common variables
            self.confidence_list.append(conf)
            self.area_list.append(area)
            self.name_list.append(name)
            self.bb_list.append(bb)
        return

    def evaluate_batch(self, heatmap: torch.Tensor, gt: torch.Tensor, label, conf, name, thr = None, **kwargs) -> None:
        self._evaluate_batch(heatmap, 'std', gt, label, conf, name, thr)

        sil_heatmap = kwargs.get('silence_heatmap', None)
        if sil_heatmap != None:
            self._evaluate_batch(sil_heatmap, 'sil', gt, label, conf, name, thr)

        noise_heatmap = kwargs.get('noise_heatmap', None)
        if noise_heatmap != None:
            self._evaluate_batch(noise_heatmap, 'noise', gt, label, conf, name, thr)

    def _evaluate_batch(self, heatmap: torch.Tensor, metric, gt: torch.Tensor, label, conf, name, thr = None):
        for i in range(heatmap.shape[0]):
            pred = heatmap[i].detach().cpu()
            target = gt[i]
            if thr is None:
                thr = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1]) // 2]
            elif thr == 'adap':
                gt_nums = (target!=0).sum()
                if int(gt_nums) == 0:
                    gt_nums = int(target.shape[0] * target.shape[1]) // 2
                thr = np.sort(target.flatten())[int(target.shape[0] * target.shape[1]) - int(gt_nums)] # adap

            bb = 1 if label[i] != 'non-sounding' else 0

            if metric in ('sil', 'noise'):
                self.cal_pIA(pred, metric, thr)
            else:
                self.update(bb, gt[i], conf[i], pred, thr, name[i], metric)

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
        return

    def finalize_AUC(self):
        """
        Calculate the Area Under the Curve (AUC).

        Returns:
            float: AUC value.
        """
        for metric in [self.std_metrics]:
            if len(metric['cIoU']) > 0:
                cious = [np.sum(np.array(metric['cIoU']) >= 0.05 * i) / len(metric['cIoU'])
                        for i in range(21)]
                thr = [0.05 * i for i in range(21)]
                auc = mt.auc(thr, cious)
                metric['metrics']['AUC'] = auc

        for metric in [self.silence_metrics, self.noise_metrics]:
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
        for metric in [self.std_metrics]:
            if len(metric['cIoU']) > 0:
                ap50 = np.mean(np.array(metric['cIoU']) >= 0.5)
                metric['metrics']['cIoU_ap50'] = ap50

        for metric in [self.silence_metrics, self.noise_metrics]:
            if len(metric['pIA']) > 0:
                ap50 = np.mean(np.array(metric['pIA']) < 0.5)
                metric['metrics']['pIA_ap50'] = ap50

    def finalize_means(self):
        """
        Calculate mean cIoU.

        Returns:
            float: Mean cIoU value.
        """
        for metric in [self.std_metrics]:
            if len(metric['cIoU']) > 0:
                ciou = np.mean(np.array(metric['cIoU']))
                metric['metrics']['cIoU_hat'] = ciou

        for metric in [self.silence_metrics, self.noise_metrics]:
            if len(metric['pIA']) > 0:
                pia = np.mean(np.array(metric['pIA']))
                metric['metrics']['pIA_hat'] = pia

    def finalize(self):
        self.finalize_stats()
        self.finalize_AUC()
        self.finalize_AP50()
        self.finalize_means()

        return self.std_metrics['metrics'], self.silence_metrics['metrics'], self.noise_metrics['metrics']

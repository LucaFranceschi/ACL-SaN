import load_utils
import viz_utils
import importlib
importlib.reload(load_utils)
importlib.reload(viz_utils)

from load_utils import *
from viz_utils import *

if "EXPERIMENTS" not in globals():
    global EXPERIMENTS
    EXPERIMENTS = {}

def cleanup():
    global EXPERIMENTS
    EXPERIMENTS = {}

def multiton(cls):
    global EXPERIMENTS
    def getinstance(name):
        if name not in EXPERIMENTS:
            EXPERIMENTS[name] = cls(name)
        return EXPERIMENTS[name]
    return getinstance

@multiton
class Experiment(object):
    def __init__(self, exp_name) -> None:
        self.name = exp_name
        self.metrics = pd.DataFrame()
        self.infer_info = pd.DataFrame()
        self.loaded_dirs = []
        self.thresholds = None

    def cleanup(self):
        global EXPERIMENTS
        del EXPERIMENTS[self.name]

    def load_eval_metrics(self, path_to_tensorboard_dir):
        if path_to_tensorboard_dir not in self.loaded_dirs:
            metrics = load_eval(path_to_tensorboard_dir, self.name)
            self.metrics = pd.concat([
                self.metrics,
                metrics
            ])
            self.loaded_dirs.append(path_to_tensorboard_dir)

    def load_eval_inference_info(self, path_to_numpy_dir):
        if path_to_numpy_dir not in self.loaded_dirs:
            self.infer_info = load_infer_info(path_to_numpy_dir, self.name)

            if self.thresholds is None:
                self.thresholds = get_thresholds(path_to_numpy_dir)

            self.loaded_dirs.append(path_to_numpy_dir)

    def _print_metrics(self, epoch, thr, seg_item):
        if self.thresholds and seg_item in self.thresholds[epoch] and thr in self.thresholds[epoch][seg_item]:
            thr = str(self.thresholds[epoch][seg_item][thr])
        print_metrics(self.metrics, epoch=epoch, thr=thr, seg_item=seg_item)

    def print_metrics(self, thr, seg_item):
        if self.thresholds is None:
            raise Exception('Load inference info first!!')
        for i in sorted(self.thresholds.keys()):
            self._print_metrics(i, thr=thr, seg_item=seg_item)

    def print_metrics_noisy(self, epoch, thr, seg_item):
        if self.thresholds and seg_item in self.thresholds[epoch] and thr in self.thresholds[epoch][seg_item]:
            thr = str(self.thresholds[epoch][seg_item][thr])
        print_metrics_noisy(self.metrics, epoch=epoch, thr=thr, seg_item=seg_item)

    def plot_all_metrics(self, epoch, thr, seg_item):
        plot_all_metrics(self.metrics[self.metrics['epoch'] == epoch], thr=thr, seg_item=seg_item, experiment_name=self.name, epoch=epoch)

    def boxplots_by_dataset(self, dataset_name, epochs, th_name, seg_item, min_max='max'):
        boxplots_by_dataset(self.infer_info, dataset_name, self.thresholds, epochs,
                            th_name=th_name, min_max=min_max, seg_item=seg_item, experiment_name=self.name)

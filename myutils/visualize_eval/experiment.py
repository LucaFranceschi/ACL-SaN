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
            self.thresholds = get_thresholds(self.infer_info)
            self.loaded_dirs.append(path_to_numpy_dir)

    def print_metrics(self, epoch, thr):
        print_metrics(self.metrics, epoch=epoch, thr=thr)

    def print_metrics_noisy(self, epoch, thr):
        print_metrics_noisy(self.metrics, epoch=epoch, thr=thr)

    def plot_all_metrics(self, epoch):
        plot_all_metrics(self.metrics[self.metrics['epoch'] == epoch])

    def boxplots_by_dataset(self, dataset_name, epochs, th_name, min_max='max', seg_item='m_i_seg'):
        boxplots_by_dataset(self.infer_info, dataset_name, self.thresholds, epochs,
                            th_name=th_name, min_max=min_max, seg_item=seg_item)

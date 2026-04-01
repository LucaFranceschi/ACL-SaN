import os
from load_utils import *
from viz_utils import *
from experiment import Experiment

if __name__ == '__main__':
    data_path = '/'.join(os.getcwd().split('/')[:-2] + ['train_outputs'])

    baseline = Experiment('baseline')
    baseline.load_eval_metrics(os.path.join(data_path, "2139484/Test_record/Test_record/ACL_ViT16_Exp_ACL_v1/tensorboard"))
    # baseline.load_eval_metrics(os.path.join(data_path, "2126070/Test_record_noisy/ACL_ViT16_Exp_ACL_v1/tensorboard"))
    baseline.load_eval_inference_info(os.path.join(data_path, "2139484/Test_record/Test_record/ACL_ViT16_Exp_ACL_v1/numpy")) # since only one epoch i can load it from noisy eval
    # baseline.focus_epoch = 20

    baseline.print_metrics_noisy('all', '0.5')
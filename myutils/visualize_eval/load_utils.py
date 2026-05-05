import re, os, gc, json
import pandas as pd
import numpy as np

from tensorboard.backend.event_processing import event_accumulator

def get_thr(ss: str):
    match = re.search(r'\((.*)\)', ss)
    if match != None:
        return match.group(1)

def get_metric(ss: str):
    match = re.search(r'\(.*\)_(.*)', ss)
    if match != None:
        return match.group(1)

def get_audio_type(ss: str):
    match = re.search(r'(positive|pos|silence|sil|noise|noi|offscreen|off)', ss)
    if match != None:
        return match.group(1)

def get_dataset(ss: str):
    match = re.search(r'(avs_ms3|avs_s4|ms3|s4|vggss|exvggss|vggsound|flickr|exflickr|avatar_one_bb|avatar_one_seg|avatar_one)', ss)
    if match != None:
        return match.group(1)

def get_epoch(ss: str):
    match = re.search(r'epoch(\d+|best)', ss)
    if match != None:
        epoch = match.group(1)
        if epoch == 'best':
            return epoch
        return int(epoch)

def get_snr(ss: str):
    match = re.search(r'snr(\d+)', ss)
    if match == None:
        return -1
    snr = match.group(1)
    return int(snr)

def get_min_max(ss: str):
    match = re.search(r'(min|max)', ss)
    if match == None:
        return ''
    snr = match.group(1)
    return snr

def get_seg_item(ss: str):
    match = re.search(r'(v_d|m_i|v_i|v_f)', ss)
    if match == None:
        return ''
    snr = match.group(1)
    return snr

def load_nested_tb_logs(root_dir):
    all_data = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        # Check if there are any tfevents files in this specific folder
        if any(f.startswith("events.out.tfevents") for f in files):
            # Extract the folder name to use as a category/run label

            # Initialize accumulator for this specific subdirectory
            acc = event_accumulator.EventAccumulator(root)
            acc.Reload()

            for tag in acc.Tags()['scalars']:
                events = acc.Scalars(tag)
                df_temp = pd.DataFrame(events)

                # We add 'metric' (e.g., value) and 'sub_dir' (e.g., test_noise_avs...)
                df_temp['metric_tag'] = tag
                df_temp['run_group'] = root

                all_data.append(df_temp)

    if not all_data:
        print("No event files found in the specified path.")
        return pd.DataFrame()

    # Combine all found data
    master_df = pd.concat(all_data, ignore_index=True)

    # Cleanup: Convert time and reorder columns
    master_df['wall_time'] = pd.to_datetime(master_df['wall_time'], unit='s')

    return master_df

def load_eval(path, run_name):
    df = load_nested_tb_logs(path)
    print(f"Loaded {len(df)} data points.")

    df['threshold'] = df['run_group'].apply(lambda x: get_thr(str(x)))
    df['metric'] = df['run_group'].apply(lambda x: get_metric(str(x)))
    df['audio_type'] = df['metric_tag'].apply(lambda x: get_audio_type(str(x)))
    df['dataset'] = df['run_group'].apply(lambda x: get_dataset(str(x)))
    df['epoch'] = df['run_group'].apply(lambda x: get_epoch(str(x)))
    df['snr'] = df['run_group'].apply(lambda x: get_snr(str(x)))
    df['seg_item'] = df['metric_tag'].apply(lambda x: get_seg_item(str(x)))
    df.drop(['wall_time', 'metric_tag', 'run_group'],axis=1, inplace=True)
    df = df.assign(run=run_name)

    return df

def load_nested_np_logs(root_dir):
    tmp_list = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('npy'):
                arr = np.load(os.path.join(root, f)).ravel()
                tmp_list.append([root, f, arr])

    return pd.DataFrame(tmp_list, columns=['run_group', 'metric_tag', 'data'])

def load_infer_info(path, run_name):
    df = load_nested_np_logs(path)
    print(f"Loaded {len(df)} data points.")

    df['epoch'] = df['run_group'].apply(lambda x: get_epoch(str(x)))
    df['audio_type'] = df['metric_tag'].apply(lambda x: get_audio_type(str(x)))
    df['snr'] = df['metric_tag'].apply(lambda x: get_snr(str(x)))
    df['dataset'] = df['metric_tag'].apply(lambda x: get_dataset(str(x)))
    df['min_max'] = df['metric_tag'].apply(lambda x: get_min_max(str(x)))
    df['seg_item'] = df['metric_tag'].apply(lambda x: get_seg_item(str(x)))
    df.drop(['metric_tag', 'run_group'], axis=1, inplace=True)
    df = df.assign(run=run_name)

    return df

# def get_thresholds(infer_info_df):

#     df = infer_info_df[(infer_info_df["dataset"] == 'vggss') &
#                        (infer_info_df["min_max"] == 'max')].copy()

#     return_thresholds = {}

#     for epoch in df['epoch'].unique():
#         df_epoch = df[df['epoch'] == epoch].copy()

#         return_thresholds[int(epoch)] = {}
#         for seg_item in df_epoch['seg_item'].unique():

#             pos_arr = df_epoch[(df_epoch['audio_type'] == 'pos') & (df_epoch['seg_item'] == seg_item)]['data'].to_numpy()
#             sil_arr = df_epoch[(df_epoch['audio_type'] == 'sil') & (df_epoch['seg_item'] == seg_item)]['data'].to_numpy()
#             noise_arr = df_epoch[(df_epoch['audio_type'] == 'noi') & (df_epoch['seg_item'] == seg_item)]['data'].to_numpy()

#             max_negatives = [list(sil_arr), list(noise_arr)]
#             max_negatives_separate = [np.percentile(list(sil_arr), 75), np.percentile(list(noise_arr), 75)]

#             thresh_for_seg_item = {
#                 'max_neg': np.mean(max_negatives),
#                 'max_neg_plus_10': np.mean(max_negatives) * 1.1,
#                 'max_q2_pos': np.percentile(list(pos_arr), 25),
#                 'max_q3_all': np.percentile(max_negatives, 75),
#                 'max_q3_separate': np.max(max_negatives_separate)
#             }

#             return_thresholds[int(epoch)][seg_item] = thresh_for_seg_item

#         del df_epoch, max_negatives, max_negatives_separate
#         gc.collect()

#     return return_thresholds

def get_thresholds(path_to_thresholds):
    thresholds_dict = {}
    for file in os.listdir(path_to_thresholds):
        if file.endswith('.json'):
            with open(os.path.join(path_to_thresholds, file), 'r') as fp:
                thresholds_dict[file] = json.load(fp)

    thresholds_dict = {k: v for d in thresholds_dict.values() for k, v in d.items()}
    thresholds_dict = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in thresholds_dict.items()}
    return thresholds_dict
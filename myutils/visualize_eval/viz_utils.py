import os, gc
from datetime import datetime

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COLOR_PALETTE = {
    'vggss': '#bcbd22',
    'flickr': '#e377c2',
    'avatar_one_seg': '#ff7f0e',
    'avs_s4': '#d62728',
    'avs_ms3': '#2ca02c',
    'avatar_one_bb': '#1f77b4',
    'exvggss': '#8c564b',
    'exflickr': '#9467bd',
    'vggsound': '#7f7f7f',
}

def print_metrics(df, epoch, thr:str='0.5', seg_item='m_i', snr=False) -> pd.DataFrame | None:
    filtered_df = df[df['epoch'] == epoch].copy()

    if len(filtered_df) == 0:
        return None

    filtered_df = filtered_df[
        (filtered_df['threshold'] == thr) &
        (filtered_df["seg_item"] == seg_item) &
        (filtered_df['metric'].isin(['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N', 'mIoU', 'Fmeasure']))
    ]

    # 2. Pivot the data
    # index: what you want as rows
    # columns: what you want as side-by-side columns
    # values: the numbers to fill the table
    pivot_df = filtered_df.pivot_table(
        index=['epoch', 'dataset'],
        columns=['audio_type', 'metric'] if not snr else ['audio_type', 'snr', 'metric'],
        values='value',
    )

    # Define the desired order for each audio_type
    if not snr:
        std_cols = [('pos', m) for m in ['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N']]
        silence_cols = [('sil', m) for m in ['pIA_hat', 'AUC_N']]
        noise_cols = [('noi', m) for m in ['pIA_hat', 'AUC_N']]
        offscreen_cols = [('off', m) for m in ['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N']]
    else:
        std_cols = [('pos', s, m) for s in [-1, 20.0, 10.0, 5.0] for m in ['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N']]
        silence_cols = [('sil', -1, m) for m in ['pIA_hat', 'AUC_N']]
        noise_cols = [('noi', -1, m) for m in ['pIA_hat', 'AUC_N']]
        offscreen_cols = [('off', -1, m) for m in ['pIA_hat', 'AUC_N']]

    # Combine them into one ordered list
    target_columns = std_cols + silence_cols + noise_cols + offscreen_cols

    # Reindex the columns to the new order
    # errors='ignore' ensures it doesn't crash if a specific metric is missing for one type
    pivot_df = pivot_df.reindex(columns=target_columns)
    # Order datasets with avatar_one above avatar_off
    custom_order = [
        'avatar_one_bb', 'avatar_one_seg',
        'avatar_off_bb', 'avatar_off_seg',
        'avs_ms3', 'avs_s4', 'flickr', 'vggss', 'exflickr', 'exvggss'
    ]

    # Create a mapping for sorting
    dataset_order = {ds: i for i, ds in enumerate(custom_order)}

    # Sort by the 'dataset' level (level 1) using the custom order
    pivot_df = pivot_df.reindex(
        sorted(pivot_df.index, key=lambda x: dataset_order.get(x[1], 999))
    )

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000 # Increased width to prevent wrapping
    # pivot_df.to_clipboard(header=True, sep='\t')

    # print(pivot_df)

    # input() # to be able to copy-paste tables into sheets :)

    return pivot_df

'''
def print_metrics_noisy(df, epoch, thr:str='0.5', seg_item='m_i'):

    filtered_df = df[df['epoch'] == epoch].copy()

    filtered_df = filtered_df[
        (filtered_df['threshold'] == thr) &
        (filtered_df["seg_item"] == seg_item) &
        (filtered_df['metric'].isin(['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N', 'mIoU', 'Fmeasure']))
    ]

    # 2. Pivot the data
    # index: what you want as rows
    # columns: what you want as side-by-side columns
    # values: the numbers to fill the table
    pivot_df = filtered_df.pivot_table(
        index=['dataset', 'epoch'],
        columns=['audio_type', 'snr', 'metric'],
        values='value',
    )

    # Define the desired order for each audio_type
    # std_cols = [('pos', m) for m in ['cIoU_hat', 'AUC', 'mIoU', 'Fmeasure']]
    std_cols = [('pos', s, m) for s in [5.0, 10.0, 20.0, -1] for m in ['cIoU_hat', 'AUC']]
    silence_cols = [('sil', -1, m) for m in ['pIA_hat', 'AUC_N']]
    noise_cols = [('noi', -1, m) for m in ['pIA_hat', 'AUC_N']]
    offscreen_cols = [('off', -1, m) for m in ['pIA_hat', 'AUC_N']]

    # Combine them into one ordered list
    target_columns = std_cols + silence_cols + noise_cols + offscreen_cols

    # Reindex the columns to the new order
    # errors='ignore' ensures it doesn't crash if a specific metric is missing for one type
    pivot_df = pivot_df.reindex(columns=target_columns)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000 # Increased width to prevent wrapping

    print(pivot_df)
    return pivot_df
'''

def print_runs(df, thr=0.5):
    filtered_df = df[
        (df['threshold'] == thr) &
        (df['metric'].isin(['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N', 'mIoU', 'Fmeasure']))
    ]

    pivot_df = filtered_df.pivot_table(
        index=['dataset', 'run'],
        columns=['audio_type', 'metric'],
        values='value',
    )

    std_cols = [('pos', m) for m in ['cIoU_hat', 'AUC']]
    silence_cols = [('sil', m) for m in ['pIA_hat', 'AUC_N']]
    noise_cols = [('noi', m) for m in ['pIA_hat', 'AUC_N']]

    target_columns = std_cols + silence_cols + noise_cols

    pivot_df = pivot_df.reindex(columns=target_columns)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000 # Increased width to prevent wrapping

    print(pivot_df)
    return pivot_df

def plot_all_metrics(df, thr='adap', seg_item = 'm_i', experiment_name = '', epoch = ''):
    df_filtered = df[df['seg_item'] == seg_item].copy()
    # color_palette = {}
    # for dataset, color in zip(sorted(df_filtered['dataset'].unique()), sns.color_palette(n_colors=df_filtered['dataset'].nunique()).as_hex()):
    #     color_palette[dataset] = color
    # 1. Setup the style
    sns.set_theme(style="whitegrid")

    # 2. Define strict mappings
    # Mapping line styles to audio types
    style_map = {
        'pos': (None, None),  # Solid
        'noi': (5, 5),      # Dashed
        'sil': (1, 2)     # Dotted
    }

    # 3. Get list of unique metrics
    # metrics = df_filtered['metric'].unique()
    metrics = ['AUC', 'AUC_N', 'pIA_hat']

    for m in metrics:
        subset = df_filtered[df_filtered['metric'] == m]
        thr_subset = subset[subset['threshold'] == thr]
        subset = subset[~((subset['threshold'] == 'adap') | (subset['threshold'] == 'None'))]
        subset['threshold'] = subset['threshold'].astype(float)
        subset = subset.sort_values('threshold')

        # Build per-dataset adap value map for this metric.
        adap_values = {}
        for dataset in thr_subset['dataset'].unique():
            vals = thr_subset[thr_subset['dataset'] == dataset]['value']
            if len(vals) == 0:
                continue
            try:
                adap_values[dataset] = float(vals.iloc[0])
            except (ValueError, TypeError):
                continue

        fig = plt.figure(figsize=(10, 6))

        # 4. Create the lineplot with the fixed palette
        ax = sns.lineplot(
            data=subset.sort_values('dataset'),
            x='threshold',
            y='value',
            hue='dataset',
            palette=COLOR_PALETTE,  # Force consistent colors
            style='audio_type',
            dashes=style_map,
            markers=True,
            linewidth=2,
        )

        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        if legend is not None:
            for text in legend.texts:
                label = text.get_text()
                if label in adap_values:
                    text.set_text(f"{label} ({thr}={adap_values[label]:.3f})")

        # 5. Formatting
        plt.title(f"{experiment_name}@{epoch}: {m} ({seg_item})", fontsize=15, fontweight='bold')
        plt.xlabel("Threshold")
        plt.ylabel("Value")
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        # precision = 1
        # plt.ylim(np.true_divide(np.floor(subset['value'].min() * 10**precision), 10**precision),
        # np.true_divide(np.ceil(subset['value'].max() * 10**precision), 10**precision))

        plt.tight_layout()
        os.makedirs(f'outputs/{experiment_name}', exist_ok=True)
        fig.savefig(f"outputs/{experiment_name}/{int(datetime.timestamp(datetime.now()))}.png", bbox_inches='tight', dpi=300)
        plt.show()

def plot_all_experiments(experiments):
    raise NotImplementedError('Changes v_d and m_i')
    merged_df = pd.concat([e.metrics[e.metrics['epoch'] == e.best_epoch] for e in experiments.values()])

    merged_df = merged_df[
        (merged_df['threshold'] == 0.5) &
        (merged_df['dataset'] == 'avatar_one_seg')
    ]

    df = merged_df[merged_df['run'] != 'baseline']

    # color_palette = {}
    # for dataset, color in zip(sorted(df['run'].unique()), sns.color_palette(n_colors=df['run'].nunique()).as_hex()):
    #     color_palette[dataset] = color

    # 1. Setup the style
    sns.set_theme(style="whitegrid")

    # 2. Define strict mappings
    # Mapping line styles to audio types
    style_map = {
        'pos': (None, None),  # Solid
        'noi': (5, 5),      # Dashed
        'sil': (1, 2)     # Dotted
    }

    df = df.sort_values('threshold')

    # 3. Get list of unique metrics
    # metrics = df['metric'].unique()
    metrics = ['AUC', 'AUC_N']

    for m in metrics:
        subset = df[df['metric'] == m].copy()
        subset = subset.sort_values('threshold')

        baseline_value = merged_df[
            (merged_df['run'] == 'baseline') &
            (merged_df['metric'] == m)
        ]['value'].to_list()[0]

        plt.figure(figsize=(10, 6))

        plt.hlines(baseline_value, 0, 20, label='baseline', linestyles='dashed')

        # 4. Create the lineplot with the fixed palette
        ax = sns.lineplot(
            data=subset.sort_values('dataset'),
            x='epoch',
            y='value',
            hue='run',
            palette=COLOR_PALETTE,  # Force consistent colors
            style='audio_type',
            dashes=style_map,
            markers=True,
            linewidth=2
        )

        # 5. Formatting
        plt.title(f"Evaluation @(threshold=0.5, avatar_seg, {m})", fontsize=15, fontweight='bold')
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(range(0, 21, 2))
        precision = 1
        plt.ylim(
            np.true_divide(np.floor(min(subset['value'].min(), baseline_value) * 10**precision), 10**precision),
            np.true_divide(np.ceil(max(subset['value'].max(), baseline_value) * 10**precision), 10**precision)
        )

        plt.tight_layout()

        plt.show()

def boxplots_by_dataset(infer_info_df, dataset_name, threshold_dict, epochs, th_name = 'max_neg', min_max = 'max', seg_item = 'm_i', experiment_name = ''):
    df = infer_info_df[(infer_info_df["dataset"] == dataset_name) &
                       (infer_info_df["min_max"] == min_max) &
                       (infer_info_df["seg_item"] == seg_item)].copy()

    df = df[[e in epochs for e in df['epoch']]]
    col_order = sorted(df["epoch"].dropna().unique())

    df = df.explode("data")

    fig = sns.catplot(df,
        y='data',
        hue='audio_type',
        hue_order=["pos", "sil", "noi", 'off'],
        kind="box",
        palette='pastel',
        col='epoch',
        col_order=col_order,
        height=4,
        aspect=0.5,
    )

    # Remove the auto-generated legend
    if fig._legend:
        fig._legend.remove()

    for ax, epoch in zip(fig.axes.flat, col_order):
        ax.set_ylim([0, 1])
        th_value = threshold_dict.get(epoch, {}).get(seg_item, {}).get(th_name)
        if th_value is not None:
            ax.axhline(float(th_value), color='crimson', linestyle='--', linewidth=1.5)
            ax.text(0.5, 1.02, f'Thr = {float(th_value):.3f}', transform=ax.transAxes, ha='center', va='bottom', fontsize=8, color='black')

    plt.suptitle(f'{experiment_name}: {dataset_name} ({seg_item} - {th_name}_{min_max})')

    hue_order = ["pos", "sil", "noi", 'off']
    palette = sns.color_palette('pastel', n_colors=len(hue_order))
    legend_elements = [Patch(facecolor=palette[i], label=hue_order[i]) for i in range(len(hue_order))]
    legend_elements.append(Line2D([0], [0], color='crimson', linestyle='--', linewidth=1.5, label=th_name))

    fig.figure.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, title=None, frameon=False)
    fig.set_titles(y=-0.1)

    plt.tight_layout()
    os.makedirs(f'outputs/{experiment_name}', exist_ok=True)
    fig.savefig(f"outputs/{experiment_name}/{int(datetime.timestamp(datetime.now()))}.png", bbox_inches='tight', dpi=300)
    plt.show()

def boxplots_by_dataset_compare(
        list_of_experiments,
        list_of_epochs,
        dataset_name = 'vggss',
        seg_item = 'm_i_seg',
        th_name = 'max_neg',
        min_max = 'max',
        experiment_name = ''
    ):
    if len(list_of_experiments) != len(list_of_epochs):
        raise Exception('Incorrect params')
    N = len(list_of_epochs)

    fig, axs = plt.subplots(nrows=1, ncols=N, sharey=True, figsize=(1.5*N, 4))

    hue_order = ["pos", "sil", "noi", "off"]
    palette = sns.color_palette('pastel', n_colors=len(hue_order))

    for i in range(N):
        ax = axs if N == 1 else axs[i]

        df = list_of_experiments[i].infer_info.copy()
        df = df[(df["dataset"] == dataset_name) &
                (df["min_max"] == min_max) &
                (df["epoch"] == list_of_epochs[i]) &
                (df["seg_item"] == seg_item)]

        df = df.explode("data")

        sns.boxplot(
            df,
            y='data',
            hue='audio_type',
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            legend=False
        )

        ax.set_title(f"{list_of_experiments[i].name.strip('ACL_')}@{list_of_epochs[i]}", fontsize=12)
        ax.set_ylim([0, 1])

        ep = list_of_epochs[i]
        try:
            ep = int(list_of_epochs[i])
        except:
            pass

        th_value = list_of_experiments[i].thresholds.get(ep, {}).get(seg_item, {}).get(th_name)
        if th_value is not None:
            ax.axhline(float(th_value), color='crimson', linestyle='--', linewidth=1.5)
            ax.text(0.5, -0.07, f'Thr={float(th_value):.3f}', transform=ax.transAxes, ha='center', va='bottom', fontsize=10, fontweight='bold', color='crimson')

    legend_elements = [Patch(facecolor=palette[i], label=hue_order[i]) for i in range(len(hue_order))]
    legend_elements.append(Line2D([0], [0], color='crimson', linestyle='--', linewidth=1.5, label=th_name))

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=5,
        frameon=False
    )
    return fig
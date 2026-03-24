import os, gc
from datetime import datetime

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from experiment import Experiment

def print_metrics(df, epoch='all', thr=0.5):
    filtered_df = df[
        (df['threshold'] == thr) &
        (df['metric'].isin(['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N', 'mIoU', 'Fmeasure']))
    ]

    if epoch != 'all':
        filtered_df = filtered_df[filtered_df['epoch'] == epoch]

    # 2. Pivot the data
    # index: what you want as rows
    # columns: what you want as side-by-side columns
    # values: the numbers to fill the table
    pivot_df = filtered_df.pivot_table(
        index=['dataset', 'epoch'],
        columns=['audio_type', 'metric'],
        values='value',
    )

    # Define the desired order for each audio_type
    # std_cols = [('std', m) for m in ['cIoU_hat', 'AUC', 'mIoU', 'Fmeasure']]
    std_cols = [('std', m) for m in ['cIoU_hat', 'AUC']]
    silence_cols = [('silence', m) for m in ['pIA_hat', 'AUC_N']]
    noise_cols = [('noise', m) for m in ['pIA_hat', 'AUC_N']]

    # Combine them into one ordered list
    target_columns = std_cols + silence_cols + noise_cols

    # Reindex the columns to the new order
    # errors='ignore' ensures it doesn't crash if a specific metric is missing for one type
    pivot_df = pivot_df.reindex(columns=target_columns)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000 # Increased width to prevent wrapping

    print(pivot_df)
    return pivot_df

def print_metrics_noisy(df, epoch='all', thr=0.5):

    filtered_df = df[
        (df['threshold'] == thr) &
        (df['metric'].isin(['cIoU_hat', 'AUC', 'pIA_hat', 'AUC_N', 'mIoU', 'Fmeasure']))
    ]

    if epoch != 'all':
        filtered_df = filtered_df[filtered_df['epoch'] == epoch]

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
    # std_cols = [('std', m) for m in ['cIoU_hat', 'AUC', 'mIoU', 'Fmeasure']]
    std_cols = [('std', s, m) for s in [5.0, 10.0, 20.0, -1] for m in ['cIoU_hat', 'AUC']]
    silence_cols = [('silence', -1, m) for m in ['pIA_hat', 'AUC_N']]
    noise_cols = [('noise', -1, m) for m in ['pIA_hat', 'AUC_N']]

    # Combine them into one ordered list
    target_columns = std_cols + silence_cols + noise_cols

    # Reindex the columns to the new order
    # errors='ignore' ensures it doesn't crash if a specific metric is missing for one type
    pivot_df = pivot_df.reindex(columns=target_columns)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000 # Increased width to prevent wrapping

    print(pivot_df)
    return pivot_df

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

    std_cols = [('std', m) for m in ['cIoU_hat', 'AUC']]
    silence_cols = [('silence', m) for m in ['pIA_hat', 'AUC_N']]
    noise_cols = [('noise', m) for m in ['pIA_hat', 'AUC_N']]

    target_columns = std_cols + silence_cols + noise_cols

    pivot_df = pivot_df.reindex(columns=target_columns)

    pd.options.display.float_format = "{:,.3f}".format
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.width = 1000 # Increased width to prevent wrapping

    print(pivot_df)
    return pivot_df

def plot_all_metrics(df):
    color_palette = {}
    for dataset, color in zip(sorted(df['dataset'].unique()), sns.color_palette(n_colors=df['dataset'].nunique()).as_hex()):
        color_palette[dataset] = color
    # 1. Setup the style
    sns.set_theme(style="whitegrid")

    # 2. Define strict mappings
    # Mapping line styles to audio types
    style_map = {
        'std': (None, None),  # Solid
        'noise': (5, 5),      # Dashed
        'silence': (1, 2)     # Dotted
    }

    # 3. Get list of unique metrics
    # metrics = df['metric'].unique()
    metrics = ['AUC', 'AUC_N']

    for m in metrics:
        subset = df[df['metric'] == m].copy()
        adap_subset = subset[subset['threshold'] == 'adap']
        none_subset = subset[subset['threshold'] == 'None']
        subset = subset[~((subset['threshold'] == 'adap') | (subset['threshold'] == 'None'))]
        subset['threshold'] = subset['threshold'].astype(float)
        subset = subset.sort_values('threshold')

        # Build per-dataset adap value map for this metric.
        adap_values = {}
        for dataset in adap_subset['dataset'].unique():
            vals = adap_subset[adap_subset['dataset'] == dataset]['value']
            if len(vals) == 0:
                continue
            try:
                adap_values[dataset] = float(vals.iloc[0])
            except (ValueError, TypeError):
                continue

        plt.figure(figsize=(10, 6))

        # 4. Create the lineplot with the fixed palette
        ax = sns.lineplot(
            data=subset,
            x='threshold',
            y='value',
            hue='dataset',
            palette=color_palette,  # Force consistent colors
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
                    text.set_text(f"{label} (adap={adap_values[label]:.3f})")

        # 5. Formatting
        plt.title(f"Metric: {m}", fontsize=15, fontweight='bold')
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

        plt.show()

def plot_all_experiments(experiments):
    merged_df = pd.concat([e.metrics[e.metrics['epoch'] == e.best_epoch] for e in experiments.values()])

    merged_df = merged_df[
        (merged_df['threshold'] == 0.5) &
        (merged_df['dataset'] == 'avatar_one_seg')
    ]

    df = merged_df[merged_df['run'] != 'baseline']

    color_palette = {}
    for dataset, color in zip(sorted(df['run'].unique()), sns.color_palette(n_colors=df['run'].nunique()).as_hex()):
        color_palette[dataset] = color

    # 1. Setup the style
    sns.set_theme(style="whitegrid")

    # 2. Define strict mappings
    # Mapping line styles to audio types
    style_map = {
        'std': (None, None),  # Solid
        'noise': (5, 5),      # Dashed
        'silence': (1, 2)     # Dotted
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
            data=subset,
            x='epoch',
            y='value',
            hue='run',
            palette=color_palette,  # Force consistent colors
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

def boxplots_by_dataset(infer_info_df, dataset_name, threshold_dict, epochs, th_name = 'max_neg', min_max = 'max'):
    df = infer_info_df[(infer_info_df["dataset"] == dataset_name) &
                                    (infer_info_df["min_max"] == min_max)].copy()

    df = df[[e in epochs for e in df['epoch']]]
    col_order = sorted(df["epoch"].dropna().unique())

    df = df.explode("data")

    fig = sns.catplot(df,
        y='data',
        hue='audio_type',
        hue_order=["pos", "sil", "noi"],
        kind="box",
        palette='pastel',
        col='epoch',
        col_order=col_order,
        height=4,
        aspect=0.5,
    )

    for ax, epoch in zip(fig.axes.flat, col_order):
        th_value = threshold_dict.get(int(epoch), {}).get(th_name)
        print(threshold_dict.get(int(epoch), {}))
        if th_value is not None:
            ax.axhline(float(th_value), color='crimson', linestyle='--', linewidth=1.5)

    plt.suptitle(f'{dataset_name} test set ({th_name}_{min_max})')
    sns.move_legend(fig, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, title=None, frameon=False)
    fig.set_titles(y=-0.1)

    plt.tight_layout()
    fig.savefig(f"outputs/{int(datetime.timestamp(datetime.now()))}.png")
    plt.show()

def boxplots_by_dataset_compare(
        list_of_experiments: list[Experiment],
        list_of_epochs,
        dataset_name = 'vggss',
        seg_item = 'm_i_seg',
        th_name = 'max_neg',
        min_max = 'max'
    ):
    if len(list_of_experiments) != len(list_of_epochs):
        return
    N = len(list_of_epochs)

    fig, axs = plt.subplots(nrows=1, ncols=N, sharey=True, figsize=(1.5*N, 4))

    hue_order = ["pos", "sil", "noi"]
    palette = sns.color_palette('pastel', n_colors=len(hue_order))

    for i in range(N):
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
            ax=axs[i],
            legend=False
        )

        axs[i].set_title(f"{list_of_experiments[i].name}@{list_of_epochs[i]}", fontsize=12)
        axs[i].set_ylim([0, 1])

        th_value = list_of_experiments[i].thresholds.get(int(list_of_epochs[i]), {}).get(th_name)
        if th_value is not None:
            axs[i].axhline(float(th_value), color='crimson', linestyle='--', linewidth=1.5)

    legend_elements = [Patch(facecolor=palette[i], label=hue_order[i]) for i in range(len(hue_order))]
    legend_elements.append(Line2D([0], [0], color='crimson', linestyle='--', linewidth=1.5, label=th_name))

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        frameon=False
    )

    # Add global title
    fig.suptitle(f'{dataset_name} ({seg_item} - {min_max})', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
from math import ceil
import os
import pandas as pd
import seaborn as sns

from utils.paths import get_thesis_playground_root_dir, get_full_cloudsc_plot_dir
from utils.plot import get_new_figure, save_plot, legend_on_lines_dict


def main():
    df = pd.read_csv(os.path.join(get_thesis_playground_root_dir(), 'c_cuda.csv')).set_index(['type', 'size', 'nproma'])
    df.drop(2**12, level='size', inplace=True)
    df.drop('cuda_hoist', level='type', inplace=True)
    run_counts = df.groupby(['type', 'size', 'nproma']).count()
    print(run_counts)

    figure = get_new_figure()
    sizes = df.reset_index()['size'].unique()
    nrows = 2
    ncols = ceil((len(sizes) / nrows))
    axes = figure.subplots(nrows, ncols, sharex=True)
    axes_1d = [a for axs in axes for a in axs]
    hue_order = ['cuda', 'cuda_k_caching']
    for idx, (ax, size) in enumerate(zip(axes_1d, sizes)):
        ax.set_title(size)
        sns.lineplot(df.xs(size, level='size'), x='nproma', y='execution_time', hue='type', ax=ax,
                     hue_order=hue_order, marker='o')
        ax.get_legend().remove()
        ax.set_ylabel('')

    axes[0][0].set_ylabel('Runtime [ms]')
    axes[1][0].set_ylabel('Runtime [ms]')
    axes[1][0].set_xlabel('KLON')
    axes[1][1].set_xlabel('KLON')
    axes[1][2].set_xlabel('KLON')
    legend_on_lines_dict(
            axes[0][0],
            {
                'CUDA': {'position': (100, 18), 'color_index': 0},
                'CUDA with K-caching': {'text_position': (70, 30), 'position': (70, 5), 'color_index': 1}
            })
    legend_on_lines_dict(
            axes[0][1],
            {
                'CUDA': {'position': (100, 25), 'color_index': 0},
                'CUDA with K-caching': {'text_position': (70, 45), 'position': (70, 7), 'color_index': 1}
            })
    legend_on_lines_dict(
            axes[0][2],
            {
                'CUDA': {'position': (100, 40), 'color_index': 0},
                'CUDA with K-caching': {'position': (100, 20), 'color_index': 1}
            })
    legend_on_lines_dict(
            axes[1][0],
            {
                'CUDA': {'position': (100, 35), 'color_index': 0},
                'CUDA with K-caching': {'text_position': (70, 100), 'position': (70, 12), 'color_index': 1}
            })
    legend_on_lines_dict(
            axes[1][1],
            {
                'CUDA': {'position': (100, 60), 'color_index': 0},
                'CUDA with K-caching': {'text_position': (70, 100), 'position': (70, 25), 'color_index': 1}
            })
    legend_on_lines_dict(
            axes[1][2],
            {
                'CUDA': {'position': (100, 90), 'color_index': 0},
                'CUDA with K-caching': {'position': (100, 65), 'color_index': 1}
            })

    figure.tight_layout()
    save_plot(os.path.join(get_full_cloudsc_plot_dir(), 'cuda_cloudsc_nproma.pdf'))


if __name__ == '__main__':
    main()

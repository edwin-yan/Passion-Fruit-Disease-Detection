import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.patches import Rectangle
import math


def plot_sample_images(df, top_n=25, n_col=5, is_train=True):
    top_n = top_n if top_n <= len(df) else len(df)
    n_row = math.ceil(top_n / n_col)
    fig_size = (5 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=fig_size)
    img_folder = f'./{"Train" if is_train else "Test"}_Images'
    for idx, row in df.iloc[:top_n, :].iterrows():
        ax = axes[idx // n_col, idx % n_col]
        image = img.imread(f'{img_folder}/{row["Image_ID"]}.jpg')
        ax.imshow(image)
        if is_train:
            rect = Rectangle((row["xmin"], row["ymin"]), row["width"], row["height"], linewidth=1, edgecolor='r',
                             facecolor='none')
            ax.add_patch(rect)
            ax.set_title(row["class"])
    plt.tight_layout()
    plt.show()

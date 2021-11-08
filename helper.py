import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.patches import Rectangle
import math

def plot_sample_images(df, top_n=25, n_col=5, show_label=True, is_train=True, dir_path='.'):
    print("Warning! Deprecated....")
    top_n = top_n if top_n <= len(df) else len(df)
    n_row = math.ceil(top_n / n_col)
    fig_size = (5 * n_col, 4 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=fig_size)
    img_folder = f'{dir_path}/{"Train" if is_train else "Test"}_Images'
    for idx, row in df.iloc[:top_n, :].iterrows():
        ax = axes[idx // n_col, idx % n_col]
        image = img.imread(f'{img_folder}/{row["Image_ID"]}.jpg')
        ax.imshow(image)
        if show_label:
            rect = Rectangle((row["xmin"], row["ymin"]), row["width"], row["height"], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(row["class"])
    plt.tight_layout()
    plt.show()

def plot_sample_images_multi(df, top_n=25, n_col=5, show_label=True, is_train=True, dir_path='.', convert=False):
    if convert:
        df = df.copy(deep=True)
        df['width'] = df['xmax'] - df['xmin']
        df['height'] = df['ymax'] - df['ymin']
    top_n = top_n if top_n <= df.Image_ID.nunique() else df.Image_ID.nunique()
    n_row = math.ceil(top_n / n_col)
    fig_size = (8 * n_col, 7 * n_row)
    fig, axes = plt.subplots(n_row, n_col, figsize=fig_size)
    img_folder = f'{dir_path}/{"Train" if is_train else "Test"}_Images'
    img_ids = df.Image_ID.value_counts().index
    colors = ['b', 'g', 'r', 'm', 'c', '']
    for idx, img_id in enumerate(img_ids[:top_n]):
        ax = axes[idx // n_col, idx % n_col]
        image = img.imread(f'{img_folder}/{img_id}.jpg')
        sub_df = df[df.Image_ID == img_id].reset_index()
        ax.imshow(image)
        if show_label:
            for idx2, row in sub_df.iterrows():
                color = colors[idx2]
                x, y, w, h = row["xmin"], row["ymin"], row["width"], row["height"]
                rect = Rectangle((x, y), w, h, linewidth=2.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x + w / 3, y, row["class"], color=color, fontsize=18, weight='bold')
    plt.tight_layout()
    plt.show()

    
def rgb_splitter(image_id, is_train=True, dir_path='.'):
    img_folder = f'{dir_path}/{"Train" if is_train else "Test"}_Images'
    image = img.imread(f'{img_folder}/{image_id}.jpg')
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 4, figsize=(25,10), sharey = True)
    for i in range(4):
        if i > 0:
            ax[i].imshow(image[:,:,i - 1], cmap = rgb_list[i - 1])
            ax[i].set_title(rgb_list[i - 1], fontsize = 22)
        else:
            ax[i].imshow(image)
            ax[i].set_title("Original", fontsize = 22)
        ax[i].axis('off')
        
    fig.tight_layout()
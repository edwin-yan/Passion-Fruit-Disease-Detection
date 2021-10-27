from tqdm import tqdm
from pathlib import Path
from config import get_class_dict, get_image_size, get_train_label_path

def create_train_label_folder():
    Path(get_train_label_path()).mkdir(parents=True, exist_ok=True)

def label_encoder_transform(df, class_var):
    class_dict = get_class_dict()
    return df[class_var].apply(lambda x: class_dict.get(x))

def convert_to_yolov5(df):
    IMG_HEIGHT, IMG_WIDTH = get_image_size()
    df['xmin'] = df['xmin'] / IMG_WIDTH
    df['ymin'] = df['ymin'] / IMG_HEIGHT
    df['width'] = df['width'] / IMG_WIDTH
    df['height'] = df['height'] / IMG_HEIGHT
    df['x_center'] = df['xmin'] + df['width']/2
    df['y_center'] = df['ymin'] + df['height']/2
    df['label'] = label_encoder_transform(df, 'class')
    create_train_label_folder()
    for img_name in tqdm(df['Image_ID'].unique()):
        df_cov = df[df['Image_ID'] == img_name]
        df_cov = df_cov[['label', 'x_center', 'y_center', 'width', 'height']]
        df_cov.to_csv(f'{get_train_label_path()}/{img_name}.txt', encoding='utf-8',header=False, index=False, sep=' ', mode='a')

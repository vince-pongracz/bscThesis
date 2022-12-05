
import pandas as pd

from IPython.display import HTML

import os
import tensorflow as tf
import rawpy


def show_errors_in_table(data: list[int], html_name:str, error_name: str = 'Errors:', index_head:str = 'Images:') -> pd.DataFrame:
    # https://www.geeksforgeeks.org/how-to-render-pandas-dataframe-as-html-table/
    # https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
    idxs = []
    for i in range(len(data)):
        label = f'img_{i}'
        idxs.append(label)
    
    tmp = {
        # "Images:": idxs,
        error_name: data
    }
    
    df = pd.DataFrame(tmp, index=idxs)
    df.index.name=index_head
    df = df.transpose()
    df.style.set_caption("Hello World")
    df.to_html(html_name)
    return df


# errors = [751.16015625, 1429.07568359,  798.04833984, 1031.99560547, 1332.60961914,
#           730.02050781,  783.16162109,  902.66943359, 1963.48242188,  773.1328125,
#           1157.49511719,  482.08544922, 1087.23779297,  560.63818359]

# df = show_errors_in_table(errors, 'wb_loss.html', 'Loss of model_idx_0:', index_head='train_x images:')
# print('Loss on training set:')
# print(df)

# print('done')

def gen_jpg_from_raw_without_prep(raw_dir: str = f'resources{os.path.sep}konstanz_test_set',
                          jpg_dir: str = f'resources{os.path.sep}raw_samples_without_postprocess'):

    files = os.listdir(raw_dir)
    
    bits_for_color = 16
    scale_color_space = float(2 ** bits_for_color)

    if not os.path.exists(jpg_dir):
        os.mkdir(jpg_dir)

    for file in files:
        name, ext = file.split('.')
        if ext != 'xmp':
            with rawpy.imread(f'{raw_dir}{os.path.sep}{file}') as rawImg:
                rgbImg = rawImg.postprocess(rawpy.Params(output_bps=bits_for_color))
                rgbNormed = rgbImg / scale_color_space
                jpgName = f'{jpg_dir}{os.path.sep}{name}.jpg'
                tf.keras.utils.save_img(jpgName, rgbNormed)
                
                
gen_jpg_from_raw_without_prep()
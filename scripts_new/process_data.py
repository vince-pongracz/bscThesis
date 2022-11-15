
import os

import numpy as np

import rawpy

import tensorflow as tf
ResizeMethod = tf.image.ResizeMethod

from exiftool import ExifToolHelper

import shutil

import datasets
from datasets import Dataset

from datetime import datetime

def process_meta(exifMeta: list):
    # this transformation is required to get the exif fields
    exifMeta = exifMeta[0]
    processed = [
        # "colorTemp":
        int(exifMeta["XMP:ColorTemperature"]),
        # "tint":
        int(exifMeta["XMP:Tint"]),
        # "brightness":
        float(exifMeta["XMP:Exposure2012"]),
        # "contrast":
        int(exifMeta["XMP:Contrast2012"]),
        # "vibrance":
        int(exifMeta["XMP:Vibrance"])
        # occasionally add other fields, as start it will be goo like this
    ]
    return processed

def data_load_and_preprocess(directory: str = f'resources{os.path.sep}raws', path_to_generated_jpgs: str = f'resources{os.path.sep}genJPGs'):

    rawImages: np.array = []
    xmpData: np.array = []
    files = os.listdir(directory)
    if not os.path.exists(path_to_generated_jpgs):
        os.mkdir(path_to_generated_jpgs)
    convertedImageSize = [133, 200]

    for file in files:
        name, ext = file.split('.')
        if ext != 'xmp':
            with rawpy.imread(f'{directory}{os.path.sep}{file}') as rawImg:
                rgbImg = rawImg.postprocess(rawpy.Params(use_camera_wb=True))
                rgbNormed = rgbImg / 255.0
                bilinear = tf.image.resize(
                    rgbNormed, convertedImageSize, method=ResizeMethod.BILINEAR)
                jpgName = f'{path_to_generated_jpgs}{os.path.sep}{name}.jpg'
                tf.keras.utils.save_img(jpgName, bilinear)
                rawImages.append(bilinear)
        else:
            with ExifToolHelper() as et:
                fileName = f'{directory}{os.path.sep}{file}'
                shutil.copyfile(fileName, f'{path_to_generated_jpgs}{os.path.sep}{file}')
                meta = et.execute_json(fileName)
                processedExif = tf.convert_to_tensor(process_meta(meta))
                xmpData.append(processedExif)

    # images = tf.keras.utils.image_dataset_from_directory(directory=directory, image_size=tuple(convertedImageSize))
    print('test tensorflow load')

    return rawImages, xmpData

def saveDatasets(imgs, xmps, target_path: str = f'datasets{os.path.sep}ds', add_timestamp: bool = True) -> None:
    ds = Dataset.from_dict({"img": imgs, "exif": xmps})
    
    if add_timestamp:
        date = datetime.now()
        currDate = '{:04d}{:02d}{:2d}_{:02d}{:2d}{:2d}'.format(date.year, date.month, date.day, date.hour, date.minute, date.second )
        target_path = f'{target_path}_{currDate}'
    
    ds.save_to_disk(target_path)
    # TODO push ds to google drive :D
    print('dataset saved!')



print('PREPROCESS DATA')
print('---script starts---')

# edit these
rawBigger_path = f'resources{os.path.sep}raw_imgs'
path_to_gen_jpgs = f'resources{os.path.sep}tempjpgs'

# can stay like that
ds_path_default = f'datasets{os.path.sep}ds'

imgs, xmps = data_load_and_preprocess(directory=rawBigger_path, path_to_generated_jpgs=path_to_gen_jpgs)
saveDatasets(imgs, xmps)

print('---script ended---')

import os
import re

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

def cleanData(directory: str = 'resources\\rawTest') -> None:
    files = os.listdir(directory)

    # TODO unique constraint for the samples

    acceptedExtensions = ['xmp', 'ARW', 'arw', 'NEF', 'nef', 'cr2', 'CR2']

    for file in files:
        fullName = os.path.join(directory, file)
        name, extension = file.split('.')
        if (extension not in acceptedExtensions):
            os.remove(fullName)
            files.remove(file)
        else:
            regex = re.compile(f'{name}.*')
            count = len(list(filter(regex.match, files)))
            if (count != 2):
                # print(f'remove: {fullName}')
                os.remove(fullName)
                files.remove(file)

def data_load_and_preprocess(directory: str = f'resources{os.path.sep}raws', path_to_generated_jpgs: str = f'resources{os.path.sep}genJPGs'):

    rawImages: np.array = []
    xmpData: np.array = []
    files = os.listdir(directory)
    if not os.path.exists(path_to_generated_jpgs):
        os.mkdir(path_to_generated_jpgs)
    convertedImageSize = [133, 200]
    
    bits_for_color = 16
    scale_color_space = float(2 ** bits_for_color)
    save_jpgs = True

    for file in files:
        name, ext = file.split('.')
        # check if it is a raw file
        if ext != 'xmp':
            with rawpy.imread(f'{directory}{os.path.sep}{file}') as rawImg:
                # additional parameter to postprocess: rawpy.Params(use_camera_wb=True)
                rgbImg = rawImg.postprocess(rawpy.Params(output_bps=bits_for_color))
                rgbNormed = rgbImg / scale_color_space
                bilinear = tf.image.resize(
                    rgbNormed, convertedImageSize, method=ResizeMethod.BILINEAR)
                if save_jpgs:
                    jpgName = f'{path_to_generated_jpgs}{os.path.sep}{name}.jpg'
                    tf.keras.utils.save_img(jpgName, bilinear)
                rawImages.append(bilinear)
        else: # so it is an .xmp file
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
    # https://huggingface.co/docs/datasets/package_reference/main_classes
    ds.save_to_disk(target_path)
    # TODO push ds to google drive :D
    print('dataset saved!')



print('PREPROCESS DATA')
print('---script starts---')

# edit these
raw_bigger_path = f'resources{os.path.sep}mixed_ds'
path_to_gen_jpgs = f'resources{os.path.sep}tempjpgs_mixed'

# can stay like that
ds_path_default = f'datasets{os.path.sep}ds_mixed'

cleanData(raw_bigger_path)
imgs, xmps = data_load_and_preprocess(directory=raw_bigger_path, path_to_generated_jpgs=path_to_gen_jpgs)
saveDatasets(imgs, xmps, target_path=ds_path_default)

print('---script ended---')
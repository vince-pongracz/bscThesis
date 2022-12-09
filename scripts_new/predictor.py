
# Imports:
from distutils.util import strtobool
import pathlib

import argparse

import datasets
from datasets import Dataset

import shutil
from copyreg import pickle

import rawpy

import pandas as pd

import os
import datetime

import numpy as np

import cv2

from exiftool import ExifToolHelper
import tensorboard

import subprocess

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
ResizeMethod = tf.image.ResizeMethod

# Functions, methods, classes

"""
Recieves a list with metadata, which is a dictonary and extracts the values from target tags.
:param exifmeta: metadata
:return: list of target values for one image
"""
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


"""
Data structure to store image corrections
"""
class Correction:
    num_pred: int = 5

    def __init__(self, pred_list: list):
        list_len = len(pred_list)
        for i in range(Correction.num_pred - list_len):
            pred_list.append(0)
            
        [self.colorTemp, self.tint, self.brightness, self.contrast, self.vibrance] = pred_list




"""
Writes the XMP file to path for pictureName file. With correction data will be the XMP filled.
"""
def generate_xmp_result(path: str = 'resources\\raws', pictureName: str = 'sample4.NEF', correction: Correction = Correction([7000, -4, 1.3, 10, 4])):
    fullFileNameRaw = os.path.join(path, pictureName)
    fullFileNameXmp = fullFileNameRaw.split(".")[0] + ".xmp"

    date = datetime.datetime.now()

    xmpString = f"""<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Adobe XMP Core 5.6-c140 79.160451, {date}        ">
        <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about=""
                xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
                crs:Temperature="{round(correction.colorTemp)}"
                crs:Tint="{round(correction.tint)}"
                crs:Exposure2012="{correction.brightness}"
                crs:Contrast2012="{round(correction.contrast)}"
                crs:Vibrance="{round(correction.vibrance)}"
                crs:AutoLateralCA="1">
            </rdf:Description>
        </rdf:RDF>
    </x:xmpmeta>"""

    with open(fullFileNameXmp, "w") as xmpFile:
        xmpFile.write(xmpString)

# image size. To this size will be converted the images.
convertedImageSize = [133, 200]



"""
prepare and preprocess images for the prediction
"""
def prepare_to_prediction(raw_dir: str = f'resources{os.path.sep}raws', jpg_dir: str = f'resources{os.path.sep}predictOnThem', save_jpg: bool = False) -> list:

    images: np.array = []
    image_names = []
    files = os.listdir(raw_dir)
    
    bits_for_color = 16
    scale_color_space = float(2 ** bits_for_color)

    if save_jpg and not os.path.exists(jpg_dir):
        os.mkdir(jpg_dir)

    for file in files:
        name, ext = file.split('.')
        if ext != 'xmp':
            with rawpy.imread(f'{raw_dir}{os.path.sep}{file}') as rawImg:
                rgbImg = rawImg.postprocess(rawpy.Params(output_bps=bits_for_color))
                rgbNormed = rgbImg / scale_color_space
                bilinear_img = tf.image.resize(rgbNormed, convertedImageSize, method=ResizeMethod.BILINEAR)
                if save_jpg:
                    jpgName = f'{jpg_dir}{os.path.sep}{name}.jpg'
                    tf.keras.utils.save_img(jpgName, bilinear_img)
                images.append(bilinear_img)
                image_names.append(name)

    # standardization step, the models have learnt on standardized data...
    images = np.asarray(images)
    images = tf.image.per_image_standardization(images).numpy()
        
    print('preparePredict')

    return images, image_names


"""
Load models from path.
"""
def load_models(path: str = 'models', file_format: str = 'h5'):
    print(f'load models from: {path}')
    model_files = os.listdir(path)
    num_models = len(model_files)
    # assert(num_models == 5)
    models = []
    for i in range(num_models):
        model = load_model(f'{path}{os.path.sep}model_{i}.{file_format}')
        models.append(model)
    return models

"""
Predict every choosen image property for every image
"""
def predict_on_models(images, models: list, verbose:bool = False):
    verbosity = 'auto'
    if verbose:
        verbosity = '1'
        
    all_predictions = []
    for img in images:
        img_predictions = []
        img = tf.expand_dims(img, axis=0)
        for m in models:
            m_res = m.predict(img, verbose=verbosity)
            m_res = np.ndarray.flatten(m_res)[0]
            img_predictions.append(m_res)
        all_predictions.append(img_predictions)

    return all_predictions

"""
Generate XMP corrections for multiple images
"""
def generate_batch_xmps(target_dir:str, img_names_and_predictions):
    for pic_name, pred in img_names_and_predictions:
        corr = Correction(pred)
        generate_xmp_result(target_dir, pictureName=pic_name, correction=corr)
    
"""
Move or copy the images, if it is specified.
"""
def move_results(source: str, target: str, create_copy: bool = False) -> None:
    allfiles = os.listdir(source)
    
    for file in allfiles:
        target_name = target + os.path.sep + file
        
        if os.path.exists(target_name):
            print(target_name,'exists in the destination path - overwrite!')
            os.remove(target_name)
        img_source_path = os.path.join(source, file)
        
        if create_copy:
            shutil.copyfile(img_source_path, target_name)
        else:
            shutil.move(img_source_path, target)

    print('move done')
    
"""
Open and execute Lightroom.exe
"""
def open_lightroom(lr_path:str):
    subprocess.run([lr_path])
    
# Main process, do the job:
if __name__ == '__main__':
    assert (pd.__version__ == '1.3.5')
    
    # Defaults
    sep = os.path.sep
    raw_source_path = f'resources{sep}konstanz_test_set'
    lr_target_path = f'resources{sep}auto_imp_dir'
    path_to_models = 'models'
    lr_executable_path = f'C:{sep}Program Files{sep}Adobe{sep}Adobe Lightroom Classic{sep}Lightroom.exe'
    
    rawBigger_path = f'resources{sep}rawTest'
    
    program_description = "This program helps speeding up retouch workflows"
    
    # keep LR exec in environment variables
    parser = argparse.ArgumentParser(description=program_description)
    parser.add_argument("raw_source_dir", type=pathlib.Path, help='path to the raw image source directory')
    parser.add_argument("--lr_target_dir", default=lr_target_path, type=pathlib.Path, help='target dir for Lightroom open, here will the predictions and the original photos land')
    parser.add_argument("-m", "--path_to_models", type=pathlib.Path, default=path_to_models, help='Specify the path to the models, which are required to predict retouch values. Default is ./models')
    parser.add_argument('-o',"--open_up_LR", choices=('True','False'), help='Switches ON/OFF auto-Lightroom start', default='True')
    parser.add_argument("-LRe", "--lr_executable_path", type=pathlib.Path, default=lr_executable_path, help=f'Specify where to find Lightroom executable, default: {lr_executable_path}')
    parser.add_argument("-c", '--create_copy', choices=('True','False'), help='Leave raw files where they were or bring them under Lightroom, default is False', default='False')
    parser.add_argument("-v", '--verbose', help='Verbosity switch', action="store_true")
    
    args = parser.parse_args()
    # argument assignments
    raw_source_path = str(args.raw_source_dir)
    lr_target_path = str(args.lr_target_dir)
    path_to_models = str(args.path_to_models)
    open_lr_after_pred = bool(strtobool(args.open_up_LR))
    create_copy = bool(strtobool(args.create_copy))
    verbose = bool(args.verbose)
    
    # get LR exe from configured env variable
    env_LR = os.environ.get('LR') # returns None if not present
    if not env_LR == None:
        lr_executable_path = str(env_LR)
    else:
        lr_executable_path = str(args.lr_executable_path)
    
    # Prompt some valueable info about the current run config
    print('--- start script ---')
    print(' with the following settings :')
    print(' ----- ')
    print(f' source dir: {raw_source_path}')
    print(f' target dir: {lr_target_path}')
    print(f' path to models: {path_to_models}')
    print(f' LR opens: {open_lr_after_pred}')
    print(f' LR executable: {lr_executable_path}')
    print(f' create copy: {create_copy}')
    print(' ----- ')
    
    if verbose:
        print('Diagnostic:')
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print('Tensorflow version: ' + tf.__version__)
    
    # load
    nn_models = load_models(path=path_to_models)

    # prepare
    images, image_names = prepare_to_prediction(raw_source_path)
    # predict
    predictions = predict_on_models(images, nn_models, verbose=verbose)
    # transform
    prediction_results = list(zip(image_names, predictions))
    # generate output
    generate_batch_xmps(raw_source_path, prediction_results)
    
    # post task, move images
    move_results(raw_source_path, lr_target_path, create_copy=create_copy)
    
    # open Lightroom, when it was asked
    if open_lr_after_pred:
        open_lightroom(lr_executable_path)
    
    print('--- script ended ---')
    
    
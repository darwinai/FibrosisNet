import cv2
from pydicom import dcmread
import tensorflow.compat.v1 as tf

import pandas as pd
import numpy as np
import random
import math
from functools import partial

import sys
import os

######################################################################################
################              Functions for eval.py                    ###############
######################################################################################

def _create_patient_data_df(csv_path):
    # Create a pandas dataframe using the given csv 
    metadata_list = []
    csv_df = pd.read_csv(csv_path)
    for patient_id in csv_df.Patient.unique():
        patient_dict = {}
        patient_data = csv_df[csv_df.Patient == patient_id]
        
        metadata_vec = _get_patient_tab_vector(patient_data)
        cur_data = patient_data.iloc[0] # Current corresponds to first available data point
        
        patient_dict['Patient'] = patient_id
        patient_dict['metadata'] = metadata_vec.astype(np.float32)
        patient_dict['cur_fvc'] = float(cur_data.FVC)
        patient_dict['cur_week'] = float(cur_data.Weeks)

        metadata_list.append(patient_dict)
    return pd.DataFrame(metadata_list)


def _get_patient_tab_vector(df):
    # This function is adapted from the Kaggle Winner's notebook
    # https://www.kaggle.com/artkulak/inference-45-55-600-epochs-tuned-effnet-b5-30-ep
    vector = [(df.Age.values[0] - 30) / 30]
    
    if df.Sex.values[0] == 'Male':
       vector.append(0)
    else:
       vector.append(1)
    
    if df.SmokingStatus.values[0] == 'Never smoked':
        vector.extend([0,0])
    elif df.SmokingStatus.values[0] == 'Ex-smoker':
        vector.extend([1,1])
    elif df.SmokingStatus.values[0] == 'Currently smokes':
        vector.extend([0,1])
    else:
        vector.extend([1,0])
        
    return np.array(vector)


def _get_file_names(image_dir):
    # Retrieve the filenames of the images we wish to process
    all_images = []
    for patient in os.listdir(image_dir):
        patient_dir = os.path.join(image_dir, patient)
        if os.path.isdir(patient_dir):
            image_files = os.listdir(patient_dir)
            image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            
            left = int(len(image_files) * 0.4)
            right = int(len(image_files) * 0.05)
            
            image_files = image_files[left:-right] if right > 0 else image_files[left:]
            image_files = [os.path.join(patient_dir, f) for f in image_files]
            all_images.extend(image_files)
    return all_images


def _parse_dicom_data(file):
    # Read in a dicom image and preprocess it 
    image, patient_id = tf.py_func(_parse_dicom_data_pyfunc, [file], Tout=[tf.float32, tf.string])
    patient_id.set_shape([1])

    image.set_shape([512, 512])
    image = tf.stack((image,)*3, axis=-1)
    image = tf.squeeze(image)

    return {'image' : image,
            'Patient' : patient_id}


def _parse_dicom_data_pyfunc(file):
    # The python function that actually does the preprocessing 
    dcm = dcmread(file.decode("utf-8"))
    image = dcm.pixel_array
    
    if image.shape[0] == 1:
        image = image.squeeze(axis=0)
    if image.shape[-1] == 1:
        image = image.squeeze(axis=-1)
    
    if abs(image.shape[0] - image.shape[1]) >= 5 and image.shape[0] > 512:
        image = _crop(image, 512, 512)
    
    image = cv2.resize(image, (512, 512)).astype(np.float32)
    image = np.reshape(image, (512, 512, 1))

    # Convert Pixel Arrays to HU via HU = m*P + b
    image = dcm.RescaleSlope * image + dcm.RescaleIntercept

    # determine clip range and scale factor
    if _has_circular_artifacting(image):
        CLIP_RANGE=[-1000,200]
    else:
        CLIP_RANGE=[-1500,200]
    SCALE_FACTORS=[650, 850]

    # clip and rescale values
    image = np.clip(image, a_min=CLIP_RANGE[0], a_max=CLIP_RANGE[1])
    image = (image + SCALE_FACTORS[0]) / SCALE_FACTORS[1]
    
    image = np.reshape(image, (512, 512, 1))
    patient_id = dcm.PatientID
    
    return image, patient_id

def _has_circular_artifacting(image):
    return (np.sum(image <= -1500, axis=None) / np.sum(np.ones(image.shape), axis=None)) > 0.20

def _crop(image: np.ndarray, height: int, width: int):
    # Crop the image to the specified height and width, using the 
    # center of the image as the origin.
    assert len(image.shape) == 2
        
    if abs(image.shape[0] - height) <= 1 and abs(image.shape[1] - width) <= 1:
        return image
    
    center_y = int(image.shape[0] / 2)
    center_x = int(image.shape[1] / 2)
    crop = image[
        int(center_y - height/2) : int(center_y + height/2), 
        int(center_x - width/2)  : int(center_x + width/2)
    ]
    return crop

######################################################################################


######################################################################################
################       Functions for fibrosis_clinical.py              ###############
######################################################################################

def set_env_seed(seed=2020):
    # Set the python environment seed, for consistent random results
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def run_single_model(model, train_df, test_df, folds, features, target, fold_num=0):
    # Run predictions using a single model
    trn_idx = folds[folds.fold!=fold_num].index
    val_idx = folds[folds.fold==fold_num].index
    
    y_tr = target.iloc[trn_idx].values
    X_tr = train_df.iloc[trn_idx][features].values
    y_val = target.iloc[val_idx].values
    X_val = train_df.iloc[val_idx][features].values
    
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))
    model.fit(X_tr, y_tr)
    
    oof[val_idx] = model.predict(X_val)
    predictions += model.predict(test_df[features])
    return oof, predictions

def run_kfold_model(model, train, test, folds, features, target, n_fold=9):   
    # Run predictions using a single model, using n-fold averaging 
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    
    feature_importance_df = pd.DataFrame()
    for fold_ in range(n_fold):
        _oof, _predictions = run_single_model(model, train, test, folds, features, target, fold_num = fold_)
        oof += _oof
        predictions += _predictions/n_fold
    
    return oof, predictions


def loss_func(weight, row):
    # Predefined loss function
    confidence = weight
    sigma_clipped = max(confidence, 70)
    diff = abs(row['FVC'] - row['FVC_pred'])
    delta = min(diff, 1000)
    score = -math.sqrt(2)*delta/sigma_clipped - np.log(math.sqrt(2)*sigma_clipped)
    return -score

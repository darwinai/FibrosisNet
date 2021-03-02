'''
FibrosisNet Evaluation Script

This python script allows users to perform evaluation
using the FibrosisNet ensemble

This script uses CT_WEIGHT=1.0 because that gives the 
best performance on the kaggle submission.
'''

import os
import sys
import random
import math
import cv2
import argparse
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import pickle
import category_encoders as ce
import scipy as sp
from tqdm import tqdm
from dsi import OSICFibrosisDSI

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# Constants (don't edit these unless you know what you're doing)
DATA_FEED_DICT = {
    'image'       : 'Placeholder:0',
    'is_training' : 'is_training:0',
    'metadata'    : 'metadata:0',
}
FETCH_DICT = {
    'prediction/slope' : 'pred_slope:0',
}
#=================================================================

def run_evaluation(model_path, input_path, ct_weight, output_file):
    # Check the file formats in the input path.
    # Should contain 1 csv file with clinical metadata, and optionally CT scan images.
    csv_path = os.path.join(input_path, "example.csv")
    assert os.path.exists(csv_path)

    # Setup the model and corresponding graph
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    # Load model
    with tf.gfile.GFile(os.path.join(model_path, "FibrosisNetCT.pb"), 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=''
    )

    # Create and attach dataset interface
    dsi = OSICFibrosisDSI(csv_path=csv_path, ct_path=input_path)
    ds_test, num_test_samples, test_batch_size = dsi.get_test_dataset()
    ds_iter_test = ds_test.make_initializable_iterator()
    test_inputs = ds_iter_test.get_next()

    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run([ds_iter_test.initializer, tf.tables_initializer()])

    # Store predictions in DataFrame
    pred = pd.DataFrame(columns=['Patient', 'cur_fvc', 'cur_week', 'slope'])

    # Run eval iters
    print('Evaluating using FibrosisNetCT...')
    eval_iters = num_test_samples // test_batch_size
    for i in tqdm(range(eval_iters)):
        data_feeds = sess.run(test_inputs)
        feed_dict = {
            graph.get_tensor_by_name(tname): data_feeds[data_key] for data_key, tname in DATA_FEED_DICT.items() \
                if data_key in data_feeds
        }

        fetch_values = sess.run(FETCH_DICT, feed_dict=feed_dict)
        pred = pred.append({
            'Patient'     : data_feeds['Patient'].item().decode("utf-8"),
            'cur_fvc'     : data_feeds['cur_fvc'].item(),
            'cur_week'    : data_feeds['cur_week'].item(),
            'slope'       : fetch_values['prediction/slope'].item(),
        }, ignore_index=True)

    # Predict FVCs (forced vital capacity)
    ct_sub = pd.DataFrame()
    all_weeks = np.array(range(-12, 134))

    for patient in pred.Patient.unique():
        filtered = pred[pred.Patient == patient]
        slope = filtered.slope.median()
        cur_week = int(filtered.cur_week.unique()[0])
        cur_fvc = filtered.cur_fvc.unique()[0]
        patient_weeks = [patient + "_" + str(w) for w in all_weeks]
        
        # predict fvc
        intercept = cur_fvc - slope * cur_week
        pred_fvcs = intercept + slope * all_weeks
        
        ct_sub = ct_sub.append(pd.DataFrame({
            "Patient_Week":patient_weeks,
            "FVC":pred_fvcs
        }))

    # Evaluation Input Construction for second round of predictions
    input_data = pd.read_csv(csv_path)
    # Duplicate each patient in the evaluation input, across the entire range of weeks we predict for
    all_data = pd.DataFrame()
    all_weeks = np.array(range(-12, 134))

    for patient in input_data.Patient.unique():
        patient_weeks = [patient + "_" + str(w) for w in all_weeks]
        patient_id = [patient for w in all_weeks]
        predict_weeks = [w for w in all_weeks]

        filtered = input_data[input_data.Patient == patient]
        base_weeks = [filtered["Weeks"].values[0] for w in all_weeks]
        base_FVC_weeks = [filtered["FVC"].values[0] for w in all_weeks]
        percent_weeks = [filtered["Percent"].values[0] for w in all_weeks]
        age_weeks = [filtered["Age"].values[0] for w in all_weeks]
        sex_weeks = [filtered["Sex"].values[0] for w in all_weeks]
        smokingstatus_weeks = [filtered["SmokingStatus"].values[0] for w in all_weeks]

        all_data = all_data.append(pd.DataFrame({
            "Patient_Week":patient_weeks,
            "Patient": patient_id,
            "predict_Week": predict_weeks,
            "base_Week": base_weeks,
            "base_FVC": base_FVC_weeks,
            "Percent": percent_weeks,
            "base_Age": age_weeks,
            "Sex": sex_weeks,
            "SmokingStatus": smokingstatus_weeks,
        }))

    all_data['Week_passed'] = all_data['predict_Week'] - all_data['base_Week']
    all_data.set_index('Patient_Week', inplace=True)
    
    # Use the saved FibrosisNetClinical model to also predict
    print('Evaluating using FibrosisNetClinical...')
    # Load the model
    model = pickle.load(open(os.path.join(model_path, "FibrosisNetClinical_FVC.sav"), 'rb'))

    all_data["FVC"] = np.nan
    # Load the Ordinal Encoder from before, and encode the test data
    ce_oe = pickle.load(open(os.path.join(model_path, "Encoder.obj"), 'rb'))
    all_data = ce_oe.transform(all_data)

    # Order the columns properly before input into the model
    fvc_pred_data = all_data[['base_FVC', 'base_Age', 'Week_passed', 'Sex', 'SmokingStatus']].copy()

    # Predict using the model
    predictions = np.zeros(len(fvc_pred_data))
    predictions += model.predict(fvc_pred_data)
    all_data['FVC_pred'] = predictions

    # Obtain confidences
    model = pickle.load(open(os.path.join(model_path, "FibrosisNetClinical_Conf.sav"), 'rb'))
    all_data['Confidence'] = np.nan
    conf_pred_data = all_data[['base_FVC', 'Percent', 'base_Age', 'Week_passed', 'Sex', 'SmokingStatus']].copy()

    predictions = np.zeros(len(conf_pred_data))
    predictions += model.predict(conf_pred_data)
    all_data['Confidence'] = predictions

    all_data = all_data.reset_index()

    all_data = all_data[["Patient_Week", "FVC_pred", "Confidence"]]
    all_data = all_data.rename(columns={'FVC_pred': 'FVC'})
    clinical_sub = all_data.copy()

    # Ensemble the results to obtain the final result
    df1 = ct_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    df2 = clinical_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    # Convert the columns to float values
    df1['FVC'] = pd.to_numeric(df1['FVC'], downcast="float")
    df2['FVC'] = pd.to_numeric(df2['FVC'], downcast="float")
    final_sub = df1[['Patient_Week']].copy()
    final_sub['FVC'] = ct_weight * df1['FVC'] + (1 - ct_weight) * df2['FVC']
    final_sub['Confidence'] = df2['Confidence']
    final_sub.head()
    final_sub.to_csv(output_file, index=False)
    print("Results generated!")
    print('**DISCLAIMER**')
    print('Do not use this prediction for self-diagnosis. You should check with your local authorities for the latest advice on seeking medical assistance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FibrosisNet Evaluation')
    parser.add_argument('--modelpath', default='models/', type=str, help='Path to the folder containing the models')
    parser.add_argument('--inputpath', default='example_input/', type=str, help='Folder containing all inputs. \
        A .csv containing clinical data, and folder containing the corresponding CT scans.')
    parser.add_argument('--ctweight', default='1.0', type=float, help='The weight the CT scan has on the final decision')
    parser.add_argument('--outputfile', default='results.csv', type=str, help='Output results .csv file')

    args = parser.parse_args()
    assert os.path.exists(args.modelpath)
    assert os.path.exists(args.inputpath)

    run_evaluation(args.modelpath, args.inputpath, args.ctweight, args.outputfile)
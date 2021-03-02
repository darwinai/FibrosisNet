'''
FibrosisNet Kaggle Submission

This python script will create the submission file that we
submitted to the Kaggle competition.

Running this script with `CT_WEIGHT=1.0` (best performance)
achieves -6.8188 private score. Running it with 
`CT_WEIGHT=0.96` achieves -6.8195 private score. Note that 
the Darwin team have reported a 0.0001 deviation in score 
between different Kaggle accounts and days of testing.
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
import category_encoders as ce
import scipy as sp
import pickle

from pydicom import dcmread
from functools import partial
from tqdm import tqdm
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, GroupKFold

from dsi import OSICFibrosisDSI
from utility import run_kfold_model, set_env_seed, loss_func

tf.disable_eager_execution()
# Suppress TensorFlow's warning messages
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

# Constants (don't edit these unless you know what you're doing)
SEED = 123
N_FOLD = 10

BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DATA_FEED_DICT = {
    'image'       : 'Placeholder:0',
    'is_training' : 'is_training:0',
    'metadata'    : 'metadata:0',
}
FETCH_DICT = {
    'prediction/slope' : 'pred_slope:0',
}

# ===================================================================

def generate_kaggle_results(data_path, model_path, ct_weight, output_file):
    # Model Checkpoint Directory
    MODEL_DIR = model_path
    # OSIC Pulmonary Fibrosis Progression Dataset
    DATA_DIR = data_path
    TRAIN_CSV = os.path.join(data_path, 'train.csv')
    TEST_CSV = os.path.join(data_path, 'test.csv')
    SUBMISSION_CSV = os.path.join(data_path, 'sample_submission.csv')

    # ===================================================================
    #                              FibrosisNet CT
    #           Uses CT scan images along with clinical data to predict
    # ===================================================================

    # Make sure the specified paths exist
    assert os.path.exists(MODEL_DIR)
    assert os.path.exists(TRAIN_CSV)
    assert os.path.exists(TEST_CSV)
    assert os.path.exists(SUBMISSION_CSV)

    # Setup the model and corresponding graph
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    # Load model
    with tf.gfile.GFile(os.path.join(MODEL_DIR, 'FibrosisNetCT.pb'), 'rb') as f:
        restored_graph_def = tf.GraphDef()
        restored_graph_def.ParseFromString(f.read())

    tf.import_graph_def(
        restored_graph_def,
        input_map=None,
        return_elements=None,
        name=''
    )

    # Create and attach dataset interface
    dsi = OSICFibrosisDSI(csv_path=TEST_CSV, ct_path=os.path.join(DATA_DIR, "test/"))
    ds_test, num_test_samples, test_batch_size = dsi.get_test_dataset()
    ds_iter_test = ds_test.make_initializable_iterator()
    test_inputs = ds_iter_test.get_next()

    sess = tf.Session()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    sess.run([ds_iter_test.initializer, tf.tables_initializer()])

    # Store predictions in DataFrame
    pred = pd.DataFrame(columns=['Patient', 'cur_fvc', 'cur_week', 'slope'])

    # Run eval iters
    eval_iters = num_test_samples // test_batch_size
    print('FibrosisNetCT')
    print('Num samples:', num_test_samples)
    print('Batch size:', test_batch_size)
    print('Evaluating for {} iters'.format(eval_iters))

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
    print("Predicting FVCs...")
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

    # ===================================================================
    #                            FibrosisNet Clinical
    #                      Uses clinical data to predict
    # ===================================================================

    print("FibrosisNet Clinical")
    set_env_seed(seed=SEED)
    train_orig = pd.read_csv(TRAIN_CSV)
    test_orig = pd.read_csv(TEST_CSV)

    # Reformat train df, by concatenating the train and test sets together 
    # We don't use the following columns: 
    # Age, Sex, SmokingStatus, or Percent
    print("Setting up training data...")
    train = pd.concat([train_orig, test_orig])
    output = pd.DataFrame()
    train_grouped = train.groupby('Patient')
    for _, usr_df in tqdm(train_grouped, total = len(train_grouped)):
        usr_output = pd.DataFrame()
        for week, tmp in usr_df.groupby("Weeks"):
            rename_cols = {'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'}
            tmp = tmp.rename(columns = rename_cols)
            drop_cols = ['Age', 'Sex', 'SmokingStatus', 'Percent'] 
            _usr_output = usr_df.drop(columns=drop_cols).rename(columns={'Weeks': 'predict_Week'}).merge(tmp, on='Patient')
            _usr_output['Week_passed'] = _usr_output['predict_Week'] - _usr_output['base_Week']
            # concat the empty DF with edited DF
            usr_output = pd.concat([usr_output, _usr_output])
        output = pd.concat([output, usr_output])

    train = output[output['Week_passed']!=0].reset_index(drop=True)

    # Use the submission sample as the test set instead
    # get patient and weeks to predict from submission sample
    print("Setting up test data...")
    submission = pd.read_csv(SUBMISSION_CSV)
    submission['Patient'] = submission['Patient_Week'].apply(lambda x:x.split('_')[0])
    submission['predict_Week'] = submission['Patient_Week'].apply(lambda x:x.split('_')[1]).astype(int)
    test = test_orig.rename(columns={'Weeks': 'base_Week', 'FVC': 'base_FVC', 'Age': 'base_Age'})
    test = submission.drop(columns = ["FVC", "Confidence"]).merge(test, on = 'Patient')
    test['Week_passed'] = test['predict_Week'] - test['base_Week']
    test.set_index('Patient_Week', inplace=True)

    # Split into folds for cross validation
    folds = train[['Patient', 'FVC']].copy()
    Fold = GroupKFold(n_splits=N_FOLD)
    groups = folds['Patient'].values
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['FVC'], groups)):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)

    target = train['FVC']
    test['FVC'] = np.nan

    # features
    cat_features = ['Sex', 'SmokingStatus'] # categorical features
    num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)] # numerical features

    features = num_features + cat_features
    drop_features = ['FVC', 'predict_Week', 'Percent', 'base_Week']
    features = [c for c in features if c not in drop_features]

    if cat_features:
        ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')
        ce_oe.fit(train)
        # Save the ordinal encoder for evaluation purposes
        save_oe_filename = os.path.join(MODEL_DIR, "Encoder.obj")
        pickle.dump(ce_oe, open(save_oe_filename, 'wb'))
        # Transform the data
        train = ce_oe.transform(train)
        test = ce_oe.transform(test)


    # Fit and Predict (K Fold)
    print("Training Model...")
    model = ElasticNet(alpha=0.3, l1_ratio=0.8)
    oof, predictions = run_kfold_model(model, train, test, folds, features, target, n_fold=N_FOLD)

    # Save the model for evaluation later
    save_model_filename = os.path.join(MODEL_DIR, "FibrosisNetClinical_FVC.sav")
    pickle.dump(model, open(save_model_filename, 'wb'))

    train['FVC_pred'] = oof
    test['FVC_pred'] = predictions

    results = []
    weight = [100]
    for _, row in tqdm(train.iterrows(), total=len(train)):
        loss_partial = partial(loss_func, row=row)
        result = sp.optimize.minimize(loss_partial, weight, method='SLSQP')
        results.append(result['x'][0])

    train['Confidence'] = results

    target = train['Confidence']
    test['Confidence'] = np.nan

    # features
    cat_features = ['Sex', 'SmokingStatus']
    num_features = [c for c in test.columns if (test.dtypes[c] != 'object') & (c not in cat_features)]
    features = num_features + cat_features
    drop_features = ['Patient_Week', 'Confidence', 'predict_Week', 'base_Week', 'FVC', 'FVC_pred']
    features = [c for c in features if c not in drop_features]
    oof, predictions = run_kfold_model(model, train, test, folds, features, target, n_fold=N_FOLD)

    # Save the model for evaluation later
    save_model_filename = os.path.join(MODEL_DIR, "FibrosisNetClinical_Conf.sav")
    pickle.dump(model, open(save_model_filename, 'wb'))

    train['Confidence'] = oof
    train['sigma_clipped'] = train['Confidence'].apply(lambda x: max(x, 70))
    train['diff'] = abs(train['FVC'] - train['FVC_pred'])
    train['delta'] = train['diff'].apply(lambda x: min(x, 1000))
    train['score'] = -math.sqrt(2)*train['delta']/train['sigma_clipped'] - np.log(math.sqrt(2)*train['sigma_clipped'])
    score = train['score'].mean()
    print('Training Score: {}'.format(score))

    test['Confidence'] = predictions
    test = test.reset_index()

    sub = submission[['Patient_Week']].merge(test[['Patient_Week', 'FVC_pred', 'Confidence']], on='Patient_Week')
    sub = sub.rename(columns={'FVC_pred': 'FVC'})

    for i in range(len(test_orig)):
        sub.loc[sub['Patient_Week']==test_orig.Patient[i]+'_'+str(test_orig.Weeks[i]), 'FVC'] = test_orig.FVC[i]
        sub.loc[sub['Patient_Week']==test_orig.Patient[i]+'_'+str(test_orig.Weeks[i]), 'Confidence'] = 0.1
        
    clinical_sub = sub[["Patient_Week","FVC","Confidence"]].copy()

    # ===================================================================
    # The final submission results uses an ensemble of the two results
    # ===================================================================

    # Ensemble
    print("Creating submission file...")
    df1 = ct_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    df2 = clinical_sub.sort_values(by=['Patient_Week'], ascending=True).reset_index(drop=True)
    final_sub = df1[['Patient_Week']].copy()
    final_sub['FVC'] = ct_weight * df1['FVC'] + (1 - ct_weight) * df2['FVC']
    final_sub['Confidence'] = df2['Confidence']
    final_sub.head()
    final_sub.to_csv(output_file, index=False)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FibrosisNet on Kaggle')
    parser.add_argument('--datapath', default='osic-pulmonary-fibrosis-progression/', type=str, help='Path to the osic-pulmonary-fibrosis-progression directory')
    parser.add_argument('--modelpath', default='models/', type=str, help='Path to the osic-pulmonary-fibrosis-progression directory')
    parser.add_argument('--ctweight', default='1.0', type=float, help='The weight the CT scan has on the final decision')
    parser.add_argument('--outputfile', default='submission.csv', type=str, help='Output results .csv file')

    args = parser.parse_args()
    assert os.path.exists(args.datapath)
    assert os.path.exists(args.modelpath)

    generate_kaggle_results(args.datapath, args.modelpath, args.ctweight, args.outputfile)
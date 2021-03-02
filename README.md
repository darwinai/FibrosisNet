# Fibrosis-Net #

### Files ###

* Fibrosis-Net is comprised of two main components, `FibrosisNet-CT` and `FibrosisNet-Clinical`. The `FibrosisNet-CT` component takes in the provided CT scans along with clinical metadata, while the `FibrosisNet-Clinical` component takes in the provided clinical metadata.
* A pre-trained model, `FibrosisNetCT.pb`, is included in the "models/" directory. 

### How to Reproduce Our Results ###

To reproduce our results for Fibrosis-Net, first clone this repository, and download the dataset [here](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)

After extraction of the code and dataset, navigate into the main folder and install dependencies using:

```
pip3 install -r requirements.txt
```

Now, you can run the following code to generate our kaggle submission file. Make sure to modify the arguments as necessary. Running the code using `CT_WEIGHT=1.0` achieves -6.8188 private score. Running it with `CT_WEIGHT=0.96` achieves -6.8195 private score. Note that the Darwin team have reported a 0.0001 deviation in score between different Kaggle accounts and days of testing.

```
python3 kaggle_submission.py --datapath="osic-pulmonary-fibrosis-progression/" --modelpath="models/" --ctweight=1.0 --outputfile="submission.csv"
```

### Requirements ###

The main requirements are listed below. A full list can be found in "requirements.txt"

* Tested with Tensorflow 1.15
* OpenCV 4.5.1
* Python 3.6
* Numpy 1.20.0
* Pandas 1.2.1

### Using the evaluation script ###

In order to perform custom evaluation using the evaluation script, the "kaggle_submission.py" script must first be run (this automatically generates the additional models required for evaluation).

Set up the evaluation inputs. Refer to the directory "example_input" for an example. There must be a .csv file inside, with the following columns:

* Patient
* Weeks
* FVC 
* Percent
* Age
* Sex
* SmokingStatus
* PredictWeek

For each patient listed in the csv file, a folder containing CT scan images should be available in the same directory. Each folder should correspond to a patient in the csv file, linked by folder names.

Run the following code, and make sure to modify the arguments accordingly. If successful, a .csv file will be generated, containing all of the predictions for 100 weeks for each patient.

```
python3 fibrosisnet_eval.py --modelpath="models/" --inputpath="example_input/" --ctweight=1.0 --outputfile="results.csv"
```
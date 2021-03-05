# Fibrosis-Net #

<p align="center">
	<img src="assets/explainability.png" alt="image classification results with critical factors highlighted by GSInquire" width="70%" height="70%">
	<br>
	<em>Example images and their associated critical factors (highlighted in white) as identified by GSInquire.</em>
</p>

Pulmonary fibrosis is a devastating chronic lung disease that causes irreparable lung tissue scarring and damage, resulting in progressive loss in lung capacity and has no known cure. A critical step in the treatment and management of pulmonary fibrosis is the assessment of lung function decline, with computed tomography (CT) imaging being a particularly effective method for determining the extent of lung damage caused by pulmonary fibrosis.  Motivated by this, we introduce Fibrosis-Net, a deep convolutional neural network design tailored for the prediction of pulmonary fibrosis progression from chest CT images. More specifically, machine-driven design exploration was leveraged to determine a strong architectural design for CT lung analysis, upon which we build a customized network design tailored for predicting forced vital capacity (FVC) based on a patient's CT scan, initial spirometry measurement, and clinical metadata.  Finally, we leverage an explainability-driven performance validation strategy to study the decision-making behaviour of Fibrosis-Net as to verify that predictions are based on relevant visual indicators in CT images.  Experiments using the OSIC Pulmonary Fibrosis Progression Challenge benchmark dataset showed that the proposed Fibrosis-Net is able to achieve a significantly higher modified Laplace Log Likelihood score than the winning solutions on the challenge leaderboard.  Furthermore, explainability-driven performance validation demonstrated that the proposed Fibrosis-Net exhibits correct decision-making behaviour by leveraging clinically-relevant visual indicators in CT images when making predictions on pulmonary fibrosis progress.  Fibrosis-Net is available to the general public in an open-source and open access manner as part of the OpenMedAI initiative. While Fibrosis-Net is not yet a production-ready clinical assessment solution, we hope that releasing the model will encourage researchers, clinicians, and citizen data scientists alike to leverage and build upon them.

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

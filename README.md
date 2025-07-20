Multimodal Sentiment Analysis in Conversational Data
This project implements a multimodal sentiment analysis model to classify emotions in conversations using text, audio, and visual data. The system uses comprehensive feature engineering and an ensemble of XGBoost classifiers to predict sentiment from the provided conversational dataset.

Implementation Steps
The project pipeline is executed in the following sequence:

Data Loading and Preparation:

The training and testing datasets (.csv files) are loaded using pandas.

Paths to the corresponding video clips are generated and appended to the dataframes.

Feature Engineering:

Textual Features: Text is cleaned while preserving emotional punctuation (!, ?, ...). Features like punctuation counts, word count, character count, and speaker-specific patterns are extracted.

Temporal Features: Conversational flow is analyzed by calculating utterance duration, speaking rate, and the pause duration between utterances.

Audio Features: The audio is extracted from each video clip using moviepy and librosa. Features such as pitch, energy, and Harmonic-to-Noise Ratio (HNR) are extracted using parselmouth.

Visual Features: An MTCNN (Multi-task Cascaded Convolutional Networks) model processes video frames to detect faces and extract facial landmarks, expression changes, and head movement.

Model Training:

An ensemble model combining three XGBoost classifiers with varying hyperparameters is used.

The model is trained using a 5-fold stratified cross-validation strategy to ensure robustness and evaluate performance across different subsets of the data.

Features are scaled using StandardScaler before being fed into the model.

Prediction and Submission:

The trained ensemble model is used to predict sentiment on the unseen test dataset.

A submission.csv file is generated containing the final predictions, formatted according to the competition's requirements.

Requirements
To run this project, you will need Python 3 and the following libraries:

pandas

numpy

scikit-learn

xgboost

matplotlib

opencv-python

mtcnn

tensorflow

librosa

parselmouth

moviepy==1.0.3

soundfile

tqdm

How to Run
Clone the Repository:

Bash

git clone https://github.com/Veereshamaragatti/ML-Hackethon.git
cd ML-Hackethon
Install Dependencies:
It is recommended to create a virtual environment first. Then, install the required packages.

Bash

pip install -r requirements.txt
(You may need to create a requirements.txt file from the libraries listed above).

Ensure Data is in Place:
Make sure the Set3_Sentiment dataset (or your custom data) is placed in the correct directory structure as expected by the notebook.

Execute the Notebook:
Open and run the ml hakethon.ipynb notebook in a Jupyter environment.

Bash

jupyter notebook "ml hakethon.ipynb"
Get the Output:
After running the notebook, a submission.csv file with the test set predictions will be generated in the root directory.

# Multimodal Sentiment Analysis in Conversational Data

This project implements a multimodal sentiment analysis model to classify emotions in conversations using **text**, **audio**, and **visual** data. The system uses comprehensive feature engineering and an ensemble of **XGBoost** classifiers to predict sentiment from the provided conversational dataset.

### Key Features

* **Multimodal Approach**: Integrates features from three data sources for a holistic analysis:
    * **Text Analysis**: Cleans text while preserving emotional markers (`!`, `?`, `...`). Extracts features like punctuation counts, word count, and speaking rate.
    * **Audio Analysis**: Processes audio using `librosa` and `parselmouth` to extract pitch, energy, and voice quality metrics (e.g., HNR).
    * **Visual Analysis**: Employs an **MTCNN** to detect faces, track facial landmarks, and analyze expressions and movements from video clips.

* **Advanced Feature Engineering**:
    * **Temporal Features**: Captures conversational flow through utterance duration, speaking rate, and pauses.
    * **Speaker & Context Features**: Creates speaker profiles based on sentiment patterns and tracks conversational context, such as previous speaker and sentiment.

### Implementation Steps

The project pipeline is executed in the following sequence:

1.  **Data Loading and Preparation**:
    * The training and testing datasets (`.csv` files) are loaded using **pandas**.
    * Paths to the corresponding video clips are generated and appended to the dataframes.

2.  **Feature Engineering**:
    * **Textual Features**: Text is cleaned while preserving emotional punctuation (`!`, `?`, `...`). Features like punctuation counts, word count, character count, and speaker-specific patterns are extracted.
    * **Temporal Features**: Conversational flow is analyzed by calculating utterance duration, speaking rate, and the pause duration between utterances.
    * **Audio Features**: The audio is extracted from each video clip using **moviepy** and **librosa**. Features such as pitch, energy, and **Harmonic-to-Noise Ratio (HNR)** are extracted using **parselmouth**.
    * **Visual Features**: An **MTCNN (Multi-task Cascaded Convolutional Networks)** model processes video frames to detect faces and extract facial landmarks, expression changes, and head movement.

3.  **Model Training**:
    * An ensemble model combining three **XGBoost** classifiers with varying hyperparameters is used.
    * The model is trained using a **5-fold stratified cross-validation** strategy to ensure robustness and evaluate performance across different subsets of the data.
    * Features are scaled using `StandardScaler` before being fed into the model.

4.  **Prediction and Submission**:
    * The trained ensemble model is used to predict sentiment on the unseen test dataset.
    * A `submission.csv` file is generated containing the final predictions, formatted according to the competition's requirements.

### Requirements

To run this project, you will need **Python 3** and the following libraries:

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `matplotlib`
* `opencv-python`
* `mtcnn`
* `tensorflow`
* `librosa`
* `parselmouth`
* `moviepy==1.0.3`
* `soundfile`
* `tqdm`

### How to Run

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/Veereshamaragatti/ML-Hackethon.git](https://github.com/Veereshamaragatti/ML-Hackethon.git)
    cd ML-Hackethon
    ```

2.  **Install Dependencies**:
    It is recommended to create a virtual environment first. Then, install the required packages. You can create a `requirements.txt` file with the list above and run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ensure Data is in Place**:
    Make sure the **Set3_Sentiment** dataset is placed in the correct directory structure as expected by the notebook (e.g., `Set3_Sentiment/train/`, `Set3_Sentiment/test/`).

4.  **Execute the Notebook**:
    Open and run the `ml hakethon.ipynb` notebook in a Jupyter environment.
    ```bash
    jupyter notebook "ml hakethon.ipynb"
    ```

5.  **Get the Output**:
    After running the notebook, a `submission.csv` file with the test set predictions will be generated in the root directory.

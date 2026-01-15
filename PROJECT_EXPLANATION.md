# Multimodal Sentiment Analysis - In-Depth Project Explanation

This document provides a detailed, step-by-step explanation of how the sentiment analysis project works, with concrete examples showing data flow through each processing stage.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Data Structure](#2-data-structure)
3. [Text Processing Pipeline](#3-text-processing-pipeline)
4. [Audio Processing Pipeline](#4-audio-processing-pipeline)
5. [Video Processing Pipeline](#5-video-processing-pipeline)
6. [Feature Integration & Model Training](#6-feature-integration--model-training)
7. [Complete Data Flow Diagram](#7-complete-data-flow-diagram)

---

## 1. Project Overview

This project classifies sentiment (positive, negative, neutral) in conversational video clips by analyzing **three modalities**:
- **Text**: What is being said (the transcript)
- **Audio**: How it is being said (voice characteristics)
- **Video**: Visual cues (facial expressions, movements)

The features from all three sources are combined and fed into an XGBoost ensemble classifier.

---

## 2. Data Structure

### Input Data (CSV File)
Each row in the dataset represents one utterance (a single spoken segment):

```
Sr No. | Utterance                    | Speaker   | Sentiment | Dialogue_ID | Utterance_ID | StartTime      | EndTime
-------+------------------------------+-----------+-----------+-------------+--------------+----------------+---------------
11     | No don't I beg of you!       | Chandler  | negative  | 0           | 10           | 00:17:02,856   | 00:17:04,858
34     | You betcha!                  | Chandler  | positive  | 2           | 10           | 0:06:10,161    | 0:06:10,973
```

### Video Files
Each utterance has a corresponding video clip:
- **Format**: `dia{Dialogue_ID}_utt{Utterance_ID}.mp4`
- **Example**: For Dialogue_ID=0, Utterance_ID=10 → `dia0_utt10.mp4`

---

## 3. Text Processing Pipeline

### Step 3.1: Text Cleaning
The raw utterance text is cleaned while **preserving emotional markers** (!, ?, ...).

```
EXAMPLE:
Input:  "No don't I beg of you!!!"
Output: "no dont i beg of you!"

What happens:
1. Convert to lowercase: "no don't i beg of you!!!"
2. Multiple ! reduced to one: "no don't i beg of you!"
3. Apostrophes removed: "no dont i beg of you!"
```

### Step 3.2: Emotion Marker Extraction
Count punctuation that indicates emotion:

```python
# For utterance: "No don't I beg of you!!!"
exclamation_count = 3  # Three ! marks
question_count = 0     # No ? marks
ellipsis_count = 0     # No ... patterns

# For utterance: "Really?!"
exclamation_count = 1
question_count = 1
```

**Why this matters**: 
- Multiple exclamation marks suggest strong emotion (often negative or very positive)
- Question marks may indicate uncertainty or curiosity
- Ellipsis (...) often indicates hesitation or trailing thoughts

### Step 3.3: Word Count & Speaking Rate

```python
# For utterance: "No don't I beg of you!"
# StartTime: 00:17:02,856  EndTime: 00:17:04,858

word_count = 6  # ["No", "don't", "I", "beg", "of", "you"]
char_count = 21
utterance_duration = 2.002 seconds  # (EndTime - StartTime)
speaking_rate = 6 / 2.002 = 2.99 words/second
```

**Why this matters**:
- Fast speaking rate may indicate excitement or agitation
- Slow rate may indicate sadness or thoughtfulness
- Short utterances with high emotion markers are often more intense

### Step 3.4: Context Features
Track the conversation flow:

```python
# Current utterance: Chandler says "No don't I beg of you!" (negative)
# Previous utterance: The Interviewer said something (neutral)

prev_sentiment = "neutral"
prev_speaker = "The Interviewer"
time_gap = 8.342 seconds  # Time since previous utterance ended
relative_position = 0.83  # This is 83% through the dialogue
```

**Why this matters**:
- Sentiment often follows patterns (negative responses to negative statements)
- Speaker changes may indicate turn-taking dynamics
- Position in dialogue matters (endings may be more emotional)

### Step 3.5: Speaker Profile Features
Build a profile for each speaker's typical behavior:

```python
# For speaker "Chandler" across all his utterances:
speaker_positive_ratio = 0.25  # 25% of Chandler's lines are positive
speaker_negative_ratio = 0.35  # 35% are negative
speaker_avg_exclamation = 0.8  # Chandler averages 0.8 exclamation marks per utterance
```

**Why this matters**:
- Some characters are typically more negative/positive
- Deviations from typical behavior may be significant

### Text Features Summary Table

| Feature | Example Value | What It Tells Us |
|---------|--------------|------------------|
| `exclamation_count` | 3 | Strong emphasis/emotion |
| `question_count` | 0 | Not a question |
| `word_count` | 6 | Short utterance |
| `speaking_rate` | 2.99 | Moderate speed |
| `prev_sentiment` | neutral | Responding to neutral |
| `relative_position` | 0.83 | Near end of dialogue |
| `speaker_positive_ratio` | 0.25 | Speaker is often negative |

---

## 4. Audio Processing Pipeline

### Step 4.1: Extract Audio from Video
```python
# Input: dia0_utt10.mp4
# Process:
1. Load video file using moviepy
2. Extract audio track
3. Save as temporary WAV file: temp_dia0_utt10.mp4.wav
4. Load audio with librosa at 16kHz sample rate
```

### Step 4.2: Pitch Analysis
Pitch (fundamental frequency F0) measures how "high" or "low" the voice sounds.

```python
# Using parselmouth (Praat):
sound = parselmouth.Sound(audio_array, sample_rate=16000)
pitch = sound.to_pitch()

# Extract pitch values (ignoring silence where pitch=0)
pitch_values = [120, 125, 130, 140, 135, 128]  # Hz

# Calculate features:
pitch_mean = 129.67 Hz  # Average pitch
pitch_std = 6.89 Hz     # Variation in pitch
```

**Why this matters**:
- Higher pitch often indicates excitement, fear, or surprise
- Low pitch may indicate sadness or calmness
- High variation (std) suggests animated speech
- Low variation suggests monotone (neutral or bored)

### Step 4.3: Energy/Loudness Analysis
```python
# Using librosa:
rms = librosa.feature.rms(y=audio_array)  # Root Mean Square energy

# Example values over time: [0.02, 0.05, 0.08, 0.12, 0.15, 0.10]

energy_mean = 0.087  # Average loudness
energy_std = 0.045   # Loudness variation
```

**Why this matters**:
- Loud speech → anger, excitement, emphasis
- Quiet speech → sadness, intimacy, uncertainty
- High variation → dynamic emotional expression

### Step 4.4: Voice Quality (HNR)
HNR = Harmonics-to-Noise Ratio measures voice clarity.

```python
# Using parselmouth:
harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
hnr = call(harmonicity, "Get mean", 0, 0)

# Example: hnr = 12.5 dB
```

**Why this matters**:
- High HNR (>15 dB) → Clear, calm voice
- Low HNR (<10 dB) → Breathy, strained, or emotional voice
- Very low HNR → Crying, shouting, or hoarse voice

### Audio Features Summary Table

| Feature | Example Value | What It Tells Us |
|---------|--------------|------------------|
| `pitch_mean` | 129.67 Hz | Voice frequency (higher = more excited) |
| `pitch_std` | 6.89 Hz | Pitch variation (higher = more animated) |
| `energy_mean` | 0.087 | Loudness (higher = stronger emotion) |
| `energy_std` | 0.045 | Loudness variation |
| `hnr` | 12.5 dB | Voice clarity (lower = more strained) |

### Audio Processing Flow
```
dia0_utt10.mp4
      │
      ▼
┌─────────────────┐
│ Extract Audio   │ ──► temp_audio.wav
│ (moviepy)       │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Load at 16kHz   │ ──► audio_array (numpy)
│ (librosa)       │
└─────────────────┘
      │
      ├──────────────────────┬──────────────────────┐
      ▼                      ▼                      ▼
┌───────────┐        ┌───────────────┐      ┌─────────────┐
│ Pitch     │        │ Energy (RMS)  │      │ HNR         │
│ Analysis  │        │ Analysis      │      │ Analysis    │
│(parselmouth)       │ (librosa)     │      │(parselmouth)│
└───────────┘        └───────────────┘      └─────────────┘
      │                      │                      │
      ▼                      ▼                      ▼
  pitch_mean             energy_mean               hnr
  pitch_std              energy_std
```

---

## 5. Video Processing Pipeline

### Step 5.1: Frame Extraction
Extract frames from video at regular intervals.

```python
# For a 2-second video at 30 FPS with sample_rate=0.5 seconds:
# Total frames in video: 60
# Frame interval: 30 * 0.5 = 15 frames

# Extracted frames: frame 0, frame 15, frame 30, frame 45
# Result: 4 frames for analysis
```

### Step 5.2: Face Detection (MTCNN)
For each frame, detect faces and their landmarks.

```python
# MTCNN outputs for one frame:
{
    'box': [120, 80, 150, 180],  # [x, y, width, height] of face
    'confidence': 0.9987,        # How confident the detection is
    'keypoints': {
        'left_eye': (165, 130),
        'right_eye': (220, 128),
        'nose': (195, 165),
        'mouth_left': (170, 195),
        'mouth_right': (215, 193)
    }
}
```

### Step 5.3: Geometric Feature Extraction
Calculate measurements from facial landmarks:

```python
# Eye distance (indicates face angle/head tilt)
left_eye = (165, 130)
right_eye = (220, 128)
eye_distance = sqrt((220-165)² + (128-130)²) = 55.04 pixels

# Mouth width (indicates expression - smile vs neutral)
mouth_left = (170, 195)
mouth_right = (215, 193)
mouth_width = sqrt((215-170)² + (193-195)²) = 45.04 pixels

# Face dimensions
face_height = 180 pixels
face_width = 150 pixels
```

**Why this matters**:
- Wide mouth → smiling (positive) or shock (surprise)
- Narrow mouth → neutral or frowning (negative)
- Eye distance changes indicate head movement/tilting

### Step 5.4: Temporal Features (Across Frames)
Track how features change over time:

```python
# Frame 1: face center at (195, 170)
# Frame 2: face center at (198, 172)
# Frame 3: face center at (200, 175)
# Frame 4: face center at (195, 173)

# Movement calculation:
movement_1_to_2 = sqrt((198-195)² + (172-170)²) = 3.6 pixels
movement_2_to_3 = sqrt((200-198)² + (175-172)²) = 3.6 pixels
movement_3_to_4 = sqrt((195-200)² + (173-175)²) = 5.4 pixels

average_movement = (3.6 + 3.6 + 5.4) / 3 = 4.2 pixels per frame

# Expression change:
# Frame 1: mouth_width = 45.0
# Frame 2: mouth_width = 48.0
# Frame 3: mouth_width = 52.0 (mouth opens more - excitement?)
# Frame 4: mouth_width = 50.0

mouth_change = avg(|48-45| + |52-48| + |50-52|) = 3.0 pixels
```

**Why this matters**:
- High movement → animated, excited, or agitated speaker
- Low movement → calm or sad speaker
- High expression change → expressive emotion
- Low expression change → flat affect (neutral or masked emotion)

### Video Features Summary Table

| Feature | Example Value | What It Tells Us |
|---------|--------------|------------------|
| `confidence` | 0.9987 | Face clearly visible |
| `eye_distance` | 55.04 px | Face angle/orientation |
| `mouth_width` | 45.04 px | Expression (wide = smile/shock) |
| `face_movement` | 4.2 px/frame | Head movement intensity |
| `expression_change` | 3.0 px | Facial animation level |

### Video Processing Flow
```
dia0_utt10.mp4
      │
      ▼
┌──────────────────────┐
│ Extract Frames       │ ──► [Frame1, Frame2, Frame3, Frame4]
│ (OpenCV)             │
│ @ 0.5 sec intervals  │
└──────────────────────┘
      │
      ▼ (for each frame)
┌──────────────────────┐
│ Face Detection       │ ──► box, keypoints, confidence
│ (MTCNN)              │
└──────────────────────┘
      │
      ▼
┌──────────────────────┐
│ Geometric Features   │ ──► eye_distance, mouth_width
│ (Calculate)          │
└──────────────────────┘
      │
      ▼ (across all frames)
┌──────────────────────┐
│ Temporal Features    │ ──► movement, expression_change
│ (Compare frames)     │
└──────────────────────┘
```

---

## 6. Feature Integration & Model Training

### Step 6.1: Combine All Features
All features from text, audio, and video are combined into a single feature vector:

```python
# For one utterance "No don't I beg of you!":

combined_features = {
    # Text features (7 features)
    'exclamation_count': 3,
    'question_count': 0,
    'ellipsis_count': 0,
    'word_count': 6,
    'char_count': 21,
    'speaking_rate': 2.99,
    'avg_word_length': 3.5,
    
    # Audio features (5 features)
    'pitch_mean': 145.2,
    'pitch_std': 25.3,
    'energy_mean': 0.12,
    'energy_std': 0.05,
    'hnr': 8.5,
    
    # Video features (5 features)
    'confidence': 0.998,
    'eye_distance': 55.0,
    'mouth_width': 42.0,
    'face_movement': 12.5,
    'expression_change': 8.2
}

# Total: 17+ features per utterance
```

### Step 6.2: Feature Scaling
Features are normalized using StandardScaler:

```python
# Before scaling:
pitch_mean = 145.2 (range: 80-300 Hz)
energy_mean = 0.12 (range: 0-1)

# After scaling (mean=0, std=1):
pitch_mean_scaled = 0.85
energy_mean_scaled = 1.23
```

**Why this matters**: XGBoost works better when features are on similar scales.

### Step 6.3: XGBoost Ensemble Training
Three XGBoost classifiers with different configurations are combined:

```python
# Model 1: Shallow trees, higher learning rate
xgb1 = XGBClassifier(
    n_estimators=200,    # 200 trees
    max_depth=5,         # Trees go 5 levels deep
    learning_rate=0.1    # Larger steps
)

# Model 2: Medium trees, medium learning rate
xgb2 = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05
)

# Model 3: Deep trees, small learning rate
xgb3 = XGBClassifier(
    n_estimators=200,
    max_depth=9,
    learning_rate=0.01
)

# Combine with weighted voting
ensemble = VotingClassifier(
    estimators=[('xgb1', xgb1), ('xgb2', xgb2), ('xgb3', xgb3)],
    weights=[0.4, 0.3, 0.3],  # xgb1 has more influence
    voting='soft'  # Average probability predictions
)
```

### Step 6.4: 5-Fold Cross-Validation
```
Training Data (1000 utterances)
     │
     ├── Fold 1: Train on 800, Validate on 200 (rows 1-200)
     ├── Fold 2: Train on 800, Validate on 200 (rows 201-400)
     ├── Fold 3: Train on 800, Validate on 200 (rows 401-600)
     ├── Fold 4: Train on 800, Validate on 200 (rows 601-800)
     └── Fold 5: Train on 800, Validate on 200 (rows 801-1000)

Average F1-Score across folds = Model Performance
```

### Step 6.5: Final Prediction
```python
# For new utterance, get probability for each class:
probabilities = model.predict_proba(features)
# Result: [0.15, 0.25, 0.60]  # [negative, neutral, positive]

# Final prediction = class with highest probability
prediction = "positive"  # 60% probability
```

---

## 7. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT DATA                                      │
│  CSV: Utterance, Speaker, Timestamps, Sentiment    +    Video: dia#_utt#.mp4 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   TEXT PROCESSING   │  │  AUDIO PROCESSING   │  │  VIDEO PROCESSING   │
│                     │  │                     │  │                     │
│ • Clean text        │  │ • Extract audio     │  │ • Extract frames    │
│ • Count punctuation │  │ • Analyze pitch     │  │ • Detect faces      │
│ • Word count/rate   │  │ • Measure energy    │  │ • Extract landmarks │
│ • Context features  │  │ • Calculate HNR     │  │ • Track movement    │
│ • Speaker profiles  │  │                     │  │ • Expression change │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
           │                        │                        │
           │   exclamation_count    │   pitch_mean           │   face_movement
           │   question_count       │   pitch_std            │   expression_change
           │   word_count           │   energy_mean          │   confidence
           │   speaking_rate        │   energy_std           │   mouth_width
           │   speaker_ratios       │   hnr                  │   eye_distance
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     FEATURE COMBINATION       │
                    │   (17+ features per sample)   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     FEATURE SCALING           │
                    │     (StandardScaler)          │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   XGBOOST ENSEMBLE            │
                    │                               │
                    │  ┌─────┐ ┌─────┐ ┌─────┐     │
                    │  │XGB1 │ │XGB2 │ │XGB3 │     │
                    │  │40%  │ │30%  │ │30%  │     │
                    │  └──┬──┘ └──┬──┘ └──┬──┘     │
                    │     └──────┼───────┘         │
                    │            ▼                 │
                    │    Weighted Average          │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         PREDICTION            │
                    │                               │
                    │   positive | neutral | negative
                    │      15%   |   25%   |   60%  │
                    │                               │
                    │   Final: "negative" (60%)     │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       submission.csv          │
                    │   Sr No. | Sentiment          │
                    │      11  | negative           │
                    └───────────────────────────────┘
```

---

## Example: Complete Processing of One Utterance

**Input**: 
- Utterance: "No don't I beg of you!"
- Speaker: Chandler
- Video: dia0_utt10.mp4

**Step 1 - Text Processing**:
```
cleaned_text = "no dont i beg of you!"
exclamation_count = 1
question_count = 0
word_count = 6
speaking_rate = 2.99 words/sec
speaker_negative_ratio = 0.35 (Chandler is often negative)
```

**Step 2 - Audio Processing**:
```
pitch_mean = 185.3 Hz (higher than normal - stressed)
pitch_std = 32.1 Hz (high variation - emotional)
energy_mean = 0.18 (loud)
energy_std = 0.08 (variable loudness)
hnr = 7.2 dB (strained voice quality)
```

**Step 3 - Video Processing**:
```
confidence = 0.997 (clear face detection)
mouth_width = 38.2 px (not wide - not smiling)
face_movement = 15.3 px/frame (high - animated)
expression_change = 9.8 (high - expressive)
```

**Step 4 - Combine & Predict**:
```
All 17 features → StandardScaler → XGBoost Ensemble

Probabilities: [negative: 0.72, neutral: 0.18, positive: 0.10]

Final Prediction: "negative" ✓ (matches ground truth)
```

**Why the model predicted "negative"**:
- High exclamation count (emphasis)
- High pitch mean and variation (stress)
- Low HNR (strained voice)
- No smile (narrow mouth)
- High movement and expression change (agitation)
- Speaker (Chandler) historically has 35% negative utterances

---

## Summary

The multimodal approach captures emotion from multiple angles:
1. **Text** tells us WHAT is said (words, punctuation, context)
2. **Audio** tells us HOW it's said (pitch, loudness, voice quality)
3. **Video** tells us what the speaker LOOKS like (expressions, movement)

By combining all three, the model can detect sentiment more accurately than using any single modality alone.

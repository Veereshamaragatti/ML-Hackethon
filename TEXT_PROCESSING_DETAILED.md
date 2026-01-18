# Complete Text Processing Guide - From Scratch

This document explains every step of text processing in detail, showing exactly what happens to each piece of data and where it goes.

---

## Overview: The Journey of Text Data

```
CSV File (Utterance column)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT PROCESSING PIPELINE                      │
│                                                                  │
│  Step 1: Text Cleaning                                          │
│  Step 2: Emotion Marker Extraction                              │
│  Step 3: Word & Character Analysis                              │
│  Step 4: Temporal Features (Duration, Speaking Rate)            │
│  Step 5: Context Features (Previous Utterance Info)             │
│  Step 6: Speaker Profile Features                               │
│  Step 7: Text Enhancement Features (Shouting, Stuttering)       │
│  Step 8: Dialogue Structure Features                            │
│  Step 9: Emotion Intensifier Features                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Feature Vector (sent to model)
```

---

## STEP 1: Text Cleaning

### What It Does
Takes raw utterance text and cleans it while **keeping emotional punctuation**.

### The Code
```python
import re
import string

def clean_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Replace multiple !!! with single !
    text = re.sub(r'!+', '!', text)
    
    # 3. Replace multiple ??? with single ?
    text = re.sub(r'\?+', '?', text)
    
    # 4. Keep ellipsis as ...
    text = re.sub(r'\.{2,}', '...', text)
    
    # 5. Remove other punctuation (commas, quotes, etc.)
    punct = string.punctuation.replace('!', '').replace('?', '').replace('.', '')
    text = ''.join(ch for ch in text if ch not in punct)
    
    # 6. Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

### Example Walk-Through

**Input:** `"No, don't!!! I beg of you!!!"`

| Step | Action | Result |
|------|--------|--------|
| 1 | Convert to lowercase | `"no, don't!!! i beg of you!!!"` |
| 2 | Multiple ! → single ! | `"no, don't! i beg of you!"` |
| 3 | Multiple ? → single ? | (no change) |
| 4 | Keep ellipsis | (no change) |
| 5 | Remove other punctuation | `"no dont! i beg of you!"` |
| 6 | Clean whitespace | `"no dont! i beg of you!"` |

**Output:** `"no dont! i beg of you!"`

### Where Does This Go?
```python
# Applied to entire DataFrame
train_df['cleaned_text'] = train_df['Utterance'].apply(clean_text)
```

The `cleaned_text` column is created and stored in the DataFrame for later use.

---

## STEP 2: Emotion Marker Extraction

### What It Does
Counts punctuation marks that indicate emotion.

### The Code
```python
# Count exclamation marks in ORIGINAL text (not cleaned)
train_df['exclamation_count'] = train_df['Utterance'].str.count(r'!')

# Count question marks
train_df['question_count'] = train_df['Utterance'].str.count(r'\?')

# Count ellipsis patterns (...)
train_df['ellipsis_count'] = train_df['Utterance'].str.count(r'\.\.\.')
```

### Example Walk-Through

| Utterance | exclamation_count | question_count | ellipsis_count |
|-----------|-------------------|----------------|----------------|
| `"No don't I beg of you!!!"` | 3 | 0 | 0 |
| `"Really?!"` | 1 | 1 | 0 |
| `"Well... I don't know..."` | 0 | 0 | 2 |
| `"What?? What??"` | 0 | 4 | 0 |
| `"Hello there"` | 0 | 0 | 0 |

### Why These Features Matter

| Feature | High Value Indicates | Low Value Indicates |
|---------|---------------------|---------------------|
| `exclamation_count` | Strong emotion, shouting, emphasis | Calm, neutral statement |
| `question_count` | Uncertainty, curiosity, interrogation | Statement, assertion |
| `ellipsis_count` | Hesitation, trailing off, uncertainty | Complete thought |

### Where Does This Go?
These become **3 numeric features** that go directly into the model:
- `exclamation_count` → Feature for model
- `question_count` → Feature for model  
- `ellipsis_count` → Feature for model

---

## STEP 3: Word & Character Analysis

### What It Does
Analyzes the basic structure of the text.

### The Code
```python
# Count words (split by whitespace)
train_df['word_count'] = train_df['Utterance'].str.split().str.len()

# Count characters
train_df['char_count'] = train_df['Utterance'].str.len()

# Calculate average word length
train_df['avg_word_length'] = train_df['char_count'] / train_df['word_count']
```

### Example Walk-Through

**Utterance:** `"No don't I beg of you!"`

| Calculation | Process | Result |
|------------|---------|--------|
| word_count | Split: `["No", "don't", "I", "beg", "of", "you!"]` → Count | 6 |
| char_count | Count all characters including spaces | 22 |
| avg_word_length | 22 / 6 | 3.67 |

### Why These Features Matter

| Feature | High Value Indicates | Low Value Indicates |
|---------|---------------------|---------------------|
| `word_count` | Long explanation, complex thought | Short response, exclamation |
| `char_count` | Detailed utterance | Brief response |
| `avg_word_length` | Complex/formal vocabulary | Simple/casual vocabulary |

### Where Does This Go?
These become **3 numeric features** that go directly into the model.

---

## STEP 4: Temporal Features (Duration & Speaking Rate)

### What It Does
Uses the StartTime and EndTime to calculate how fast someone is speaking.

### The Code
```python
# Convert time strings to datetime
train_df['start_time'] = pd.to_datetime(train_df['StartTime'], format='%H:%M:%S,%f')
train_df['end_time'] = pd.to_datetime(train_df['EndTime'], format='%H:%M:%S,%f')

# Calculate duration in seconds
train_df['utterance_duration'] = (
    train_df['end_time'] - train_df['start_time']
).dt.total_seconds()

# Calculate speaking rate (words per second)
train_df['speaking_rate'] = train_df['word_count'] / train_df['utterance_duration']

# Calculate pause after this utterance (time until next person speaks)
# ... (calculated per dialogue)

# Identify rapid exchanges
train_df['is_rapid_exchange'] = (
    (train_df['utterance_duration'] < median_duration) & 
    (train_df['pause_after'] < median_pause)
)
```

### Example Walk-Through

**Utterance:** `"No don't I beg of you!"`  
**StartTime:** `00:17:02,856`  
**EndTime:** `00:17:04,858`

| Calculation | Process | Result |
|------------|---------|--------|
| start_time | Parse `00:17:02,856` | 17:02:02.856 |
| end_time | Parse `00:17:04,858` | 17:02:04.858 |
| utterance_duration | end - start | 2.002 seconds |
| word_count | (from Step 3) | 6 words |
| speaking_rate | 6 / 2.002 | 2.997 words/sec |

### Why These Features Matter

| Feature | High Value Indicates | Low Value Indicates |
|---------|---------------------|---------------------|
| `utterance_duration` | Long speech, explanation | Quick response, exclamation |
| `speaking_rate` | Excitement, urgency, agitation | Sadness, thoughtfulness, calm |
| `pause_after` | Thinking, dramatic pause | Rapid conversation, interruption |
| `is_rapid_exchange` | Heated argument, excitement | Normal conversation pace |

### Where Does This Go?
These become **4 numeric features** that go directly into the model:
- `utterance_duration`
- `speaking_rate`
- `pause_after`
- `is_rapid_exchange` (boolean → 0 or 1)

---

## STEP 5: Context Features (Conversation Flow)

### What It Does
Looks at what was said BEFORE this utterance to understand context.

### The Code
```python
def add_context_features(df):
    # For each utterance, find the previous one in the same dialogue
    df['prev_sentiment'] = None
    df['prev_speaker'] = None
    df['time_gap'] = 0.0
    
    for idx, row in df.iterrows():
        # Get previous utterances in same dialogue
        prev_utts = df[
            (df['Dialogue_ID'] == row['Dialogue_ID']) & 
            (df['Utterance_ID'] < row['Utterance_ID'])
        ].sort_values('Utterance_ID', ascending=False)
        
        if not prev_utts.empty:
            # Get the immediately previous utterance
            df.at[idx, 'prev_sentiment'] = prev_utts.iloc[0]['Sentiment']
            df.at[idx, 'prev_speaker'] = prev_utts.iloc[0]['Speaker']
            
            # Time gap since previous utterance ended
            current_start = pd.to_datetime(row['StartTime'])
            prev_end = pd.to_datetime(prev_utts.iloc[0]['EndTime'])
            df.at[idx, 'time_gap'] = (current_start - prev_end).total_seconds()
    
    # Calculate position in dialogue
    dialogue_lengths = df.groupby('Dialogue_ID').size()
    df['dialogue_length'] = df['Dialogue_ID'].map(dialogue_lengths)
    df['utterance_position'] = df.groupby('Dialogue_ID')['Utterance_ID'].rank()
    df['relative_position'] = df['utterance_position'] / df['dialogue_length']
    
    return df
```

### Example Walk-Through

**Current Dialogue (Dialogue_ID = 0):**

| Utterance_ID | Speaker | Utterance | Sentiment | EndTime |
|--------------|---------|-----------|-----------|---------|
| 7 | The Interviewer | "But there'll be perhaps 30 people under you..." | neutral | 00:16:54,514 |
| 10 | Chandler | "No don't I beg of you!" | negative | 00:17:04,858 |
| 11 | The Interviewer | "All right then, we'll have a definite answer..." | neutral | 00:17:13,324 |

**For Chandler's utterance (Utterance_ID = 10):**

| Feature | How Calculated | Value |
|---------|----------------|-------|
| prev_sentiment | Previous utterance's sentiment | "neutral" |
| prev_speaker | Previous utterance's speaker | "The Interviewer" |
| time_gap | 00:17:02,856 - 00:16:54,514 | 8.342 seconds |
| dialogue_length | Total utterances in dialogue | 12 |
| utterance_position | This is the 10th utterance | 10 |
| relative_position | 10 / 12 | 0.833 (83% through) |

### Why These Features Matter

| Feature | What It Tells Us |
|---------|-----------------|
| `prev_sentiment` | People often respond in patterns (negative → negative) |
| `prev_speaker` | Same speaker continuing vs. responding to someone |
| `time_gap` | Long gap = thinking; Short gap = quick response/interruption |
| `relative_position` | Endings often more emotional; Beginnings more neutral |

### Where Does This Go?
These become **6 features** for the model:
- `prev_sentiment` → Encoded as number (positive=0, neutral=1, negative=2)
- `prev_speaker` → One-hot encoded or label encoded
- `time_gap` → Numeric feature
- `dialogue_length` → Numeric feature
- `utterance_position` → Numeric feature
- `relative_position` → Numeric feature (0 to 1)

---

## STEP 6: Speaker Profile Features

### What It Does
Builds a profile of how each speaker typically behaves across ALL their utterances.

### The Code
```python
def create_speaker_profiles(df):
    # Calculate what percentage of each speaker's lines are positive/negative/neutral
    speaker_sentiment = pd.crosstab(
        df['Speaker'], 
        df['Sentiment'], 
        normalize='index'  # Normalize by row (per speaker)
    )
    
    # Calculate average emotion markers per speaker
    speaker_stats = df.groupby('Speaker').agg({
        'exclamation_count': 'mean',
        'question_count': 'mean',
        'ellipsis_count': 'mean',
        'Utterance': 'count'
    }).rename(columns={'Utterance': 'total_utterances'})
    
    return pd.concat([speaker_sentiment, speaker_stats], axis=1)

# Add speaker features to each utterance
def add_speaker_features(df, speaker_profiles):
    for sentiment in ['positive', 'negative', 'neutral']:
        df[f'speaker_{sentiment}_ratio'] = df['Speaker'].map(
            speaker_profiles[sentiment]
        )
    
    for marker in ['exclamation_count', 'question_count', 'ellipsis_count']:
        df[f'speaker_avg_{marker}'] = df['Speaker'].map(
            speaker_profiles[marker]
        )
    
    return df
```

### Example Walk-Through

**First, build speaker profiles from ALL data:**

| Speaker | positive | negative | neutral | avg_exclamation | avg_question | total_utterances |
|---------|----------|----------|---------|-----------------|--------------|------------------|
| Chandler | 0.25 | 0.35 | 0.40 | 0.8 | 0.3 | 150 |
| Ross | 0.30 | 0.30 | 0.40 | 0.5 | 0.4 | 180 |
| Rachel | 0.35 | 0.25 | 0.40 | 0.9 | 0.5 | 160 |

**Then, for each utterance, add the speaker's profile:**

For Chandler's utterance `"No don't I beg of you!"`:

| Feature | Value | Meaning |
|---------|-------|---------|
| speaker_positive_ratio | 0.25 | Chandler is positive 25% of time |
| speaker_negative_ratio | 0.35 | Chandler is negative 35% of time |
| speaker_neutral_ratio | 0.40 | Chandler is neutral 40% of time |
| speaker_avg_exclamation_count | 0.8 | Chandler averages 0.8 ! per utterance |
| speaker_avg_question_count | 0.3 | Chandler averages 0.3 ? per utterance |

### Why These Features Matter

| Feature | What It Tells Us |
|---------|-----------------|
| `speaker_positive_ratio` | Is this speaker typically positive? Deviation may be significant |
| `speaker_negative_ratio` | Is this speaker typically negative? |
| `speaker_avg_exclamation` | Does this speaker typically use exclamations? |

### Where Does This Go?
These become **6 features** for the model (per utterance):
- `speaker_positive_ratio`
- `speaker_negative_ratio`
- `speaker_neutral_ratio`
- `speaker_avg_exclamation_count`
- `speaker_avg_question_count`
- `speaker_avg_ellipsis_count`

---

## STEP 7: Text Enhancement Features

### What It Does
Detects special text patterns like shouting, stuttering, repetition, and laughter.

### The Code
```python
def add_text_enhancement_features(df):
    for idx, row in df.iterrows():
        text = row['Utterance']
        words = text.split()
        
        # 1. SHOUTING: Words in ALL CAPS (like "NO" or "STOP")
        caps_words = [w for w in words if w.isupper() and len(w) > 1]
        df.at[idx, 'has_shouting'] = len(caps_words) > 0
        df.at[idx, 'shouting_word_count'] = len(caps_words)
        
        # 2. REPETITION: Like "no-no-no" 
        repetitions = [w for w in words if w.count('-') >= 2]
        df.at[idx, 'has_repetition'] = len(repetitions) > 0
        df.at[idx, 'repetition_count'] = len(repetitions)
        
        # 3. STUTTERING: Like "I-I" or "w-what"
        stutters = [w for w in words if len(w) <= 4 and '-' in w]
        df.at[idx, 'has_stuttering'] = len(stutters) > 0
        df.at[idx, 'stutter_count'] = len(stutters)
        
        # 4. LAUGHTER: Like "haha", "hehe", "lol"
        laughter_patterns = ['haha', 'hehe', 'lol', 'lmao']
        has_laugh = any(p in text.lower() for p in laughter_patterns)
        df.at[idx, 'has_laughter'] = has_laugh
    
    return df
```

### Example Walk-Through

| Utterance | has_shouting | has_repetition | has_stuttering | has_laughter |
|-----------|--------------|----------------|----------------|--------------|
| `"GO, GO, GO!"` | True (GO) | False | False | False |
| `"No-no-no-no, no!"` | False | True (no-no-no-no) | False | False |
| `"W-what did you say?"` | False | False | True (W-what) | False |
| `"Haha, that's funny!"` | False | False | False | True |
| `"Hello there"` | False | False | False | False |

### Why These Features Matter

| Feature | What It Indicates |
|---------|------------------|
| `has_shouting` | Strong emotion (anger, excitement) |
| `has_repetition` | Emphasis, frustration, urgency |
| `has_stuttering` | Nervousness, fear, surprise |
| `has_laughter` | Positive emotion, humor |

### Where Does This Go?
These become **8 features** for the model:
- `has_shouting` (boolean)
- `shouting_word_count` (numeric)
- `has_repetition` (boolean)
- `repetition_count` (numeric)
- `has_stuttering` (boolean)
- `stutter_count` (numeric)
- `has_laughter` (boolean)
- `laughter_count` (numeric)

---

## STEP 8: Dialogue Structure Features

### What It Does
Understands the structure of the conversation.

### The Code
```python
def add_dialogue_structure_features(df):
    # Is this utterance a question?
    df['is_question'] = df['Utterance'].str.contains(r'\?')
    
    # Is this an answer (follows a question)?
    df['is_answer'] = False
    
    # Is this an interruption (very short pause before)?
    df['is_interruption'] = False
    
    # How many people are in this conversation?
    df['conversation_size'] = 0
    
    # Is this the first or last utterance?
    df['is_opening'] = False
    df['is_closing'] = False
    
    for dialogue_id in df['Dialogue_ID'].unique():
        dialogue = df[df['Dialogue_ID'] == dialogue_id].sort_values('Utterance_ID')
        
        # Count unique speakers
        df.loc[dialogue.index, 'conversation_size'] = dialogue['Speaker'].nunique()
        
        # Mark first and last
        df.loc[dialogue.index[0], 'is_opening'] = True
        df.loc[dialogue.index[-1], 'is_closing'] = True
        
        # Find answers (utterances after questions)
        for i in range(len(dialogue) - 1):
            if dialogue.iloc[i]['is_question']:
                df.loc[dialogue.index[i + 1], 'is_answer'] = True
        
        # Find interruptions (gap < 0.5 seconds)
        for i in range(1, len(dialogue)):
            start = pd.to_datetime(dialogue.iloc[i]['StartTime'])
            prev_end = pd.to_datetime(dialogue.iloc[i-1]['EndTime'])
            if (start - prev_end).total_seconds() < 0.5:
                df.loc[dialogue.index[i], 'is_interruption'] = True
    
    return df
```

### Example Walk-Through

**Dialogue:**
```
[0] Ross: "Did you hear about Monica?" (question, opening)
[1] Rachel: "What happened?" (question, answer to previous)
[2] Ross: "She got the job!" (answer to previous)
[3] Rachel: "Oh my god!" (closing)
```

| Utterance | is_question | is_answer | is_opening | is_closing | conversation_size |
|-----------|-------------|-----------|------------|------------|-------------------|
| "Did you hear about Monica?" | True | False | True | False | 2 |
| "What happened?" | True | True | False | False | 2 |
| "She got the job!" | False | True | False | False | 2 |
| "Oh my god!" | False | False | False | True | 2 |

### Why These Features Matter

| Feature | What It Indicates |
|---------|------------------|
| `is_question` | Seeking information, uncertainty |
| `is_answer` | Responding to question, providing information |
| `is_opening` | Conversation starters often neutral |
| `is_closing` | Endings can be more emotional |
| `is_interruption` | Excitement, urgency, conflict |
| `conversation_size` | Group dynamics vs one-on-one |

### Where Does This Go?
These become **7 features** for the model.

---

## STEP 9: Emotion Intensifier Features

### What It Does
Detects words and patterns that intensify or indicate emotion.

### The Code
```python
def add_emotion_intensifier_features(df):
    emphasis_words = ['very', 'so', 'really', 'extremely', 'totally', 
                      'absolutely', 'completely', 'literally', 'definitely']
    
    positive_emotions = ['love', 'happy', 'excited', 'glad', 'wonderful', 
                         'great', 'amazing', 'fantastic', 'awesome']
    
    negative_emotions = ['hate', 'angry', 'sad', 'upset', 'terrible', 
                         'horrible', 'awful', 'furious', 'annoyed']
    
    for idx, row in df.iterrows():
        text = row['Utterance'].lower()
        words = text.split()
        
        # Count repeated punctuation like "!!!" or "???"
        df.at[idx, 'repeated_punct_count'] = len(re.findall(r'[!?]{2,}', text))
        
        # Count emphasis words
        df.at[idx, 'emphasis_word_count'] = sum(
            1 for w in emphasis_words if w in words
        )
        
        # Count positive emotion words
        df.at[idx, 'positive_emotion_count'] = sum(
            1 for w in positive_emotions if w in words
        )
        
        # Count negative emotion words
        df.at[idx, 'negative_emotion_count'] = sum(
            1 for w in negative_emotions if w in words
        )
    
    return df
```

### Example Walk-Through

| Utterance | emphasis_count | positive_emotion | negative_emotion |
|-----------|----------------|------------------|------------------|
| `"I really love this!"` | 1 (really) | 1 (love) | 0 |
| `"I'm so angry right now"` | 1 (so) | 0 | 1 (angry) |
| `"This is absolutely amazing!"` | 1 (absolutely) | 1 (amazing) | 0 |
| `"I hate this terrible thing"` | 0 | 0 | 2 (hate, terrible) |

### Where Does This Go?
These become **4 features** for the model.

---

## FINAL SUMMARY: All Text Features

Here's every text feature that gets extracted and sent to the model:

### From Text Cleaning (Step 1)
- `cleaned_text` (used for further processing, not directly in model)

### From Emotion Markers (Step 2)
1. `exclamation_count`
2. `question_count`
3. `ellipsis_count`

### From Word Analysis (Step 3)
4. `word_count`
5. `char_count`
6. `avg_word_length`

### From Temporal Features (Step 4)
7. `utterance_duration`
8. `speaking_rate`
9. `pause_after`
10. `is_rapid_exchange`

### From Context Features (Step 5)
11. `prev_sentiment` (encoded)
12. `prev_speaker` (encoded)
13. `time_gap`
14. `relative_position`

### From Speaker Profiles (Step 6)
15. `speaker_positive_ratio`
16. `speaker_negative_ratio`
17. `speaker_neutral_ratio`
18. `speaker_avg_exclamation_count`

### From Text Enhancement (Step 7)
19. `has_shouting`
20. `has_repetition`
21. `has_stuttering`
22. `has_laughter`

### From Dialogue Structure (Step 8)
23. `is_question`
24. `is_answer`
25. `is_opening`
26. `is_closing`
27. `conversation_size`

### From Emotion Intensifiers (Step 9)
28. `emphasis_word_count`
29. `positive_emotion_count`
30. `negative_emotion_count`

---

## COMPLETE FLOW DIAGRAM

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           RAW CSV DATA                                      │
│  Utterance: "No don't I beg of you!!!"                                     │
│  Speaker: Chandler    StartTime: 00:17:02,856    EndTime: 00:17:04,858     │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   STEP 1-3       │    │    STEP 4        │    │   STEP 5-6       │
│   Text Analysis  │    │    Timing        │    │   Context        │
│                  │    │                  │    │                  │
│ • Clean text     │    │ • Duration       │    │ • prev_sentiment │
│ • Count !/?/...  │    │ • Speaking rate  │    │ • prev_speaker   │
│ • Word count     │    │ • Pause after    │    │ • Speaker profile│
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ exclamation: 3   │    │ duration: 2.002s │    │ prev_sent: neutral│
│ question: 0      │    │ rate: 2.99 w/s   │    │ position: 0.83   │
│ word_count: 6    │    │ pause: 8.34s     │    │ spkr_neg: 0.35   │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │     FEATURE VECTOR        │
                    │                           │
                    │ [3, 0, 0, 6, 22, 3.67,    │
                    │  2.002, 2.99, 8.34, 0,    │
                    │  1, ..., 0.35, 0.8, ...]  │
                    │                           │
                    │    (~30 text features)    │
                    └───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │   COMBINE WITH AUDIO &    │
                    │   VIDEO FEATURES          │
                    └───────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────┐
                    │      XGBOOST MODEL        │
                    │                           │
                    │  Prediction: "negative"   │
                    └───────────────────────────┘
```

---

## EXAMPLE: Complete Processing of One Utterance

**Input:**
- Utterance: `"No don't I beg of you!!!"`
- Speaker: `Chandler`
- StartTime: `00:17:02,856`
- EndTime: `00:17:04,858`

**Step-by-Step Processing:**

| Step | Feature | Value | How Calculated |
|------|---------|-------|----------------|
| 1 | cleaned_text | "no dont i beg of you!" | Lowercase + remove punctuation except !?. |
| 2 | exclamation_count | 3 | Count `!` in original |
| 2 | question_count | 0 | Count `?` in original |
| 2 | ellipsis_count | 0 | Count `...` in original |
| 3 | word_count | 6 | Split and count |
| 3 | char_count | 22 | Length of string |
| 3 | avg_word_length | 3.67 | 22/6 |
| 4 | utterance_duration | 2.002 | EndTime - StartTime |
| 4 | speaking_rate | 2.99 | 6 / 2.002 |
| 5 | prev_sentiment | neutral | Look up previous utterance |
| 5 | time_gap | 8.34 | Time since previous ended |
| 5 | relative_position | 0.83 | 10th of 12 utterances |
| 6 | speaker_negative_ratio | 0.35 | Chandler is 35% negative overall |
| 7 | has_shouting | False | No ALL CAPS words |
| 7 | has_repetition | False | No repeated words |
| 8 | is_question | False | No `?` in text |
| 8 | is_answer | True | Follows a question |
| 9 | emphasis_word_count | 0 | No "really", "very", etc. |

**Final Feature Vector (text portion):**
```
[3, 0, 0, 6, 22, 3.67, 2.002, 2.99, 8.34, False, "neutral", "The Interviewer", 
 0.83, 0.35, 0.25, 0.8, False, False, False, False, False, True, ...]
```

This vector is then combined with audio and video features before being sent to the XGBoost model for prediction.

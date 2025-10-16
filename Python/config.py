
RUN_STEP12_SANITY = True   # set False to run the normal professor pipeline

#SIGNAL
TARGET_FS = 100            # We set at 100 Hz because this is what the instructions tells us to do 
EPOCH_LEN_S = 30           # 30-second epochs , again because it is what the instructions tells us to do 

# CHANNELS
EEG_CHANNELS = ["EEG-C1", "EEG-C2"]
EOG_CHANNELS = ["EOG-L", "EOG-R"]
EMG_CHANNELS = ["EMG-Chin"]


CURRENT_ITERATION = 1

# Set to True to use cached data for preprocessing and feature extraction.
USE_CACHE = False  

# -- File Paths --
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   #Here he looks for a file in the same path as the one where the code is which is why we don't need to specify where in the laptop the files are located 
# Then it will seperate each path depending on what we require
DATA_DIR     = os.path.join(BASE_DIR, "data")   + os.sep
TRAINING_DIR = os.path.join(DATA_DIR, "training") + os.sep
HOLDOUT_DIR  = os.path.join(DATA_DIR, "holdout")  + os.sep
SAMPLE_DIR   = os.path.join(DATA_DIR, "sample")   + os.sep
CACHE_DIR    = os.path.join(BASE_DIR, "cache")    + os.sep

# Validate and create directories if needed
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

# PREPROCESSING
LOW_PASS_FILTER_FREQ = 40  # Hz

# -- Feature Extraction --
# (Add feature-specific parameters here)

# -- Classification --
# Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1:
    # Iteration 1: Basic pipeline with k-NN
    CLASSIFIER_TYPE = 'knn'
    KNN_N_NEIGHBORS = 5
elif CURRENT_ITERATION == 2:
    # Iteration 2: Enhanced EEG processing with SVM
    CLASSIFIER_TYPE = 'svm'
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'
elif CURRENT_ITERATION == 3:
    # Iteration 3: Multi-signal processing with Random Forest
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
elif CURRENT_ITERATION == 4:
    # Iteration 4: Full system optimization
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 5
else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")

# -- Submission --
SUBMISSION_FILE = 'submission.csv'
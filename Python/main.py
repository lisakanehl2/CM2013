#pip install -r requirements.txt ALWAYS START BY PASTING THIS TO START CODING

import os
import sys
import io

THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(THIS_FILE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import config
from src.data_loader import load_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache

# ---- BEGIN: Minimal Step1–2 sanity runner (safe / self-contained) ----
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import mne
from lxml import etree
from scipy.signal import butter, filtfilt, iirnotch

from config import (
    RUN_STEP12_SANITY, TRAINING_DIR, TARGET_FS, EPOCH_LEN_S,
    EEG_CHANNELS, EOG_CHANNELS, EMG_CHANNELS,
    EEG_BAND, EOG_BAND, EMG_BAND, NOTCH_FREQ
)


#--------------ALL PAIRS EDF LOADING: ITERATION 1 ; PAHE 1.1-----------------------------
def _pair_all(training_dir: str) -> List[Tuple[str, str]]:
    """
    Find ALL matching EDF/XML pairs in the training directory.
    
    Returns:
        List of tuples: [(edf1_path, xml1_path), (edf2_path, xml2_path), ...]
    """
    edfs = sorted(glob(str(Path(training_dir) / "*.edf")))
    pairs = []
    
    for edf in edfs:
        xml = str(Path(training_dir) / (Path(edf).stem + ".xml"))
        print(f"🔍 DEBUG: Checking {Path(edf).name} → Looking for {Path(xml).name}")
        
        if Path(xml).exists():
            pairs.append((edf, xml))
            print(f"  ✅ Found matching XML")
        else:
            print(f"  ❌ No XML found, skipping...")
    
    print(f"\n🔍 DEBUG: Total valid pairs found: {len(pairs)}")
    
    if not pairs:
        raise FileNotFoundError(f"No matching EDF/XML pairs in {training_dir}")
    
    return pairs



# -----------XML ANOTATION LOADING: ITERATION 1 ; PHASE1.2--------------------




def _parse_xml(xml_path: str):         #here it finds the xml files, returning a list of (start time , duration, and label )
    tree = etree.parse(xml_path)
    ann = []
    for ev in tree.findall(".//ScoredEvent"):
        s = float(ev.findtext("Start", "0"))
        d = float(ev.findtext("Duration", "0"))
        lab = ev.findtext("EventConcept", "Wake")
        ann.append((s, d, lab))
    return ann

def _majority_label(ann, t0, t1, default="Wake"):
    # Map string labels to numeric
    label_map = {
        "SDO:WakeState": 0,
        "SDO:NonRapidEyeMovementSleep-N1": 1,
        "SDO:NonRapidEyeMovementSleep-N2": 2,
        "SDO:NonRapidEyeMovementSleep-N3": 3,
        "SDO:RapidEyeMovementSleep": 4,
        "Wake": 0,  # Fallback names
        "N1": 1,
        "N2": 2,
        "N3": 3,
        "REM": 4
    }
    
    best = {}
    for s, d, lab in ann:
        a0, a1 = s, s + d
        ov = max(0.0, min(t1, a1) - max(t0, a0))
        if ov > 0:
            best[lab] = best.get(lab, 0.0) + ov
    
    if not best:
        return label_map.get(default, 0)
    
    winning_label = max(best, key=best.get)
    return label_map.get(winning_label, 0)

def _pick_existing(raw, wanted: List[str]) -> List[str]:
    names = set(raw.ch_names)
    return [w for w in wanted if w in names]



def _epochs_from_channels(raw, picks: List[str], fs: int, n_epochs: int, samples_per_epoch: int):
    if not picks:
        return None
    X = raw.copy().pick(picks).get_data()   # [n_ch, T]
    x = X.mean(axis=0)                      # average channels → mono (v1)
    x = x[:n_epochs * samples_per_epoch]
    return x.reshape(n_epochs, samples_per_epoch)




def run_step12_sanity():
    # Find ALL EDF/XML pairs
    pairs = _pair_all(TRAINING_DIR)
    print(f"[Step1–2] Found {len(pairs)} EDF/XML pairs in {TRAINING_DIR}")
    
    # Storage for all recordings
    all_data = {"EEG": [], "EOG": [], "EMG": []}
    all_labels = []
    all_record_ids = []  # Track which recording each epoch came from
    
    # Process each recording
    for idx, (edf_path, xml_path) in enumerate(pairs, 1):
        record_id = Path(edf_path).stem  # e.g., "R1", "R2", ...
        print(f"\n[Step1–2] Processing {idx}/{len(pairs)}: {record_id}")
        
        # Load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        print(f"  🔍 Available channels: {raw.ch_names}")
        if int(raw.info["sfreq"]) != int(TARGET_FS):
            raw.resample(TARGET_FS)
        fs = int(raw.info["sfreq"])
        
        # Pick channels
        eeg_p = _pick_existing(raw, EEG_CHANNELS)
        eog_p = _pick_existing(raw, EOG_CHANNELS)
        emg_p = _pick_existing(raw, EMG_CHANNELS)
        print(f"  Picks: EEG {eeg_p} | EOG {eog_p} | EMG {emg_p}")
        
        # Calculate epochs
        total_sec = float(raw.times[-1])
        n_epochs = int(np.floor(total_sec / EPOCH_LEN_S))
        spe = int(fs * EPOCH_LEN_S)
        print(f"  Total duration: {total_sec:.1f}s → {n_epochs} epochs")
        
        # Extract epoch data
        data = {}
        data["EEG"] = _epochs_from_channels(raw, eeg_p, fs, n_epochs, spe)
        data["EOG"] = _epochs_from_channels(raw, eog_p, fs, n_epochs, spe)
        data["EMG"] = _epochs_from_channels(raw, emg_p, fs, n_epochs, spe)
        data = {k: v for k, v in data.items() if v is not None}
        
        ann = _parse_xml(xml_path) 
        labels = [_majority_label(ann, i*EPOCH_LEN_S, (i+1)*EPOCH_LEN_S) for i in range(n_epochs)]

        out=data #uses epoch data as it is, without filtering
        
        # Verify epoch/label alignment
        if out:
            n_ep = next(iter(out.values())).shape[0]
            assert n_ep == len(labels), f"{record_id}: labels {len(labels)} != epochs {n_ep}"
        
        # Store data from this recording
        for k, v in out.items():
            all_data[k].append(v)
        all_labels.append(np.array(labels))
        all_record_ids.extend([record_id] * len(labels))
        
        print(f"  ✅ {record_id}: {len(labels)} epochs loaded and filtered")
    
    # Concatenate all recordings
    print(f"\n[Step1–2] Combining all {len(pairs)} recordings...")
    combined_data = {}
    for k in all_data:
        if all_data[k]:  # If this signal type exists
            combined_data[k] = np.concatenate(all_data[k], axis=0)
            print(f"  Combined {k}: {combined_data[k].shape}")
    
    combined_labels = np.concatenate(all_labels, axis=0)
    print(f"  Combined labels: {combined_labels.shape}")
    
    # Show label distribution
    unique, counts = np.unique(combined_labels, return_counts=True)
    print(f"\n[Step1–2] Label distribution across all recordings:")
    stage_names = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM'}
    for stage, count in zip(unique, counts):
        stage_name = stage_names.get(stage, f'Unknown({stage})')
        print(f"  {stage_name}: {count} epochs ({100*count/len(combined_labels):.1f}%)")
    
    print(f"\n[Step1–2] ✅ All recordings loaded — {len(pairs)} subjects, {len(combined_labels)} total epochs")
    print(f"[Step1–2] Record IDs tracked for LOSO cross-validation")


# Run the sanity block and exit cleanly (professor's code untouched)
if RUN_STEP12_SANITY:
    run_step12_sanity()
    sys.exit(0)
# ---- END: Minimal Step1–2 sanity runner ----


def _pick_training_files(training_dir: str, edf_name="R1.edf", xml_name="R1.xml"):
    """Return (edf_path, xml_path). Falls back to first .edf/.xml in the folder if R1.* not found."""
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(
            f"TRAINING_DIR does not exist: {training_dir}\n"
            f"Fix config.TRAINING_DIR or create the folder."
        )

    preferred_edf = os.path.join(training_dir, edf_name)
    preferred_xml = os.path.join(training_dir, xml_name)

    if os.path.isfile(preferred_edf) and os.path.isfile(preferred_xml):
        return preferred_edf, preferred_xml

    # Fallback: first .edf and .xml in directory
    edfs = sorted([os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.lower().endswith(".edf")])
    xmls = sorted([os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.lower().endswith(".xml")])

    if not edfs:
        raise FileNotFoundError(
            f"No .edf file found in {training_dir}. Put your EDF there or update config.TRAINING_DIR."
        )
    if not xmls:
        raise FileNotFoundError(
            f"No .xml file found in {training_dir}. Put your annotation XML there or update config.TRAINING_DIR."
        )

    # Naive pairing: if different basenames, we still take the first of each
    return edfs[0], xmls[0]


def _ensure_2d_epochs(signal, name="signal"):
    """
    Normalize shape to (n_epochs, n_samples) if possible.
    Accepts:
      - (n_epochs, n_samples)
      - (n_epochs, n_channels, n_samples) -> takes channel 0
    """
    import numpy as np

    arr = np.asarray(signal)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # take channel 0 by default
        return arr[:, 0, :]
    raise ValueError(
        f"{name} has unsupported shape {arr.shape}. Expected (epochs,samples) or (epochs,channels,samples)."
    )


def main():
    # Capture stdout safely
    stdout_buffer = io.StringIO()
    original_stdout = sys.stdout
    try:
        sys.stdout = stdout_buffer

        print("\n=== PROCESSING LOG ===")
        print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

        # 1) Data loading
        print("\n=== STEP 1: DATA LOADING ===")
        edf_file, xml_file = _pick_training_files(config.TRAINING_DIR, "R1.edf", "R1.xml")
        print(f"Using EDF: {edf_file}")
        print(f"Using XML: {xml_file}")

        try:
            multi_channel_data, labels, channel_info = load_training_data(edf_file, xml_file)
            print("Detected multi-channel loader return.")
            # Expect keys 'eeg','eog','emg'
            eeg_data = _ensure_2d_epochs(multi_channel_data["eeg"], name="EEG")             
            print(f"EEG shape: {eeg_data.shape}")
            if "eog" in multi_channel_data:
                print(f"EOG shape: {multi_channel_data['eog'].shape}")
            if "emg" in multi_channel_data:
                print(f"EMG shape: {multi_channel_data['emg'].shape}")
            print(f"Labels shape: {labels.shape}")
        except (TypeError, ValueError, KeyError):
            print("Falling back to single-channel loader return.")
            #Here the code is only focusing on the EEG data since we are still in Iteration 1:
            eeg_data, labels = load_training_data(edf_file, xml_file)
            eeg_data = _ensure_2d_epochs(eeg_data, name="EEG")
            print(f"EEG shape: {eeg_data.shape}")
            print(f"Labels shape: {labels.shape}")

        # Basic alignment check
        if eeg_data.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Epoch/label count mismatch: EEG epochs={eeg_data.shape[0]} vs labels={labels.shape[0]}.\n"
                f"Ensure your epoching matches the XML annotations (e.g., 30s epochs)."
            )

        # 2) Preprocessing (+ cache)
        print("\n=== STEP 2: PREPROCESSING ===")
        preprocessed_data = None
        cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"
        if getattr(config, "USE_CACHE", False):
            preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
            if preprocessed_data is not None:
                print("Loaded preprocessed data from cache.")

        if preprocessed_data is None:
            preprocessed_data = preprocess(eeg_data, config)
            print(f"Preprocessed data shape: {preprocessed_data.shape}")
            if getattr(config, "USE_CACHE", False):
                save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
                print("Saved preprocessed data to cache.")

        # 3) Feature Extraction (+ cache)
        print("\n=== STEP 3: FEATURE EXTRACTION ===")
        features = None
        cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
        if getattr(config, "USE_CACHE", False):
            features = load_cache(cache_filename_features, config.CACHE_DIR)
            if features is not None:
                print("Loaded features from cache.")

        if features is None:
            features = extract_features(preprocessed_data, config)
            print(f"Extracted features shape: {features.shape}")
            if features.shape[1] == 0:
                print("⚠️  WARNING: No features extracted! Implement feature extraction.")
            if getattr(config, "USE_CACHE", False):
                save_cache(features, cache_filename_features, config.CACHE_DIR)
                print("Saved features to cache.")

        # 4) Feature Selection
        print("\n=== STEP 4: FEATURE SELECTION ===")
        selected_features = select_features(features, labels, config)
        print(f"Selected features shape: {selected_features.shape}")

        # 5) Classification
        print("\n=== STEP 5: CLASSIFICATION ===")
        model = None
        if selected_features.shape[1] > 0:
            model = train_classifier(selected_features, labels, config)
            print(f"Trained {config.CLASSIFIER_TYPE} classifier.")
        else:
            print("⚠️  WARNING: Cannot train classifier - no features available!")
            print("Implement feature extraction to proceed.")

        # 6) Visualization
        print("\n=== STEP 6: VISUALIZATION ===")
        if model is not None:
            visualize_results(model, selected_features, labels, config)
        else:
            print("Skipping visualization - no trained model.")

        # 7) Report
        print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")
    finally:
        # Always restore stdout even if something crashes above
        sys.stdout = original_stdout

    # Emit / use captured log
    processing_log = stdout_buffer.getvalue()
    stdout_buffer.close()

    if 'model' in locals() and model is not None:
        generate_report(model, selected_features, labels, config, processing_log)
    else:
        print("Skipping report - no trained model.")

    print("\n" + "=" * 50)
    print("PIPELINE FINISHED")
    if 'model' not in locals() or model is None:
        print("⚠️  Students need to implement missing components!")
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Extract F0 features and metadata from CREMA-D dataset.

This script:
1. Scans AudioWAV directory for all audio files
2. Parses filenames to extract emotion and sentence information
3. Loads actor demographics from VideoDemographics.csv
4. Extracts F0 features using CREPE (deep learning-based pitch tracker)
5. Generates a comprehensive CSV with all metadata, F0 statistics, and confidence scores

Usage:
    python extract_f0_with_metadata_from_crema-d.py \
        --audio_dir crema-d-mirror/AudioWAV \
        --demographics_csv crema-d-mirror/VideoDemographics.csv \
        --output_csv data/crema_d_with_f0.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import crepe
import warnings
import librosa

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


# Mappings from the CREMA-D README
SENTENCE_MAP = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

EMOTION_MAP = {
    "ANG": "Anger",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
    "SAD": "Sad",
}

LEVEL_MAP = {
    "LO": "Low",
    "MD": "Medium",
    "HI": "High",
    "XX": "Unspecified",
}


def parse_filename(filename: str) -> dict:
    """
    Parse a CREMA-D filename to extract metadata.

    Example: 1001_IEO_NEU_XX.wav

    Parameters
    ----------
    filename : str
        Audio filename (with or without .wav extension)

    Returns
    -------
    dict
        Parsed metadata including ActorID, sentence, emotion, and level codes
    """
    # Remove .wav extension if present
    fname = filename.replace(".wav", "")

    parts = fname.split("_")
    if len(parts) != 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    actor_id, sentence_code, emotion_code, level_code = parts

    return {
        "Filename": filename,
        "ActorID": actor_id.zfill(4),
        "SentenceCode": sentence_code,
        "Sentence": SENTENCE_MAP.get(sentence_code, sentence_code),
        "EmotionCode": emotion_code,
        "Emotion": EMOTION_MAP.get(emotion_code, emotion_code),
        "LevelCode": level_code,
        "Level": LEVEL_MAP.get(level_code, level_code),
    }


def load_demographics(demographics_csv: str) -> pd.DataFrame:
    """
    Load and process actor demographics.

    Parameters
    ----------
    demographics_csv : str
        Path to VideoDemographics.csv

    Returns
    -------
    pd.DataFrame
        Demographics with ActorID normalized and Hispanic ethnicity folded into Race
    """
    demo = pd.read_csv(demographics_csv)

    # Normalize ActorID to 4-digit string
    demo["ActorID"] = demo["ActorID"].astype(str).str.zfill(4)

    # Fold Ethnicity into Race: if Ethnicity == "Hispanic", set Race = "Hispanic"
    if "Ethnicity" in demo.columns and "Race" in demo.columns:
        is_hispanic = (
            demo["Ethnicity"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("hispanic")
        )
        demo.loc[is_hispanic, "Race"] = "Hispanic"
        demo = demo.drop(columns=["Ethnicity"])

    return demo


def scan_audio_files(audio_dir: str) -> list:
    """
    Scan the audio directory for all .wav files.

    Parameters
    ----------
    audio_dir : str
        Path to AudioWAV directory

    Returns
    -------
    list
        List of audio file paths
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    wav_files = sorted(audio_path.glob("*.wav"))
    return [str(f) for f in wav_files]


def extract_f0(audio_path: str, sr=16000) -> dict:
    """
    Extract F0 from an audio file using CREPE (deep learning-based pitch tracker).

    Parameters
    ----------
    audio_path : str
        Path to the audio file
    sr : int
        Sample rate in Hz (default: 16000 for CREMA-D)

    Returns
    -------
    dict with:
        - meanF0: mean F0 of voiced frames (Hz)
        - sdF0: standard deviation of voiced F0 (Hz)
        - iqrF0: interquartile range of voiced F0 (Hz)
        - confidence: mean confidence score for F0 predictions
    """
    try:
        # Load audio file using librosa
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)

        # CREPE predict returns time, frequency, confidence, activation
        _, frequency, confidence, _ = crepe.predict(
            audio,
            sr,
            model_capacity='medium',
            viterbi=True,
            verbose=0  # Suppress output
        )

        # Filter by confidence threshold and exclude unvoiced frames (freq > 0)
        # Common confidence threshold is 0.5
        confidence_threshold = 0.5
        voiced_mask = (frequency > 0) & (confidence > confidence_threshold)
        voiced_f0 = frequency[voiced_mask]
        voiced_confidence = confidence[voiced_mask]

        if len(voiced_f0) == 0:
            return {
                "meanF0": np.nan,
                "sdF0": np.nan,
                "iqrF0": np.nan,
                "confidence": np.nan
            }

        return {
            "meanF0": np.mean(voiced_f0),
            "sdF0": np.std(voiced_f0),
            "iqrF0": np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25),
            "confidence": np.mean(voiced_confidence),
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return {
            "meanF0": np.nan,
            "sdF0": np.nan,
            "iqrF0": np.nan,
            "confidence": np.nan
        }


def build_dataset(audio_dir: str, demographics_csv: str) -> pd.DataFrame:
    """
    Build complete dataset with metadata and F0 features.

    Parameters
    ----------
    audio_dir : str
        Path to AudioWAV directory
    demographics_csv : str
        Path to VideoDemographics.csv

    Returns
    -------
    pd.DataFrame
        Complete dataset with all metadata, F0 features, and confidence scores
    """
    # Load demographics
    print(f"Loading demographics from {demographics_csv}...")
    demographics = load_demographics(demographics_csv)

    # Scan audio files
    print(f"Scanning audio files in {audio_dir}...")
    audio_files = scan_audio_files(audio_dir)
    print(f"Found {len(audio_files)} audio files")

    # Process each audio file
    rows = []
    print(f"Processing audio files and extracting F0 using CREPE medium...")

    for audio_path in tqdm(audio_files):
        filename = Path(audio_path).name

        # Parse filename for metadata
        try:
            metadata = parse_filename(filename)
        except ValueError as e:
            print(f"Skipping {filename}: {e}")
            continue

        # Extract F0 features using CREPE
        f0_features = extract_f0(audio_path)

        # Combine metadata and F0 features
        row = {**metadata, **f0_features, "AudioPath": audio_path}
        rows.append(row)

    # Create dataframe
    df = pd.DataFrame(rows)

    # Merge with demographics
    print("Merging with demographics...")
    df = df.merge(demographics, on="ActorID", how="left")

    # Reorder columns for better readability
    column_order = [
        "Filename",
        "AudioPath",
        "ActorID",
        "Age",
        "Sex",
        "Race",
        "SentenceCode",
        "Sentence",
        "EmotionCode",
        "Emotion",
        "LevelCode",
        "Level",
        "meanF0",
        "sdF0",
        "iqrF0",
        "confidence",
    ]

    # Reorder, keeping any extra columns at the end
    available_cols = [col for col in column_order if col in df.columns]
    extra_cols = [col for col in df.columns if col not in column_order]
    df = df[available_cols + extra_cols]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract F0 features and metadata from CREMA-D dataset using CREPE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python3 QML2025/code/helper_scripts/extract_f0_with_metadata_from_crema-d.py \\
        --audio_dir crema-d-mirror/AudioWAV \\
        --demographics_csv crema-d-mirror/VideoDemographics.csv \\
        --output_csv QML2025/data/crema_d_with_f0.csv
        """
    )

    parser.add_argument(
        "--audio_dir",
        required=True,
        help="Path to AudioWAV directory containing .wav files",
    )
    parser.add_argument(
        "--demographics_csv",
        required=True,
        help="Path to VideoDemographics.csv",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to write output CSV with metadata and F0 features",
    )

    args = parser.parse_args()

    # Build dataset
    df = build_dataset(args.audio_dir, args.demographics_csv)

    # Save to CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSuccessfully processed {len(df)} audio files")
    print(f"Output saved to: {args.output_csv}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSample statistics:")
    print(f"  - Unique actors: {df['ActorID'].nunique()}")
    print(f"  - Unique sentences: {df['SentenceCode'].nunique()}")
    print(f"  - Unique emotions: {df['EmotionCode'].nunique()}")
    print(f"\nF0 statistics:")
    print(f"  - Mean F0: {df['meanF0'].mean():.2f} Hz (±{df['meanF0'].std():.2f})")
    print(f"  - Mean F0 SD: {df['sdF0'].mean():.2f} Hz")
    print(f"  - Mean F0 IQR: {df['iqrF0'].mean():.2f} Hz")
    print(f"  - Mean Confidence: {df['confidence'].mean():.3f}")


if __name__ == "__main__":
    main()

"""
NoXi Session Preprocessing for ASAP Model
Usage: python preprocessing/preprocess.py <session_name>
"""

import os
import numpy as np
import pandas as pd
import opensmile
from pathlib import Path


def load_visual_features(csv_path):
    print(f"  Loading {csv_path.name}")
    df = pd.read_csv(csv_path, index_col=0)
    print(f"    {len(df)} frames (~{len(df)/25:.1f}s)")
    return df


def extract_audio_features(audio_path):
    print(f"  Extracting {audio_path.name}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    audio_features = smile.process_file(audio_path)
    audio_resampled = audio_features.iloc[::4]  # 100fps → 25fps

    print(f"    {len(audio_resampled)} frames")
    return audio_resampled


def select_participant_features(visual_df, audio_df):
    """
    Select 28 features per participant for ASAP.
    Visual (12): rotation(3) + AUs(7) + gaze(2)
    Audio (16): prosodic and spectral features
    """

    visual_features = [
        'head_x', 'head_y', 'head_z',
        'AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU12',
        'gaze_x', 'gaze_y',
    ]

    missing = [f for f in visual_features if f not in visual_df.columns]
    if missing:
        print(f"    Warning: Missing {missing}, filling with zeros")
        for f in missing:
            visual_df[f] = 0.0

    visual_selected = visual_df[visual_features].values

    audio_keywords = ['F0semitone', 'loudness', 'jitterLocal', 'shimmer', 'HNR', 'mfcc']

    audio_cols = []
    for kw in audio_keywords:
        matching = [col for col in audio_df.columns if kw.lower() in col.lower()]
        audio_cols.extend(matching[:3])
        if len(audio_cols) >= 16:
            break

    audio_cols = audio_cols[:16]

    if len(audio_cols) < 16:
        print(f"    Warning: Only {len(audio_cols)} audio features, padding with zeros")
        audio_selected = np.zeros((len(audio_df), 16))
        if audio_cols:
            audio_selected[:, :len(audio_cols)] = audio_df[audio_cols].values
    else:
        audio_selected = audio_df[audio_cols].values

    min_len = min(len(visual_selected), len(audio_selected))
    visual_sync = visual_selected[:min_len]
    audio_sync = audio_selected[:min_len]

    combined = np.concatenate([visual_sync, audio_sync], axis=1)

    return combined, visual_features, audio_cols


def create_sequences(features, seq_length=100, stride=1):
    """Create sliding window sequences. Y is shifted by 1 frame."""
    print(f"\nCreating sequences (length={seq_length}, stride={stride})")

    X, Y = [], []
    for i in range(0, len(features) - seq_length, stride):
        X.append(features[i:i+seq_length])
        Y.append(features[i+1:i+seq_length+1])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"  Created {len(X)} sequences")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")

    return X, Y


def preprocess_session(session_path, seq_length=100, stride=1):
    """
    Preprocess one NoXi session into dyadic format for ASAP.
    Returns X and Y with shape (sequences, 100, 56) where 56 = expert(28) + novice(28).
    """
    session_path = Path(session_path)

    print(f"\n{'='*60}")
    print(f"Processing session: {session_path.name}")
    print(f"{'='*60}\n")

    # Process expert
    print("Expert:")
    exp_vis = load_visual_features(session_path / "non_varbal_expert.csv")
    exp_aud = extract_audio_features(session_path / "audio_expert.wav")
    expert_feat, exp_vis_names, exp_aud_names = select_participant_features(exp_vis, exp_aud)
    print(f"  Expert features: {expert_feat.shape}")

    # Process novice
    print("\nNovice:")
    nov_vis = load_visual_features(session_path / "non_varbal_novice.csv")
    nov_aud = extract_audio_features(session_path / "audio_novice.wav")
    novice_feat, nov_vis_names, nov_aud_names = select_participant_features(nov_vis, nov_aud)
    print(f"  Novice features: {novice_feat.shape}")

    # Synchronize and combine into dyadic format
    min_len = min(len(expert_feat), len(novice_feat))
    dyadic = np.concatenate([expert_feat[:min_len], novice_feat[:min_len]], axis=1)

    print(f"\nDyadic features: {dyadic.shape} ({min_len/25:.1f}s)")

    X, Y = create_sequences(dyadic, seq_length, stride)

    feature_names = (
        [f"expert_{name}" for name in exp_vis_names + exp_aud_names] +
        [f"novice_{name}" for name in nov_vis_names + nov_aud_names]
    )

    return {
        'session_name': session_path.name,
        'X': X,
        'Y': Y,
        'feature_names': feature_names,
        'num_sequences': len(X),
        'duration_seconds': min_len / 25,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess NoXi for ASAP model')
    parser.add_argument('session_name', type=str, help='Session name (e.g., Augsburg_01)')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    session_path = project_root / "data" / "raw" / "noxi" / args.session_name

    result = preprocess_session(session_path)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Session: {result['session_name']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")
    print(f"Sequences: {result['num_sequences']}")
    print(f"\nFeatures (56 total = 28 expert + 28 novice):")
    for i, name in enumerate(result['feature_names'], 1):
        print(f"  {i:2d}. {name}")

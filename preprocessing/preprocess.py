"""
NoXi Session Preprocessing for ASAP Model
Usage:
    python preprocessing/preprocess.py Augsburg_01
    python preprocessing/preprocess.py --city Augsburg
    python preprocessing/preprocess.py --all
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
    """
    Create sliding window sequences.
    """
    print(f"\nCreating sequences (length={seq_length}, stride={stride})")

    visual_indices = list(range(0, 12)) + list(range(28, 40))

    X, Y = [], []
    for i in range(0, len(features) - seq_length, stride):
        X.append(features[i:i+seq_length])
        Y.append(features[i+1:i+seq_length+1, visual_indices])

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

    print("Expert:")
    exp_vis = load_visual_features(session_path / "non_varbal_expert.csv")
    exp_aud = extract_audio_features(session_path / "audio_expert.wav")
    expert_feat, exp_vis_names, exp_aud_names = select_participant_features(exp_vis, exp_aud)
    print(f"  Expert features: {expert_feat.shape}")

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


def preprocess_city(data_dir, output_dir, city_name):
    """Preprocess all sessions for a specific city."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir) / city_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sessions for the city
    sessions = sorted([d.name for d in data_dir.iterdir()
                      if d.is_dir() and d.name.startswith(city_name)])

    if not sessions:
        print(f"No sessions found for city: {city_name}")
        return

    print(f"\nFound {len(sessions)} sessions for {city_name}")

    X_list, Y_list = [], []
    for session in sessions:
        try:
            result = preprocess_session(data_dir / session)
            X_list.append(result['X'])
            Y_list.append(result['Y'])
        except Exception as e:
            print(f"Skipping {session}: {e}")

    # Combine all sessions
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)

    print(f"\n{'='*60}")
    print(f"Combined: {X_all.shape}")
    print(f"{'='*60}")

    mean_X = X_all.mean(axis=(0, 1))
    std_X = X_all.std(axis=(0, 1)) + 1e-8
    X_norm = (X_all - mean_X) / std_X

    visual_indices = list(range(0, 12)) + list(range(28, 40))
    mean_Y = mean_X[visual_indices]
    std_Y = std_X[visual_indices]
    Y_norm = (Y_all - mean_Y) / std_Y

    # Split into train (80%) and val (20%)
    split_idx = int(len(X_norm) * 0.8)
    X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
    Y_train, Y_val = Y_norm[:split_idx], Y_norm[split_idx:]

    # Save
    np.save(output_dir / "Xij_train.npy", X_train)
    np.save(output_dir / "Yij_train.npy", Y_train)
    np.save(output_dir / "Xij_val.npy", X_val)
    np.save(output_dir / "Yij_val.npy", Y_val)
    np.save(output_dir / "stats.npy", {'mean_X': mean_X, 'std_X': std_X, 'mean_Y': mean_Y, 'std_Y': std_Y})

    print(f"\nSaved to {output_dir}:")
    print(f"  Xij_train.npy: {X_train.shape}")
    print(f"  Yij_train.npy: {Y_train.shape}")
    print(f"  Xij_val.npy: {X_val.shape}")
    print(f"  Yij_val.npy: {Y_val.shape}")


def preprocess_all(data_dir, output_dir):
    """Preprocess all sessions in data_dir and save combined output."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all sessions
    sessions = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(sessions)} sessions")
    print("="*60)

    X_list, Y_list = [], []
    for session in sessions:
        try:
            result = preprocess_session(data_dir / session)
            X_list.append(result['X'])
            Y_list.append(result['Y'])
        except Exception as e:
            print(f"Skipping {session}: {e}")

    # Combine all sessions
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)

    print(f"\n{'='*60}")
    print(f"Combined: {X_all.shape}")
    print(f"{'='*60}")

    mean_X = X_all.mean(axis=(0, 1))
    std_X = X_all.std(axis=(0, 1)) + 1e-8
    X_norm = (X_all - mean_X) / std_X

    visual_indices = list(range(0, 12)) + list(range(28, 40))
    mean_Y = mean_X[visual_indices]
    std_Y = std_X[visual_indices]
    Y_norm = (Y_all - mean_Y) / std_Y

    # Split into train (80%) and val (20%)
    split_idx = int(len(X_norm) * 0.8)
    X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
    Y_train, Y_val = Y_norm[:split_idx], Y_norm[split_idx:]

    # Save
    np.save(output_dir / "Xij_train.npy", X_train)
    np.save(output_dir / "Yij_train.npy", Y_train)
    np.save(output_dir / "Xij_val.npy", X_val)
    np.save(output_dir / "Yij_val.npy", Y_val)
    np.save(output_dir / "stats.npy", {'mean_X': mean_X, 'std_X': std_X, 'mean_Y': mean_Y, 'std_Y': std_Y})

    print(f"\nSaved to {output_dir}:")
    print(f"  Xij_train.npy: {X_train.shape}")
    print(f"  Yij_train.npy: {Y_train.shape}")
    print(f"  Xij_val.npy: {X_val.shape}")
    print(f"  Yij_val.npy: {Y_val.shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess NoXi for ASAP model')
    parser.add_argument('session_name', type=str, nargs='?', help='Single session (e.g., Augsburg_01)')
    parser.add_argument('--all', action='store_true', help='Process all sessions')
    parser.add_argument('--city', type=str, help='Process all sessions for a city (e.g., Augsburg)')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "raw" / "noxi"

    if args.city:
        output_dir = project_root / "data" / "processed"
        preprocess_city(data_dir, output_dir, args.city)
    elif args.all:
        output_dir = project_root / "data" / "processed"
        preprocess_all(data_dir, output_dir)
    else:
        if not args.session_name:
            print("Usage: python preprocessing/preprocess.py [SESSION_NAME | --city CITY | --all]")
            exit(1)

        result = preprocess_session(data_dir / args.session_name)

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Session: {result['session_name']}")
        print(f"Duration: {result['duration_seconds']:.2f}s")
        print(f"Sequences: {result['num_sequences']}")
        print(f"\nFeatures (56 total = 28 expert + 28 novice):")
        for i, name in enumerate(result['feature_names'], 1):
            print(f"  {i:2d}. {name}")

        # Save single session to disk
        output_dir = project_root / "data" / "processed" / result['session_name']
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X.npy", result['X'])
        np.save(output_dir / "Y.npy", result['Y'])
        np.save(output_dir / "metadata.npy", {
            'session_name': result['session_name'],
            'feature_names': result['feature_names'],
            'num_sequences': result['num_sequences'],
            'duration_seconds': result['duration_seconds']
        })

        print(f"\n{'='*60}")
        print(f"Saved to {output_dir}:")
        print(f"  X.npy: {result['X'].shape}")
        print(f"  Y.npy: {result['Y'].shape}")
        print(f"  metadata.npy")
        print(f"{'='*60}")

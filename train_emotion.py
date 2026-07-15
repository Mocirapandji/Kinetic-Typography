import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    return (
        Path,
        RandomForestClassifier,
        StratifiedGroupKFold,
        accuracy_score,
        call,
        classification_report,
        confusion_matrix,
        cross_val_predict,
        mo,
        np,
        os,
        parselmouth,
        pd,
        plt,
    )


@app.cell
def _():
    data_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_emotions = {"calm"}  # 7-class scheme drops calm
    features_csv = "outputs/features.csv"
    return data_dir, drop_emotions, emotion_map, features_csv


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Clip-level prosodic feature extraction
    """)
    return


@app.cell
def _(call, np, parselmouth):
    def extract_clip_features(path):
        snd = parselmouth.Sound(str(path))
        dur = snd.get_total_duration()

        # --- pitch (F0), voiced frames only ---
        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0v = f0[f0 > 0]
        if len(f0v) > 0:
            f0_mean = float(np.mean(f0v)); f0_std = float(np.std(f0v))
            f0_min = float(np.min(f0v)); f0_max = float(np.max(f0v))
            f0_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = 0.0
        voiced_fraction = float(len(f0v) / len(f0)) if len(f0) > 0 else 0.0

        # --- intensity (dB) ---
        try:
            intensity = snd.to_intensity()
            ivals = intensity.values.flatten()
            ivals = ivals[np.isfinite(ivals)]
            int_mean = float(np.mean(ivals)); int_std = float(np.std(ivals))
            int_max = float(np.max(ivals)); int_min = float(np.min(ivals))
        except Exception:
            int_mean = int_std = int_max = int_min = 0.0

        # --- overall RMS energy ---
        rms = float(call(snd, "Get root-mean-square", 0, 0))

        # --- HNR (voice quality) ---
        try:
            harm = snd.to_harmonicity()
            hv = harm.values[harm.values != -200]
            hnr_mean = float(np.mean(hv)) if len(hv) > 0 else 0.0
            hnr_std = float(np.std(hv)) if len(hv) > 0 else 0.0
        except Exception:
            hnr_mean = hnr_std = 0.0

        return {
            "duration": dur, "voiced_fraction": voiced_fraction,
            "f0_mean": f0_mean, "f0_std": f0_std, "f0_min": f0_min,
            "f0_max": f0_max, "f0_range": f0_range,
            "rms": rms, "intensity_mean": int_mean, "intensity_std": int_std,
            "intensity_max": int_max, "intensity_min": int_min,
            "hnr_mean": hnr_mean, "hnr_std": hnr_std,
        }

    return (extract_clip_features,)


@app.cell
def _(
    Path,
    data_dir,
    drop_emotions,
    emotion_map,
    extract_clip_features,
    features_csv,
    os,
    pd,
):
    # cache: extract once, reuse afterwards (extraction is the slow part)
    if os.path.exists(features_csv):
        df = pd.read_csv(features_csv)
        print(f"Loaded cached features: {len(df)} clips from {features_csv}")
    else:
        os.makedirs("outputs", exist_ok=True)
        wav_paths = sorted(Path(data_dir).glob("Actor_*/*.wav"))
        print(f"Found {len(wav_paths)} wav files. Extracting features...")
        rows = []
        skipped = 0
        for n, p in enumerate(wav_paths, 1):
            parts = p.stem.split("-")
            emotion = emotion_map.get(parts[2], "unknown")
            if emotion in drop_emotions or emotion == "unknown":
                continue
            actor = int(parts[6])
            try:
                feats = extract_clip_features(p)
            except Exception as e:
                skipped += 1
                continue
            feats["file"] = p.name
            feats["emotion"] = emotion
            feats["actor"] = actor
            rows.append(feats)
            if n % 200 == 0:
                print(f"  ...processed {n}/{len(wav_paths)}")
        df = pd.DataFrame(rows)
        df.to_csv(features_csv, index=False)
        print(f"Done. Kept {len(df)} clips, skipped {skipped}. Saved to {features_csv}")
    df
    return (df,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Class balance check
    """)
    return


@app.cell
def _(df):
    df["emotion"].value_counts()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Train Random Forest (speaker-independent CV)
    """)
    return


@app.cell
def _(df):
    feature_cols = [c for c in df.columns if c not in ("file", "emotion", "actor")]
    X = df[feature_cols].values
    y = df["emotion"].values
    groups = df["actor"].values
    print("Feature matrix:", X.shape)
    print("Features used:", feature_cols)
    return X, feature_cols, groups, y


@app.cell
def _(
    RandomForestClassifier,
    StratifiedGroupKFold,
    X,
    accuracy_score,
    classification_report,
    cross_val_predict,
    groups,
    np,
    y,
):
    clf = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)

    acc = accuracy_score(y, y_pred)
    n_classes = len(np.unique(y))
    print(f"Speaker-independent 5-fold accuracy: {acc:.3f}")
    print(f"Random-chance baseline (1/{n_classes}):     {1/n_classes:.3f}")
    print()
    print(classification_report(y, y_pred))
    return (y_pred,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Confusion matrix
    """)
    return


@app.cell
def _(confusion_matrix, np, pd, y, y_pred):
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{l}" for l in labels_sorted],
        columns=[f"pred_{l}" for l in labels_sorted],
    )
    cm_df
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Feature importances (which prosody features carry emotion)
    """)
    return


@app.cell
def _(RandomForestClassifier, X, feature_cols, pd, y):
    clf_full = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ).fit(X, y)
    imp = pd.DataFrame(
        {"feature": feature_cols, "importance": clf_full.feature_importances_}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    imp
    return clf_full, imp


@app.cell
def _(imp, plt):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importances (prosody to emotion)")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 6: Visual Inference (Create MP4 with Emotion Label)
    """)
    return


@app.cell
def _(clf_full, extract_clip_features, feature_cols, pd):
    # 1. Select demo file
    demo_audio_rf = "data/raw/Actor_01/03-01-06-01-02-01-01.wav"
    feats_rf = extract_clip_features(demo_audio_rf)

    # 2. Format features
    df_demo_rf = pd.DataFrame([feats_rf])[feature_cols]

    # 3. Predict using the full Random Forest model
    pred_emotion_rf = clf_full.predict(df_demo_rf)[0]

    print(f"Predicted RF Emotion: {pred_emotion_rf.upper()}")
    return demo_audio_rf, pred_emotion_rf


@app.cell
def _(demo_audio_rf, pred_emotion_rf):
    def render_rf_video(audio_path, emotion, out_dir="outputs"):
        import os, subprocess
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720

        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {width}\n"
            f"PlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,96,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,1,5,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        color_map = {
            "happy": "&H0000FF00&",     
            "sad": "&H00FF0000&",       
            "angry": "&H000000FF&",     
            "fearful": "&H00FF00FF&",   
            "disgust": "&H0000A5FF&",   
            "surprised": "&H00FFFF00&", 
            "neutral": "&H00FFFFFF&"    
        }
        ass_color = color_map.get(emotion, "&H00FFFFFF&")

        start = "0:00:00.00"
        end = "0:01:00.00" 
        text = f"{{\\c{ass_color}\\an5}}RF Prediction:\\N{emotion.upper()}"
        line = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"

        ass_path = f"{out_dir}/ass/rf_demo.ass"
        with open(ass_path, "w") as f:
            f.write(header + line)

        out_path = f"{out_dir}/video/rf_demo.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=25:d=60",
            "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return out_path

    out_vid_rf = render_rf_video(demo_audio_rf, pred_emotion_rf)
    print("Wrote RF visual inference to:", out_vid_rf)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

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
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    return (
        GridSearchCV,
        MLPClassifier,
        Path,
        Pipeline,
        StandardScaler,
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
    )


@app.cell
def _():
    data_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_emotions = {"calm"}
    features_csv = "outputs/features.csv"
    return data_dir, drop_emotions, emotion_map, features_csv


@app.cell
def _(mo):
    mo.md("""
    ## Step 1: Load features (from cache, or extract if missing)
    """)
    return


@app.cell
def _(call, np, parselmouth):
    def extract_clip_features(path):
        snd = parselmouth.Sound(str(path))
        dur = snd.get_total_duration()

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

        try:
            intensity = snd.to_intensity()
            ivals = intensity.values.flatten()
            ivals = ivals[np.isfinite(ivals)]
            int_mean = float(np.mean(ivals)); int_std = float(np.std(ivals))
            int_max = float(np.max(ivals)); int_min = float(np.min(ivals))
        except Exception:
            int_mean = int_std = int_max = int_min = 0.0

        rms = float(call(snd, "Get root-mean-square", 0, 0))

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
    if os.path.exists(features_csv):
        df = pd.read_csv(features_csv)
        print(f"Loaded cached features: {len(df)} clips from {features_csv}")
    else:
        os.makedirs("outputs", exist_ok=True)
        wav_paths = sorted(Path(data_dir).glob("Actor_*/*.wav"))
        print(f"No cache found. Extracting from {len(wav_paths)} wav files...")
        rows = []
        for n, p in enumerate(wav_paths, 1):
            parts = p.stem.split("-")
            emotion = emotion_map.get(parts[2], "unknown")
            if emotion in drop_emotions or emotion == "unknown":
                continue
            actor = int(parts[6])
            try:
                feats = extract_clip_features(p)
            except Exception:
                continue
            feats["file"] = p.name
            feats["emotion"] = emotion
            feats["actor"] = actor
            rows.append(feats)
            if n % 200 == 0:
                print(f"  ...processed {n}/{len(wav_paths)}")
        df = pd.DataFrame(rows)
        df.to_csv(features_csv, index=False)
        print(f"Saved {len(df)} clips to {features_csv}")
    df
    return (df,)


@app.cell
def _(df):
    feature_cols = [c for c in df.columns if c not in ("file", "emotion", "actor")]
    X = df[feature_cols].values
    y = df["emotion"].values
    groups = df["actor"].values
    print("Feature matrix:", X.shape)
    print("Classes:", sorted(set(y)))
    return X, groups, y


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: MLP - genuine best-effort search (scaled, tuned, regularised)
    """)
    return


@app.cell
def _(
    GridSearchCV,
    MLPClassifier,
    Pipeline,
    StandardScaler,
    StratifiedGroupKFold,
    X,
    groups,
    y,
):
    # StandardScaler is INSIDE the pipeline so it is fit per-fold (no leakage).
    mlp_base = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            solver="adam",
            max_iter=1500,
            early_stopping=True,       # stop when held-out val stops improving
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64), (128, 64, 32)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],            # L2 regularisation strength
        "mlp__learning_rate_init": [1e-3, 5e-4],
    }
    grid_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    mlp_search = GridSearchCV(
        mlp_base, param_grid, cv=grid_cv, scoring="accuracy", n_jobs=-1
    )
    mlp_search.fit(X, y, groups=groups)

    print("Best MLP config found:")
    for k, v in mlp_search.best_params_.items():
        print(f"  {k} = {v}")
    print(f"\nBest MLP CV accuracy (3-fold grouped, during search): {mlp_search.best_score_:.3f}")
    return (mlp_search,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: Evaluate best MLP (speaker-independent 5-fold)
    """)
    return


@app.cell
def _(
    StratifiedGroupKFold,
    X,
    accuracy_score,
    classification_report,
    cross_val_predict,
    groups,
    mlp_search,
    np,
    y,
):
    best_mlp = mlp_search.best_estimator_
    eval_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_mlp = cross_val_predict(best_mlp, X, y, cv=eval_cv, groups=groups)

    acc_mlp = accuracy_score(y, y_pred_mlp)
    n_cls = len(np.unique(y))
    print(f"MLP speaker-independent 5-fold accuracy: {acc_mlp:.3f}")
    print(f"Random-chance baseline (1/{n_cls}): {1/n_cls:.3f}")
    print()
    print(classification_report(y, y_pred_mlp))
    return acc_mlp, y_pred_mlp


@app.cell
def _(mo):
    mo.md("""
    ## Step 4: Confusion matrix
    """)
    return


@app.cell
def _(confusion_matrix, np, pd, y, y_pred_mlp):
    labels_mlp = sorted(np.unique(y))
    cm_mlp = confusion_matrix(y, y_pred_mlp, labels=labels_mlp)
    cm_mlp_df = pd.DataFrame(
        cm_mlp,
        index=[f"true_{l}" for l in labels_mlp],
        columns=[f"pred_{l}" for l in labels_mlp],
    )
    cm_mlp_df
    return


@app.cell
def _(acc_mlp, mo):
    mo.md(f"""
    ### MLP result\n\n"
    f"- Best-config MLP, speaker-independent 5-fold accuracy: **{acc_mlp:.3f}**\n"
    f"- (Comparison against the Random Forest is done in the separate comparison notebook.)\n
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 5: Visual Inference (Create MP4 with Emotion Label)
    """)
    return


@app.cell
def _(df, extract_clip_features, mlp_search, pd):
    # 1. Select a demo file and extract features
    demo_audio_mlp = "data/raw/Actor_01/03-01-06-01-02-01-01.wav"
    feats_mlp = extract_clip_features(demo_audio_mlp)

    # 2. Format features into a DataFrame matching training columns
    feature_cols_mlp = [c for c in df.columns if c not in ("file", "emotion", "actor")]
    df_demo_mlp = pd.DataFrame([feats_mlp])[feature_cols_mlp]

    # 3. Predict using the best MLP from grid search
    best_mlp = mlp_search.best_estimator_
    pred_emotion_mlp = best_mlp.predict(df_demo_mlp)[0]

    print(f"Predicted MLP Emotion: {pred_emotion_mlp.upper()}")
    return demo_audio_mlp, pred_emotion_mlp


@app.cell
def _(demo_audio_mlp, pred_emotion_mlp):
    def render_mlp_video(audio_path, emotion, out_dir="outputs"):
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

        # Color coding by emotion (ASS format is &HBBGGRR&)
        color_map = {
            "happy": "&H0000FF00&",     # Green
            "sad": "&H00FF0000&",       # Blue
            "angry": "&H000000FF&",     # Red
            "fearful": "&H00FF00FF&",   # Magenta
            "disgust": "&H0000A5FF&",   # Orange
            "surprised": "&H00FFFF00&", # Cyan
            "neutral": "&H00FFFFFF&"    # White
        }
        ass_color = color_map.get(emotion, "&H00FFFFFF&")

        start = "0:00:00.00"
        end = "0:01:00.00" # Arbitrary long end time, ffmpeg -shortest handles the trim
        text = f"{{\\c{ass_color}\\an5}}MLP Prediction:\\N{emotion.upper()}"
        line = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n"

        ass_path = f"{out_dir}/ass/mlp_demo.ass"
        with open(ass_path, "w") as f:
            f.write(header + line)

        out_path = f"{out_dir}/video/mlp_demo.mp4"
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

    out_vid_mlp = render_mlp_video(demo_audio_mlp, pred_emotion_mlp)
    print("Wrote MLP visual inference to:", out_vid_mlp)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

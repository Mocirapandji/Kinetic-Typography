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
        RandomForestClassifier,
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
        plt,
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
    return X, groups, y


@app.cell
def _(mo):
    mo.md("""
    ## Step 2: Random Forest baseline (same CV as the MLP will use)
    """)
    return


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
    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    rf_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_rf = cross_val_predict(rf, X, y, cv=rf_cv, groups=groups)

    acc = accuracy_score(y, y_pred_rf)
    print(f"Random Forest speaker-independent 5-fold accuracy: {acc:.3f}")
    print(f"Random-chance baseline (1/{len(np.unique(y))}): {1/len(np.unique(y)):.3f}")
    print()
    print(classification_report(y, y_pred_rf))
    return (acc,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3: MLP - genuine best-effort search (scaled, tuned, regularised)
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
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (64, 32), (128, 64), (128, 64, 32)],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
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
    ## Step 4: Evaluate best MLP on the SAME 5-fold speaker-independent CV
    """)
    return


@app.cell
def _(
    StratifiedGroupKFold,
    X,
    acc,
    accuracy_score,
    classification_report,
    cross_val_predict,
    groups,
    mlp_search,
    y,
):
    best_mlp = mlp_search.best_estimator_
    eval_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_mlp = cross_val_predict(best_mlp, X, y, cv=eval_cv, groups=groups)

    acc_mlp = accuracy_score(y, y_pred_mlp)
    print(f"MLP (best config)  speaker-independent 5-fold accuracy: {acc_mlp:.3f}")
    print(f"Random Forest      speaker-independent 5-fold accuracy: {acc:.3f}")
    print()
    print("MLP per-class report:")
    print(classification_report(y, y_pred_mlp))
    return acc_mlp, y_pred_mlp


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
def _(mo):
    mo.md("""
    ## Step 5: Head-to-head
    """)
    return


@app.cell
def _(acc, acc_mlp, np, plt, y):
    n_cls = len(np.unique(y))
    fig_cmp, ax_cmp = plt.subplots(figsize=(5, 4))
    names = ["Random Forest", "MLP (best)"]
    vals = [acc, acc_mlp]
    bars = ax_cmp.bar(names, vals, color=["#4C72B0", "#DD8452"])
    ax_cmp.axhline(1 / n_cls, ls="--", color="gray", label=f"chance ({1/n_cls:.2f})")
    ax_cmp.set_ylabel("Speaker-independent accuracy")
    ax_cmp.set_ylim(0, max(vals) * 1.25)
    ax_cmp.set_title("Random Forest vs MLP (same features, same CV)")
    ax_cmp.legend()
    for b, a in zip(bars, vals):
        ax_cmp.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a:.3f}", ha="center")
    fig_cmp.tight_layout()
    fig_cmp
    return


@app.cell
def _(acc, acc_mlp, mo):
    _winner = "Random Forest" if acc >= acc_mlp else "MLP"
    _delta = abs(acc - acc_mlp)
    mo.md(
        f"### Verdict\n\n"
        f"- Random Forest: **{acc:.3f}**\n"
        f"- MLP (best config found): **{acc_mlp:.3f}**\n"
        f"- Winner: **{_winner}** by {_delta:.3f}\n"
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

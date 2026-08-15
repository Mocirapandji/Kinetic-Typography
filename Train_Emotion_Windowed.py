import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # =====================================================================
    # CELL 0 — IMPORTS
    # =====================================================================
    import marimo as mo
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    mo.md("""
    # Windowed training: sub-clip features instead of one whole-clip vector

    One 3-second clip currently collapses to ONE row of 14 numbers. Averaging
    pitch over a whole sentence throws away the shape of the delivery — a rise
    into an angry peak and a flat sad line can end up with a similar mean.

    This version slides a window across each clip, extracts the same 14
    features per window, and lets each window vote. Same features, same
    classifier, more rows and finer time resolution.
    """)

    # =====================================================================
    # CELL 1 — CONFIG
    # =====================================================================
    ravdess_root = "/run/media/s5812886/T7 Shield/RAVDESS"
    win_features_csv = "outputs/features_windowed.csv"

    # Window geometry. WIN_SEC is roughly "a few words" of speech at normal
    # speaking rate — the contextual chunk you wanted, but defined in time so
    # it needs no transcription. HOP_SEC < WIN_SEC means windows overlap, so a
    # feature straddling a boundary still lands whole inside some window.
    WIN_SEC = 1.5
    HOP_SEC = 0.75
    MIN_WIN = 0.6      # Praat needs a floor to measure pitch at all
    MIN_VOICED = 0.10  # drop windows that are essentially silence

    emotion_codes = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                     "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_set = {"calm"}

    # =====================================================================
    # CELL 2 — FEATURE EXTRACTOR (identical 14 features, now on a Sound slice)
    # Same function as extract_clip_features_from_sound in the V8 notebook.
    # Kept standalone so this script runs on its own.
    # =====================================================================
    def feats_from_sound(snd):
        dur = snd.get_total_duration()

        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0v = f0[f0 > 0]
        if len(f0v) > 0:
            f0_mean = float(np.mean(f0v)); f0_std = float(np.std(f0v))
            f0_min = float(np.min(f0v));   f0_max = float(np.max(f0v))
            f0_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = 0.0
        voiced_fraction = float(len(f0v) / len(f0)) if len(f0) > 0 else 0.0

        try:
            inten = snd.to_intensity()
            iv = inten.values.flatten()
            iv = iv[np.isfinite(iv)]
            int_mean = float(np.mean(iv)); int_std = float(np.std(iv))
            int_max = float(np.max(iv));   int_min = float(np.min(iv))
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

        return {"duration": dur, "voiced_fraction": voiced_fraction,
                "f0_mean": f0_mean, "f0_std": f0_std, "f0_min": f0_min,
                "f0_max": f0_max, "f0_range": f0_range,
                "rms": rms, "intensity_mean": int_mean, "intensity_std": int_std,
                "intensity_max": int_max, "intensity_min": int_min,
                "hnr_mean": hnr_mean, "hnr_std": hnr_std}

    # =====================================================================
    # CELL 3 — SLIDING WINDOW OVER ONE CLIP
    # =====================================================================
    def windows_from_clip(path, win_sec, hop_sec, min_win, min_voiced):
        """Slide a window across one clip, return one feature dict per window.

        Windows OVERLAP (hop < win) so nothing falls between two of them.
        The final window is clamped to the end of the clip rather than running
        past it, and is dropped if that leaves it shorter than min_win.
        """
        snd = parselmouth.Sound(str(path))
        total = snd.get_total_duration()
        rows, t, idx = [], 0.0, 0

        while t < total:
            t_end = min(t + win_sec, total)
            if t_end - t >= min_win:
                try:
                    _slice = snd.extract_part(from_time=t, to_time=t_end,
                                              preserve_times=True)
                    _f = feats_from_sound(_slice)
                    # RAVDESS clips have leading/trailing silence. A window of
                    # pure silence has no emotion in it and is just noise in
                    # the training set, so drop it.
                    if _f["voiced_fraction"] >= min_voiced:
                        _f["win_index"] = idx
                        _f["win_start"] = round(t, 3)
                        rows.append(_f)
                        idx += 1
                except parselmouth.PraatError:
                    pass  # too short / unanalysable, skip this window
            if t_end >= total:
                break
            t += hop_sec

        # Fallback: a very short or very quiet clip can yield nothing.
        # Rather than losing the clip entirely, fall back to the whole thing
        # as a single window — exactly what the old pipeline did.
        if not rows:
            try:
                _f = feats_from_sound(snd)
                _f["win_index"] = 0
                _f["win_start"] = 0.0
                rows.append(_f)
            except parselmouth.PraatError:
                pass
        return rows

    # =====================================================================
    # CELL 4 — BUILD THE WINDOWED TRAINING SET
    # =====================================================================
    if os.path.exists(win_features_csv):
        win_df = pd.read_csv(win_features_csv)
        print(f"Loaded cache: {len(win_df)} windows from {win_features_csv}")
    else:
        os.makedirs("outputs", exist_ok=True)
        _paths = sorted(Path(ravdess_root).glob("Actor_*/*.wav"))
        print(f"{len(_paths)} wav files. Windowing (win={WIN_SEC}s hop={HOP_SEC}s)...")
        _rows, _skipped = [], 0
        for _n, _p in enumerate(_paths, 1):
            _parts = _p.stem.split("-")
            _emo = emotion_codes.get(_parts[2], "unknown")
            if _emo in drop_set or _emo == "unknown":
                continue
            _actor = int(_parts[6])
            try:
                _wins = windows_from_clip(_p, WIN_SEC, HOP_SEC, MIN_WIN, MIN_VOICED)
            except Exception:
                _skipped += 1
                continue
            for _w in _wins:
                _w["file"] = _p.name       # clip id — needed to aggregate votes
                _w["emotion"] = _emo       # window inherits the clip's label
                _w["actor"] = _actor       # group key for speaker-independent CV
                _rows.append(_w)
            if _n % 200 == 0:
                print(f"  ...{_n}/{len(_paths)} clips, {len(_rows)} windows so far")
        win_df = pd.DataFrame(_rows)
        win_df.to_csv(win_features_csv, index=False)
        print(f"Done. {len(win_df)} windows from {win_df['file'].nunique()} clips, "
              f"{_skipped} skipped. Saved to {win_features_csv}")

    print(f"\nwindows per clip: mean "
          f"{win_df.groupby('file').size().mean():.1f}, "
          f"min {win_df.groupby('file').size().min()}, "
          f"max {win_df.groupby('file').size().max()}")
    win_df.head()

    # =====================================================================
    # CELL 5 — SPEAKER-INDEPENDENT CV, SCORED AT CLIP LEVEL
    # This is the honest number. Window-level accuracy would be flattering
    # and meaningless — you never classify a window in production, you
    # classify a clip (or in V8, a segment).
    # =====================================================================
    meta_cols = {"file", "emotion", "actor", "win_index", "win_start"}
    win_feature_cols = [c for c in win_df.columns if c not in meta_cols]

    X_win = win_df[win_feature_cols].values
    y_win = win_df["emotion"].values
    g_win = win_df["actor"].values          # group by actor: no speaker in both
                                            # sides, and every window of a clip
                                            # shares an actor, so clips can't
                                            # straddle folds either

    rf_win = RandomForestClassifier(n_estimators=300, random_state=42,
                                    n_jobs=-1, class_weight="balanced")
    cv_win = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # predict_proba, not predict — we need soft votes to average per clip
    proba_win = cross_val_predict(rf_win, X_win, y_win, groups=g_win,
                                  cv=cv_win, method="predict_proba", n_jobs=-1)
    class_order = np.unique(y_win)          # cross_val_predict column order

    # --- window level (reported only for comparison, NOT the headline) ---
    win_level_acc = accuracy_score(y_win, class_order[proba_win.argmax(axis=1)])

    # --- clip level: average each clip's window probabilities, then argmax ---
    _agg = pd.DataFrame(proba_win, columns=class_order)
    _agg["file"] = win_df["file"].values
    _clip_proba = _agg.groupby("file")[list(class_order)].mean()
    _clip_pred = class_order[_clip_proba.values.argmax(axis=1)]
    _clip_true = win_df.groupby("file")["emotion"].first().loc[_clip_proba.index].values

    clip_level_acc = accuracy_score(_clip_true, _clip_pred)

    print(f"chance                    : {1/len(class_order):.3f}")
    print(f"window-level accuracy     : {win_level_acc:.3f}   (context only)")
    print(f"CLIP-level accuracy       : {clip_level_acc:.3f}   <-- the honest number")
    print(f"whole-clip baseline (V7/8): 0.468")
    print()
    print(classification_report(_clip_true, _clip_pred, zero_division=0))

    # =====================================================================
    # CELL 6 — CONFUSION MATRIX at clip level
    # =====================================================================
    cm_win = pd.DataFrame(
        confusion_matrix(_clip_true, _clip_pred, labels=class_order),
        index=[f"true_{c}" for c in class_order],
        columns=[f"pred_{c}" for c in class_order],
    )
    cm_win

    # =====================================================================
    # CELL 7 — FIT THE DEPLOYABLE MODEL on all windows
    # Drop this into the V8 notebook in place of clf_full, and pair it with
    # the aggregation helper below.
    # =====================================================================
    clf_windowed = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ).fit(X_win, y_win)
    print(f"Fitted clf_windowed on {len(win_df)} windows / "
          f"{len(win_feature_cols)} features.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

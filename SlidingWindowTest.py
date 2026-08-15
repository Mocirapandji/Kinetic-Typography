import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    """
    Sliding-window vs whole-clip RandomForest — reproduces the V13 windowing
    experiment (documented as a rejected iteration): does chunking each clip
    into 1.5s / 0.75s-hop windows and training on the windowed sub-clips beat
    training on one whole-clip feature vector?

    Both models use ONE fixed actor->fold map (see fixed_folds), so fold 3 means
    the same five actors in every run. GroupKFold was replaced because it orders
    folds by sample count: clip_df has equal rows per actor and window_df does
    not, so the 4-actor fold landed in a different slot in each run and the
    per-fold numbers were not row-comparable.
    """

    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    from pathlib import Path
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    # ---------------------------------------------------------------------
    # CONFIG
    # ---------------------------------------------------------------------
    RAVDESS_DIR = "/run/media/s5812886/T7 Shield/RAVDESS"   # adjust if needed
    WIN_SEC = 1.5
    HOP_SEC = 0.75
    MIN_WIN_SEC = 0.6        # drop a trailing window shorter than this
    N_FOLDS = 5
    RANDOM_STATE = 42

    EMOTION_MAP = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                   "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
    DROP_EMOTIONS = {"calm"}

    # ---------------------------------------------------------------------
    # SHARED 14-FEATURE EXTRACTOR (identical to extract_clip_features_from_sound
    # in TranscriberV19 CELL 4, so a whole-clip vector and a window vector
    # are directly comparable)
    # ---------------------------------------------------------------------
    def extract_14_features(snd):
        dur = snd.get_total_duration()

        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0v = f0[f0 > 0]
        if len(f0v) > 0:
            f0_mean, f0_std = float(np.mean(f0v)), float(np.std(f0v))
            f0_min, f0_max = float(np.min(f0v)), float(np.max(f0v))
            f0_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = 0.0
        voiced_fraction = float(len(f0v) / len(f0)) if len(f0) > 0 else 0.0

        try:
            intensity = snd.to_intensity()
            ivals = intensity.values.flatten()
            ivals = ivals[np.isfinite(ivals)]
            int_mean, int_std = float(np.mean(ivals)), float(np.std(ivals))
            int_max, int_min = float(np.max(ivals)), float(np.min(ivals))
        except Exception:
            int_mean = int_std = int_max = int_min = 0.0

        try:
            rms = float(call(snd, "Get root-mean-square", 0, 0))
        except Exception:
            rms = 0.0

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

    FEATURE_COLS = ["duration", "voiced_fraction", "f0_mean", "f0_std", "f0_min",
                     "f0_max", "f0_range", "rms", "intensity_mean", "intensity_std",
                     "intensity_max", "intensity_min", "hnr_mean", "hnr_std"]

    # ---------------------------------------------------------------------
    # WINDOW BOUNDARIES (as originally tested)
    # ---------------------------------------------------------------------
    def window_bounds(dur, win_sec=WIN_SEC, hop_sec=HOP_SEC, min_win=MIN_WIN_SEC):
        """-> list of (start, end). A clip shorter than one window still
        gets a single window spanning the whole clip, so nothing is dropped."""
        if dur <= win_sec:
            return [(0.0, dur)]
        bounds = []
        t = 0.0
        while t < dur:
            end = min(t + win_sec, dur)
            if end - t >= min_win:
                bounds.append((t, end))
            t += hop_sec
        return bounds if bounds else [(0.0, dur)]

    # ---------------------------------------------------------------------
    # SCAN RAVDESS
    # ---------------------------------------------------------------------
    def build_ravdess_index(root):
        rows = []
        for p in sorted(Path(root).glob("Actor_*/*.wav")):
            parts = p.stem.split("-")
            if len(parts) != 7 or parts[1] != "01":   # speech only, not song
                continue
            emotion = EMOTION_MAP.get(parts[2], "unknown")
            if emotion in DROP_EMOTIONS or emotion == "unknown":
                continue
            rows.append({"path": str(p), "emotion": emotion, "actor": int(parts[6])})
        return pd.DataFrame(rows)

    index_df = build_ravdess_index(RAVDESS_DIR)
    print(f"RAVDESS: {len(index_df)} clips, {index_df['emotion'].nunique()} classes, "
          f"{index_df['actor'].nunique()} actors")

    # ---------------------------------------------------------------------
    # FIXED ACTOR -> FOLD MAP, shared by every evaluation below
    # ---------------------------------------------------------------------
    # Assigned once, from the actor list only, so it does not depend on how
    # many rows a dataframe happens to have. Folds are still whole actors, so
    # the speaker-independent guarantee is unchanged. Dealing round-robin over
    # sorted actors puts 5 actors in folds 0-3 and 4 in fold 4, every run.
    _ACTORS = np.sort(index_df["actor"].unique())
    FOLD_OF_ACTOR = {int(a): i % N_FOLDS for i, a in enumerate(_ACTORS)}

    print("\nfold assignment (actor -> fold):")
    for f in range(N_FOLDS):
        members = [a for a, k in FOLD_OF_ACTOR.items() if k == f]
        print(f"  fold {f + 1}: {len(members)} actors  {members}")

    def fixed_folds(df, n_folds=N_FOLDS):
        """Yield (train_idx, test_idx) using the shared actor->fold map."""
        fold_id = df["actor"].map(FOLD_OF_ACTOR).to_numpy()
        for f in range(n_folds):
            te = np.where(fold_id == f)[0]
            tr = np.where(fold_id != f)[0]
            yield tr, te

    # ---------------------------------------------------------------------
    # EXTRACT: whole-clip (baseline) AND windowed (test) features, one pass
    # ---------------------------------------------------------------------
    clip_rows, window_rows = [], []
    for n, rec in enumerate(index_df.to_dict("records"), 1):
        try:
            snd = parselmouth.Sound(rec["path"])
        except Exception as e:
            print(f"  skipped {rec['path']}: {e}")
            continue

        feats = extract_14_features(snd)
        feats.update(emotion=rec["emotion"], actor=rec["actor"], path=rec["path"])
        clip_rows.append(feats)

        dur = snd.get_total_duration()
        for wi, (s, e) in enumerate(window_bounds(dur)):
            try:
                seg = snd.extract_part(from_time=s, to_time=e, preserve_times=True)
                wfeats = extract_14_features(seg)
            except Exception:
                continue
            wfeats.update(emotion=rec["emotion"], actor=rec["actor"],
                          path=rec["path"], window_idx=wi)
            window_rows.append(wfeats)

        if n % 200 == 0:
            print(f"  ...{n}/{len(index_df)} clips processed")

    clip_df = pd.DataFrame(clip_rows)
    window_df = pd.DataFrame(window_rows)
    print(f"\nwhole-clip rows : {len(clip_df)}")
    print(f"windowed rows   : {len(window_df)} "
          f"({len(window_df) / max(len(clip_df), 1):.1f} windows/clip average)")

    # ---------------------------------------------------------------------
    # EVAL A — whole-clip baseline, speaker-independent, fixed folds
    # ---------------------------------------------------------------------
    def eval_whole_clip(df, verbose=True, seed=RANDOM_STATE):
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df["emotion"].to_numpy()

        accs, f1s = [], []
        for fold, (tr, te) in enumerate(fixed_folds(df), 1):
            clf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                         n_jobs=-1, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            acc = accuracy_score(y[te], pred)
            f1 = f1_score(y[te], pred, average="macro", zero_division=0)
            accs.append(acc); f1s.append(f1)
            if verbose:
                print(f"  fold {fold}: acc={acc:.3f}  macroF1={f1:.3f}  (n={len(te)})")
        return accs, f1s

    # ---------------------------------------------------------------------
    # EVAL B — windowed training, scored by majority vote per clip
    # (so it's compared on the same clip-level basis as the baseline)
    # ---------------------------------------------------------------------
    def eval_windowed(df, verbose=True, seed=RANDOM_STATE):
        X = df[FEATURE_COLS].to_numpy(dtype=float)
        y = df["emotion"].to_numpy()
        paths = df["path"].to_numpy()
        true_by_path = df.drop_duplicates("path").set_index("path")["emotion"]

        accs, f1s = [], []
        for fold, (tr, te) in enumerate(fixed_folds(df), 1):
            clf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                         n_jobs=-1, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            win_pred = clf.predict(X[te])

            vote_df = pd.DataFrame({"path": paths[te], "pred": win_pred})
            clip_pred = vote_df.groupby("path")["pred"].agg(
                lambda s: s.value_counts().idxmax())
            clip_true = true_by_path.loc[clip_pred.index]

            acc = accuracy_score(clip_true, clip_pred)
            f1 = f1_score(clip_true, clip_pred, average="macro", zero_division=0)
            accs.append(acc); f1s.append(f1)
            if verbose:
                print(f"  fold {fold}: acc={acc:.3f}  macroF1={f1:.3f}  "
                      f"(n={len(clip_pred)} clips)")
        return accs, f1s

    # ---------------------------------------------------------------------
    # RUN BOTH, PRINT THE COMPARISON
    # ---------------------------------------------------------------------
    print("\n=== A: whole-clip baseline ===")
    whole_acc, whole_f1 = eval_whole_clip(clip_df)

    print("\n=== B: windowed training (1.5s / 0.75s hop), majority-vote at test ===")
    win_acc, win_f1 = eval_windowed(window_df)

    a_acc, a_std, a_f1 = np.mean(whole_acc), np.std(whole_acc), np.mean(whole_f1)
    b_acc, b_std, b_f1 = np.mean(win_acc), np.std(win_acc), np.mean(win_f1)

    print("\n" + "=" * 60)
    print(f"whole-clip : acc {a_acc:.3f} ± {a_std:.3f}   macroF1 {a_f1:.3f}")
    print(f"windowed   : acc {b_acc:.3f} ± {b_std:.3f}   macroF1 {b_f1:.3f}")
    print(f"delta      : acc {b_acc - a_acc:+.3f}   macroF1 {b_f1 - a_f1:+.3f}")
    if abs(b_acc - a_acc) < max(a_std, b_std):
        print("-> difference is smaller than one fold's own standard deviation: "
              "not a meaningful improvement.")

    # ---------------------------------------------------------------------
    # PER-FOLD ARRAYS, now row-comparable because both used the same map
    # ---------------------------------------------------------------------
    print("\nper-fold, same actors in each row:")
    print("whole-clip  acc:", [round(a, 3) for a in whole_acc])
    print("windowed    acc:", [round(a, 3) for a in win_acc])
    print("whole-clip  f1 :", [round(f, 3) for f in whole_f1])
    print("windowed    f1 :", [round(f, 3) for f in win_f1])

    # ---------------------------------------------------------------------
    # LATEX TABLE, ready to paste
    # ---------------------------------------------------------------------
    print("\n" + "-" * 60)
    for i in range(N_FOLDS):
        print(f"{i+1} & {whole_acc[i]:.3f} & {win_acc[i]:.3f} & "
              f"{whole_f1[i]:.3f} & {win_f1[i]:.3f} \\\\")
    print(f"\\midrule")
    print(f"\\textbf{{Mean}} & \\textbf{{{a_acc:.3f}}} & \\textbf{{{b_acc:.3f}}} & "
          f"\\textbf{{{a_f1:.3f}}} & \\textbf{{{b_f1:.3f}}} \\\\")
    print(f"Spread & $\\pm${a_std:.3f} & $\\pm${b_std:.3f} & & \\\\")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

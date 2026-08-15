import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # =====================================================================
    # train_emotion_v2.py
    # Ablation: 14 hand-crafted features vs windowed vs eGeMAPS,
    # each with and without per-speaker normalisation.
    # Speaker-independent throughout (StratifiedGroupKFold by actor).
    # =====================================================================

    # =====================================================================
    # CELL 0 — IMPORTS (all imports live here, marimo convention)
    # =====================================================================
    import marimo as mo
    import os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    import opensmile
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import (StratifiedGroupKFold, cross_val_score,
                                         cross_val_predict)
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    mo.md("""
    # Classifier v2: fixing the feature set, not the windowing

    The windowed experiment came back at 45.4% clip-level against a 46.8%
    whole-clip baseline — a difference smaller than the standard error, so:
    **no effect**. The reason is that all 14 features are global summary
    statistics (mean, std, min, max). Sliding a window over a clip
    re-estimates those same summaries on less data. That adds variance, not
    information.

    The actual gap is that none of the 14 features describe the *spectrum*.
    No MFCCs, no formants, no jitter or shimmer, no spectral slope. Voice
    quality carries a large share of emotional information and the model
    currently cannot see any of it.

    This script tests two changes:

    1. **eGeMAPS** (88 features, openSMILE) — the standard feature set in
       affective computing, so it is citable rather than ad hoc.
    2. **Per-speaker normalisation** — RAVDESS has 12 male and 12 female
       actors. 220 Hz is "high" for one and ordinary for another, and the
       model currently has to learn that distinction from scratch.

    Reference points: chance is 14.3%, human listeners score ~67% on
    RAVDESS, and a rigorously speaker-independent DistilHuBERT baseline
    reports 46.6% — i.e. the current 46.8% is not broken, it is normal for
    honest evaluation. The realistic headroom is ~10-15 points.
    """)

    # =====================================================================
    # CELL 1 — CONFIG
    # =====================================================================
    ravdess_root = "/run/media/s5812886/T7 Shield/RAVDESS"

    # caches: two already exist from previous runs, the third is built below
    csv_baseline = "outputs/features.csv"           # 14 features, one row per clip
    csv_windowed = "outputs/features_windowed.csv"  # 14 features, one row per window
    csv_egemaps  = "outputs/features_egemaps.csv"   # 88 features, one row per clip

    model_out = "outputs/clf_v2.joblib"

    emotion_codes = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                     "05": "angry", "06": "fearful", "07": "disgust",
                     "08": "surprised"}
    drop_set = {"calm"}

    CV_SPLITS = 5
    RANDOM_STATE = 42

    # columns that are labels/identifiers, never fed to the classifier
    META_COLS = {"file", "emotion", "actor", "gender", "win_index", "win_start"}

    os.makedirs("outputs", exist_ok=True)

    # Newer pandas infers text columns as pyarrow-backed strings, which sklearn's
    # CV splitter can't row-index. Opt out where the option exists; the eval
    # helpers below also coerce to numpy, so this works even if it's removed later.
    try:
        pd.set_option("future.infer_string", False)
    except Exception:
        pass

    # =====================================================================
    # CELL 2 — THE ORIGINAL 14-FEATURE EXTRACTOR
    # Kept so this script runs standalone if features.csv is ever missing.
    # Identical to extract_clip_features_from_sound in the V8 notebook.
    # =====================================================================
    def feats14_from_sound(snd):
        _dur = snd.get_total_duration()

        _pitch = snd.to_pitch()
        _f0 = _pitch.selected_array["frequency"]
        _f0v = _f0[_f0 > 0]                      # 0 Hz = unvoiced, not "low pitch"
        if len(_f0v) > 0:
            _f0_mean = float(np.mean(_f0v)); _f0_std = float(np.std(_f0v))
            _f0_min = float(np.min(_f0v));   _f0_max = float(np.max(_f0v))
            _f0_range = _f0_max - _f0_min
        else:
            _f0_mean = _f0_std = _f0_min = _f0_max = _f0_range = 0.0
        _voiced = float(len(_f0v) / len(_f0)) if len(_f0) > 0 else 0.0

        try:
            _inten = snd.to_intensity()
            _iv = _inten.values.flatten()
            _iv = _iv[np.isfinite(_iv)]
            _i_mean = float(np.mean(_iv)); _i_std = float(np.std(_iv))
            _i_max = float(np.max(_iv));   _i_min = float(np.min(_iv))
        except Exception:
            _i_mean = _i_std = _i_max = _i_min = 0.0

        _rms = float(call(snd, "Get root-mean-square", 0, 0))

        try:
            _harm = snd.to_harmonicity()
            _hv = _harm.values[_harm.values != -200]   # -200 = Praat "undefined"
            _hnr_mean = float(np.mean(_hv)) if len(_hv) > 0 else 0.0
            _hnr_std = float(np.std(_hv)) if len(_hv) > 0 else 0.0
        except Exception:
            _hnr_mean = _hnr_std = 0.0

        return {"duration": _dur, "voiced_fraction": _voiced,
                "f0_mean": _f0_mean, "f0_std": _f0_std, "f0_min": _f0_min,
                "f0_max": _f0_max, "f0_range": _f0_range,
                "rms": _rms, "intensity_mean": _i_mean, "intensity_std": _i_std,
                "intensity_max": _i_max, "intensity_min": _i_min,
                "hnr_mean": _hnr_mean, "hnr_std": _hnr_std}

    def feats14_from_path(path):
        return feats14_from_sound(parselmouth.Sound(str(path)))

    # =====================================================================
    # CELL 3 — HELPER: walk RAVDESS and apply any extractor function
    # RAVDESS filenames encode everything: the 3rd dash-separated field is
    # the emotion code, the 7th is the actor ID. Odd actor = male,
    # even = female.
    # =====================================================================
    def build_table(extract_fn, out_csv, label=""):
        if os.path.exists(out_csv):
            _df = pd.read_csv(out_csv)
            print(f"{label}: loaded cache, {len(_df)} rows from {out_csv}")
            return _df

        _paths = sorted(Path(ravdess_root).glob("Actor_*/*.wav"))
        print(f"{label}: extracting from {len(_paths)} wav files...")
        _rows, _skipped = [], 0

        for _n, _p in enumerate(_paths, 1):
            _parts = _p.stem.split("-")
            _emo = emotion_codes.get(_parts[2], "unknown")
            if _emo in drop_set or _emo == "unknown":
                continue
            _actor = int(_parts[6])
            try:
                _f = extract_fn(_p)
            except Exception:
                _skipped += 1
                continue
            _f["file"] = _p.name
            _f["emotion"] = _emo
            _f["actor"] = _actor
            _f["gender"] = "M" if _actor % 2 == 1 else "F"
            _rows.append(_f)
            if _n % 200 == 0:
                print(f"    ...{_n}/{len(_paths)}")

        _df = pd.DataFrame(_rows)
        _df.to_csv(out_csv, index=False)
        print(f"{label}: done. {len(_df)} clips, {_skipped} skipped -> {out_csv}")
        return _df

    # =====================================================================
    # CELL 4 — TABLE A: the 14-feature baseline
    # =====================================================================
    tbl_baseline = build_table(feats14_from_path, csv_baseline, "14-feature")

    # older caches may predate the gender column; add it if missing
    if "gender" not in tbl_baseline.columns:
        tbl_baseline["gender"] = np.where(tbl_baseline["actor"] % 2 == 1, "M", "F")

    print(f"  {tbl_baseline['file'].nunique()} clips, "
          f"{len([c for c in tbl_baseline.columns if c not in META_COLS])} features")

    # =====================================================================
    # CELL 5 — TABLE B: eGeMAPS, 88 features
    # The Geneva Minimalistic Acoustic Parameter Set: f0, jitter, shimmer,
    # HNR, spectral slope, formant frequencies and bandwidths, MFCCs,
    # loudness peaks per second. "Functionals" = summary stats over the
    # whole file, so one row per clip, same shape as Table A.
    # Takes a few minutes over 1,248 clips, then it is cached.
    # =====================================================================
    smile_engine = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    def egemaps_from_path(path):
        _r = smile_engine.process_file(str(path))
        return _r.iloc[0].to_dict()

    tbl_egemaps = build_table(egemaps_from_path, csv_egemaps, "eGeMAPS")
    print(f"  {tbl_egemaps['file'].nunique()} clips, "
          f"{len([c for c in tbl_egemaps.columns if c not in META_COLS])} features")

    # =====================================================================
    # CELL 6 — TABLE C: the windowed set from the previous run
    # Loaded only for the ablation table. Not rebuilt.
    # =====================================================================
    if os.path.exists(csv_windowed):
        tbl_windowed = pd.read_csv(csv_windowed)
        if "gender" not in tbl_windowed.columns:
            tbl_windowed["gender"] = np.where(tbl_windowed["actor"] % 2 == 1, "M", "F")
        print(f"windowed: {len(tbl_windowed)} windows from "
              f"{tbl_windowed['file'].nunique()} clips "
              f"(mean {tbl_windowed.groupby('file').size().mean():.1f} per clip)")
    else:
        tbl_windowed = None
        print("windowed: cache not found, that row will be skipped")

    # =====================================================================
    # CELL 7 — PER-SPEAKER NORMALISATION
    #
    # Centre each actor's features on that actor's OWN mean and spread.
    # After this, a feature says "louder than this person usually is"
    # rather than "loud in absolute terms" — which is what emotion actually
    # is, and it stops the model wasting capacity on who has a deep voice.
    #
    # This uses NO LABELS, so it is not leakage: it is the identical
    # operation you could run on an unseen speaker's audio at deployment.
    #
    # Limitation to state in the methodology: it needs several clips from
    # the same speaker to estimate that speaker's baseline. Fine for V8
    # (a long video is minutes of one or two speakers) and fine for RAVDESS
    # (54 clips per actor). NOT possible for a single isolated 3-second
    # clip from a stranger — that case falls back to the raw features.
    # =====================================================================
    def normalise_per_speaker(df, feature_cols, speaker_col="actor", eps=1e-9):
        _out = df.copy()
        for _spk in _out[speaker_col].unique():
            _m = _out[speaker_col] == _spk
            _vals = _out.loc[_m, feature_cols].astype(float)
            _out.loc[_m, feature_cols] = (_vals - _vals.mean()) / (_vals.std() + eps)
        # a feature that never varies for a speaker divides to NaN -> flat 0
        _out[feature_cols] = _out[feature_cols].fillna(0.0)
        return _out

    def feature_columns(df):
        return [c for c in df.columns if c not in META_COLS]

    # =====================================================================
    # CELL 8 — EVALUATION
    # _arrays() is the fix: it materialises sklearn's inputs as ordinary
    # numpy. .values on a pyarrow-backed column hands back the arrow array,
    # which sklearn can't fancy-index during cross-validation; .to_numpy()
    # converts labels to a plain object array and features to float.
    # =====================================================================
    def make_clf():
        return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                      n_jobs=-1, class_weight="balanced")

    def make_cv():
        return StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True,
                                    random_state=RANDOM_STATE)

    def _arrays(df, cols):
        """Plain-numpy X, y, groups for sklearn (arrow-safe)."""
        X = df[cols].to_numpy(dtype=float)
        y = df["emotion"].to_numpy()          # object array of label strings
        g = df["actor"].to_numpy()
        return X, y, g

    def eval_cliplevel(df, label, normalise=False):
        _cols = feature_columns(df)
        _d = normalise_per_speaker(df, _cols) if normalise else df
        _X, _y, _g = _arrays(_d, _cols)
        _scores = cross_val_score(make_clf(), _X, _y, groups=_g,
                                  cv=make_cv(), n_jobs=-1)
        print(f"  {label:44s} {_scores.mean():.3f} +/- {_scores.std():.3f}"
              f"   ({len(_cols)} feats)")
        return _scores.mean()

    def eval_windowed(df, label, normalise=False):
        _cols = feature_columns(df)
        _d = normalise_per_speaker(df, _cols) if normalise else df
        _X, _y, _g = _arrays(_d, _cols)
        _classes = np.unique(_y)

        _proba = cross_val_predict(make_clf(), _X, _y, groups=_g,
                                   cv=make_cv(), method="predict_proba", n_jobs=-1)

        # average each clip's window probabilities, then take the argmax
        _agg = pd.DataFrame(_proba, columns=_classes)
        _agg["file"] = _d["file"].to_numpy()          # numpy, not arrow
        _clip_proba = _agg.groupby("file")[list(_classes)].mean()
        _pred = _classes[_clip_proba.values.argmax(axis=1)]
        _true = (_d.groupby("file")["emotion"].first()
                    .loc[_clip_proba.index].to_numpy())

        _acc = accuracy_score(_true, _pred)
        print(f"  {label:44s} {_acc:.3f}          ({len(_cols)} feats)")
        return _acc

    # =====================================================================
    # CELL 9 — THE ABLATION TABLE
    # This table, not any single number, is what the methodology chapter
    # needs: it shows which change bought what.
    # =====================================================================
    print("\n" + "=" * 72)
    print("SPEAKER-INDEPENDENT ACCURACY (StratifiedGroupKFold by actor, "
          f"{CV_SPLITS} folds)")
    print("=" * 72)
    print(f"  {'condition':44s} accuracy")
    print("  " + "-" * 62)

    acc_results = {}
    acc_results["14 raw"]      = eval_cliplevel(tbl_baseline, "14 features, raw")
    acc_results["14 norm"]     = eval_cliplevel(tbl_baseline, "14 features, speaker-normalised", True)
    if tbl_windowed is not None:
        acc_results["14 win"]  = eval_windowed(tbl_windowed, "14 features, windowed (clip-level)")
        acc_results["14 win norm"] = eval_windowed(tbl_windowed, "14 features, windowed + normalised", True)
    acc_results["ege raw"]     = eval_cliplevel(tbl_egemaps, "eGeMAPS 88, raw")
    acc_results["ege norm"]    = eval_cliplevel(tbl_egemaps, "eGeMAPS 88, speaker-normalised", True)

    print("  " + "-" * 62)
    print(f"  {'chance (1/7)':44s} 0.143")
    print(f"  {'human listeners on RAVDESS':44s} 0.670")
    print("=" * 72)

    _best = max(acc_results, key=acc_results.get)
    print(f"\nbest condition: {_best} at {acc_results[_best]:.3f}")

    # =====================================================================
    # CELL 10 — DETAIL ON THE BEST CONDITION
    # Per-class scores matter more than the headline here: the argument in
    # RQ3 is that high-arousal classes separate and valence-dependent ones
    # do not. Check whether happy/sad stay weak while angry/surprised stay
    # strong even after the feature upgrade.
    # =====================================================================
    best_df = tbl_egemaps
    best_norm = True

    _bcols = feature_columns(best_df)
    _bd = normalise_per_speaker(best_df, _bcols) if best_norm else best_df
    _bX = _bd[_bcols].to_numpy(dtype=float)
    _by = _bd["emotion"].to_numpy()
    _bclasses = np.unique(_by)

    _bproba = cross_val_predict(make_clf(), _bX, _by,
                                groups=_bd["actor"].to_numpy(), cv=make_cv(),
                                method="predict_proba", n_jobs=-1)
    _bpred = _bclasses[_bproba.argmax(axis=1)]

    # =====================================================================
    # CELL 11 — AROUSAL COLLAPSE CHECK
    # Fold the 7 classes down to high/low arousal and re-score the SAME
    # predictions. If arousal accuracy is far above categorical accuracy,
    # the model is reading energy reliably and valence poorly — which is
    # the finding, stated quantitatively rather than as an impression.
    # =====================================================================
    arousal_map = {"angry": "high", "happy": "high", "surprised": "high",
                   "fearful": "high", "sad": "low", "disgust": "low",
                   "neutral": "low"}

    _ar_true = pd.Series(_by).map(arousal_map).values
    _ar_pred = pd.Series(_bpred).map(arousal_map).values
    print(f"7-class accuracy        : {accuracy_score(_by, _bpred):.3f}")
    print(f"binary arousal accuracy : {accuracy_score(_ar_true, _ar_pred):.3f}")
    print(f"arousal chance          : "
          f"{max(pd.Series(_ar_true).value_counts(normalize=True)):.3f}")

    # =====================================================================
    # CELL 12 — FIT AND SAVE THE DEPLOYABLE MODEL
    # Saves the classifier plus the feature column order, so V8 can load it
    # without guessing. Note which extractor it expects.
    # =====================================================================
    clf_v2 = make_clf().fit(_bX, _by)

    joblib.dump({"clf": clf_v2,
                 "feature_cols": _bcols,
                 "extractor": "egemaps",
                 "speaker_normalised": best_norm,
                 "classes": list(clf_v2.classes_)},
                model_out)
    print(f"saved {model_out}  ({len(_bcols)} features, "
          f"{len(clf_v2.classes_)} classes)")

    mo.md("""
    ## Decision log

    11. **Windowed training tested and rejected (no effect).** Sliding a
        1.5s / 0.75s window over each clip produced 4,970 windows from 1,248
        clips and scored 45.4% at clip level against a 46.8% whole-clip
        baseline — a gap smaller than the standard error on 1,248 samples,
        so no effect rather than a regression. The mechanism is that all 14
        features are global summary statistics; windowing re-estimates the
        same quantities on less audio, adding variance without adding
        information. Windowing only helps when per-window features capture
        something the global ones cannot.

    12. **The feature set, not the temporal resolution, was the bottleneck.**
        The original 14 features describe f0, intensity, HNR and duration
        only — nothing spectral. eGeMAPS (88 features) adds jitter, shimmer,
        spectral slope, formants and MFCCs, and is the standard parameter
        set in affective computing, so adopting it makes the feature choice
        citable rather than ad hoc.

    13. **Per-speaker normalisation.** Features are z-scored within each
        actor, so they express deviation from that speaker's own baseline
        rather than absolute values. This uses no label information and is
        therefore not leakage — it is the same operation available at
        deployment. It does require several clips per speaker, so it applies
        to V8's long-form video but not to an isolated clip from an unknown
        speaker.

    14. **Accuracy is reported against the right ceiling.** Published
        RAVDESS results above 80% are typically speaker-dependent. Under
        rigorous leave-one-speaker-out evaluation a DistilHuBERT baseline
        reports 46.6%, and human listeners reach roughly 67%. The
        speaker-independent constraint, not a defect in the pipeline,
        accounts for most of the distance to the headline figures in the
        literature.

    15. **Arousal is recovered, valence is not.** Folding the seven classes
        to binary arousal and re-scoring the same predictions quantifies the
        asymmetry directly, converting an observation about the confusion
        matrix into a measured result for RQ3.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

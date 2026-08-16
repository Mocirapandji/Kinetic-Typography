import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # train_emotionsV3 — IEMOCAP primary, RAVDESS supporting

    Trains the emotion classifier the transcriber loads. IEMOCAP sets the label
    space and RAVDESS only fills classes IEMOCAP already has, entering at a
    reduced sample weight. Evaluation is IEMOCAP-only, leave-one-session-out,
    so RAVDESS never sits in a test fold.

    Output is `outputs/clf_v3.joblib`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cell 0 — Imports
    """)
    return


@app.cell
def _():

    import os
    import re
    import time
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    try:
        import opensmile
        HAVE_OPENSMILE = True
    except Exception:
        opensmile = None
        HAVE_OPENSMILE = False

    print(f"opensmile available: {HAVE_OPENSMILE}")
    return (
        HAVE_OPENSMILE,
        Path,
        RandomForestClassifier,
        accuracy_score,
        call,
        confusion_matrix,
        f1_score,
        joblib,
        np,
        opensmile,
        os,
        parselmouth,
        pd,
        re,
        time,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Cell 1 — Configuration

    Every setting the notebook depends on lives here, so nothing important is
    buried further down. It sets where the two datasets sit on disk, which
    feature extractor to use, how features get normalised, the weight RAVDESS
    enters training at, and the Random Forest settings. It also builds the cache
    filenames per extractor, so switching extractor cannot silently reuse the
    wrong cached features. Change a value here and everything below re-runs
    against it.
    """)
    return


@app.cell
def _(os):
    ravdess_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    iemocap_dir = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"

    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/audio", exist_ok=True)

    # "egemaps" = 88-dim eGeMAPSv02 functionals. "praat14" = the older
    # 14-feature Praat set. Must match what the transcriber extracts.
    EXTRACTOR = "egemaps"

    # Grouping used to z-score features. "dialog" is closest to inference,
    # where the transcriber z-scores across all segments of one clip.
    NORM_SCOPE = "dialog"
    NORM_MIN_ROWS = 4          # groups smaller than this are left raw
    NORM_STD_FLOOR = 1e-6      # guards divide-by-zero on a constant feature

    # Weight on every RAVDESS row at fit time. 0.0 excludes RAVDESS entirely.
    # Chosen from the sweep in Cell 10 / Cell 11.
    RAVDESS_WEIGHT = 0.20
    RAVDESS_WEIGHT_SWEEP = [
        0.0, 0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.20, 0.22, 0.25, 0.28, 0.30,
    ]

    # IEMOCAP's excited has no RAVDESS counterpart. Merging it into happy
    # gives a bigger class; keeping it separate preserves a distinction the
    # pipeline can render.
    MERGE_EXCITED_INTO_HAPPY = False

    # Drop IEMOCAP classes below this count.
    MIN_CLASS_COUNT = 40

    # Extraction is the slow step, so it is cached per extractor.
    iemocap_feat_csv = f"{OUT_DIR}/features_iemocap_{EXTRACTOR}.csv"
    ravdess_feat_csv = f"{OUT_DIR}/features_ravdess_{EXTRACTOR}.csv"
    combined_csv = f"{OUT_DIR}/features_combined_{EXTRACTOR}.csv"

    # Fallback source for IEMOCAP labels only; its feature columns are ignored.
    iemocap_label_csv = f"{OUT_DIR}/features_iemocap.csv"
    if not os.path.exists(iemocap_label_csv):
        iemocap_label_csv = ("/run/media/s5812886/T7 Shield/kinetic_outputs/"
                             "features_iemocap.csv")

    MODEL_OUT = f"{OUT_DIR}/clf_v3.joblib"

    RF_KWARGS = dict(n_estimators=600, random_state=42, n_jobs=-1,
                     class_weight="balanced_subsample", min_samples_leaf=2)

    print(f"extractor={EXTRACTOR}  norm_scope={NORM_SCOPE}  "
          f"ravdess_weight={RAVDESS_WEIGHT}")
    print(f"iemocap cache -> {iemocap_feat_csv}")
    print(f"ravdess cache -> {ravdess_feat_csv}")
    print(f"model out     -> {MODEL_OUT}")
    return (
        EXTRACTOR,
        MERGE_EXCITED_INTO_HAPPY,
        MIN_CLASS_COUNT,
        MODEL_OUT,
        NORM_MIN_ROWS,
        NORM_SCOPE,
        NORM_STD_FLOOR,
        RAVDESS_WEIGHT,
        RAVDESS_WEIGHT_SWEEP,
        RF_KWARGS,
        combined_csv,
        iemocap_dir,
        iemocap_feat_csv,
        iemocap_label_csv,
        ravdess_dir,
        ravdess_feat_csv,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Cell 2 — Label harmonisation

    Writes out both datasets' emotion codes and what each one becomes, rather
    than inferring it at runtime. IEMOCAP's vocabulary is the reference, so
    RAVDESS labels are translated into it. RAVDESS's "calm" maps to nothing
    because IEMOCAP has no equivalent, so those clips are dropped rather than
    folded into neutral, which would change what neutral means. The cell prints
    the resulting label space and flags which classes RAVDESS can support.
    """)
    return


@app.cell
def _():
    IEMOCAP_CODE_TO_LABEL = {
        "neu": "neutral",
        "hap": "happy",
        "sad": "sad",
        "ang": "angry",
        "fru": "frustrated",
        "exc": "excited",
        "fea": "fearful",
        "sur": "surprised",
        "dis": "disgust",
        # "oth" and "xxx" (no annotator agreement) are left out.
    }

    # RAVDESS filename digit -> canonical name. None means drop the clip.
    RAVDESS_DIGIT_TO_LABEL = {
        "01": "neutral",
        "02": None,        # calm. IEMOCAP has no such category.
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised",
    }

    RAVDESS_ONLY_DROPPED = {"calm"}

    print("canonical label space (from IEMOCAP):")
    for _c, _l in sorted(IEMOCAP_CODE_TO_LABEL.items(), key=lambda kv: kv[1]):
        _sup = "RAVDESS can support" if _l in set(
            v for v in RAVDESS_DIGIT_TO_LABEL.values() if v
        ) else "IEMOCAP only"
        print(f"  {_c} -> {_l:12s}  ({_sup})")
    print(f"RAVDESS classes dropped, no IEMOCAP counterpart: "
          f"{sorted(RAVDESS_ONLY_DROPPED)}")
    return IEMOCAP_CODE_TO_LABEL, RAVDESS_DIGIT_TO_LABEL


@app.cell
def _(mo):
    mo.md("""
    ## Cell 3 — Feature extractors

    Turns one audio file into one row of numbers. In eGeMAPS mode openSMILE
    produces the 88 standard values the shipped model uses; the praat14 path is
    the older hand-picked set, kept so models trained earlier stay comparable.
    `extract_features` is the single entry point, so nothing below needs to know
    which mode is on. It is built with the same call the transcriber uses at
    inference, so the feature columns line up at prediction time.
    """)
    return


@app.cell
def _(EXTRACTOR, HAVE_OPENSMILE, call, np, opensmile, parselmouth):
    def extract_clip_features_from_sound(snd):
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

    # Same eGeMAPSv02 / Functionals construction the transcriber uses, so the
    # column names match.
    if EXTRACTOR == "egemaps":
        if not HAVE_OPENSMILE:
            raise RuntimeError(
                "EXTRACTOR='egemaps' but opensmile didn't import.\n"
                "  pip install opensmile\n"
                "or set EXTRACTOR='praat14' in CELL 1 to train the 14-feature "
                "model instead."
            )
        smile_v3 = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile_v3 = None

    def extract_features(path):
        """One utterance in, one flat dict of features out."""
        if EXTRACTOR == "egemaps":
            return smile_v3.process_file(str(path)).iloc[0].to_dict()
        return extract_clip_features_from_sound(parselmouth.Sound(str(path)))

    print(f"feature extractor ready: {EXTRACTOR}")
    return (extract_features,)


@app.cell
def _(mo):
    mo.md("""
    ## Cell 4 — Dataset indexing

    Builds the list of clips to work from before any audio is read. For IEMOCAP
    it reads the EmoEvaluation text files and produces one row per utterance
    with its label, speaker, dialogue and session, falling back to an existing
    CSV for labels if the drive is not mounted. For RAVDESS it parses the
    filename instead, keeping only speech and dropping the song half and calm.
    Rows whose wav is missing on disk are dropped and counted, and both label
    distributions are printed.
    """)
    return


@app.cell
def _(
    IEMOCAP_CODE_TO_LABEL,
    Path,
    RAVDESS_DIGIT_TO_LABEL,
    iemocap_dir,
    iemocap_label_csv,
    os,
    pd,
    ravdess_dir,
    re,
):
    EMO_LINE_RE = re.compile(
        r"^\[(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\]\s+"
        r"(?P<utt>\S+)\s+(?P<code>\S+)\s+\["
    )

    def speaker_of(utt_id):
        """Speaker id = session number + who is talking in this turn.

        The trailing field of an utterance id ('..._F012' / '..._M042') gives
        the turn's speaker. The 'F' in the session prefix ('Ses01F') only says
        whose script the dialog followed, so it isn't the speaker.
        """
        sess = utt_id[3:5]
        tail = utt_id.rsplit("_", 1)[-1]
        gender = tail[0].upper() if tail and tail[0].upper() in ("F", "M") else "?"
        return f"Ses{sess}_{gender}"

    def dialog_of(utt_id):
        return utt_id.rsplit("_", 1)[0]

    def wav_path_of(utt_id, root):
        sess = f"Session{int(utt_id[3:5])}"
        return f"{root}/{sess}/sentences/wav/{dialog_of(utt_id)}/{utt_id}.wav"

    def scan_emoevaluation(root):
        rows, skipped_codes = [], {}
        for sess_n in range(1, 6):
            eval_dir = Path(root) / f"Session{sess_n}" / "dialog" / "EmoEvaluation"
            if not eval_dir.is_dir():
                continue
            for txt in sorted(eval_dir.glob("*.txt")):
                try:
                    text = txt.read_text(errors="ignore")
                except Exception:
                    continue
                for line in text.splitlines():
                    m = EMO_LINE_RE.match(line.strip())
                    if not m:
                        continue
                    code = m.group("code").lower()
                    label = IEMOCAP_CODE_TO_LABEL.get(code)
                    if label is None:
                        skipped_codes[code] = skipped_codes.get(code, 0) + 1
                        continue
                    utt = m.group("utt")
                    rows.append({
                        "utt_id": utt,
                        "path": wav_path_of(utt, root),
                        "label": label,
                        "raw_code": code,
                        "speaker": speaker_of(utt),
                        "dialog": dialog_of(utt),
                        "session": int(utt[3:5]),
                        "source": "iemocap",
                    })
        return pd.DataFrame(rows), skipped_codes

    iem_index, _skipped = scan_emoevaluation(iemocap_dir)

    if len(iem_index) == 0 and os.path.exists(iemocap_label_csv):
        print(f"No EmoEvaluation files under {iemocap_dir}.\n"
              f"  Falling back to labels from {iemocap_label_csv}.")
        _csv = pd.read_csv(iemocap_label_csv)
        _utt = (_csv["file"].astype(str)
                .str.replace(".wav", "", regex=False))
        iem_index = pd.DataFrame({
            "utt_id": _utt,
            "path": [wav_path_of(u, iemocap_dir) for u in _utt],
            "label": _csv["emotion"].astype(str).str.lower(),
            "raw_code": _csv["emotion"].astype(str).str.lower(),
            "speaker": [speaker_of(u) for u in _utt],
            "dialog": [dialog_of(u) for u in _utt],
            "session": [int(u[3:5]) for u in _utt],
            "source": "iemocap",
        })
        # the CSV may hold full names or codes, so normalise either way
        iem_index["label"] = iem_index["label"].map(
            lambda v: IEMOCAP_CODE_TO_LABEL.get(v, v))
        iem_index = iem_index[
            iem_index["label"].isin(set(IEMOCAP_CODE_TO_LABEL.values()))]

    if len(iem_index) == 0:
        raise RuntimeError(
            f"Found no IEMOCAP labels.\n"
            f"  Checked EmoEvaluation under: {iemocap_dir}\n"
            f"  Checked label CSV at:       {iemocap_label_csv}\n"
            f"Mount the T7 drive, or point iemocap_dir at the release."
        )

    _exists = iem_index["path"].map(os.path.exists)
    print(f"IEMOCAP: {len(iem_index)} labelled utterances, "
          f"{int(_exists.sum())} with a wav on disk")
    if _skipped:
        print(f"  excluded codes: "
              f"{ {k: v for k, v in sorted(_skipped.items())} }")
    iem_index = iem_index[_exists].reset_index(drop=True)
    print(f"  speakers: {sorted(iem_index['speaker'].unique())}")
    print(iem_index["label"].value_counts().to_string())

    # Filename: modality-vocal-emotion-intensity-statement-repetition-actor
    # e.g. 03-01-06-01-02-01-12.wav -> emotion 06 (fearful), actor 12.
    def build_ravdess_index(root):
        rows, dropped = [], {}
        wavs = sorted(Path(root).glob("Actor_*/*.wav"))
        for p in wavs:
            parts = p.stem.split("-")
            if len(parts) != 7:
                continue
            if parts[1] != "01":          # 01=speech, 02=song
                dropped["song"] = dropped.get("song", 0) + 1
                continue
            label = RAVDESS_DIGIT_TO_LABEL.get(parts[2], "unknown")
            if label is None:
                dropped["calm (no IEMOCAP counterpart)"] = dropped.get(
                    "calm (no IEMOCAP counterpart)", 0) + 1
                continue
            if label == "unknown":
                dropped["unknown code"] = dropped.get("unknown code", 0) + 1
                continue
            actor = int(parts[6])
            rows.append({
                "utt_id": p.stem,
                "path": str(p),
                "label": label,
                "raw_code": parts[2],
                # RAVDESS has no dialogs, so the actor doubles as the
                # normalisation group.
                "speaker": f"RAV{actor:02d}",
                "dialog": f"RAV{actor:02d}",
                "session": 0,
                "intensity": parts[3],
                "statement": parts[4],
                "source": "ravdess",
            })
        return pd.DataFrame(rows), dropped, len(wavs)

    if os.path.isdir(ravdess_dir):
        rav_index, _rav_dropped, _n_wavs = build_ravdess_index(ravdess_dir)
        print(f"RAVDESS: scanned {_n_wavs} wavs, kept {len(rav_index)}")
        for _k, _v in sorted(_rav_dropped.items()):
            print(f"  dropped {_v:5d}  {_k}")
        if len(rav_index):
            print(rav_index["label"].value_counts().to_string())
    else:
        rav_index = pd.DataFrame(
            columns=["utt_id", "path", "label", "raw_code", "speaker",
                     "dialog", "session", "source"])
        print(f"RAVDESS not found at {ravdess_dir}, continuing IEMOCAP-only. "
              f"Everything below still runs, the support set is just empty.")
    return iem_index, rav_index


@app.cell
def _(mo):
    mo.md("""
    ## Cell 5 — Feature extraction with cache

    This is the slow step, roughly ten thousand IEMOCAP utterances plus around
    fourteen hundred RAVDESS clips, so results are written to CSV and reused on
    every later run. The cell walks each index, extracts features per clip,
    carries the label columns along beside them, and prints progress as it goes.
    A clip whose extraction throws is dropped and counted rather than written as
    zeros, because a row of zeros looks like a plausible feature vector with no
    signal in it.
    """)
    return


@app.cell
def _(
    extract_features,
    iem_index,
    iemocap_feat_csv,
    os,
    pd,
    rav_index,
    ravdess_feat_csv,
    time,
):
    def extract_for_index(index_df, cache_csv, tag):
        if os.path.exists(cache_csv):
            cached = pd.read_csv(cache_csv)
            print(f"[{tag}] loaded cache: {len(cached)} rows from {cache_csv}")
            return cached
        if len(index_df) == 0:
            print(f"[{tag}] nothing to extract")
            return index_df.copy()

        print(f"[{tag}] extracting features for {len(index_df)} clips...")
        t0 = time.time()
        rows, failed = [], 0
        meta_cols = [c for c in index_df.columns if c != "path"]
        for n, rec in enumerate(index_df.to_dict("records"), 1):
            try:
                feats = extract_features(rec["path"])
            except Exception:
                failed += 1
                continue
            for c in meta_cols:
                feats[c] = rec[c]
            rows.append(feats)
            if n % 250 == 0:
                _el = time.time() - t0
                print(f"    {n}/{len(index_df)}  ({_el:.0f}s elapsed, "
                      f"{failed} failed)")
        out = pd.DataFrame(rows)
        out.to_csv(cache_csv, index=False)
        print(f"[{tag}] done in {time.time() - t0:.0f}s: kept {len(out)}, "
              f"failed {failed}. Cached -> {cache_csv}")
        return out

    iem_feats = extract_for_index(iem_index, iemocap_feat_csv, "iemocap")
    rav_feats = extract_for_index(rav_index, ravdess_feat_csv, "ravdess")
    print(f"\niemocap rows: {len(iem_feats)}   ravdess rows: {len(rav_feats)}")
    return iem_feats, rav_feats


@app.cell
def _(mo):
    mo.md("""
    ## Cell 6 — Combine and filter

    Joins the two feature tables into the one table everything downstream uses.
    Which classes survive is decided from IEMOCAP counts alone, so RAVDESS
    cannot keep alive a class that only really exists in acted speech. It also
    drops any feature column one corpus never fills, since the model could
    otherwise learn to tell which dataset a clip came from instead of learning
    emotion. The combined table is written to disk and the rows per class per
    source are printed.
    """)
    return


@app.cell
def _(
    MERGE_EXCITED_INTO_HAPPY,
    MIN_CLASS_COUNT,
    combined_csv,
    iem_feats,
    np,
    pd,
    rav_feats,
):
    # Order matters: merge first so the merged class is counted at its real
    # size, then pick surviving labels from IEMOCAP counts, then concatenate.
    META_COLS = ["utt_id", "path", "label", "raw_code", "speaker", "dialog",
                 "session", "source", "intensity", "statement"]

    def _merge_exc(df):
        if not MERGE_EXCITED_INTO_HAPPY or len(df) == 0:
            return df
        out = df.copy()
        out["label"] = out["label"].replace({"excited": "happy"})
        return out

    _iem = _merge_exc(iem_feats)
    _rav = _merge_exc(rav_feats)

    _iem_counts = _iem["label"].value_counts()
    keep_labels = sorted(_iem_counts[_iem_counts >= MIN_CLASS_COUNT].index)
    _too_rare = sorted(_iem_counts[_iem_counts < MIN_CLASS_COUNT].index)

    print(f"IEMOCAP class counts:\n{_iem_counts.to_string()}")
    if _too_rare:
        print(f"\ndropped for < {MIN_CLASS_COUNT} IEMOCAP utterances: "
              f"{_too_rare}")
        print("  (RAVDESS doesn't rescue these on purpose. A class held up only "
              "by acted clips won't survive real dialogue.)")
    print(f"\nfinal label space ({len(keep_labels)} classes): {keep_labels}")

    _iem = _iem[_iem["label"].isin(keep_labels)]
    _rav = _rav[_rav["label"].isin(keep_labels)] if len(_rav) else _rav

    combined = pd.concat([_iem, _rav], ignore_index=True, sort=False)

    # feature columns = anything not metadata, and numeric
    feature_cols = [c for c in combined.columns if c not in META_COLS]
    feature_cols = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(combined[c])]
    # a feature has to exist on both sides, otherwise the model can learn
    # "which corpus is this"
    if len(_rav):
        _iem_ok = set(_iem.columns)
        _rav_ok = set(_rav.columns)
        _only_one = [c for c in feature_cols
                     if c not in _iem_ok or c not in _rav_ok]
        if _only_one:
            print(f"\ndropping {len(_only_one)} feature(s) present in only one "
                  f"corpus: {_only_one[:8]}{'...' if len(_only_one) > 8 else ''}")
            feature_cols = [c for c in feature_cols if c not in _only_one]

    combined[feature_cols] = (combined[feature_cols]
                              .apply(pd.to_numeric, errors="coerce")
                              .replace([np.inf, -np.inf], np.nan))
    _bad = combined[feature_cols].isna().any(axis=1)
    if int(_bad.sum()):
        print(f"dropping {int(_bad.sum())} row(s) with non-finite features")
        combined = combined[~_bad].reset_index(drop=True)

    combined.to_csv(combined_csv, index=False)

    print(f"\ncombined: {len(combined)} rows x {len(feature_cols)} features "
          f"-> {combined_csv}")
    print("\nrows per class per source:")
    print(pd.crosstab(combined["label"], combined["source"]).to_string())
    return combined, feature_cols, keep_labels


@app.cell
def _(mo):
    mo.md("""
    ## Cell 7 — Normalisation

    Rescales every feature against its own group's baseline, so a speaker who is
    simply loud stops looking angry. Grouping is per dialogue by default with
    both speakers pooled, because that matches what the transcriber does at
    inference when it z-scores across all the segments of one clip. Groups too
    small for reliable statistics are left raw rather than normalised on noise.
    Only features are touched and no labels are involved, so running this before
    the split leaks nothing.
    """)
    return


@app.cell
def _(NORM_MIN_ROWS, NORM_SCOPE, NORM_STD_FLOOR, combined, feature_cols, pd):
    # RAVDESS has no dialogs, so its group key falls back to the actor under
    # every scope.
    def normalise(df, cols, scope, min_rows=4, std_floor=1e-6):
        if scope == "off":
            return df.copy(), False, {}
        out = df.copy()
        if scope == "speaker":
            key = out["speaker"]
        elif scope == "dialog":
            key = out["dialog"]
        elif scope == "dialog_speaker":
            key = out["dialog"].astype(str) + "|" + out["speaker"].astype(str)
        else:
            raise ValueError(f"unknown NORM_SCOPE: {scope!r}")
        out["_norm_key"] = key

        sizes = out["_norm_key"].value_counts()
        small = set(sizes[sizes < min_rows].index)
        stats = {"groups": int(sizes.size),
                 "groups_left_raw": len(small),
                 "rows_left_raw": int(sizes[sizes < min_rows].sum())}

        big = out[~out["_norm_key"].isin(small)]
        if len(big):
            g = big.groupby("_norm_key")[cols]
            mu = g.transform("mean")
            sd = g.transform("std").fillna(0.0)
            sd = sd.where(sd.abs() > std_floor, 1.0)
            out.loc[big.index, cols] = ((big[cols] - mu) / sd).fillna(0.0)
        out = out.drop(columns=["_norm_key"])
        return out, True, stats

    norm_df, SPEAKER_NORMALISED, _nstats = normalise(
        combined, feature_cols, NORM_SCOPE,
        min_rows=NORM_MIN_ROWS, std_floor=NORM_STD_FLOOR)

    print(f"normalisation scope={NORM_SCOPE}  applied={SPEAKER_NORMALISED}")
    if _nstats:
        print(f"  {_nstats['groups']} groups; "
              f"{_nstats['groups_left_raw']} too small to normalise "
              f"({_nstats['rows_left_raw']} rows left raw)")
    print("\ncheck — mean/std of the first 3 features after normalisation:")
    print(pd.DataFrame({
        "mean": norm_df[feature_cols[:3]].mean().round(3),
        "std": norm_df[feature_cols[:3]].std().round(3),
    }).to_string())
    return SPEAKER_NORMALISED, norm_df


@app.cell
def _(mo):
    mo.md("""
    ## Cell 8 — Cross-validation protocol

    Defines the evaluation every experiment below runs through. IEMOCAP's five
    sessions are the folds: four train and one tests, and whole sessions leave
    together so a conversation partner never ends up in both halves. RAVDESS
    goes into every training fold at its weight and never into a test fold,
    because the question being asked is whether the model works on conversation,
    not on acted speech. Macro-F1 is reported alongside accuracy, since accuracy
    alone rewards coasting on the biggest classes.
    """)
    return


@app.cell
def _(RandomForestClassifier, accuracy_score, f1_score, np, pd):
    def run_cv(df, cols, ravdess_weight, rf_kwargs, verbose=True):
        iem = df[df["source"] == "iemocap"]
        rav = df[df["source"] == "ravdess"]
        sessions = sorted(iem["session"].unique())

        per_fold, preds, truths = [], [], []
        for s in sessions:
            te = iem[iem["session"] == s]
            tr_iem = iem[iem["session"] != s]
            if ravdess_weight > 0 and len(rav):
                tr = pd.concat([tr_iem, rav], ignore_index=True, sort=False)
            else:
                tr = tr_iem
            if te.empty or tr.empty:
                continue

            w = np.where(tr["source"].to_numpy() == "ravdess",
                         float(ravdess_weight), 1.0)
            clf = RandomForestClassifier(**rf_kwargs)
            clf.fit(tr[cols].to_numpy(dtype=float),
                    tr["label"].to_numpy(), sample_weight=w)

            yp = clf.predict(te[cols].to_numpy(dtype=float))
            yt = te["label"].to_numpy()
            preds.extend(list(yp)); truths.extend(list(yt))
            per_fold.append({
                "held_out_session": s,
                "n_test": len(te),
                "n_train_iemocap": int((tr["source"] == "iemocap").sum()),
                "n_train_ravdess": int((tr["source"] == "ravdess").sum()),
                "accuracy": accuracy_score(yt, yp),
                "macro_f1": f1_score(yt, yp, average="macro",
                                     zero_division=0),
            })
            if verbose:
                print(f"    session {s}: acc={per_fold[-1]['accuracy']:.3f}  "
                      f"macroF1={per_fold[-1]['macro_f1']:.3f}  "
                      f"(n={len(te)})")

        fold_df = pd.DataFrame(per_fold)
        pooled = {
            "accuracy": accuracy_score(truths, preds),
            "macro_f1": f1_score(truths, preds, average="macro",
                                 zero_division=0),
            "n": len(truths),
        }
        return fold_df, pooled, np.array(truths), np.array(preds)

    return (run_cv,)


@app.cell
def _(mo):
    mo.md("""
    ## Cell 9 — RAVDESS-only baseline (testing)

    This cell is for testing and reporting only. Nothing it produces feeds the
    saved model. It trains on acted speech alone and scores it on the same five
    conversational folds used everywhere else, which answers how far an
    acted-speech-only model gets on real dialogue. Because RAVDESS never had
    frustrated or excited, two pairs of numbers are printed per fold: one over
    the full label space where those classes count as always wrong, and one
    restricted to the classes RAVDESS actually has a concept of.
    """)
    return


@app.cell
def _(
    RF_KWARGS,
    RandomForestClassifier,
    accuracy_score,
    f1_score,
    feature_cols,
    norm_df,
    np,
    pd,
):
    def run_ravdess_only_baseline(df, cols, rf_kwargs, verbose=True):
        iem = df[df["source"] == "iemocap"]
        rav = df[df["source"] == "ravdess"]
        if len(rav) == 0:
            raise RuntimeError(
                "No RAVDESS rows in this dataframe — nothing to train the "
                "baseline on. Check ravdess_dir in CELL 1."
            )

        rav_classes = sorted(rav["label"].unique())
        sessions = sorted(iem["session"].unique())
        print(f"RAVDESS-only model trained on {len(rav)} rows, "
              f"{len(rav_classes)} classes: {rav_classes}\n")

        per_fold, preds, truths = [], [], []
        for s in sessions:
            te = iem[iem["session"] == s]
            if te.empty:
                continue

            # Training set is identical every fold; IEMOCAP only contributes
            # to the test side.
            clf = RandomForestClassifier(**rf_kwargs)
            clf.fit(rav[cols].to_numpy(dtype=float), rav["label"].to_numpy())

            yp = clf.predict(te[cols].to_numpy(dtype=float))
            yt = te["label"].to_numpy()
            preds.extend(list(yp))
            truths.extend(list(yt))

            acc_full = accuracy_score(yt, yp)
            f1_full = f1_score(yt, yp, average="macro", zero_division=0)

            in_vocab = np.isin(yt, rav_classes)
            if in_vocab.sum():
                acc_ov = accuracy_score(yt[in_vocab], yp[in_vocab])
                f1_ov = f1_score(yt[in_vocab], yp[in_vocab],
                                  average="macro", zero_division=0)
            else:
                acc_ov = f1_ov = float("nan")

            per_fold.append({
                "held_out_session": s,
                "n_test": len(te),
                "n_test_in_vocab": int(in_vocab.sum()),
                "acc_full_label_space": acc_full,
                "macro_f1_full_label_space": f1_full,
                "acc_overlap_classes_only": acc_ov,
                "macro_f1_overlap_classes_only": f1_ov,
            })
            if verbose:
                print(f"    session {s}: "
                      f"full acc={acc_full:.3f} macroF1={f1_full:.3f}  |  "
                      f"overlap-only acc={acc_ov:.3f} macroF1={f1_ov:.3f}  "
                      f"(n={len(te)}, in_vocab={int(in_vocab.sum())})")

        fold_df = pd.DataFrame(per_fold)
        truths_a, preds_a = np.array(truths), np.array(preds)

        pooled_full = {
            "accuracy": accuracy_score(truths_a, preds_a),
            "macro_f1": f1_score(truths_a, preds_a, average="macro",
                                  zero_division=0),
            "n": len(truths_a),
        }
        ov_mask = np.isin(truths_a, rav_classes)
        pooled_overlap = {
            "accuracy": accuracy_score(truths_a[ov_mask], preds_a[ov_mask]),
            "macro_f1": f1_score(truths_a[ov_mask], preds_a[ov_mask],
                                  average="macro", zero_division=0),
            "n": int(ov_mask.sum()),
        }

        print(f"\n{'=' * 66}")
        print("RAVDESS-only model, scored on IEMOCAP (same folds as CELL 9/10)")
        print(f"  full label space      : acc={pooled_full['accuracy']:.4f}  "
              f"macroF1={pooled_full['macro_f1']:.4f}  (n={pooled_full['n']})")
        print(f"  overlap classes only ({len(rav_classes)}) : "
              f"acc={pooled_overlap['accuracy']:.4f}  "
              f"macroF1={pooled_overlap['macro_f1']:.4f}  "
              f"(n={pooled_overlap['n']})")
        print(f"  never-reachable classes: "
              f"{sorted(set(iem['label'].unique()) - set(rav_classes))}")
        return fold_df, pooled_full, pooled_overlap, truths_a, preds_a

    rav_only_folds, rav_only_full, rav_only_overlap, _ro_yt, _ro_yp = (
        run_ravdess_only_baseline(norm_df, feature_cols, RF_KWARGS)
    )
    print("\nper-fold detail:")
    print(rav_only_folds.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cell 10 — RAVDESS weight sweep (testing)

    An experiment. It runs the whole
    cross-validation once per candidate RAVDESS weight, with weight 0.0 acting
    as the IEMOCAP-only baseline so the comparison is like for like. For each
    weight it prints pooled accuracy, pooled macro-F1 and the spread across
    folds. The best pooled score is reported but should not be trusted on its
    own, which is what the fold spread column and the next cell are for.
    """)
    return


@app.cell
def _(RAVDESS_WEIGHT_SWEEP, RF_KWARGS, feature_cols, norm_df, pd, run_cv):
    sweep_rows, sweep_detail = [], {}
    for _w in RAVDESS_WEIGHT_SWEEP:
        _tag = "IEMOCAP only" if _w == 0.0 else f"+RAVDESS @ w={_w}"
        print(f"\n=== {_tag} ===")
        _folds, _pooled, _yt, _yp = run_cv(norm_df, feature_cols, _w, RF_KWARGS)
        sweep_detail[_w] = (_folds, _yt, _yp)
        sweep_rows.append({
            "ravdess_weight": _w,
            "label": _tag,
            "pooled_accuracy": round(_pooled["accuracy"], 4),
            "pooled_macro_f1": round(_pooled["macro_f1"], 4),
            "fold_acc_mean": round(_folds["accuracy"].mean(), 4),
            "fold_acc_std": round(_folds["accuracy"].std(), 4),
            "n_test": _pooled["n"],
        })

    sweep_df = pd.DataFrame(sweep_rows)
    _n_classes = norm_df["label"].nunique()
    _base = sweep_df.loc[sweep_df["ravdess_weight"] == 0.0]
    print(f"\n{'=' * 66}")
    print(f"leave-one-session-out on IEMOCAP  |  {_n_classes} classes, "
          f"chance = {1.0 / _n_classes:.3f}")
    print(sweep_df.to_string(index=False))
    if len(_base):
        _b_acc = float(_base["pooled_accuracy"].iloc[0])
        _b_f1 = float(_base["pooled_macro_f1"].iloc[0])
        sweep_df["acc_vs_iemocap_only"] = (
            sweep_df["pooled_accuracy"] - _b_acc).round(4)
        sweep_df["f1_vs_iemocap_only"] = (
            sweep_df["pooled_macro_f1"] - _b_f1).round(4)
        _best = sweep_df.loc[sweep_df["pooled_macro_f1"].idxmax()]
        print(f"\nbest macro-F1: {_best['label']} "
              f"({_best['pooled_macro_f1']:.4f})")
        if float(_best["ravdess_weight"]) == 0.0:
            print("  -> RAVDESS didn't help on this label space. Set "
                  "RAVDESS_WEIGHT=0.0 and use the IEMOCAP-only model.")
        else:
            print(f"  -> RAVDESS helps at w={_best['ravdess_weight']}. Set "
                  f"RAVDESS_WEIGHT to that in CELL 1 before CELL 12.")
        print("  Check the gap against fold_acc_std. With 5 folds, +0.01 on a "
              "std of 0.05 isn't a result. See CELL 10b for a robustness-based "
              "pick that doesn't rely on this pooled argmax alone.")
    return sweep_detail, sweep_df


@app.cell
def _(mo):
    mo.md("""
    ## Cell 11 — Choosing the weight (testing)

    A diagnostic; the value it recommends is typed back into Cell 1 by hand, so
    nothing here changes the model automatically. Rather than picking the best
    pooled score, it asks a different question per weight: what is the worst
    thing that happened to any single session compared with the no-RAVDESS
    baseline. Weights that never hurt a fold are marked clean, and the best of
    those by mean gain is reported. This rule exists because a pooled number can
    be carried by one lucky fold while another session is quietly regressing.
    """)
    return


@app.cell
def _(RAVDESS_WEIGHT_SWEEP, pd, sweep_detail):
    _base_folds = (sweep_detail[0.0][0]
                   .set_index("held_out_session")["macro_f1"])

    rows = []
    for w in RAVDESS_WEIGHT_SWEEP:
        folds_w = (sweep_detail[w][0]
                   .set_index("held_out_session")["macro_f1"])
        deltas = folds_w - _base_folds
        rows.append({
            "ravdess_weight": w,
            "min_session_delta": round(float(deltas.min()), 4),
            "max_session_delta": round(float(deltas.max()), 4),
            "mean_session_delta": round(float(deltas.mean()), 4),
            "n_sessions_regressed": int((deltas < 0).sum()),
        })

    robustness_df = pd.DataFrame(rows).sort_values("ravdess_weight")
    print("per-weight fold-delta summary (macro-F1 vs w=0.0 baseline):\n")
    print(robustness_df.to_string(index=False))

    _clean = robustness_df[robustness_df["n_sessions_regressed"] == 0]
    print(f"\n{'=' * 66}")
    if len(_clean):
        _pick = _clean.loc[_clean["mean_session_delta"].idxmax()]
        print(f"weights with zero regressed sessions: "
              f"{sorted(_clean['ravdess_weight'].tolist())}")
        print(f"best of those by mean gain: w={_pick['ravdess_weight']} "
              f"(mean delta {_pick['mean_session_delta']:+.4f}, "
              f"worst session {_pick['min_session_delta']:+.4f})")
        print("\nSet RAVDESS_WEIGHT to that value in CELL 1 before CELL 12.")
    else:
        print("no weight in this sweep avoided regressing at least one "
              "session -- 0.0 (IEMOCAP-only) is the defensible choice, or "
              "widen the sweep below the smallest weight tried here")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cell 12 — Per-class diagnostics (testing)

    Reporting only; it inspects the cross-validation results at the chosen
    weight and does not affect training. It prints recall, precision, support
    and how often each class was predicted, worst recall first, followed by the
    confusion matrix and the per-fold table. Recall is the number that matters
    for this pipeline, because a class the model never predicts is a colour that
    never appears on screen whatever the headline accuracy says. Any class that
    was never predicted at all gets called out explicitly.
    """)
    return


@app.cell
def _(RAVDESS_WEIGHT, confusion_matrix, keep_labels, np, pd, sweep_detail):
    _w_used = (RAVDESS_WEIGHT if RAVDESS_WEIGHT in sweep_detail
               else sorted(sweep_detail)[0])
    _folds, _yt, _yp = sweep_detail[_w_used]

    _labels = [l for l in keep_labels if l in set(_yt) | set(_yp)]
    _cm = confusion_matrix(_yt, _yp, labels=_labels)
    cm_df = pd.DataFrame(_cm, index=[f"true_{l}" for l in _labels],
                         columns=[f"pred_{l}" for l in _labels])

    _support = _cm.sum(axis=1)
    _recall = np.divide(np.diag(_cm), np.where(_support == 0, 1, _support))
    _pred_n = _cm.sum(axis=0)
    _precision = np.divide(np.diag(_cm), np.where(_pred_n == 0, 1, _pred_n))
    per_class = pd.DataFrame({
        "class": _labels,
        "support": _support,
        "recall": np.round(_recall, 3),
        "precision": np.round(_precision, 3),
        "times_predicted": _pred_n,
    }).sort_values("recall")

    print(f"diagnostics at ravdess_weight={_w_used}\n")
    print("per-class, worst recall first:")
    print(per_class.to_string(index=False))
    _dead = per_class[per_class["times_predicted"] == 0]["class"].tolist()
    if _dead:
        print(f"\nnever predicted: {_dead}. These colours won't render. Either "
              f"drop the class (raise MIN_CLASS_COUNT) or accept the pipeline "
              f"can't express them.")
    print("\nconfusion matrix (rows = truth):")
    print(cm_df.to_string())
    print("\nper-fold:")
    print(_folds.to_string(index=False))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Cell 13 — Fit the shipping model and save the bundle

    Trains the model that actually ships, using everything: all five IEMOCAP
    sessions plus weighted RAVDESS. The cross-validation above only estimated
    how this model behaves on unseen speakers, it was never the model itself.
    The result is saved as a joblib bundle containing the four keys the
    transcriber reads, plus provenance such as the training date, row counts,
    weight, protocol and CV scores, so the file can always say what it was
    trained on.
    """)
    return


@app.cell
def _(
    EXTRACTOR,
    MERGE_EXCITED_INTO_HAPPY,
    MIN_CLASS_COUNT,
    MODEL_OUT,
    NORM_SCOPE,
    RAVDESS_WEIGHT,
    RF_KWARGS,
    RandomForestClassifier,
    SPEAKER_NORMALISED,
    feature_cols,
    joblib,
    keep_labels,
    norm_df,
    np,
    sweep_df,
    time,
):
    _w = np.where(norm_df["source"].to_numpy() == "ravdess",
                  float(RAVDESS_WEIGHT), 1.0)
    _X = norm_df[feature_cols].to_numpy(dtype=float)
    _y = norm_df["label"].to_numpy()

    clf_v3 = RandomForestClassifier(**RF_KWARGS)
    clf_v3.fit(_X, _y, sample_weight=_w)

    _cv = sweep_df.loc[sweep_df["ravdess_weight"] == RAVDESS_WEIGHT]
    bundle = {
        # the 4 keys the transcriber needs
        "clf": clf_v3,
        "feature_cols": list(feature_cols),
        "extractor": EXTRACTOR,
        "speaker_normalised": bool(SPEAKER_NORMALISED),
        # provenance, ignored at inference
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trained_by": "train_emotionsV3.py",
        "classes": list(clf_v3.classes_),
        "label_space_source": "iemocap",
        "n_rows_total": int(len(norm_df)),
        "n_rows_iemocap": int((norm_df["source"] == "iemocap").sum()),
        "n_rows_ravdess": int((norm_df["source"] == "ravdess").sum()),
        "ravdess_weight": float(RAVDESS_WEIGHT),
        "norm_scope": NORM_SCOPE,
        "merge_excited_into_happy": bool(MERGE_EXCITED_INTO_HAPPY),
        "min_class_count": int(MIN_CLASS_COUNT),
        "rf_kwargs": dict(RF_KWARGS),
        "cv_protocol": "leave-one-IEMOCAP-session-out; RAVDESS train-only",
        "cv_pooled_accuracy": (float(_cv["pooled_accuracy"].iloc[0])
                               if len(_cv) else None),
        "cv_pooled_macro_f1": (float(_cv["pooled_macro_f1"].iloc[0])
                               if len(_cv) else None),
        "dropped_ravdess_classes": ["calm"],
    }
    joblib.dump(bundle, MODEL_OUT)

    print(f"wrote {MODEL_OUT}")
    print(f"  classes ({len(clf_v3.classes_)}): {list(clf_v3.classes_)}")
    print(f"  features: {len(feature_cols)} ({EXTRACTOR})")
    print(f"  rows: {bundle['n_rows_iemocap']} iemocap + "
          f"{bundle['n_rows_ravdess']} ravdess @ w={RAVDESS_WEIGHT}")
    print(f"  speaker_normalised={bundle['speaker_normalised']} "
          f"(scope={NORM_SCOPE})")
    print(f"  cv: acc={bundle['cv_pooled_accuracy']} "
          f"macroF1={bundle['cv_pooled_macro_f1']}")
    print(f"  label space from IEMOCAP: {keep_labels}")
    return


if __name__ == "__main__":
    app.run()

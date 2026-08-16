import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # =====================================================================
    # CELL 0 — IMPORTS (all imports live here, marimo convention)
    # =====================================================================
    import marimo as mo
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
        mo,
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
    # train_emotionsV3 — IEMOCAP primary, RAVDESS supporting

    Replaces `train_emotion.py`, which trained on RAVDESS only (46.8%
    speaker-independent against a 14.3% chance line). V8 and V17 both test on
    IEMOCAP without ever training on it. V8's config has a commented-out
    `features_combined.csv` switch for a combined model that was never written.

    ## Rules

    IEMOCAP sets the label space. RAVDESS can only fill classes IEMOCAP already
    has. The target domain is conversational speech, so IEMOCAP is the
    reference and RAVDESS is extra data.

    - RAVDESS classes with no IEMOCAP counterpart get dropped, not remapped.
      `calm` is the main one. Folding it into `neutral` would change what
      `neutral` means, and `neutral` is IEMOCAP's biggest class.
    - RAVDESS rows are down-weighted at fit time (`RAVDESS_WEIGHT`).
    - Evaluation is IEMOCAP-only, leave-one-session-out. RAVDESS never goes
      into a test fold. Pooled accuracy would hide a model that improved on
      acted speech and not on conversation.
    - CELL 10 trains both IEMOCAP-only and IEMOCAP+RAVDESS on the same folds,
      so you can check whether RAVDESS actually helped.

    ## Output

    `outputs/clf_v3.joblib`, in the 4-key bundle format V17's CELL 6 already
    loads (`clf` / `feature_cols` / `extractor` / `speaker_normalised`), plus
    metadata keys V17 ignores.
    """)
    return


@app.cell
def _(os):
    # =====================================================================
    # CELL 1 — CONFIG
    # =====================================================================
    ravdess_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    iemocap_dir = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"

    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/audio", exist_ok=True)

    # ---------------- EXTRACTOR ----------------
    # "egemaps" -> 88-dim eGeMAPSv02 functionals, what V17's clf_v2 expects
    # "praat14" -> the 14 Praat features from V8 CELL 4
    # Has to match what the consuming notebook extracts at inference, or the
    # vector it builds won't line up with the one fitted here.
    EXTRACTOR = "egemaps"

    # ---------------- SPEAKER NORMALISATION ----------------
    # V17 normalises at inference by z-scoring across all segments of one clip
    # (predict_segment_emotions_v9). The training-time equivalent is per-dialog:
    # one IEMOCAP dialog is the structural match for one video handed to
    # process_any_video.
    #   "dialog"         group by dialog, both speakers pooled. Closest to what
    #                    V17 does. Default.
    #   "dialog_speaker" group by (dialog, speaker). Cleaner stats, but tighter
    #                    than inference can reproduce with diarisation off.
    #   "speaker"        group by speaker across the corpus. Best stats,
    #                    furthest from inference.
    #   "off"            no normalisation. Sets speaker_normalised=False so V17
    #                    skips its own step too.
    NORM_SCOPE = "dialog"
    NORM_MIN_ROWS = 4          # groups smaller than this are left raw
    NORM_STD_FLOOR = 1e-6      # guards divide-by-zero on a constant feature

    # ---------------- SUPPORT-SET WEIGHTING ----------------
    # Weight on every RAVDESS row at fit time. 1.0 means a RAVDESS clip counts
    # as much as an IEMOCAP one; 0.0 excludes RAVDESS (same as the IEMOCAP-only
    # ablation). 0.3 is a starting guess. CELL 10 sweeps it so you can pick
    # from the fold numbers instead.
    #
    # Sweep widened around 0.0-0.3, where the coarse pass (0.0/0.15/0.3/0.5/1.0)
    # showed every fold improving at 0.15 and session 5 starting to regress by
    # 0.3. CELL 10b picks from this sweep by worst-case fold, not pooled argmax.
    RAVDESS_WEIGHT = 0.20
    RAVDESS_WEIGHT_SWEEP = [
        0.0, 0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.20, 0.22, 0.25, 0.28, 0.30,
    ]

    # ---------------- LABEL-SPACE OPTIONS ----------------
    # IEMOCAP's exc (excited) has no RAVDESS counterpart. A lot of published
    # IEMOCAP work merges exc into hap because annotators confused the two.
    # Merging gives a bigger happy class that RAVDESS can support; keeping them
    # separate preserves a distinction this pipeline could render differently.
    MERGE_EXCITED_INTO_HAPPY = False

    # Drop IEMOCAP classes below this count. fea/sur/dis are rare enough in
    # IEMOCAP that without RAVDESS they're barely learnable, and with it they
    # become mostly-acted classes. Counts print either way.
    MIN_CLASS_COUNT = 40

    # ---------------- CACHING ----------------
    # Extraction over ~10k utterances is the slow part. Cached per extractor so
    # switching EXTRACTOR doesn't reuse the wrong cache.
    iemocap_feat_csv = f"{OUT_DIR}/features_iemocap_{EXTRACTOR}.csv"
    ravdess_feat_csv = f"{OUT_DIR}/features_ravdess_{EXTRACTOR}.csv"
    combined_csv = f"{OUT_DIR}/features_combined_{EXTRACTOR}.csv"

    # Existing V8/V17 IEMOCAP CSV, if present. Used for labels only; its
    # feature columns are praat14 and get ignored.
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
def _():
    # =====================================================================
    # CELL 2 — LABEL HARMONISATION
    # Both mappings written out rather than inferred at runtime.
    # =====================================================================
    # IEMOCAP's 3-letter codes -> canonical names. This is the label space;
    # nothing downstream adds to it.
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
        # "oth" (other) and "xxx" (no annotator agreement) are left out. An
        # utterance the annotators couldn't agree on is a noisy label.
    }

    # RAVDESS filename digit -> canonical name, using IEMOCAP's vocabulary.
    # None means no IEMOCAP counterpart, so drop the clip.
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

    # In both corpora : neutral, happy, sad, angry, fearful, disgust, surprised
    # IEMOCAP only    : frustrated, excited (learned from IEMOCAP alone)
    # RAVDESS only    : calm (dropped)
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
def _(EXTRACTOR, HAVE_OPENSMILE, call, np, opensmile, parselmouth):
    # =====================================================================
    # CELL 3 — FEATURE EXTRACTORS
    # praat14 is copied from TranscriberV8 CELL 4 unchanged, so a praat14
    # model trained here is comparable with the old one.
    # =====================================================================
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

    # Same eGeMAPSv02 / Functionals construction as V17 CELL 7, so the column
    # names match and feature_cols lines up at inference.
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
def _(IEMOCAP_CODE_TO_LABEL, Path, iemocap_dir, iemocap_label_csv, os, pd, re):
    # =====================================================================
    # CELL 4 — IEMOCAP LABEL INDEX
    # Two sources, in order:
    #  1. The EmoEvaluation .txt files inside IEMOCAP_full_release. This is
    #     what extract_iemocap.py was presumably wrapping.
    #  2. features_iemocap.csv, if the release isn't mounted. Labels only.
    # =====================================================================
    EMO_LINE_RE = re.compile(
        r"^\[(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\]\s+"
        r"(?P<utt>\S+)\s+(?P<code>\S+)\s+\["
    )

    def speaker_of(utt_id):
        """Speaker id = session number + who is talking in this turn.

        The trailing field of an utterance id ('..._F012' / '..._M042') gives
        the turn's speaker. The 'F' in the session prefix ('Ses01F') only says
        whose script the dialog followed, so it isn't the speaker. Using it
        halves the apparent speaker count and lets the same voice land in both
        train and test.
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
    return (iem_index,)


@app.cell
def _(Path, RAVDESS_DIGIT_TO_LABEL, os, pd, ravdess_dir):
    # =====================================================================
    # CELL 5 — RAVDESS INDEX (overlapping classes only)
    # Filename: modality-vocal-emotion-intensity-statement-repetition-actor
    # e.g. 03-01-06-01-02-01-12.wav -> emotion 06 (fearful), actor 12.
    # Speech only (vocal channel 01). The song half is a different production
    # style and isn't what this pipeline captions.
    # =====================================================================
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
                # RAVDESS actors are the speakers, and there are no dialogs, so
                # the actor doubles as the normalisation group (CELL 8).
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
    return (rav_index,)


@app.cell
def _(extract_features, os, pd, time):
    # =====================================================================
    # CELL 6 — FEATURE EXTRACTION WITH CACHE
    # The only slow step (~10k IEMOCAP utterances + ~1.4k RAVDESS speech
    # clips). Cached to CSV keyed by extractor. A clip whose extraction throws
    # is dropped and counted, not written as zeros: a row of zeros looks like a
    # plausible feature vector with no signal in it.
    # =====================================================================
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

    return (extract_for_index,)


@app.cell
def _(
    extract_for_index,
    iem_index,
    iemocap_feat_csv,
    rav_index,
    ravdess_feat_csv,
):
    # =====================================================================
    # CELL 6b — RUN EXTRACTION
    # =====================================================================
    iem_feats = extract_for_index(iem_index, iemocap_feat_csv, "iemocap")
    rav_feats = extract_for_index(rav_index, ravdess_feat_csv, "ravdess")
    print(f"\niemocap rows: {len(iem_feats)}   ravdess rows: {len(rav_feats)}")
    return iem_feats, rav_feats


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
    # =====================================================================
    # CELL 7 — COMBINE
    # Order matters here:
    #   1. merge exc->hap (if enabled) before counting, so the merged class is
    #      judged at its real size
    #   2. pick surviving labels from IEMOCAP counts alone. RAVDESS shouldn't
    #      rescue a class IEMOCAP can't support, or you get a class the
    #      pipeline only recognises in acted speech.
    #   3. filter both frames to that set, then concatenate
    # =====================================================================
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
    # "which corpus is this" from a column one corpus never fills
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
def _(NORM_MIN_ROWS, NORM_SCOPE, NORM_STD_FLOOR, combined, feature_cols, pd):
    # =====================================================================
    # CELL 8 — SPEAKER / DIALOG NORMALISATION
    # Features only, no labels, so running it before the split leaks nothing.
    # It's the same transform V17 applies at inference, where there are no
    # labels at all. What it buys: a speaker who is just loud stops looking
    # angry, because every feature is relative to that speaker's (or that
    # dialog's) own baseline.
    #
    # RAVDESS has no dialogs, so its group key falls back to the actor under
    # every scope. IEMOCAP rows get conversation-relative features, RAVDESS
    # rows get actor-relative ones.
    # =====================================================================
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
def _(RandomForestClassifier, accuracy_score, f1_score, np, pd):
    # =====================================================================
    # CELL 9 — EVALUATION: leave-one-IEMOCAP-session-out
    #
    # Test folds are IEMOCAP only, RAVDESS stays in train. The question is
    # whether the model works on conversational speech, and a pooled number
    # that went up because it got better at studio-acted sentences answers a
    # different question.
    #
    # Folds are whole sessions, so both speakers of a conversation leave
    # together. Splitting on speaker alone leaves their partner's half of the
    # same recording in train: same room, same mic, sometimes crosstalk.
    #
    # All RAVDESS rows go into every train fold, weighted. They're support, not
    # held-out data.
    #
    # Reports macro-F1 as well as accuracy, because accuracy alone rewards
    # coasting on neutral and frustrated, where IEMOCAP's mass sits.
    # =====================================================================
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
def _(RandomForestClassifier, accuracy_score, f1_score, np, pd):
    # =====================================================================
    # CELL 9c — RAVDESS-ONLY BASELINE, SCORED ON THE SAME IEMOCAP FOLDS
    #
    # This is the fair comparison to the old 46.8% figure, which was measured
    # on RAVDESS itself — same acted-speech domain the model was trained on.
    # This instead trains only on RAVDESS and tests on the same 5 IEMOCAP
    # leave-one-session-out folds CELL 9/10 use, so the row sits in the same
    # table as the rest of the sweep.
    #
    # Caveat: this trains on the RAVDESS rows CELL 5 already filtered (calm
    # dropped, song dropped) — the same rows CELL 10 uses as support. It is
    # not a re-run of the original train_emotion.py, which likely kept calm
    # and was only ever tested on RAVDESS. Read this as "how far does an
    # acted-speech-only model get on conversation," not as reproducing 46.8%.
    #
    # RAVDESS never had frustrated/excited, so this model can never predict
    # them — that's an out-of-vocabulary penalty, not confusion. Two numbers
    # are reported per fold: accuracy/macro-F1 over IEMOCAP's full label
    # space (frustrated/excited count as always-wrong, matching CELL 9's
    # rules so the row is comparable), and the same pair restricted to the
    # classes RAVDESS actually has a concept of.
    # =====================================================================
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

            # Training set is identical every fold — IEMOCAP contributes
            # nothing to this model, only to the test side.
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

    return (run_ravdess_only_baseline,)


@app.cell
def _(RF_KWARGS, feature_cols, norm_df, run_ravdess_only_baseline):
    # =====================================================================
    # CELL 9d — RUN THE RAVDESS-ONLY BASELINE
    # =====================================================================
    rav_only_folds, rav_only_full, rav_only_overlap, _ro_yt, _ro_yp = (
        run_ravdess_only_baseline(norm_df, feature_cols, RF_KWARGS)
    )
    print("\nper-fold detail:")
    print(rav_only_folds.to_string(index=False))
    return


@app.cell
def _(RAVDESS_WEIGHT_SWEEP, RF_KWARGS, feature_cols, norm_df, pd, run_cv):
    # =====================================================================
    # CELL 10 — ABLATION: does RAVDESS help?
    # weight 0.0 is the IEMOCAP-only baseline, same folds and hyperparameters
    # with RAVDESS just absent, so the comparison is like for like.
    #
    # Chance is 1/n_classes. The old RAVDESS-only pipeline's 46.8% was measured
    # on RAVDESS, so it isn't a like-for-like target. The comparison that
    # matters here is weight 0.0 against weight > 0.0.
    # =====================================================================
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
def _(RAVDESS_WEIGHT_SWEEP, pd, sweep_detail):
    # =====================================================================
    # CELL 10b — PICK BY WORST-CASE FOLD, NOT POOLED ARGMAX
    #
    # Pooled macro-F1 can be won by one lucky fold while others quietly
    # regress -- that's what happened at w=1.0 in the coarse sweep, where
    # session 3 alone (+0.057) carried a pooled number that session 5
    # (-0.013) was actively working against. This asks a different question
    # per weight: what is the worst thing that happened to any single
    # session, relative to the w=0.0 baseline? The weight that maximises
    # that worst case is the one where RAVDESS is never hurting a fold, not
    # just helping on average.
    # =====================================================================
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
def _(RAVDESS_WEIGHT, confusion_matrix, keep_labels, np, pd, sweep_detail):
    # =====================================================================
    # CELL 11 — PER-CLASS DIAGNOSTICS at the chosen weight
    # Recall per class is what matters for this pipeline: a class the model
    # never predicts is a colour that never appears on screen, whatever the
    # headline accuracy says.
    # =====================================================================
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
    # =====================================================================
    # CELL 12 — FIT THE SHIPPING MODEL + SAVE THE BUNDLE
    # Everything gets used here: all 5 IEMOCAP sessions plus weighted RAVDESS.
    # The CV above estimates how this model will behave on unseen speakers, it
    # isn't the model itself.
    #
    # V17's CELL 6 reads 4 keys: clf, feature_cols, extractor,
    # speaker_normalised. The rest is provenance. V17 ignores extra keys, and a
    # model file that can't say what it was trained on is how the 60.6% figure
    # ended up untraceable.
    # =====================================================================
    _w = np.where(norm_df["source"].to_numpy() == "ravdess",
                  float(RAVDESS_WEIGHT), 1.0)
    _X = norm_df[feature_cols].to_numpy(dtype=float)
    _y = norm_df["label"].to_numpy()

    clf_v3 = RandomForestClassifier(**RF_KWARGS)
    clf_v3.fit(_X, _y, sample_weight=_w)

    _cv = sweep_df.loc[sweep_df["ravdess_weight"] == RAVDESS_WEIGHT]
    bundle = {
        # --- the 4 keys V17 CELL 6 needs ---
        "clf": clf_v3,
        "feature_cols": list(feature_cols),
        "extractor": EXTRACTOR,
        "speaker_normalised": bool(SPEAKER_NORMALISED),
        # --- provenance, ignored by V17 ---
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


@app.cell
def _(mo):
    mo.md("""
 
    """)
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


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

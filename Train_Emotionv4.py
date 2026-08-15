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
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    try:
        import opensmile
        HAVE_OPENSMILE = True
    except Exception:
        opensmile = None
        HAVE_OPENSMILE = False

    print(f"opensmile available: {HAVE_OPENSMILE}")

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
    RAVDESS_WEIGHT = 0.3
    RAVDESS_WEIGHT_SWEEP = [
        0.0, 0.05, 0.08, 0.10, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18,
        0.19, 0.20, 0.22, 0.25, 0.28, 0.30,
    ]

    # ---------------- VAD TARGET SCALE ----------------
    # The known issue from the handoff (§7): features are dialog-z-scored in
    # CELL 8, but IEMOCAP's V/A/D targets are absolute 1-5 ratings. Predicting
    # an absolute quantity from relative features caps CCC and makes any
    # comparison against published numbers unfair.
    #
    # BUT THE FRAMING IN THE HANDOFF IS BACKWARDS FOR THIS PIPELINE, and it
    # is worth being explicit about why, because it changes which setting is
    # "correct" rather than merely which is higher-scoring.
    #
    # Look at how the renderer actually consumes arousal. TranscriberV19's
    # segment_arousal_floor() takes the per-segment mean of its arousal
    # features, robust-z-scores it ACROSS THE SEGMENTS OF THAT CLIP, and
    # tanh-squashes the result. The quantity the styling has always used is
    # therefore *clip-relative*: "loud for this speaker, in this clip". It has
    # never been an absolute rating and there is nowhere for one to go.
    #
    # So a dialog-z target is not the compromise. It is the closer match to
    # the thing being replaced. "raw" is the number to quote in a write-up
    # against published CCCs; "dialog_z" is the number that says whether this
    # would work on screen. Run both -- CELL 10c does -- and expect them to
    # disagree.
    #
    #   "raw"       targets as annotated, 1-5.
    #   "dialog_z"  targets z-scored within dialog, matching the features.
    #               Not shippable as an absolute predictor: at inference there
    #               are no per-dialog target statistics to invert with. Ship
    #               it only if the consumer treats the output as relative,
    #               which segment_arousal_floor already does.
    VAD_TARGET_SCOPE = "raw"          # "raw" | "dialog_z"

    # Regressor settings mirror RF_KWARGS, minus class_weight -- that argument
    # is classifier-only and RandomForestRegressor raises TypeError on it, so
    # RF_KWARGS cannot be reused directly.
    VAD_RF_KWARGS = dict(n_estimators=1000, random_state=42, n_jobs=-1,
                         min_samples_leaf=2)

    VAD_DIMENSIONS = ["valence", "arousal", "dominance"]

    # ---------------- WHEN IS A DIMENSION GOOD ENOUGH TO STYLE? ----------
    # CCC floor below which a dimension is recorded but flagged unfit to drive
    # anything visible. 0.45 is a judgement call, stated so it can be argued
    # with: in dimensional affect recognition, CCC above ~0.6 is generally
    # considered good, 0.4-0.6 moderate, below 0.4 weak. 0.45 sits at the
    # bottom of "moderate" -- low enough not to reject a usable arousal
    # regressor, high enough that a valence regressor at 0.2 cannot quietly
    # start driving hue.
    VAD_CCC_MIN_FOR_STYLING = 0.45

    # Arousal has an extra condition, and it is the important one. The
    # renderer ALREADY has a working arousal estimate: the hand-weighted
    # AROUSAL_FEATURES proxy. Replacing a heuristic that works with a model
    # that merely ties it is added machinery, a second thing to keep
    # calibrated, and a new dependency on the bundle, for nothing. So a
    # trained arousal regressor must BEAT the proxy by this margin, not just
    # clear the floor above.
    VAD_MUST_BEAT_PROXY_BY = 0.05

    # eGeMAPS column names standing in for the renderer's AROUSAL_FEATURES
    # ({intensity_db: 1.0, f0_mean: 0.6}). Checked at run time, not assumed --
    # eGeMAPSv02 names are long and easy to get subtly wrong, and a silently
    # missing column would make the proxy look worse than it is and hand the
    # comparison to the trained model by default.
    PROXY_AROUSAL_FEATURES = {
        "loudness_sma3_amean": 1.0,                      # ~ intensity_db
        "F0semitoneFrom27.5Hz_sma3nz_amean": 0.6,        # ~ f0_mean
    }


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

    RF_KWARGS = dict(n_estimators=1000, random_state=42, n_jobs=-1,
                     class_weight="balanced", min_samples_leaf=2)

    print(f"extractor={EXTRACTOR}  norm_scope={NORM_SCOPE}  "
          f"ravdess_weight={RAVDESS_WEIGHT}")
    print(f"iemocap cache -> {iemocap_feat_csv}")
    print(f"ravdess cache -> {ravdess_feat_csv}")
    print(f"model out     -> {MODEL_OUT}")

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

    # =====================================================================
    # =====================================================================
    # CELL 4b — RECOVER V/A/D FROM EmoEvaluation
    # ---------------------------------------------------------------------
    # The values have been on disk and parsed-past all along. CELL 4's
    # EMO_LINE_RE matches up to the opening "[" of the dimensional bracket and
    # stops:
    #
    #   [6.2901 - 8.2357]  Ses01F_impro01_F000  neu  [2.5000, 2.5000, 2.5000]
    #                                                 valence arousal dominance
    #
    # IEMOCAP calls arousal "activation"; the bracket order is
    # [valence, activation, dominance]. This is a second pass over the same
    # text files -- cheap, and it keeps CELL 4's regex untouched so the
    # categorical path cannot break as a side effect of adding this.
    #
    # Verified against real line variants including releases that write whole
    # numbers without decimals ("[4, 3.5000, 3]"), which a \d+\.\d+ pattern
    # silently drops.
    # =====================================================================
    EMO_VAD_LINE_RE = re.compile(
        r"^\[(?P<start>\d+\.\d+)\s*-\s*(?P<end>\d+\.\d+)\]\s+"
        r"(?P<utt>\S+)\s+(?P<code>\S+)\s+"
        r"\[\s*(?P<valence>\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<arousal>\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<dominance>\d+(?:\.\d+)?)\s*\]"
    )


    def scan_emoevaluation_vad(root):
        """Every utterance's V/A/D, regardless of categorical agreement.

        Deliberately does NOT filter on the emotion code. An utterance the
        annotators labelled 'xxx' (no categorical agreement) still carries
        dimensional ratings, and those ratings are not obviously worthless --
        disagreeing about whether something is 'frustrated' or 'angry' is
        consistent with agreeing it is low-valence and high-arousal.

        Whether any of that extra data is reachable is a separate question,
        answered in 8b: CELL 6 only extracted features for rows that survived
        CELL 4's categorical filter, so unmatched rows have no feature vector
        to join to. They are reported, not used. Using them would mean
        re-running opensmile, which is explicitly out of scope here.
        """
        rows, n_lines, n_bad = [], 0, 0
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
                    line = line.strip()
                    if not line.startswith("["):
                        continue
                    n_lines += 1
                    m = EMO_VAD_LINE_RE.match(line)
                    if not m:
                        n_bad += 1
                        continue
                    rows.append({
                        "utt_id": m.group("utt"),
                        "raw_code": m.group("code").lower(),
                        "valence": float(m.group("valence")),
                        "arousal": float(m.group("arousal")),
                        "dominance": float(m.group("dominance")),
                    })
        return pd.DataFrame(rows), n_lines, n_bad


    vad_index, _vad_lines, _vad_unparsed = scan_emoevaluation_vad(iemocap_dir)

    if len(vad_index) == 0:
        print(f"No V/A/D recovered from {iemocap_dir}.")
        print("  Everything below will no-op. This needs the EmoEvaluation "
              "text files -- the features_iemocap.csv fallback CELL 4 uses "
              "carries labels only, never the dimensional bracket.")
    else:
        vad_index = vad_index.drop_duplicates(subset="utt_id", keep="first")
        print(f"V/A/D recovered for {len(vad_index)} utterances "
              f"({_vad_lines} candidate lines, {_vad_unparsed} unparsed)")
        print(vad_index[VAD_DIMENSIONS].describe().round(3).to_string())


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

    # =====================================================================
    # CELL 6b — RUN EXTRACTION
    # =====================================================================
    iem_feats = extract_for_index(iem_index, iemocap_feat_csv, "iemocap")
    rav_feats = extract_for_index(rav_index, ravdess_feat_csv, "ravdess")
    print(f"\niemocap rows: {len(iem_feats)}   ravdess rows: {len(rav_feats)}")

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

    # =====================================================================
    # =====================================================================
    # CELL 8b — JOIN ONTO FEATURES, COVERAGE, INTER-DIMENSION CORRELATION
    # (the handoff's "CELL 4c" — renumbered, it has to follow CELL 8)
    # ---------------------------------------------------------------------
    # Joins onto norm_df, not onto the raw feature cache, so the regressors
    # see exactly the features the classifier sees -- same normalisation, same
    # dropped rows. IEMOCAP only: RAVDESS has no dimensional annotations, so
    # there is no support set here and RAVDESS_WEIGHT plays no part.
    #
    # The correlation table is the one that decides whether dominance is worth
    # keeping. If dominance correlates with valence above ~0.8 it is not a
    # third axis, it is valence with extra steps, and a third regressor buys a
    # third thing to maintain and nothing on screen.
    # =====================================================================
    if len(vad_index):
        _iem_rows = norm_df[norm_df["source"] == "iemocap"].copy()
        vad_df = _iem_rows.merge(vad_index[["utt_id"] + VAD_DIMENSIONS],
                                 on="utt_id", how="inner")

        _n_feat = len(_iem_rows)
        _n_join = len(vad_df)
        _n_ann = len(vad_index)
        print(f"coverage")
        print(f"  IEMOCAP rows with features (post-CELL 8) : {_n_feat}")
        print(f"  utterances with V/A/D annotations        : {_n_ann}")
        print(f"  joined, usable for regression            : {_n_join} "
              f"({_n_join / max(_n_feat, 1):.1%} of feature rows)")

        _unreachable = _n_ann - _n_join
        if _unreachable > 0:
            print(f"  annotated but no feature vector          : {_unreachable}")
            print(f"    Mostly 'xxx'/'oth' utterances CELL 4 excluded for "
                  f"having no categorical agreement. Their V/A/D is fine; "
                  f"CELL 6 just never extracted features for them. Reaching "
                  f"them means re-running opensmile on ~{_unreachable} clips "
                  f"-- a real option for the regressors, out of scope here.")

        if _n_join < _n_feat * 0.9:
            print(f"  WARNING: joined fewer than 90% of feature rows. Check "
                  f"utt_id formatting on both sides before trusting anything "
                  f"below.")

        print(f"\nper-session counts:")
        print(vad_df.groupby("session").size().to_string())

        print(f"\ninter-dimension correlation (Pearson):")
        _corr = vad_df[VAD_DIMENSIONS].corr().round(3)
        print(_corr.to_string())
        _vd = abs(float(_corr.loc["valence", "dominance"]))
        if _vd >= 0.8:
            print(f"\n  dominance/valence r={_vd:.3f}. Dominance is not adding "
                  f"a third axis at this level -- it is close to a rescaled "
                  f"valence. Two regressors, not three, unless the CCC numbers "
                  f"below say otherwise.")
        elif _vd >= 0.6:
            print(f"\n  dominance/valence r={_vd:.3f}: substantially but not "
                  f"fully redundant. Worth keeping through CV to see whether "
                  f"it predicts better than valence does.")

        print(f"\ncorrelation with the categorical label (sanity check):")
        print(vad_df.groupby("label")[VAD_DIMENSIONS].mean().round(2).to_string())
        print("  angry/frustrated should sit low-valence high-arousal, happy/"
              "excited high-valence high-arousal, sad low-arousal. If they do "
              "not, the join is wrong.")
    else:
        vad_df = pd.DataFrame()


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

    # =====================================================================
    # =====================================================================
    # CELL 9e — CCC, AND A PROXY BASELINE TO BEAT
    # ---------------------------------------------------------------------
    # CCC (Lin's concordance correlation coefficient), NOT Pearson r.
    #
    # Pearson asks "does it move in the right direction". CCC asks "is it the
    # right number". For styling that difference is the whole thing: a model
    # that tracks arousal perfectly but reads half a point high renders every
    # caption over-saturated, and Pearson scores it 1.000. The self-test below
    # makes that concrete -- shift, scale and shrink cases all score a perfect
    # Pearson while CCC correctly penalises each.
    # =====================================================================
    def concordance_cc(y_true, y_pred):
        """Lin's CCC: 2*cov / (var_t + var_p + (mean_t - mean_p)^2)."""
        t = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
        if len(t) < 2 or len(t) != len(p):
            return float("nan")
        mt, mp = t.mean(), p.mean()
        cov = ((t - mt) * (p - mp)).mean()
        denom = t.var() + p.var() + (mt - mp) ** 2
        return float(2.0 * cov / denom) if denom > 0 else float("nan")


    def _ccc_self_test():
        """Assert CCC penalises the errors Pearson is blind to."""
        _rng = np.random.default_rng(0)
        y = _rng.normal(3.0, 0.9, 2000).clip(1, 5)
        cases = {
            "perfect":        y.copy(),
            "+0.5 shift":     y + 0.5,
            "1.5x scale":     y * 1.5,
            "shrunk to mean": y.mean() + 0.3 * (y - y.mean()),
            "pure noise":     _rng.normal(3.0, 0.9, 2000),
        }
        print(f"  {'case':18s} {'CCC':>7s} {'Pearson':>8s}")
        got = {}
        for k, v in cases.items():
            c = concordance_cc(y, v)
            r = float(np.corrcoef(y, v)[0, 1])
            got[k] = c
            print(f"  {k:18s} {c:7.3f} {r:8.3f}")
        assert got["perfect"] > 0.999, "CCC should be 1.0 on an exact match"
        assert abs(got["pure noise"]) < 0.10, "CCC should be ~0 on noise"
        for k in ("+0.5 shift", "1.5x scale", "shrunk to mean"):
            assert got[k] < 0.95, f"CCC failed to penalise {k}"
        print("  self-test passed: every case Pearson scores 1.000 is "
              "penalised by CCC.")


    print("CCC self-test")
    _ccc_self_test()


    def _apply_target_scope(df, dims, scope):
        """Return a copy with targets on the requested scale."""
        if scope == "raw":
            return df.copy()
        if scope != "dialog_z":
            raise ValueError(f"unknown VAD_TARGET_SCOPE: {scope!r}")
        out = df.copy()
        g = out.groupby("dialog")[dims]
        mu = g.transform("mean")
        sd = g.transform("std").fillna(0.0)
        sd = sd.where(sd.abs() > 1e-6, 1.0)
        out[dims] = ((out[dims] - mu) / sd).fillna(0.0)
        return out


    def proxy_arousal_baseline(df, cols, weights=None):
        """Rebuild the renderer's hand-weighted arousal estimate.

        TranscriberV19 computes arousal as a fixed weighted sum of loudness
        and pitch (AROUSAL_FEATURES), z-scored across the segments of a clip.
        This is the eGeMAPS equivalent of that sum, so a trained regressor has
        the incumbent to beat rather than only a chance line. Without this the
        comparison is against nothing and any positive CCC looks like a win.

        Returns (prediction, columns_used), z-scored and therefore WITHOUT a
        meaningful offset or unit. run_vad_cv calibrates it against the target
        on the training fold before scoring; see the note there for why
        scoring it uncalibrated is not a fair test.
        """
        weights = weights or PROXY_AROUSAL_FEATURES
        have = {c: w for c, w in weights.items() if c in cols}
        missing = sorted(set(weights) - set(have))
        if missing:
            print(f"    proxy: {len(missing)} configured column(s) not in the "
                  f"feature set: {missing}")
            _cands = [c for c in cols
                      if "loudness" in c.lower() or "F0semitone" in c]
            if _cands:
                print(f"    closest available: {_cands[:6]}")
        if not have:
            return None, []
        acc = np.zeros(len(df), dtype=float)
        for c, w in have.items():
            v = df[c].to_numpy(dtype=float)
            sd = v.std()
            acc += w * ((v - v.mean()) / (sd if sd > 1e-9 else 1.0))
        tot = sum(have.values())
        return acc / (tot if tot > 0 else 1.0), sorted(have)


    def run_vad_cv(df, cols, dim, rf_kwargs, verbose=True):
        """Leave-one-IEMOCAP-session-out regression CV, same folds as CELL 9.

        Same fold convention as run_cv on purpose: a VAD number and a
        classifier number quoted in the same write-up have to have been
        measured the same way, or the comparison is rhetorical.
        """
        df = df.reset_index(drop=True)
        sessions = sorted(df["session"].unique())
        per_fold, preds, truths, proxies = [], [], [], []

        # Proxy computed ONCE over every row, so its z-scoring is one
        # consistent transform rather than a different one per fold.
        proxy_all, _proxy_cols = (None, [])
        if dim == "arousal":
            proxy_all, _proxy_cols = proxy_arousal_baseline(df, cols)
            if proxy_all is not None and verbose:
                print(f"    proxy built from {len(_proxy_cols)} column(s): "
                      f"{_proxy_cols}")

        for s in sessions:
            te_mask = (df["session"] == s).to_numpy()
            te, tr = df[te_mask], df[~te_mask]
            if te.empty or tr.empty:
                continue

            reg = RandomForestRegressor(**rf_kwargs)
            reg.fit(tr[cols].to_numpy(dtype=float),
                    tr[dim].to_numpy(dtype=float))
            yp = reg.predict(te[cols].to_numpy(dtype=float))
            yt = te[dim].to_numpy(dtype=float)

            row = {
                "held_out_session": s,
                "n_test": len(te),
                "ccc": concordance_cc(yt, yp),
                "pearson_r": (float(np.corrcoef(yt, yp)[0, 1])
                              if len(yt) > 1 else float("nan")),
                "mae": float(np.mean(np.abs(yt - yp))),
            }

            if proxy_all is not None:
                # CALIBRATE ON THE TRAINING FOLD, THEN SCORE.
                #
                # The proxy is a weighted sum of z-scored features: it has a
                # shape but no offset and no unit. Scoring that directly
                # against a 1-5 target makes CCC's (mean_t - mean_p)^2 term do
                # almost all the work, and the proxy loses for being unscaled
                # rather than for being wrong -- on synthetic data where the
                # proxy had the BETTER shape, that artefact still handed the
                # trained model a +0.66 CCC advantage.
                #
                # Two reasons that would have been the wrong number to act on:
                # it is not a property of either estimator, and the renderer
                # never consumes the proxy unscaled anyway --
                # segment_arousal_floor z-scores it across the clip, which is
                # this same calibration done implicitly.
                #
                # Least squares fitted on train rows only, applied to the held
                # out fold, so no target information crosses the split.
                _px_tr = proxy_all[~te_mask]
                _px_te = proxy_all[te_mask]
                _A = np.vstack([_px_tr, np.ones_like(_px_tr)]).T
                _coef, *_ = np.linalg.lstsq(
                    _A, tr[dim].to_numpy(dtype=float), rcond=None)
                _px_cal = _coef[0] * _px_te + _coef[1]

                row["proxy_ccc"] = concordance_cc(yt, _px_cal)
                row["proxy_ccc_uncal"] = concordance_cc(yt, _px_te)
                row["proxy_pearson_r"] = (float(np.corrcoef(yt, _px_te)[0, 1])
                                          if len(yt) > 1 else float("nan"))
                proxies.extend(list(_px_cal))

            per_fold.append(row)
            preds.extend(list(yp))
            truths.extend(list(yt))
            if verbose:
                _px = (f"  proxy_ccc={row['proxy_ccc']:.3f}"
                       f" (uncal {row['proxy_ccc_uncal']:+.3f})"
                       if "proxy_ccc" in row else "")
                print(f"    session {s}: ccc={row['ccc']:.3f}  "
                      f"r={row['pearson_r']:.3f}  mae={row['mae']:.3f}{_px} "
                      f"(n={len(te)})")

        fold_df = pd.DataFrame(per_fold)
        t, p = np.array(truths), np.array(preds)
        pooled = {
            "dimension": dim,
            "ccc": concordance_cc(t, p),
            "pearson_r": float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else np.nan,
            "mae": float(np.mean(np.abs(t - p))),
            "n": len(t),
            "proxy_ccc": (concordance_cc(t, np.array(proxies))
                          if len(proxies) == len(t) and len(proxies) else np.nan),
            "proxy_ccc_uncal": (float(fold_df["proxy_ccc_uncal"].mean())
                                if "proxy_ccc_uncal" in fold_df.columns
                                else np.nan),
        }
        return fold_df, pooled


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

    # =====================================================================
    # CELL 9d — RUN THE RAVDESS-ONLY BASELINE
    # =====================================================================
    rav_only_folds, rav_only_full, rav_only_overlap, _ro_yt, _ro_yp = (
        run_ravdess_only_baseline(norm_df, feature_cols, RF_KWARGS)
    )
    print("\nper-fold detail:")
    print(rav_only_folds.to_string(index=False))

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

    # =====================================================================
    # =====================================================================
    # CELL 10c — RUN ALL THREE DIMENSIONS, BOTH TARGET SCALES
    # ---------------------------------------------------------------------
    # Both scales every time rather than whichever VAD_TARGET_SCOPE happens to
    # be set to. The gap between them IS the answer to the handoff's §7
    # question -- if raw is poor and dialog_z is decent, the ceiling was the
    # normalisation mismatch and not the acoustics, and that is the single
    # most useful thing this cell can tell you.
    # =====================================================================
    if len(vad_df):
        _vad_feature_cols = [c for c in feature_cols if c in vad_df.columns]
        print(f"regressing on {len(_vad_feature_cols)} features, "
              f"{len(vad_df)} IEMOCAP utterances, "
              f"{vad_df['session'].nunique()} session folds\n")

        _rows, vad_fold_detail = [], {}
        for _scope in ("raw", "dialog_z"):
            _scoped = _apply_target_scope(vad_df, VAD_DIMENSIONS, _scope)
            for _dim in VAD_DIMENSIONS:
                print(f"=== {_dim}  (targets: {_scope}) ===")
                _folds, _pooled = run_vad_cv(
                    _scoped, _vad_feature_cols, _dim, VAD_RF_KWARGS)
                _pooled["target_scope"] = _scope
                _pooled["ccc_fold_std"] = round(float(_folds["ccc"].std()), 4)
                _rows.append(_pooled)
                vad_fold_detail[(_scope, _dim)] = _folds
                print()

        vad_cv_df = pd.DataFrame(_rows)[
            ["dimension", "target_scope", "ccc", "ccc_fold_std", "pearson_r",
             "mae", "proxy_ccc", "proxy_ccc_uncal", "n"]].round(4)

        print("=" * 72)
        print("VAD regression, leave-one-IEMOCAP-session-out")
        print(vad_cv_df.to_string(index=False))

        # ---- the arousal comparison this whole exercise turns on ----
        print(f"\n{'=' * 72}\ntrained arousal vs the renderer's existing proxy")
        for _sc in ("raw", "dialog_z"):
            _r = vad_cv_df[(vad_cv_df["dimension"] == "arousal")
                           & (vad_cv_df["target_scope"] == _sc)]
            if not len(_r):
                continue
            _t = float(_r["ccc"].iloc[0])
            _p = float(_r["proxy_ccc"].iloc[0])
            if np.isnan(_p):
                print(f"  {_sc:9s}: trained CCC={_t:.3f}, proxy unavailable "
                      f"(configured columns missing -- see PROXY_AROUSAL_"
                      f"FEATURES)")
                continue
            _d = _t - _p
            _verdict = ("BEATS the proxy" if _d >= VAD_MUST_BEAT_PROXY_BY
                        else "does NOT clear the margin over the proxy")
            _u = float(_r["proxy_ccc_uncal"].iloc[0])
            print(f"  {_sc:9s}: trained CCC={_t:.3f}  proxy CCC={_p:.3f}  "
                  f"delta={_d:+.3f}  -> {_verdict}")
            if not np.isnan(_u) and abs(_p - _u) > 0.05:
                print(f"             (proxy uncalibrated would score {_u:+.3f} "
                      f"-- the gap is offset/scale, not shape. The calibrated "
                      f"number is the honest one; the renderer z-scores the "
                      f"proxy across the clip, which is this same fit.)")

        # ---- did the normalisation mismatch cap anything? ----
        print(f"\n{'=' * 72}\nraw vs dialog_z (the §7 question)")
        for _dim in VAD_DIMENSIONS:
            _raw = vad_cv_df[(vad_cv_df["dimension"] == _dim)
                             & (vad_cv_df["target_scope"] == "raw")]["ccc"]
            _dz = vad_cv_df[(vad_cv_df["dimension"] == _dim)
                            & (vad_cv_df["target_scope"] == "dialog_z")]["ccc"]
            if len(_raw) and len(_dz):
                _r0, _d0 = float(_raw.iloc[0]), float(_dz.iloc[0])
                print(f"  {_dim:10s} raw={_r0:+.3f}  dialog_z={_d0:+.3f}  "
                      f"delta={_d0 - _r0:+.3f}")
        print("  A large positive delta means absolute-vs-relative was the "
              "ceiling, not the acoustics. Since segment_arousal_floor already "
              "consumes arousal clip-relatively, dialog_z is the number that "
              "predicts on-screen behaviour; raw is the one comparable to "
              "published CCCs.")
        print("  Check ccc_fold_std before believing any gap. Five folds, and "
              "a 0.03 delta on a 0.08 std is not a result.")
    else:
        vad_cv_df = pd.DataFrame()
        print("no VAD data -- skipping regression CV")


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

    # =====================================================================
    # CELL 12c — FIT FINAL REGRESSORS, SAVE ALONGSIDE THE CLASSIFIER
    # ---------------------------------------------------------------------
    # Separate bundle keys. The 4 keys TranscriberV19 CELL 6 reads (clf /
    # feature_cols / extractor / speaker_normalised) and every provenance key
    # CELL 12 wrote are untouched, so an old loader reads this file exactly as
    # before and a new one can look for `vad_models` and not find it without
    # anything breaking.
    #
    # Run this AFTER CELL 12 -- it re-reads and extends `bundle`.
    # =====================================================================
    if len(vad_df) and len(vad_cv_df):
        _scoped_final = _apply_target_scope(vad_df, VAD_DIMENSIONS,
                                            VAD_TARGET_SCOPE)
        _cols = [c for c in feature_cols if c in vad_df.columns]

        vad_models, vad_meta = {}, {}
        for _dim in VAD_DIMENSIONS:
            _row = vad_cv_df[(vad_cv_df["dimension"] == _dim)
                             & (vad_cv_df["target_scope"] == VAD_TARGET_SCOPE)]
            _ccc = float(_row["ccc"].iloc[0]) if len(_row) else float("nan")
            _proxy = float(_row["proxy_ccc"].iloc[0]) if len(_row) else float("nan")
            _std = float(_row["ccc_fold_std"].iloc[0]) if len(_row) else float("nan")

            _ok = bool(_ccc >= VAD_CCC_MIN_FOR_STYLING)
            _why = ("" if _ok else
                    f"CCC {_ccc:.3f} < floor {VAD_CCC_MIN_FOR_STYLING}")
            if _ok and _dim == "arousal" and not np.isnan(_proxy):
                if (_ccc - _proxy) < VAD_MUST_BEAT_PROXY_BY:
                    _ok = False
                    _why = (f"clears the CCC floor but only beats the existing "
                            f"hand-weighted proxy by {_ccc - _proxy:+.3f} "
                            f"(needs {VAD_MUST_BEAT_PROXY_BY:+.3f}) -- not "
                            f"worth replacing a working heuristic")

            _reg = RandomForestRegressor(**VAD_RF_KWARGS)
            _reg.fit(_scoped_final[_cols].to_numpy(dtype=float),
                     _scoped_final[_dim].to_numpy(dtype=float))
            vad_models[_dim] = _reg
            vad_meta[_dim] = {
                "cv_ccc": _ccc,
                "cv_ccc_fold_std": _std,
                "cv_proxy_ccc": _proxy,
                "recommended_for_styling": _ok,
                "not_recommended_because": _why,
                "target_scope": VAD_TARGET_SCOPE,
                "target_range": ("1-5 as annotated" if VAD_TARGET_SCOPE == "raw"
                                 else "z-scored within dialog"),
            }

        bundle["vad_models"] = vad_models
        bundle["vad_feature_cols"] = list(_cols)
        bundle["vad_meta"] = vad_meta
        bundle["vad_target_scope"] = VAD_TARGET_SCOPE
        bundle["vad_n_rows"] = int(len(vad_df))
        bundle["vad_cv_protocol"] = ("leave-one-IEMOCAP-session-out; "
                                     "IEMOCAP only, no RAVDESS support set")
        bundle["vad_ccc_floor"] = float(VAD_CCC_MIN_FOR_STYLING)

        # ---- the warning that has to travel WITH the model ----------------
        # TranscriberV19's shout gate calls _arousal_to_z(), which inverts
        # segment_arousal_floor's tanh squash and therefore assumes its input
        # sits in [0, SEGMENT_AROUSAL_FLOOR] (0.45 by default). A prediction
        # from this regressor does not: it is a 1-5 rating, or a z-score.
        # Feeding one in unconverted gives frac = 3.0/0.45 = 6.7, clipped to
        # 0.999999, arctanh -> z ~ +10.9 for EVERY segment. The gate does not
        # error. It opens completely, and every angry segment renders in
        # capitals regardless of delivery.
        #
        # This lives in the bundle rather than only in a document because the
        # bundle is what someone will still have in six months.
        bundle["vad_scale_warning"] = (
            f"pred_arousal is on the '{VAD_TARGET_SCOPE}' scale "
            f"({vad_meta['arousal']['target_range']}). It is NOT on the "
            f"[0, SEGMENT_AROUSAL_FLOOR] scale that TranscriberV19's "
            f"arousal_floor uses. Do not substitute it into arousal_floor "
            f"without rescaling: _arousal_to_z() would arctanh an "
            f"out-of-range value, saturate at ~+11 sigma for every segment, "
            f"and silently open the ALL-CAPS gate. Two consumers read "
            f"arousal_floor -- saturation (assign_styles) and the shout gate "
            f"(apply_emotion_shout) -- so a swap affects both."
        )

        joblib.dump(bundle, MODEL_OUT)

        print(f"extended {MODEL_OUT} with VAD regressors")
        print(f"  target scope : {VAD_TARGET_SCOPE}")
        print(f"  rows         : {len(vad_df)} IEMOCAP utterances")
        print(f"  features     : {len(_cols)}")
        for _dim, _m in vad_meta.items():
            _flag = "STYLE-READY" if _m["recommended_for_styling"] else "hold"
            print(f"  {_dim:10s} CCC={_m['cv_ccc']:.3f} "
                  f"(fold std {_m['cv_ccc_fold_std']:.3f})  -> {_flag}")
            if _m["not_recommended_because"]:
                print(f"             {_m['not_recommended_because']}")
        print(f"\n  classifier keys untouched: existing loaders read this file "
              f"exactly as before.")
        print(f"  {bundle['vad_scale_warning']}")
    else:
        print("no VAD models fitted -- nothing to save")


    mo.md("""
    ## Wiring this into TranscriberV17

    V17 CELL 6 hardcodes `model_bundle_path = "outputs/clf_v2.joblib"`. Either
    change that to `"outputs/clf_v3.joblib"` (keeps both on disk, so a
    regression is one string away from being undone), or copy clf_v3 over
    clf_v2 (nothing to edit, nothing to roll back to).

    ### Three things to check on the first run

    - **`EXTRACTOR` has to match.** V17 builds its inference vector with
      `opensmile.FeatureSet.eGeMAPSv02` / `FeatureLevel.Functionals`, and this
      script uses the same constructor so the column names line up. Train with
      `praat14` and hand it to a V17 expecting `egemaps` and you get silent
      garbage rather than an error, since V17 reads `extractor` from the bundle
      and follows whatever this script declares.
    - **The label space changed, so `EMOTION_STYLES` needs updating.** It was
      built around RAVDESS names. This model can emit `frustrated` and (unless
      merged) `excited`, which RAVDESS never had, and can't emit `calm` any
      more. A label with no entry falls through to the default branch.
    - **Chance level moved.** V17 has a comment reasoning about "7 classes,
      ~60.6% accuracy" and a 0.40/0.50 confidence threshold tuned against that.
      If this model ships a different class count, that threshold is calibrated
      against a distribution that no longer exists. `confidence_scale` takes the
      class count as an argument for this reason. Re-check the shout triggers.

    ### Not in here, deliberately

    - No `calm`, and no folding of `calm` into `neutral`.
    - No pooled train/test accuracy as the headline number, only
      IEMOCAP-held-out. See CELL 9.
    - No RAVDESS song files.
    - No claim that combining helped. CELL 10 measures it. Read the sweep
      before setting `RAVDESS_WEIGHT`, and compare any gap against
      `fold_acc_std`.

    ### Also in this version

    - **CELL 9c/9d** — a RAVDESS-only model, trained on nothing but RAVDESS,
      scored on the same 5 IEMOCAP folds CELL 9/10 use. This is the honest
      comparison to the archived 46.8% figure, which was measured on RAVDESS
      itself. It reports both the full 8-class number (frustrated/excited
      count as always-wrong, since RAVDESS never had them) and a number
      restricted to the 6 classes RAVDESS actually knows.
    - **CELL 10b** — picks `RAVDESS_WEIGHT` by worst-case fold delta against
      the w=0.0 baseline, not by pooled macro-F1 argmax. Pooled argmax can be
      won by one outlier fold; this instead finds the weight where no single
      IEMOCAP session gets worse than IEMOCAP-only.
    - `RAVDESS_WEIGHT_SWEEP` in CELL 1 was widened to fine steps between
      0.0 and 0.3, since that's the range where the coarse sweep showed the
      interesting behavior (every fold improving at 0.15, session 5 starting
      to regress by 0.3).
    """)
    return (
        OUT_DIR,
        RAVDESS_WEIGHT,
        confusion_matrix,
        keep_labels,
        mo,
        norm_df,
        np,
        os,
        pd,
        rav_only_full,
        rav_only_overlap,
        robustness_df,
        sweep_detail,
        sweep_df,
    )


@app.cell
def _(
    OUT_DIR,
    RAVDESS_WEIGHT,
    confusion_matrix,
    keep_labels,
    mo,
    norm_df,
    np,
    os,
    pd,
    rav_only_full,
    rav_only_overlap,
    robustness_df,
    sweep_detail,
    sweep_df,
):
    # =====================================================================
    # CELL 13 — WEIGHT-SWEEP FIGURES FOR THE WRITE-UP
    # ---------------------------------------------------------------------
    # Renders what CELL 9d / 10 / 10b already computed. Fits nothing, trains
    # nothing, re-extracts nothing, so it is cheap to re-run while fiddling
    # with styling.
    #
    # Reads (all defined upstream, none of them modified here):
    #   OUT_DIR, sweep_df, sweep_detail, robustness_df, rav_only_full,
    #   rav_only_overlap, norm_df, keep_labels, RAVDESS_WEIGHT
    #   + os / np / pd / confusion_matrix / mo from CELL 0
    #
    # Naming: everything this cell puts into the notebook namespace is
    # V4_/v4_ prefixed, and every temporary is underscore-prefixed (marimo
    # treats leading-underscore names as cell-local), so nothing here can
    # collide with a name an earlier cell already owns.
    #
    # Writes to outputs/figures, 300-dpi PNG (to look at) + vector PDF (to
    # put in the paper), plus the CSV behind each figure:
    #   fig1_weight_sweep       macro-F1 and accuracy vs w, with a +/-1 SD
    #                           across-session band so a 0.01 gap is visibly
    #                           inside the noise
    #   fig2_per_session_delta  each held-out session's macro-F1 change vs
    #                           the w=0 baseline, plus the worst-case curve
    #                           that CELL 10b actually selects on
    #   fig3_headline           RAVDESS-only vs IEMOCAP-only vs both
    #   fig4_per_class_recall   which classes the gain landed in
    # =====================================================================
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ---------------------------------------------------------------------
    # House style. rc_context rather than rcParams.update, so re-running this
    # cell can't leave the notebook's global matplotlib state modified.
    # pdf.fonttype 42 embeds TrueType instead of outlining the glyphs, which
    # keeps the text selectable and searchable in the submitted PDF.
    # ---------------------------------------------------------------------
    V4_FIG_DIR = f"{OUT_DIR}/figures"
    os.makedirs(V4_FIG_DIR, exist_ok=True)

    V4_RC = {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",      # "serif" to match a LaTeX body font
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "lines.linewidth": 1.4,
        "lines.solid_capstyle": "round",
    }

    # Okabe-Ito. Distinguishable in the three common colour-vision
    # deficiencies and, more to the point for a journal, in greyscale print.
    V4_C = {
        "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
        "grey": "#7A7A7A", "ink": "#1A1A1A",
    }
    V4_SESSION_COLOURS = [V4_C["blue"], V4_C["orange"], V4_C["green"],
                          V4_C["purple"], V4_C["sky"]]


    def v4_save(fig, stem):
        """PDF for the paper, PNG for the screen."""
        _paths = []
        for _ext in ("pdf", "png"):
            _p = f"{V4_FIG_DIR}/{stem}.{_ext}"
            fig.savefig(_p, dpi=300, bbox_inches="tight")
            _paths.append(_p)
        return _paths


    def v4_recall_table(y_true, y_pred, labels):
        """Per-class recall over the pooled leave-one-session-out predictions."""
        _cm = confusion_matrix(y_true, y_pred, labels=labels)
        _sup = _cm.sum(axis=1)
        return pd.DataFrame({
            "class": labels,
            "support": _sup,
            "recall": np.diag(_cm) / np.where(_sup == 0, 1, _sup),
            "times_predicted": _cm.sum(axis=0),
        })


    # =====================================================================
    # Reshape the sweep into two tidy frames
    # =====================================================================
    _v4_curve_rows, _v4_fold_rows = [], []
    for _v4_w in sorted(sweep_detail):
        _v4_f = sweep_detail[_v4_w][0]
        _v4_pooled = sweep_df.loc[sweep_df["ravdess_weight"] == _v4_w]
        _v4_curve_rows.append({
            "w": float(_v4_w),
            "pooled_macro_f1": float(_v4_pooled["pooled_macro_f1"].iloc[0]),
            "pooled_accuracy": float(_v4_pooled["pooled_accuracy"].iloc[0]),
            "fold_f1_mean": float(_v4_f["macro_f1"].mean()),
            "fold_f1_std": float(_v4_f["macro_f1"].std(ddof=1)),
            "fold_acc_mean": float(_v4_f["accuracy"].mean()),
            "fold_acc_std": float(_v4_f["accuracy"].std(ddof=1)),
        })
        for _v4_r in _v4_f.to_dict("records"):
            _v4_fold_rows.append({
                "w": float(_v4_w),
                "session": int(_v4_r["held_out_session"]),
                "macro_f1": float(_v4_r["macro_f1"]),
                "accuracy": float(_v4_r["accuracy"]),
            })

    v4_curve = pd.DataFrame(_v4_curve_rows).sort_values("w").reset_index(drop=True)
    v4_folds = pd.DataFrame(_v4_fold_rows)

    _v4_base = v4_folds[v4_folds["w"] == 0.0].set_index("session")
    v4_folds["d_macro_f1"] = (v4_folds["macro_f1"]
                              - v4_folds["session"].map(_v4_base["macro_f1"]))
    v4_folds["d_accuracy"] = (v4_folds["accuracy"]
                              - v4_folds["session"].map(_v4_base["accuracy"]))

    # The weight CELL 10b's rule selects: no session regresses, best mean
    # gain among those. Falls back to pooled argmax if nothing is clean.
    _v4_clean = robustness_df[robustness_df["n_sessions_regressed"] == 0]
    if len(_v4_clean):
        v4_pick_w = float(_v4_clean.loc[_v4_clean["mean_session_delta"].idxmax(),
                                        "ravdess_weight"])
    else:
        v4_pick_w = float(sweep_df.loc[sweep_df["pooled_macro_f1"].idxmax(),
                                       "ravdess_weight"])
    # snap to an actual sweep key, so float equality can't miss
    v4_pick_w = min(sweep_detail, key=lambda k: abs(float(k) - v4_pick_w))
    v4_clean_weights = sorted(float(x) for x in _v4_clean["ravdess_weight"])

    # If the honest pick is 0.0, figures 3 and 4 would compare a model with
    # itself. Show the best non-zero weight instead and say so on the axis.
    _v4_nonzero = [k for k in sorted(sweep_detail) if float(k) > 0]
    if float(v4_pick_w) > 0 or not _v4_nonzero:
        v4_compare_w = v4_pick_w
        v4_compare_note = "selected by worst-case fold (CELL 10b)"
    else:
        v4_compare_w = max(_v4_nonzero, key=lambda k: float(
            sweep_df.loc[sweep_df["ravdess_weight"] == k,
                         "pooled_macro_f1"].iloc[0]))
        v4_compare_note = "best non-zero w — NOT recommended, it regresses a fold"

    v4_n_classes = int(norm_df["label"].nunique())
    v4_chance = 1.0 / v4_n_classes
    v4_rav_labels = set(norm_df.loc[norm_df["source"] == "ravdess",
                                    "label"].unique())


    # =====================================================================
    # FIGURE 1 — the sweep itself
    # =====================================================================
    def v4_fig_sweep(curve, pick_w):
        _b = curve.loc[curve["w"] == 0.0].iloc[0]
        _specs = [
            ("(a) macro-F1", "fold_f1_mean", "fold_f1_std", "pooled_macro_f1"),
            ("(b) accuracy", "fold_acc_mean", "fold_acc_std", "pooled_accuracy"),
        ]
        with plt.rc_context(V4_RC):
            fig, _axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
            for _ax, (_name, _m, _s, _p) in zip(_axs, _specs):
                _ax.fill_between(curve["w"], curve[_m] - curve[_s],
                                 curve[_m] + curve[_s],
                                 color=V4_C["blue"], alpha=0.14, lw=0)
                _ax.plot(curve["w"], curve[_m], color=V4_C["blue"],
                         marker="o", ms=3.0, zorder=4)
                _ax.plot(curve["w"], curve[_p], color=V4_C["ink"], lw=1.0,
                         ls=(0, (1, 1.7)), zorder=3)
                _ax.axhline(float(_b[_m]), color=V4_C["grey"], lw=0.9,
                            ls="--", zorder=1)
                _ax.axvline(pick_w, color=V4_C["red"], lw=0.9, ls=":", zorder=1)
                _ax.set_title(_name)
                _ax.set_xlabel("RAVDESS sample weight $w$")
                _ax.set_xticks(curve["w"].tolist(), minor=True)
                _ax.margins(x=0.03)
                _ax.annotate(f"IEMOCAP-only ({float(_b[_m]):.3f})",
                             xy=(curve["w"].max(), float(_b[_m])),
                             xytext=(-4, -3), textcoords="offset points",
                             ha="right", va="top", fontsize=6.4,
                             color=V4_C["grey"])
            _axs[0].annotate(f"$w$ = {pick_w:g}", xy=(pick_w, 1.0),
                             xycoords=("data", "axes fraction"),
                             xytext=(3, -9), textcoords="offset points",
                             fontsize=6.6, color=V4_C["red"])
            _handles = [
                Line2D([], [], color=V4_C["blue"], marker="o", ms=3.0,
                       label="mean over the 5 held-out sessions"),
                Patch(facecolor=V4_C["blue"], alpha=0.14,
                      label="$\\pm$1 SD across sessions"),
                Line2D([], [], color=V4_C["ink"], lw=1.0, ls=(0, (1, 1.7)),
                       label="pooled (micro-average)"),
                Line2D([], [], color=V4_C["grey"], lw=0.9, ls="--",
                       label="IEMOCAP-only baseline ($w=0$)"),
                Line2D([], [], color=V4_C["red"], lw=0.9, ls=":",
                       label="selected $w$"),
            ]
            fig.legend(handles=_handles, ncol=3, loc="upper center",
                       bbox_to_anchor=(0.5, 0.06))
            fig.tight_layout()
        return fig


    # =====================================================================
    # FIGURE 2 — per-session deltas, and the worst case they imply
    # =====================================================================
    def v4_fig_session_delta(folds, pick_w, clean_ws):
        _sessions = sorted(folds["session"].unique())
        _worst = folds.groupby("w")["d_macro_f1"].min().sort_index()
        with plt.rc_context(V4_RC):
            fig, _ax = plt.subplots(figsize=(6.6, 3.3))
            for _i, _s in enumerate(_sessions):
                _d = folds[folds["session"] == _s].sort_values("w")
                _ax.plot(_d["w"], _d["d_macro_f1"], marker="o", ms=2.6, lw=1.1,
                         alpha=0.9,
                         color=V4_SESSION_COLOURS[_i % len(V4_SESSION_COLOURS)],
                         label=f"session {_s}")
            _ax.plot(_worst.index, _worst.values, color=V4_C["ink"], lw=2.0,
                     marker="s", ms=3.0, zorder=5,
                     label="worst session (selection criterion)")
            _ax.axhline(0.0, color=V4_C["ink"], lw=0.9)
            _ax.axvline(pick_w, color=V4_C["red"], lw=0.9, ls=":", zorder=1)

            _lo, _hi = _ax.get_ylim()
            _ax.axhspan(0.0, _hi, color=V4_C["green"], alpha=0.05, lw=0,
                        zorder=0)
            if clean_ws:
                _ax.plot(clean_ws, [_lo + 0.02 * (_hi - _lo)] * len(clean_ws),
                         ls="none", marker="s", ms=3.4, color=V4_C["green"],
                         zorder=5, label="no session regressed (axis marks)")
            _ax.set_ylim(_lo, _hi)
            _ax.set_xlabel("RAVDESS sample weight $w$")
            _ax.set_ylabel("$\\Delta$ macro-F1 vs IEMOCAP-only")
            _ax.set_title("Per-session change from adding RAVDESS")
            _ax.set_xticks(sorted(folds["w"].unique()), minor=True)
            _ax.margins(x=0.03)
            _ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
                       borderaxespad=0.0)
            fig.tight_layout()
        return fig


    # =====================================================================
    # FIGURE 3 — the headline comparison
    # =====================================================================
    def v4_fig_headline(curve, compare_w, note):
        _row = curve.loc[curve["w"] == float(compare_w)].iloc[0]
        _b = curve.loc[curve["w"] == 0.0].iloc[0]
        _models = [
            ("RAVDESS-only\n(trained on acted speech)",
             rav_only_full["accuracy"], rav_only_full["macro_f1"], False),
            (f"RAVDESS-only\n(its {len(v4_rav_labels)} classes only)",
             rav_only_overlap["accuracy"], rav_only_overlap["macro_f1"], True),
            ("IEMOCAP-only\n($w=0$)",
             float(_b["pooled_accuracy"]), float(_b["pooled_macro_f1"]), False),
            (f"IEMOCAP + RAVDESS\n($w={float(compare_w):g}$)",
             float(_row["pooled_accuracy"]), float(_row["pooled_macro_f1"]),
             False),
        ]
        _x = np.arange(len(_models))
        _wid = 0.38
        with plt.rc_context(V4_RC):
            fig, _ax = plt.subplots(figsize=(6.2, 3.1))
            _b1 = _ax.bar(_x - _wid / 2, [m[1] for m in _models], _wid,
                          color=V4_C["blue"], label="accuracy",
                          edgecolor="white", linewidth=0.4)
            _b2 = _ax.bar(_x + _wid / 2, [m[2] for m in _models], _wid,
                          color=V4_C["orange"], label="macro-F1",
                          edgecolor="white", linewidth=0.4)
            for _i, _m in enumerate(_models):
                if _m[3]:
                    for _bar in (_b1[_i], _b2[_i]):
                        _bar.set_hatch("///")
                        _bar.set_alpha(0.55)
            _ax.axhline(v4_chance, color=V4_C["grey"], lw=0.9, ls="--")
            _ax.annotate(f"chance\n({v4_chance:.3f})",
                         xy=(1.0, v4_chance),
                         xycoords=("axes fraction", "data"), xytext=(5, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=6.4, color=V4_C["grey"],
                         annotation_clip=False)
            for _c in (_b1, _b2):
                if hasattr(_ax, "bar_label"):
                    _ax.bar_label(_c, fmt="%.3f", padding=1.5, fontsize=6.3)
            _ax.set_xticks(_x)
            _ax.set_xticklabels([m[0] for m in _models])
            _ax.set_ylabel("score")
            _ax.set_ylim(0, max(max(m[1], m[2]) for m in _models) * 1.22)
            _ax.set_title("Scored on the same IEMOCAP folds")
            _ax.legend(loc="upper left", ncol=2)
            _ax.annotate(
                "hatched = restricted test subset, different denominator"
                f"\nselected $w$: {note}",
                xy=(0.0, -0.30), xycoords="axes fraction", fontsize=6.2,
                color=V4_C["grey"], va="top")
            fig.tight_layout()
        return fig


    # =====================================================================
    # FIGURE 4 — where the change landed, class by class
    # =====================================================================
    def v4_fig_per_class(base_tbl, cmp_tbl, compare_w):
        _t = base_tbl.merge(cmp_tbl, on="class", suffixes=("_base", "_cmp"))
        _t = _t.sort_values("recall_base").reset_index(drop=True)
        _y = np.arange(len(_t))
        with plt.rc_context(V4_RC):
            fig, _ax = plt.subplots(figsize=(6.0, 3.5))
            for _i, _r in _t.iterrows():
                _up = _r["recall_cmp"] >= _r["recall_base"]
                _ax.annotate(
                    "", xy=(_r["recall_cmp"], _i), xytext=(_r["recall_base"], _i),
                    arrowprops=dict(
                        arrowstyle="-|>", shrinkA=0, shrinkB=0, lw=1.6,
                        color=V4_C["green"] if _up else V4_C["red"]))
            _ax.scatter(_t["recall_base"], _y, s=26, facecolors="white",
                        edgecolors=V4_C["ink"], linewidths=1.0, zorder=5,
                        label="IEMOCAP-only ($w=0$)")
            _ax.scatter(_t["recall_cmp"], _y, s=26, color=V4_C["ink"],
                        zorder=5, label=f"+ RAVDESS ($w={float(compare_w):g}$)")
            _ax.set_yticks(_y)
            _ax.set_yticklabels([
                f"{_c}$^\\ast$" if _c in v4_rav_labels else _c
                for _c in _t["class"]])
            for _i, _r in _t.iterrows():
                _ax.annotate(f"n={int(_r['support_base'])}",
                             xy=(1.0, _i), xycoords=("axes fraction", "data"),
                             xytext=(4, 0), textcoords="offset points",
                             va="center", fontsize=6.2, color=V4_C["grey"],
                             annotation_clip=False)
            _ax.set_xlabel("recall, pooled over the 5 held-out sessions")
            _ax.set_title("Per-class recall, IEMOCAP-only $\\rightarrow$ + RAVDESS")
            _ax.set_xlim(0, max(1.0, float(_t[["recall_base",
                                               "recall_cmp"]].max().max()) * 1.08))
            _ax.grid(axis="y", visible=False)
            _ax.legend(loc="upper left")
            _ax.annotate("$^\\ast$ class RAVDESS can support; the rest are "
                         "IEMOCAP-only",
                         xy=(0.0, -0.16), xycoords="axes fraction",
                         fontsize=6.2, color=V4_C["grey"], va="top")
            fig.tight_layout()
        return fig


    # =====================================================================
    # Render, save, report
    # =====================================================================
    _v4_yt_base, _v4_yp_base = sweep_detail[0.0][1], sweep_detail[0.0][2]
    _v4_yt_cmp, _v4_yp_cmp = (sweep_detail[v4_compare_w][1],
                              sweep_detail[v4_compare_w][2])
    _v4_labels = [_l for _l in keep_labels if _l in set(_v4_yt_base)]
    v4_recall_base = v4_recall_table(_v4_yt_base, _v4_yp_base, _v4_labels)
    v4_recall_cmp = v4_recall_table(_v4_yt_cmp, _v4_yp_cmp, _v4_labels)

    v4_figs = {
        "fig1_weight_sweep": v4_fig_sweep(v4_curve, v4_pick_w),
        "fig2_per_session_delta": v4_fig_session_delta(
            v4_folds, v4_pick_w, v4_clean_weights),
        "fig3_headline": v4_fig_headline(v4_curve, v4_compare_w,
                                         v4_compare_note),
        "fig4_per_class_recall": v4_fig_per_class(
            v4_recall_base, v4_recall_cmp, v4_compare_w),
    }
    v4_fig_paths = {_k: v4_save(_f, _k) for _k, _f in v4_figs.items()}

    # the numbers behind the figures, so the paper's table and its plots
    # cannot drift apart
    v4_curve.round(4).to_csv(f"{V4_FIG_DIR}/sweep_pooled.csv", index=False)
    v4_folds.round(4).to_csv(f"{V4_FIG_DIR}/sweep_per_session.csv", index=False)
    (v4_recall_base.merge(v4_recall_cmp, on="class", suffixes=("_base", "_cmp"))
     .round(4).to_csv(f"{V4_FIG_DIR}/per_class_recall.csv", index=False))

    _v4_row = v4_curve.loc[v4_curve["w"] == float(v4_compare_w)].iloc[0]
    _v4_b = v4_curve.loc[v4_curve["w"] == 0.0].iloc[0]
    _v4_worst = float(v4_folds[v4_folds["w"] == float(v4_compare_w)]
                      ["d_macro_f1"].min())

    print(f"figures -> {V4_FIG_DIR}")
    for _k, _v in v4_fig_paths.items():
        print(f"  {_k:24s} {_v[0]}  (+ .png)")
    print(f"\nselected w = {v4_pick_w:g}  ({v4_compare_note})")
    print(f"  macro-F1  {float(_v4_b['pooled_macro_f1']):.4f} -> "
          f"{float(_v4_row['pooled_macro_f1']):.4f}  "
          f"({float(_v4_row['pooled_macro_f1']) - float(_v4_b['pooled_macro_f1']):+.4f})")
    print(f"  accuracy  {float(_v4_b['pooled_accuracy']):.4f} -> "
          f"{float(_v4_row['pooled_accuracy']):.4f}  "
          f"({float(_v4_row['pooled_accuracy']) - float(_v4_b['pooled_accuracy']):+.4f})")
    print(f"  worst single session: {_v4_worst:+.4f} macro-F1")
    print(f"  across-session SD at that w: "
          f"{float(_v4_row['fold_f1_std']):.4f} (macro-F1)")
    if abs(float(_v4_row['pooled_macro_f1'])
           - float(_v4_b['pooled_macro_f1'])) < float(_v4_row['fold_f1_std']):
        print("  NOTE: the gain is smaller than one across-session SD. Say so "
              "in the caption rather than letting the reader assume otherwise.")
    if float(RAVDESS_WEIGHT) != float(v4_pick_w):
        print(f"\n  CELL 1 currently has RAVDESS_WEIGHT={RAVDESS_WEIGHT}, the "
              f"figures mark w={v4_pick_w:g}. Reconcile before CELL 12.")

    _v4_caps = f"""
    ### Draft captions

    **Figure 1.** Emotion-classification performance against the weight $w$
    applied to RAVDESS rows at fit time, under leave-one-IEMOCAP-session-out
    cross-validation ({v4_n_classes} classes, chance {v4_chance:.3f}). Solid
    line: mean over the five held-out sessions; band: $\\pm$1 SD across those
    sessions; dotted: pooled micro-average. $w=0$ is the IEMOCAP-only
    baseline (dashed). RAVDESS is present in every training fold and in no
    test fold.

    **Figure 2.** Change in macro-F1 for each held-out IEMOCAP session
    relative to the $w=0$ baseline. The bold line is the worst-performing
    session at each weight, which is the quantity the weight is selected on:
    a pooled improvement carried by one session while another regresses is
    not a result we act on. Green markers mark weights at which no session
    regressed.

    **Figure 3.** The same five folds, three training regimes. The
    RAVDESS-only model is trained exclusively on acted speech and never sees
    IEMOCAP; it cannot emit `frustrated` or `excited`, which the full-label-
    space bars count as errors. The hatched pair restricts the test set to
    the {len(v4_rav_labels)} classes RAVDESS has a concept of and is therefore
    computed over a different denominator.

    **Figure 4.** Per-class recall pooled across the five held-out sessions,
    IEMOCAP-only (hollow) to $w={float(v4_compare_w):g}$ (filled). Starred
    classes are the ones RAVDESS can supply examples for; the remainder are
    learned from IEMOCAP alone and serve as a control on whether the support
    set is helping specifically where it adds data.
    """

    mo.vstack([
        mo.md("## Weight-sweep figures"),
        mo.as_html(v4_figs["fig1_weight_sweep"]),
        mo.as_html(v4_figs["fig2_per_session_delta"]),
        mo.as_html(v4_figs["fig3_headline"]),
        mo.as_html(v4_figs["fig4_per_class_recall"]),
        mo.md(_v4_caps),
    ])

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

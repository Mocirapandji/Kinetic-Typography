import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # =====================================================================
    # TranscriberV9: V8 pipeline + the eGeMAPS / speaker-normalised model
    # Only the classifier PATH changes vs V8. Budget, motion, styling and
    # rendering are identical. V9 loads clf_v2.joblib (eGeMAPS 88, speaker-
    # normalised, 60.6% speaker-independent) and feeds each segment the
    # features that model expects, normalising across the video's own
    # segments when there are enough of them to do so safely.
    # =====================================================================

    # =====================================================================
    # CELL 0 — IMPORTS (all imports live here, marimo convention)
    # =====================================================================
    import marimo as mo
    import os
    import colorsys
    import subprocess
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    import whisperx
    from sklearn.ensemble import RandomForestClassifier
    import json
    import opensmile          # NEW in V9: eGeMAPS extraction
    import joblib             # NEW in V9: load the saved clf_v2 bundle

    mo.md(r"""
    # TranscriberV9: better classifier, same expressive pipeline

    V8 answered "can this run on any real video" — extract audio, transcribe,
    per-word prosody, salience budget, per-segment emotion, burn onto the
    footage. V9 changes exactly one thing inside that: the emotion classifier.

    V8 shipped the 46.8% 14-feature RAVDESS model. `train_emotion_v2.py`
    established that **eGeMAPS (88 features) + per-speaker normalisation**
    reaches 60.6% speaker-independent — +13.8 points, about two-thirds of the
    way to the ~67% human ceiling. V9 loads that model (`clf_v2.joblib`) and
    feeds each WhisperX segment eGeMAPS features instead of the 14 Praat ones.

    Two consequences the code has to handle, both flagged in the decision log
    at the bottom:

    - eGeMAPS is extracted per segment via openSMILE, so a segment is written
      to a temp wav (openSMILE reads files). Slower than V8's in-memory Praat.
    - The model was trained on speaker-normalised features, so V9 normalises
      each segment's features across the video's segments before classifying.
      That needs several segments to be meaningful, so `SEGMENT_NORM="auto"`
      only normalises when enough segments exist, and falls back to raw
      eGeMAPS otherwise. On a single-emotion clip, normalising would subtract
      the emotion out — set `SEGMENT_NORM="off"` for those.
    """)

    # =====================================================================
    # CELL 1 — CONFIG + DATASET SWITCH
    # =====================================================================

    # ---------------- DATASET SWITCH ----------------
    # "ravdess" -> acted studio speech (what the model was trained on)
    # "iemocap" -> conversational speech (the harder, realistic test)
    DATASET = "iemocap"

    ravdess_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    iemocap_dir = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"
    data_dir = ravdess_dir  # Part A's glob still uses this if the cache is missing

    emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_emotions = {"calm"}

    # ---------------- MODEL SWITCH ----------------
    features_csv = "outputs/features.csv"              # RAVDESS-only 14-feature cache
    # features_csv = "outputs/features_combined.csv"   # RAVDESS + IEMOCAP model

    # IEMOCAP labels CSV (built by extract_iemocap.py); local first, T7 fallback.
    iemocap_csv = "outputs/features_iemocap.csv"
    if not os.path.exists(iemocap_csv):
        iemocap_csv = "/run/media/s5812886/T7 Shield/kinetic_outputs/features_iemocap.csv"

    device = "cpu"
    compute_type = "int8"

    # ---------------- TEST CLIP ----------------
    if DATASET == "ravdess":
        audio_file = f"{ravdess_dir}/Actor_01/03-01-06-01-02-01-01.wav"
    else:
        _utt = "Ses01F_impro01_F012"   # paste any "file" value from CELL 2's table
        _utt = _utt.replace(".wav", "")
        _sess = f"Session{int(_utt[3:5])}"
        _dialog = _utt.rsplit("_", 1)[0]
        audio_file = f"{iemocap_dir}/{_sess}/sentences/wav/{_dialog}/{_utt}.wav"

    out_tag = "v9_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
    print(f"dataset={DATASET}\nclip={audio_file}\nfeature cache: {features_csv}")

    # =====================================================================
    # CELL 2 — IEMOCAP CLIP PICKER
    # Browse labelled utterances, copy a "file" value into CELL 1's _utt.
    # =====================================================================
    _iem_all = pd.read_csv(iemocap_csv)
    print(f"{len(_iem_all)} labelled IEMOCAP utterances available")
    _iem_all.groupby("emotion").head(3)[["file", "emotion", "actor"]]

    # =====================================================================
    # CELL 3 — GROUND-TRUTH LABEL for the current clip
    # =====================================================================
    _clip_name = audio_file.split("/")[-1].replace(".wav", "")
    if DATASET == "ravdess":
        # RAVDESS bakes the emotion into the filename (3rd dash-separated field)
        true_emotion = emotion_map.get(_clip_name.split("-")[2], "unknown")
    else:
        # IEMOCAP: look the utterance up in the CSV built by extract_iemocap.py
        _lookup = pd.read_csv(iemocap_csv)
        _hit = _lookup[_lookup["file"].astype(str).str.contains(_clip_name)]
        true_emotion = _hit["emotion"].iloc[0] if len(_hit) else "NOT IN CSV"
    print(f"dataset label for this clip: {true_emotion}")

    mo.md("""
    ## Part A: Clip-level classifier (load the V2 model)
    """)

    # =====================================================================
    # CELL 4 — CLIP-LEVEL 14-FEATURE EXTRACTOR (kept for the fallback model
    # and reused by the 14-feature path in CELL 6b)
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

    def extract_clip_features(path):
        return extract_clip_features_from_sound(parselmouth.Sound(str(path)))

    # =====================================================================
    # CELL 5 — LOAD 14-FEATURE CACHE (used only by the fallback model in
    # CELL 6; the eGeMAPS path does not need it, but it stays loaded so the
    # notebook still works if clf_v2.joblib is absent)
    # =====================================================================
    if os.path.exists(features_csv):
        clip_df = pd.read_csv(features_csv)
        print(f"Loaded cached features: {len(clip_df)} clips from {features_csv}")
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
            except Exception:
                skipped += 1
                continue
            feats["file"] = p.name
            feats["emotion"] = emotion
            feats["actor"] = actor
            rows.append(feats)
            if n % 200 == 0:
                print(f"  ...processed {n}/{len(wav_paths)}")
        clip_df = pd.DataFrame(rows)
        clip_df.to_csv(features_csv, index=False)
        print(f"Done. Kept {len(clip_df)} clips, skipped {skipped}. Saved to {features_csv}")
    clip_df

    # =====================================================================
    # CELL 6 — LOAD THE V2 MODEL (eGeMAPS + speaker-normalised, 60.6%)
    # Replaces V8's "train clf_full from features.csv". The bundle saved by
    # train_emotion_v2.py carries the classifier AND the metadata V9 needs
    # to use it correctly: which feature columns, in what order, whether it
    # expects normalised input, and which extractor produced them.
    # Falls back to the old 14-feature model if the joblib is missing, so
    # this notebook still runs on a machine that never ran train_emotion_v2.
    # =====================================================================
    model_bundle_path = "outputs/clf_v2.joblib"

    if os.path.exists(model_bundle_path):
        _bundle = joblib.load(model_bundle_path)
        clf_full         = _bundle["clf"]
        clf_feature_cols = _bundle["feature_cols"]        # 88 eGeMAPS names, ordered
        CLF_EXTRACTOR    = _bundle["extractor"]           # "egemaps"
        CLF_NORMALISED   = _bundle["speaker_normalised"]  # True
        print(f"Loaded clf_v2: extractor={CLF_EXTRACTOR}, "
              f"{len(clf_feature_cols)} features, "
              f"speaker_normalised={CLF_NORMALISED}, "
              f"classes={list(clf_full.classes_)}")
    else:
        # ---- fallback: the original 14-feature RAVDESS model ----
        print("clf_v2.joblib not found -> falling back to the 14-feature model.")
        _cols14 = [c for c in clip_df.columns if c not in ("file", "emotion", "actor")]
        _X = clip_df[_cols14].to_numpy()
        _y = clip_df["emotion"].to_numpy()
        clf_full = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        ).fit(_X, _y)
        clf_feature_cols = _cols14
        CLF_EXTRACTOR  = "praat14"
        CLF_NORMALISED = False
        print(f"Trained 14-feature clf_full on {len(clip_df)} clips.")

    # =====================================================================
    # CELL 6b (NEW) — eGeMAPS extraction + per-segment prediction for V9
    #
    # This cell IS the V9 change. Two switches control the normalisation
    # that clf_v2 expects:
    #   SEGMENT_NORM:
    #     "auto" -> normalise only if >= NORM_MIN_SEGMENTS segments exist,
    #               else fall back to raw features (safe default)
    #     "on"   -> always normalise (only for genuine multi-turn, multi-
    #               emotion conversations)
    #     "off"  -> never normalise (correct for a single-emotion clip; a
    #               per-speaker baseline would subtract the emotion out)
    # If the loaded model is the 14-feature fallback, CLF_NORMALISED is False
    # and normalisation is skipped regardless, reproducing V8 exactly.
    # =====================================================================
    SEGMENT_NORM = "auto"      # "auto" | "on" | "off"
    NORM_MIN_SEGMENTS = 4      # below this, "auto" declines to normalise

    # openSMILE engine — built only if the loaded model actually wants eGeMAPS
    if CLF_EXTRACTOR == "egemaps":
        _smile_v9 = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        _smile_v9 = None

    def clf_features_from_path(path):
        """Whole-file feature dict matching the loaded model (used by CELL 13)."""
        if CLF_EXTRACTOR == "egemaps":
            return _smile_v9.process_file(str(path)).iloc[0].to_dict()
        return extract_clip_features(path)

    def clf_features_from_sound(seg_snd, tmp_wav="outputs/audio/_seg_tmp.wav"):
        """Segment feature dict matching the loaded model.
        eGeMAPS path writes the slice to a temp wav because openSMILE reads
        files; the 14-feature path works on the Sound object directly."""
        if CLF_EXTRACTOR == "egemaps":
            os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)
            call(seg_snd, "Save as WAV file", tmp_wav)   # Praat writer, version-proof
            return _smile_v9.process_file(tmp_wav).iloc[0].to_dict()
        return extract_clip_features_from_sound(seg_snd)

    def predict_segment_emotions_v9(audio_path, segments, clf, feature_cols,
                                    normalise_mode="auto", norm_min_segments=4):
        """Per-segment emotion, matching whatever clf_v2 was trained on.

        Return shape matches V8's predict_segment_emotions (segment_id, start,
        end, pred_emotion, p_top, conf_scale) so CELL 20/21/22 need no change.
        Adds one column, 'normalised', recording what actually happened.
        """
        snd_full = parselmouth.Sound(audio_path)
        chance_level = 1.0 / len(clf.classes_)

        # ---- pass 1: extract features for every segment ----
        feats_per_seg, meta = [], []
        for sid, seg in enumerate(segments):
            s0, s1 = float(seg["start"]), float(seg["end"])
            try:
                seg_snd = snd_full.extract_part(from_time=s0, to_time=s1,
                                                preserve_times=True)
                feats_per_seg.append(clf_features_from_sound(seg_snd))
                meta.append({"segment_id": sid, "start": s0, "end": s1, "ok": True})
            except Exception as e:
                print(f"  segment {sid} ({s0:.2f}-{s1:.2f}s) feature-extract failed: {e}")
                feats_per_seg.append(None)
                meta.append({"segment_id": sid, "start": s0, "end": s1, "ok": False})

        feat_df = pd.DataFrame([f for f in feats_per_seg if f is not None])
        n_ok = len(feat_df)

        # ---- decide whether to normalise ----
        if normalise_mode == "off":
            do_norm = False
        elif normalise_mode == "on":
            do_norm = True
        else:  # "auto"
            do_norm = (n_ok >= norm_min_segments)
        do_norm = do_norm and CLF_NORMALISED   # only meaningful if model expects it

        if not feat_df.empty and do_norm:
            _cols = [c for c in feature_cols if c in feat_df.columns]
            # z-score across THIS video's segments = the deployment-time
            # equivalent of the per-speaker normalisation used in training
            _mu = feat_df[_cols].mean()
            _sd = feat_df[_cols].std().replace(0.0, 1.0)
            feat_df[_cols] = ((feat_df[_cols] - _mu) / _sd).fillna(0.0)

        # ---- pass 2: classify ----
        rows, fi = [], 0
        for m in meta:
            if not m["ok"]:
                rows.append({"segment_id": m["segment_id"], "start": m["start"],
                             "end": m["end"], "pred_emotion": "neutral",
                             "p_top": 0.0, "conf_scale": 0.5, "normalised": False})
                continue
            _vec = (feat_df.iloc[fi][[c for c in feature_cols if c in feat_df.columns]]
                    .to_numpy(dtype=float).reshape(1, -1))
            fi += 1
            pred  = str(clf.predict(_vec)[0])
            proba = clf.predict_proba(_vec)[0]
            p_top = float(np.max(proba))
            conf  = float(0.5 + 0.5 * np.clip(
                (p_top - chance_level) / (0.5 - chance_level), 0.0, 1.0))
            rows.append({"segment_id": m["segment_id"], "start": m["start"],
                         "end": m["end"], "pred_emotion": pred,
                         "p_top": round(p_top, 3), "conf_scale": round(conf, 3),
                         "normalised": bool(do_norm)})

        out = pd.DataFrame(rows)
        print(f"  predicted {len(out)} segment(s) | extractor={CLF_EXTRACTOR} | "
              f"normalised={do_norm} (mode='{normalise_mode}', {n_ok} usable segs)")
        return out

    mo.md("""
    ## Part B: WhisperX transcription → word timestamps + per-word prosody
    """)

    # =====================================================================
    # CELL 7 — WHISPERX TRANSCRIPTION
    # =====================================================================
    asr_model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = asr_model.transcribe(audio, batch_size=16)
    print("LANGUAGE:", result["language"])
    print("SEGMENTS:", result["segments"])

    # =====================================================================
    # CELL 8 — WORD-LEVEL ALIGNMENT
    # =====================================================================
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )
    aligned["word_segments"]

    # =====================================================================
    # CELL 9 — PER-WORD PROSODY EXTRACTOR (+ f0 slope) — GUARDED
    # Praat's pitch/harmonicity analysis needs a minimum window length
    # (roughly 0.04-0.06s at default settings). Curated single-sentence
    # clips never hit this, but a raw conversation has fast function words,
    # interjections, and forced-alignment artifacts that can land under that
    # floor -- and Praat raises instead of returning "no pitch." Each word is
    # guarded and falls back to the same zero used for "unvoiced".
    # =====================================================================
    def extract_word_features(audio_path, word_segments):
        snd = parselmouth.Sound(audio_path)
        rows = []

        for i, w in enumerate(word_segments):
            start, end = float(w["start"]), float(w["end"])
            duration = end - start

            if i < len(word_segments) - 1:
                pause_after = float(word_segments[i + 1]["start"]) - end
            else:
                pause_after = 0.0

            word_snd = snd.extract_part(from_time=start, to_time=end, preserve_times=True)

            # pitch (F0) -- guarded against "too short to analyse"
            try:
                pitch = word_snd.to_pitch()
                f0 = pitch.selected_array["frequency"]
                f0v = f0[f0 > 0]
                f0_mean = float(f0v.mean()) if len(f0v) else 0.0
                f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0
            except parselmouth.PraatError:
                f0v = np.array([])
                f0_mean = 0.0
                f0_range = 0.0

            # f0 slope -- naturally falls back to 0.0 when f0v is empty
            if len(f0v) >= 3:
                _t = np.linspace(0.0, max(duration, 1e-6), len(f0v))
                _semitones = 12.0 * np.log2(f0v / f0v[0])
                f0_slope = float(np.polyfit(_t, _semitones, 1)[0])
            else:
                f0_slope = 0.0

            # intensity / loudness (RMS) -- rarely fails, guarded anyway
            try:
                rms = call(word_snd, "Get root-mean-square", 0, 0)
                rms = 0.0 if rms != rms else rms   # nan guard
            except Exception:
                rms = 0.0

            # voice quality (HNR) -- guarded, same reason as pitch
            try:
                harm = word_snd.to_harmonicity()
                hnr_vals = harm.values[harm.values != -200]
                hnr = float(hnr_vals.mean()) if len(hnr_vals) else 0.0
            except parselmouth.PraatError:
                hnr = 0.0

            rows.append({
                "word": w["word"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "pause_after": round(pause_after, 3),
                "f0_mean": round(f0_mean, 1),
                "f0_range": round(f0_range, 1),
                "f0_slope": round(f0_slope, 2),
                "rms": round(rms, 4),
                "hnr": round(hnr, 1),
            })

        return pd.DataFrame(rows)

    # =====================================================================
    # CELL 10 — RUN PER-WORD EXTRACTION
    # =====================================================================
    word_df = extract_word_features(audio_file, aligned["word_segments"])
    word_df

    mo.md("""
    ## Part C: The expressive budget + all styling dials
    """)

    # =====================================================================
    # CELL 11 — TUNABLE PARAMETERS
    # =====================================================================
    # ----- budget (unchanged since V3) -----
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}
    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5
    USE_CONFIDENCE_SCALING = True

    # ----- TYPOGRAPHY channel: the sole carrier of intensity -----
    BASE_FONT_SIZE = 32
    FONT_SWING = 32            # 32..64 px
    BOLD_THRESHOLD = 0.60

    # ----- OPACITY channel: time only -----
    CAPTION_MODE = "sentence"  # "sentence" (bottom line) | "word" (centre, legacy)
    REVEAL_MODE = "wipe"       # "wipe" (bloom over spoken duration) | "snap" | "none"
    DIM_ALPHA = 150            # unspoken words (0 = opaque, 255 = invisible)
    HOLD_MAX_TAIL = 0.6
    MIN_LINE_DURATION = 1.0

    # ----- MOTION channel: pitch direction -----
    MOTION_SOURCE = "pitch"        # "pitch" | "none"
    MOTION_MIN_INTENSITY = 0.45    # a word must clear this to be allowed to move
    SLOPE_DEADZONE = 2.0           # |semitones/sec| below this counts as flat
    SLOPE_FULL = 12.0              # |slope| at or above this = full-strength move
    WOBBLE_RANGE_HZ = 90.0         # f0_range above this = wobble instead of lift/drop

    MOTION_STYLE = "scale"         # "scale" (fscy swell, safe) | "tilt" (frz, risky)
    MOTION_SWELL_PEAK = 30         # lift: max % of extra height at full strength
    MOTION_TILT_DEG = 7            # tilt: max rotation in degrees at full strength
    MOTION_MIN_MS, MOTION_MAX_MS = 200, 700
    MOTION_TEMPO = {"pop": 0.80, "soft": 1.50, "flat": 1.00}

    # ----- COLOUR channel: one fixed colour per emotion, whole clip -----
    # hue      : Jonauskaite 37-nation survey (n=8,615), max-weight assignment
    # sat/value: Valdez & Mehrabian (1994) PAD regressions inverted per emotion
    EMOTION_STYLES = {
        #             hue     sat   val  italic  anim    font
        "angry":     {"h": 0.000, "s": 0.85, "v": 0.80, "i": 0, "anim": "pop",  "font": "DejaVu Sans Condensed"},
        "happy":     {"h": 0.140, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",  "font": "DejaVu Sans"},
        "surprised": {"h": 0.075, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",  "font": "DejaVu Sans"},
        "sad":       {"h": 0.610, "s": 0.55, "v": 0.82, "i": 0, "anim": "soft", "font": "Liberation Serif"},
        "fearful":   {"h": 0.700, "s": 0.28, "v": 0.88, "i": 0, "anim": "soft", "font": "DejaVu Serif"},
        "disgust":   {"h": 0.110, "s": 0.55, "v": 0.72, "i": 1, "anim": "flat", "font": "Liberation Mono"},
        "neutral":   {"h": 0.000, "s": 0.00, "v": 0.95, "i": 0, "anim": "flat", "font": "Liberation Sans"},
    }

    # =====================================================================
    # CELL 12 — FONT AVAILABILITY CHECK
    # libass silently falls back to a default when a font is missing.
    # =====================================================================
    try:
        _installed = subprocess.run(["fc-list"], capture_output=True, text=True).stdout.lower()
        for _emo, _fam in EMOTION_STYLES.items():
            _ok = _fam["font"].lower() in _installed
            print(f"{_emo:10s} -> {_fam['font']:24s} {'OK' if _ok else 'MISSING (libass will fall back)'}")
    except FileNotFoundError:
        print("fc-list not found; cannot verify fonts on this machine.")

    mo.md("""
    ### Step C1: Clip-level emotion prediction (legacy single-clip path)
    """)

    # =====================================================================
    # CELL 13 — PREDICT THE CLIP'S EMOTION (pred_emotion + conf_scale)
    # V9: extracts whatever features clf_full expects (eGeMAPS or 14).
    # NOTE: this whole-file path CANNOT speaker-normalise (there is only one
    # vector, nothing to normalise against). Since clf_v2 was trained on
    # normalised features, this legacy single-clip prediction is LESS
    # reliable than the per-segment path in process_any_video, which does
    # normalise across a video's segments. Use process_any_video for real work.
    # =====================================================================
    clip_pred_feats = clf_features_from_path(audio_file)
    clip_pred_vec = pd.DataFrame([clip_pred_feats])[clf_feature_cols].to_numpy()

    pred_emotion = str(clf_full.predict(clip_pred_vec)[0])
    pred_proba = clf_full.predict_proba(clip_pred_vec)[0]
    p_top = float(np.max(pred_proba))

    if USE_CONFIDENCE_SCALING:
        chance_level = 1.0 / len(clf_full.classes_)
        conf_scale = float(0.5 + 0.5 * np.clip(
            (p_top - chance_level) / (0.5 - chance_level), 0.0, 1.0))
    else:
        conf_scale = 1.0

    if CLF_NORMALISED:
        print("note: single-clip path cannot speaker-normalise; this prediction "
              "is less reliable than process_any_video's per-segment path.")
    print(f"Clip-level prediction: {pred_emotion.upper()}  "
          f"(top probability {p_top:.2f}, confidence scale {conf_scale:.2f})")
    proba_table = pd.DataFrame(
        {"emotion": clf_full.classes_, "probability": np.round(pred_proba, 3)}
    ).sort_values("probability", ascending=False).reset_index(drop=True)
    proba_table

    # =====================================================================
    # CELL 14 — PREDICTION vs LABEL VERDICT
    # =====================================================================
    _verdict = "MATCH" if pred_emotion == true_emotion else "MISMATCH"
    print(f"model predicted : {pred_emotion}")
    print(f"dataset label   : {true_emotion}")
    print(f"confidence scale: {conf_scale:.2f}")
    print(f"--> {_verdict}")

    mo.md("""
    ### Step C2: Salience → competitive 100-point budget
    """)

    # =====================================================================
    # CELL 15 — BUDGET MACHINERY (synthetic-test validated, unchanged)
    # =====================================================================
    def assign_words_to_segments(words_df, segments):
        """Tag each word with the id of the sentence/segment it belongs to."""
        bounds = [(float(s["start"]), float(s["end"])) for s in segments]

        def locate(mid):
            for si, (s0, s1) in enumerate(bounds):
                if s0 <= mid <= s1:
                    return si
            return min(range(len(bounds)),
                       key=lambda i: min(abs(mid - bounds[i][0]), abs(mid - bounds[i][1])))

        mids = (words_df["start"].astype(float) + words_df["end"].astype(float)) / 2.0
        return [locate(m) for m in mids]

    def compute_salience(words_df, weights, zero_missing=frozenset(), eps=1e-9):
        """Salience = weighted sum of |z| of each feature vs the word's OWN sentence."""
        df = words_df.copy()
        salience = np.zeros(len(df))
        seg_ids = df["segment_id"].values
        for feat, w in weights.items():
            vals = df[feat].astype(float).values
            z = np.zeros(len(df))
            missing = (vals == 0.0) if feat in zero_missing else np.zeros(len(df), dtype=bool)
            for sid in np.unique(seg_ids):
                m = (seg_ids == sid) & ~missing
                if m.sum() >= 2:
                    mu, sd = vals[m].mean(), vals[m].std()
                    if sd > eps:
                        z[m] = (vals[m] - mu) / sd
            salience += w * np.abs(z)
        df["salience"] = salience
        return df

    def allocate_points(words_df, temperature, min_points, eps=1e-9):
        """Softmax split of a fixed 100-point budget per segment, with a floor."""
        df = words_df.copy()
        df["points"] = 0.0
        seg_ids = df["segment_id"].values
        for sid in np.unique(seg_ids):
            m = seg_ids == sid
            s = df.loc[m, "salience"].values
            n = int(m.sum())
            floor = min(min_points, 100.0 / n)
            pool = 100.0 - floor * n
            logits = s / max(temperature, eps)
            logits = logits - logits.max()
            w = np.exp(logits)
            w = w / w.sum()
            df.loc[m, "points"] = floor + pool * w
        fair = df.groupby("segment_id")["points"].transform(lambda p: 100.0 / len(p))
        df["share_ratio"] = df["points"] / fair
        return df

    # =====================================================================
    # CELL 16 — RUN THE BUDGET on this clip's words
    # =====================================================================
    seg_list = aligned.get("segments") or result["segments"]

    tagged_word_df = word_df.copy()
    tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

    salient_word_df = compute_salience(
        tagged_word_df, SALIENCE_WEIGHTS, zero_missing=ZERO_MEANS_MISSING
    )
    budget_df = allocate_points(
        salient_word_df, temperature=SOFTMAX_TEMPERATURE, min_points=MIN_POINTS
    )

    budget_df["intensity"] = np.clip(
        (budget_df["share_ratio"] - 1.0) / (FULL_DRAMA_RATIO - 1.0), 0.0, 1.0
    ) * conf_scale

    for _sid in np.unique(budget_df["segment_id"].values):
        _m = budget_df["segment_id"] == _sid
        print(f"Segment {_sid}: {int(_m.sum())} words, "
              f"points sum = {budget_df.loc[_m, 'points'].sum():.2f}, "
              f"top word = '{budget_df.loc[_m].sort_values('points').iloc[-1]['word'].strip()}'")

    budget_df[["word", "segment_id", "salience", "points", "share_ratio", "intensity"]].round(2)

    mo.md("""
    ## Part D: Style mapping — one variable per channel
    """)

    # =====================================================================
    # CELL 17 — STYLE MAPPING v6 (colour=emotion, size=intensity,
    #                             motion=pitch direction) — legacy single-clip
    # =====================================================================
    def emotion_color(emotion, styles):
        """One fixed colour for the whole clip. Colour says WHICH emotion."""
        fam = styles.get(emotion, styles["neutral"])
        r, g, b = (int(round(c * 255))
                   for c in colorsys.hsv_to_rgb(fam["h"], fam["s"], fam["v"]))
        return f"&H{b:02X}{g:02X}{r:02X}&"   # ASS colour is &HBBGGRR&, NOT RGB

    def assign_styles_v6(words_df, emotion, styles,
                         base_font, font_swing, bold_thresh,
                         motion_source, motion_min_intensity,
                         slope_deadzone, slope_full, wobble_range_hz):
        df = words_df.copy()
        fam = styles.get(emotion, styles["neutral"])

        df["color_ass"] = emotion_color(emotion, styles)
        df["italic"] = int(fam["i"])

        df["font_size"] = (base_font + font_swing * df["intensity"]).round().astype(int)
        df["bold"] = (df["intensity"] >= bold_thresh).astype(int)

        def _gesture(r):
            if motion_source != "pitch":
                return "none"
            if float(r["intensity"]) < motion_min_intensity:
                return "none"
            if float(r["f0_range"]) >= wobble_range_hz:
                return "wobble"
            s = float(r["f0_slope"])
            if abs(s) < slope_deadzone:
                return "none"
            return "lift" if s > 0 else "drop"

        df["gesture"] = df.apply(_gesture, axis=1)
        df["motion_strength"] = (
            ((df["f0_slope"].abs() - slope_deadzone) / max(slope_full - slope_deadzone, 1e-6))
            .clip(0.0, 1.0)
        )
        _wob = df["gesture"] == "wobble"
        df.loc[_wob, "motion_strength"] = (
            (df.loc[_wob, "f0_range"] / (wobble_range_hz * 2.0)).clip(0.3, 1.0)
        )
        df.loc[df["gesture"] == "none", "motion_strength"] = 0.0
        return df

    styled_word_df = assign_styles_v6(
        budget_df, pred_emotion, EMOTION_STYLES,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING,
        bold_thresh=BOLD_THRESHOLD,
        motion_source=MOTION_SOURCE, motion_min_intensity=MOTION_MIN_INTENSITY,
        slope_deadzone=SLOPE_DEADZONE, slope_full=SLOPE_FULL,
        wobble_range_hz=WOBBLE_RANGE_HZ,
    )
    _fam = EMOTION_STYLES.get(pred_emotion, EMOTION_STYLES["neutral"])
    _moving = int((styled_word_df["gesture"] != "none").sum())
    print(f"emotion={pred_emotion} | colour={styled_word_df['color_ass'].iloc[0]} "
          f"| font={_fam['font']}")
    print(f"moving words: {_moving}/{len(styled_word_df)}  "
          f"{styled_word_df['gesture'].value_counts().to_dict()}")
    styled_word_df[["word", "intensity", "f0_slope", "f0_range",
                    "gesture", "motion_strength", "font_size", "bold"]].round(2)

    mo.md(r"""
    ## Part E: Render .ass + FFmpeg burn-in

    - **Every word re-declares every tag it uses.** .ass override tags persist
      along a Dialogue line, so a word that animates scale or rotation must be
      followed by words that explicitly reset them — hence `\fscx100\fscy100\frz0`
      in every word's tag block.
    - **`\fscy` swell is the default motion**: width never changes (no reflow),
      baseline stays flat. `\frz` tilt is more literal but propagates along a
      line and breaks the baseline unless every word resets `\frz0`.
    - **There is no per-word position tag in .ass.** `\pos` and `\move` are
      line-level, like `\fad`, so a "lift" is a swell or a tilt, never a true
      translation.
    """)

    # =====================================================================
    # CELL 18 — RENDER: wipe (time) + typography (intensity) + pitch motion
    # (legacy single-clip render, black screen)
    # =====================================================================
    def render_budget_video(audio_path, df, emotion, styles, out_dir="outputs",
                            caption_mode="sentence", reveal_mode="wipe",
                            hold_max_tail=0.6, min_line_dur=1.0, dim_alpha=150,
                            motion_style="scale", swell_peak=30, tilt_deg=7,
                            motion_min_ms=200, motion_max_ms=700,
                            motion_tempo=None, tag="v9_demo"):
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720
        fam = styles.get(emotion, styles["neutral"])
        tempo = (motion_tempo or {"pop": 0.8, "soft": 1.5, "flat": 1.0})[fam["anim"]]

        def sec_to_ass(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            cs = int(round((t - int(t)) * 100))
            if cs == 100: cs = 0; s += 1
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            f"PlayResX: {width}\n"
            f"PlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Cap,Liberation Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)
        lines = []

        def word_tags(rw, line_start):
            tags = (f"\\fn{fam['font']}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}"
                    f"\\fscx100\\fscy100\\frz0")

            d0 = max(0, int(round((float(rw["start"]) - line_start) * 1000)))
            d1 = max(d0 + 120, int(round((float(rw["end"]) - line_start) * 1000)))
            if reveal_mode == "wipe":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d1},\\alpha&H00&)"
            elif reveal_mode == "snap":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d0 + 90},\\alpha&H00&)"

            g = str(rw.get("gesture", "none"))
            k = float(rw.get("motion_strength", 0.0))
            if g != "none" and k > 0.0:
                dur = int(min(max((float(rw["end"]) - float(rw["start"])) * 1000,
                                  motion_min_ms), motion_max_ms) * tempo)
                mid = d0 + int(dur * 0.45)
                end = d0 + dur
                if motion_style == "scale":
                    if g == "lift":
                        peak = int(100 + swell_peak * k)
                        tags += f"\\t({d0},{mid},\\fscy{peak})\\t({mid},{end},\\fscy100)"
                    elif g == "drop":
                        trough = int(100 - (swell_peak * 0.6) * k)
                        tags += f"\\t({d0},{mid},\\fscy{trough})\\t({mid},{end},\\fscy100)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        hi = int(100 + (swell_peak * 0.5) * k)
                        lo = int(100 - (swell_peak * 0.4) * k)
                        tags += (f"\\t({d0},{d0 + q},\\fscy{hi})"
                                 f"\\t({d0 + q},{d0 + 2 * q},\\fscy{lo})"
                                 f"\\t({d0 + 2 * q},{end},\\fscy100)")
                else:  # "tilt" — rotation
                    t_ = int(round(tilt_deg * k))
                    if g == "lift":
                        tags += f"\\frz{t_}\\t({d0},{end},\\frz0)"
                    elif g == "drop":
                        tags += f"\\frz-{t_}\\t({d0},{end},\\frz0)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        tags += (f"\\t({d0},{d0 + q},\\frz{t_})"
                                 f"\\t({d0 + q},{d0 + 2 * q},\\frz-{t_})"
                                 f"\\t({d0 + 2 * q},{end},\\frz0)")
            return tags

        if caption_mode == "word":
            for i, rw in rows.iterrows():
                start_t = float(rw["start"])
                if i < len(rows) - 1:
                    end_t = min(float(rows.loc[i + 1, "start"]),
                                float(rw["end"]) + hold_max_tail)
                else:
                    end_t = float(rw["end"]) + 0.35
                text = "{\\an5" + word_tags(rw, start_t) + "}" + str(rw["word"]).strip()
                lines.append(f"Dialogue: 0,{sec_to_ass(start_t)},{sec_to_ass(end_t)},"
                             f"Cap,,0,0,0,,{text}")
        else:
            seg_ids = list(dict.fromkeys(rows["segment_id"].tolist()))
            seg_starts = {s: float(rows.loc[rows["segment_id"] == s, "start"].min())
                          for s in seg_ids}
            for si, sid in enumerate(seg_ids):
                seg = rows[rows["segment_id"] == sid].sort_values("start")
                s0 = float(seg["start"].min())
                last_end = float(seg["end"].max())
                e0 = max(last_end + hold_max_tail, s0 + min_line_dur)
                if si < len(seg_ids) - 1:
                    e0 = min(e0, seg_starts[seg_ids[si + 1]])
                e0 = max(e0, s0 + 0.10)
                parts = ["{" + word_tags(rw, s0) + "}" + str(rw["word"]).strip()
                         for _, rw in seg.iterrows()]
                lines.append(f"Dialogue: 0,{sec_to_ass(s0)},{sec_to_ass(e0)},"
                             f"Cap,,0,0,0,," + "{\\fad(120,120)}" + " ".join(parts))

        ass_path = f"{out_dir}/ass/{tag}.ass"
        with open(ass_path, "w") as f:
            f.write(header + "\n".join(lines) + "\n")

        duration = float(rows["end"].max()) + 0.8
        out_path = f"{out_dir}/video/{tag}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=25:d={duration}",
            "-i", audio_path,
            "-map", "0:v", "-map", "1:a",
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest",
            out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print("FFmpeg return code:", proc.returncode)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
        _mv = int((df["gesture"] != "none").sum())
        print(f"{len(lines)} caption line(s) | {_mv} moving word(s) "
              f"({caption_mode} / {reveal_mode} / motion={motion_style})")
        return out_path, ass_path

    v9_video, v9_ass = render_budget_video(
        audio_file, styled_word_df, pred_emotion, EMOTION_STYLES,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE,
        hold_max_tail=HOLD_MAX_TAIL, min_line_dur=MIN_LINE_DURATION,
        dim_alpha=DIM_ALPHA, motion_style=MOTION_STYLE,
        swell_peak=MOTION_SWELL_PEAK, tilt_deg=MOTION_TILT_DEG,
        motion_min_ms=MOTION_MIN_MS, motion_max_ms=MOTION_MAX_MS,
        motion_tempo=MOTION_TEMPO,
        tag=out_tag + "_" + CAPTION_MODE + "_" + MOTION_STYLE,
    )
    print("Wrote:", v9_video, "and", v9_ass)

    # =====================================================================
    # CELL 19 — RUN PER-SEGMENT EMOTION (V9) on the single test clip, as a
    # standalone sanity display. Uses the eGeMAPS + conditional-norm predictor.
    # =====================================================================
    seg_emotion_df = predict_segment_emotions_v9(
        audio_file, seg_list, clf_full, clf_feature_cols,
        normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS,
    )
    seg_emotion_df

    # =====================================================================
    # CELL 20 — MOTION (pulled out of assign_styles_v6) + STYLE MAPPING v8:
    # colour/font/anim vary PER SEGMENT, not once per clip.
    # =====================================================================
    def attach_motion(words_df, motion_source, motion_min_intensity,
                      slope_deadzone, slope_full, wobble_range_hz):
        df = words_df.copy()
        def _gesture(r):
            if motion_source != "pitch":
                return "none"
            if float(r["intensity"]) < motion_min_intensity:
                return "none"
            if float(r["f0_range"]) >= wobble_range_hz:
                return "wobble"
            s = float(r["f0_slope"])
            if abs(s) < slope_deadzone:
                return "none"
            return "lift" if s > 0 else "drop"
        df["gesture"] = df.apply(_gesture, axis=1)
        df["motion_strength"] = (
            ((df["f0_slope"].abs() - slope_deadzone) / max(slope_full - slope_deadzone, 1e-6))
            .clip(0.0, 1.0)
        )
        _wob = df["gesture"] == "wobble"
        df.loc[_wob, "motion_strength"] = (
            (df.loc[_wob, "f0_range"] / (wobble_range_hz * 2.0)).clip(0.3, 1.0)
        )
        df.loc[df["gesture"] == "none", "motion_strength"] = 0.0
        return df

    def assign_styles_v8(words_df, seg_emotion_df, styles, base_font, font_swing, bold_thresh):
        df = words_df.copy()
        emo_lookup  = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df["pred_emotion"]))
        conf_lookup = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df["conf_scale"]))

        df["emotion"]    = df["segment_id"].map(emo_lookup).fillna("neutral")
        df["conf_scale"] = df["segment_id"].map(conf_lookup).fillna(1.0)

        df["intensity"] = (df["intensity_raw"] * df["conf_scale"]).clip(0.0, 1.0)

        df["color_ass"] = df["emotion"].apply(lambda e: emotion_color(e, styles))
        df["italic"]    = df["emotion"].apply(lambda e: int(styles.get(e, styles["neutral"])["i"]))
        df["font"]      = df["emotion"].apply(lambda e: styles.get(e, styles["neutral"])["font"])
        df["anim"]      = df["emotion"].apply(lambda e: styles.get(e, styles["neutral"])["anim"])

        df["font_size"] = (base_font + font_swing * df["intensity"]).round().astype(int)
        df["bold"] = (df["intensity"] >= bold_thresh).astype(int)
        return df

    # =====================================================================
    # CELL 21 — RENDER onto a REAL video, colour/font change per segment.
    # =====================================================================
    def get_video_info(path):
        """width, height, duration, has_audio -- via ffprobe, no guessing."""
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True,
        )
        info = json.loads(probe.stdout)
        w = info["streams"][0]["width"]; h = info["streams"][0]["height"]
        dur = float(info["format"]["duration"])
        audio_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "json", path],
            capture_output=True, text=True,
        )
        has_audio = len(json.loads(audio_probe.stdout).get("streams", [])) > 0
        return w, h, dur, has_audio

    def render_long_video(audio_path, df, styles, out_dir="outputs",
                          caption_mode="sentence", reveal_mode="wipe",
                          hold_max_tail=0.6, min_line_dur=1.0, dim_alpha=150,
                          motion_style="scale", swell_peak=30, tilt_deg=7,
                          motion_min_ms=200, motion_max_ms=700,
                          motion_tempo=None, bg_video_path=None, tag="v9_demo"):
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        motion_tempo = motion_tempo or {"pop": 0.8, "soft": 1.5, "flat": 1.0}

        if bg_video_path:
            width, height, video_dur, bg_has_audio = get_video_info(bg_video_path)
        else:
            width, height = 1280, 720
            bg_has_audio = False

        def sec_to_ass(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            cs = int(round((t - int(t)) * 100))
            if cs == 100: cs = 0; s += 1
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        header = (
            "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {width}\nPlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Cap,Liberation Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)

        def word_tags(rw, line_start):
            tempo = motion_tempo.get(str(rw.get("anim", "flat")), 1.0)
            tags = (f"\\fn{rw['font']}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}"
                    f"\\fscx100\\fscy100\\frz0")

            d0 = max(0, int(round((float(rw["start"]) - line_start) * 1000)))
            d1 = max(d0 + 120, int(round((float(rw["end"]) - line_start) * 1000)))
            if reveal_mode == "wipe":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d1},\\alpha&H00&)"
            elif reveal_mode == "snap":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d0 + 90},\\alpha&H00&)"

            g = str(rw.get("gesture", "none"))
            k = float(rw.get("motion_strength", 0.0))
            if g != "none" and k > 0.0:
                dur = int(min(max((float(rw["end"]) - float(rw["start"])) * 1000,
                                  motion_min_ms), motion_max_ms) * tempo)
                mid = d0 + int(dur * 0.45); end = d0 + dur
                if motion_style == "scale":
                    if g == "lift":
                        peak = int(100 + swell_peak * k)
                        tags += f"\\t({d0},{mid},\\fscy{peak})\\t({mid},{end},\\fscy100)"
                    elif g == "drop":
                        trough = int(100 - (swell_peak * 0.6) * k)
                        tags += f"\\t({d0},{mid},\\fscy{trough})\\t({mid},{end},\\fscy100)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        hi = int(100 + (swell_peak * 0.5) * k); lo = int(100 - (swell_peak * 0.4) * k)
                        tags += (f"\\t({d0},{d0 + q},\\fscy{hi})\\t({d0 + q},{d0 + 2*q},\\fscy{lo})"
                                 f"\\t({d0 + 2*q},{end},\\fscy100)")
                else:
                    t_ = int(round(tilt_deg * k))
                    if g == "lift":
                        tags += f"\\frz{t_}\\t({d0},{end},\\frz0)"
                    elif g == "drop":
                        tags += f"\\frz-{t_}\\t({d0},{end},\\frz0)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        tags += (f"\\t({d0},{d0 + q},\\frz{t_})\\t({d0 + q},{d0 + 2*q},\\frz-{t_})"
                                 f"\\t({d0 + 2*q},{end},\\frz0)")
            return tags

        seg_ids = list(dict.fromkeys(rows["segment_id"].tolist()))
        seg_starts = {s: float(rows.loc[rows["segment_id"] == s, "start"].min()) for s in seg_ids}
        lines = []
        for si, sid in enumerate(seg_ids):
            seg = rows[rows["segment_id"] == sid].sort_values("start")
            s0 = float(seg["start"].min())
            e0 = max(float(seg["end"].max()) + hold_max_tail, s0 + min_line_dur)
            if si < len(seg_ids) - 1:
                e0 = min(e0, seg_starts[seg_ids[si + 1]])
            e0 = max(e0, s0 + 0.10)
            parts = ["{" + word_tags(rw, s0) + "}" + str(rw["word"]).strip() for _, rw in seg.iterrows()]
            lines.append(f"Dialogue: 0,{sec_to_ass(s0)},{sec_to_ass(e0)},Cap,,0,0,0,,"
                         + "{\\fad(120,120)}" + " ".join(parts))

        ass_path = f"{out_dir}/ass/{tag}.ass"
        with open(ass_path, "w") as f:
            f.write(header + "\n".join(lines) + "\n")

        out_path = f"{out_dir}/video/{tag}.mp4"
        if bg_video_path:
            if bg_has_audio:
                cmd = ["ffmpeg", "-y", "-i", bg_video_path, "-vf", f"ass={ass_path}",
                      "-map", "0:v", "-map", "0:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-shortest", out_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", bg_video_path, "-i", audio_path,
                      "-vf", f"ass={ass_path}", "-map", "0:v", "-map", "1:a",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                      "-shortest", out_path]
        else:
            duration = float(rows["end"].max()) + 0.8
            cmd = ["ffmpeg", "-y", "-f", "lavfi",
                  "-i", f"color=c=black:s={width}x{height}:r=25:d={duration}",
                  "-i", audio_path, "-map", "0:v", "-map", "1:a", "-vf", f"ass={ass_path}",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        print("FFmpeg return code:", proc.returncode)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
        print(f"{len(lines)} caption line(s) across {len(seg_ids)} segment(s)  |  "
              f"background: {'real video' if bg_video_path else 'black (no video given)'}")
        return out_path, ass_path

    # =====================================================================
    # CELL 22 — "INSERT ANY VIDEO": full pipeline, real video out (V9)
    # Only [5/6] changed vs V8: predict_segment_emotions -> _v9.
    # Needs CELLS 0-6b already run once (clf_full, clf_feature_cols,
    # asr_model, the V9 predictor must exist).
    # =====================================================================
    def process_any_video(video_path, out_tag=None, use_bg_video=True):
        os.makedirs("outputs/audio", exist_ok=True)
        stem = Path(video_path).stem
        extracted_audio = f"outputs/audio/{stem}.wav"

        print(f"[1/6] extracting audio <- {video_path}")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                        extracted_audio], capture_output=True, text=True, check=True)

        print("[2/6] transcribing (whisperx)")
        audio = whisperx.load_audio(extracted_audio)
        result = asr_model.transcribe(audio, batch_size=16)
        align_model, align_meta = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], align_model, align_meta, audio, device,
                                 return_char_alignments=False)
        seg_list = aligned.get("segments") or result["segments"]
        print(f"        {len(seg_list)} segment(s), {len(aligned['word_segments'])} word(s)")

        print("[3/6] per-word prosody")
        word_df = extract_word_features(extracted_audio, aligned["word_segments"])
        tagged_word_df = word_df.copy()
        tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

        print("[4/6] salience budget")
        salient_word_df = compute_salience(tagged_word_df, SALIENCE_WEIGHTS, zero_missing=ZERO_MEANS_MISSING)
        budget_df = allocate_points(salient_word_df, temperature=SOFTMAX_TEMPERATURE, min_points=MIN_POINTS)
        budget_df["intensity_raw"] = np.clip(
            (budget_df["share_ratio"] - 1.0) / (FULL_DRAMA_RATIO - 1.0), 0.0, 1.0)

        print("[5/6] emotion per segment (V9: eGeMAPS + conditional norm) + styling")
        seg_emotion_df = predict_segment_emotions_v9(
            extracted_audio, seg_list, clf_full, clf_feature_cols,
            normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS,
        )
        styled_df = assign_styles_v8(budget_df, seg_emotion_df, EMOTION_STYLES,
                                     base_font=BASE_FONT_SIZE, font_swing=FONT_SWING,
                                     bold_thresh=BOLD_THRESHOLD)
        styled_df = attach_motion(styled_df, motion_source=MOTION_SOURCE,
                                  motion_min_intensity=MOTION_MIN_INTENSITY,
                                  slope_deadzone=SLOPE_DEADZONE, slope_full=SLOPE_FULL,
                                  wobble_range_hz=WOBBLE_RANGE_HZ)

        print("[6/6] rendering + burning onto the original video")
        tag = out_tag or ("anyvideo_" + stem)
        out_path, ass_path = render_long_video(
            extracted_audio, styled_df, EMOTION_STYLES,
            caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE, hold_max_tail=HOLD_MAX_TAIL,
            min_line_dur=MIN_LINE_DURATION, dim_alpha=DIM_ALPHA, motion_style=MOTION_STYLE,
            swell_peak=MOTION_SWELL_PEAK, tilt_deg=MOTION_TILT_DEG, motion_min_ms=MOTION_MIN_MS,
            motion_max_ms=MOTION_MAX_MS, motion_tempo=MOTION_TEMPO,
            bg_video_path=(video_path if use_bg_video else None), tag=tag,
        )
        print("emotions found:", seg_emotion_df["pred_emotion"].value_counts().to_dict())
        if "normalised" in seg_emotion_df:
            print("normalisation applied:", bool(seg_emotion_df["normalised"].any()))
        print("wrote:", out_path)
        return out_path, ass_path, seg_emotion_df, styled_df

    # =====================================================================
    # CELL 23 — DEMO: pull a whole conversation straight off IEMOCAP
    # =====================================================================
    def find_iemocap_dialog_video(utt_id, iemocap_dir):
        sess = f"Session{int(utt_id[3:5])}"
        dialog = utt_id.rsplit("_", 1)[0]
        candidate = f"{iemocap_dir}/{sess}/dialog/avi/DivX/{dialog}.avi"
        return candidate if os.path.exists(candidate) else None

    _demo_utt = "Ses01F_impro01_F012"
    _demo_video = find_iemocap_dialog_video(_demo_utt, iemocap_dir)
    print("found dialog video:", _demo_video)

    if _demo_video:
        long_out, long_ass, long_seg_emotions, long_styled_df = process_any_video(_demo_video)
    else:
        print("No dialog video at the expected path -- check "
              f"{iemocap_dir}/SessionX/dialog/avi/DivX/ exists, or just pass any "
              "video path straight into process_any_video('/path/to/your/clip.mp4').")

    # =====================================================================
    # CELL 24 — RUN ON A COMPLETELY NEW VIDEO: 12AngryMenTest.mp4
    # =====================================================================
    new_video_path = "12AngryMenTest.mp4"   # relative to where marimo was launched

    if not os.path.exists(new_video_path):
        print(f"Can't find '{new_video_path}' from the current working directory "
              f"({os.getcwd()}). If marimo wasn't launched from "
              "~/Desktop/Kinetic-typography, either cd there first, or replace "
              "new_video_path above with the full path to the file.")
    else:
        angry_men_out, angry_men_ass, angry_men_seg_emotions, angry_men_styled_df = \
            process_any_video(new_video_path)

    mo.md(r"""
    ## Decision log — V8 (carried forward)

    11. **Emotion is predicted per segment, not per clip.**
    12. **Colour/font/italic/anim vary by segment.** Motion needed no change —
        it never touched the emotion label.
    13. **Per-segment confidence replaces per-clip confidence.**
    14. **Rendering can burn onto a real video, not just black.** Duration
        comes from the video file via ffprobe.
    15. **The dataset can supply that video directly** (`find_iemocap_dialog_video`).
    16. **Per-word prosody extraction is guarded** against Praat's minimum
        window length on short words.
    """)

    mo.md(r"""
    ## Decision log — V9 (this version)

    17. **Loaded the eGeMAPS / speaker-normalised model (60.6%).** V9 replaces
        V8's inline 14-feature training (46.8%) with `clf_v2.joblib` from
        `train_emotion_v2.py`. The bundle carries the classifier plus the
        feature-column order, the extractor name, and the normalisation flag,
        so the pipeline uses it without guessing. If the joblib is absent the
        notebook falls back to training the 14-feature model, so it still runs
        standalone. Nothing downstream of classification changed: budget,
        motion, styling and rendering are byte-for-byte V8. The visible
        captions are still per-segment colour, now driven by a 60.6% model
        instead of a 46.8% one.

    18. **Segments are classified on eGeMAPS with conditional normalisation.**
        `predict_segment_emotions_v9` extracts eGeMAPS per segment (openSMILE,
        via a temp wav) and, because the model was trained on speaker-
        normalised features, z-scores each segment's features across the
        video's own segments before classifying. Normalisation needs several
        segments to estimate a baseline, so `SEGMENT_NORM="auto"` only
        normalises when `>= NORM_MIN_SEGMENTS` segments exist and otherwise
        falls back to raw eGeMAPS (~47.8%). The `normalised` column records
        what happened per run.

    19. **Per-speaker normalisation subtracts sustained emotion — documented
        limitation.** On a clip where the speaker holds one emotion throughout,
        that emotion becomes the "baseline" and normalising removes it. This is
        why RAVDESS (actors cycling all seven emotions) trained cleanly but a
        single-emotion monologue must use `SEGMENT_NORM="off"`. The legacy
        single-clip path (CELL 13) cannot normalise at all — there is only one
        vector — so it is intentionally less reliable than `process_any_video`
        and prints a caveat when the loaded model expects normalisation.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

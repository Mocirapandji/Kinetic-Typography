import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # CELL 0 — IMPORTS
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
    import opensmile
    import joblib

    return (
        Path,
        RandomForestClassifier,
        call,
        colorsys,
        joblib,
        json,
        mo,
        np,
        opensmile,
        os,
        parselmouth,
        pd,
        subprocess,
        whisperx,
    )


@app.cell
def _(mo):
    mo.md("""
    # V0.7: four more channels

    V0.6 bought a much better classifier and then spent nothing on it. Its
    output was still argmax'd down to one label per segment and rendered as one
    flat hue. This version spends it, and adds one dimension nothing in the
    system had touched.

    **Saturation carries intensity.** Hue already says *which* emotion. Because
    hue and saturation are separable, saturation can say *how strongly* without
    contradicting it — a word never drops below half the emotion's saturation,
    and the budget winner sits at full.

    **Hue blending carries uncertainty.** The classifier is frequently close to
    a coin flip between two emotions, and argmax throws that away. When the top
    two probabilities are within `BLEND_MARGIN`, the word colour interpolates
    between both hues, weighted by the probabilities. The model's doubt becomes
    visible instead of being hidden behind a confident colour.

    **Letter spacing carries calm.** The low-arousal end of the range had
    nothing of its own — slow, quiet speech was expressed only by the absence of
    everything else. Words that are both slow and quiet relative to the clip now
    open their tracking up.

    **Held space carries silence.** A pause of `PAUSE_HOLD_THRESH` or more
    widens the gap before the next word, scaled by the pause and capped. This is
    the first genuinely *temporal* signal in the system: every other channel
    encodes something about the words, not the space between them.

    Deliberately not dots. Inserting "..." would put characters on screen the
    speaker never said, desyncing a deaf viewer's read from the audio and
    fighting the wipe timing. Widening the gap keeps the caption word-for-word
    faithful while still making the silence visible.
    """)
    return


@app.cell
def _(os):
    # CELL 1 — CONFIG + DATASET SWITCH
    DATASET = "iemocap"                       # "ravdess" | "iemocap"

    ravdess_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    iemocap_dir = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"
    data_dir = ravdess_dir

    emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_emotions = {"calm"}

    features_csv = "outputs/features.csv"     # 14-feature cache (fallback model)

    iemocap_csv = "outputs/features_iemocap.csv"
    if not os.path.exists(iemocap_csv):
        iemocap_csv = "/run/media/s5812886/T7 Shield/kinetic_outputs/features_iemocap.csv"

    device = "cpu"
    compute_type = "int8"

    if DATASET == "ravdess":
        audio_file = f"{ravdess_dir}/Actor_01/03-01-06-01-02-01-01.wav"
    else:
        _utt = "Ses01F_impro01_F012"          # any "file" value from CELL 2's table
        _utt = _utt.replace(".wav", "")
        _sess = f"Session{int(_utt[3:5])}"
        _dialog = _utt.rsplit("_", 1)[0]
        audio_file = f"{iemocap_dir}/{_sess}/sentences/wav/{_dialog}/{_utt}.wav"

    out_tag = "v0_7_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
    print(f"dataset={DATASET}\nclip={audio_file}\nfeature cache: {features_csv}")
    return (
        DATASET,
        audio_file,
        compute_type,
        data_dir,
        device,
        drop_emotions,
        emotion_map,
        features_csv,
        iemocap_csv,
        iemocap_dir,
        out_tag,
    )


@app.cell
def _(iemocap_csv, pd):
    # CELL 2 — IEMOCAP CLIP PICKER
    _iem_all = pd.read_csv(iemocap_csv)
    print(f"{len(_iem_all)} labelled IEMOCAP utterances available")
    _iem_all.groupby("emotion").head(3)[["file", "emotion", "actor"]]
    return


@app.cell
def _(DATASET, audio_file, emotion_map, iemocap_csv, pd):
    # CELL 3 — GROUND-TRUTH LABEL for the current clip
    _clip_name = audio_file.split("/")[-1].replace(".wav", "")
    if DATASET == "ravdess":
        # RAVDESS bakes the emotion into the filename (3rd dash-separated field)
        true_emotion = emotion_map.get(_clip_name.split("-")[2], "unknown")
    else:
        _lookup = pd.read_csv(iemocap_csv)
        _hit = _lookup[_lookup["file"].astype(str).str.contains(_clip_name)]
        true_emotion = _hit["emotion"].iloc[0] if len(_hit) else "NOT IN CSV"
    print(f"dataset label for this clip: {true_emotion}")
    return (true_emotion,)


@app.cell
def _(mo):
    mo.md("""
    ## Part A: Clip-level classifier (load the saved model)
    """)
    return


@app.cell
def _(call, np, parselmouth):
    # CELL 4 — CLIP-LEVEL 14-FEATURE EXTRACTOR
    # Not the main feature path any more. Kept for the fallback model in CELL 6
    # and for the 14-feature branch in CELL 7.
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

    return extract_clip_features, extract_clip_features_from_sound


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
    # CELL 5 — LOAD 14-FEATURE CACHE
    # The eGeMAPS model does not need this; it stays loaded so the notebook
    # still works where clf_v2.joblib is absent.
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
    return (clip_df,)


@app.cell
def _(RandomForestClassifier, clip_df, joblib, os):
    # CELL 6 — LOAD THE SAVED MODEL (eGeMAPS + speaker-normalised, 60.6%)
    # The bundle carries the classifier AND the metadata needed to use it
    # correctly: feature-column order, the extractor name, and whether it
    # expects normalised input. Guessing any of those produces garbage rather
    # than an error. Falls back to the 14-feature model if the joblib is absent.
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
        print("clf_v2.joblib not found -> falling back to the 14-feature model.")
        _cols14 = [c for c in clip_df.columns if c not in ("file", "emotion", "actor")]
        clf_full = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        ).fit(clip_df[_cols14].to_numpy(), clip_df["emotion"].to_numpy())
        clf_feature_cols = _cols14
        CLF_EXTRACTOR  = "praat14"
        CLF_NORMALISED = False
        print(f"Trained 14-feature clf_full on {len(clip_df)} clips.")
    return CLF_EXTRACTOR, CLF_NORMALISED, clf_feature_cols, clf_full


@app.cell
def _(
    CLF_EXTRACTOR,
    CLF_NORMALISED,
    call,
    extract_clip_features,
    extract_clip_features_from_sound,
    np,
    opensmile,
    os,
    pd,
    parselmouth,
):
    # CELL 7 — eGeMAPS EXTRACTION + PER-SEGMENT PREDICTION (now returns top-2)
    # The predictor keeps the runner-up class and its probability, because the
    # hue blend in CELL 13 needs them. Everything else is as before: features
    # per segment through openSMILE, z-scored across the video's own segments
    # when there are enough of them for a baseline to mean anything.
    #   "auto" -> normalise only if >= NORM_MIN_SEGMENTS segments exist
    #   "on"   -> always. Only for genuine multi-turn, multi-emotion audio.
    #   "off"  -> never. Correct for a single-emotion clip, where a per-speaker
    #             baseline would subtract the emotion out.
    SEGMENT_NORM = "auto"      # "auto" | "on" | "off"
    NORM_MIN_SEGMENTS = 4

    if CLF_EXTRACTOR == "egemaps":
        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        _smile = None

    def clf_features_from_path(path):
        """Whole-file feature dict matching the loaded model (used by CELL 15)."""
        if CLF_EXTRACTOR == "egemaps":
            return _smile.process_file(str(path)).iloc[0].to_dict()
        return extract_clip_features(path)

    def clf_features_from_sound(seg_snd, tmp_wav="outputs/audio/_seg_tmp.wav"):
        """Segment feature dict matching the loaded model. The eGeMAPS path
        writes the slice to a temp wav because openSMILE reads files, not
        arrays; the 14-feature path works on the Sound directly."""
        if CLF_EXTRACTOR == "egemaps":
            os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)
            call(seg_snd, "Save as WAV file", tmp_wav)   # Praat writer, version-proof
            return _smile.process_file(tmp_wav).iloc[0].to_dict()
        return extract_clip_features_from_sound(seg_snd)

    def predict_segment_emotions_v9(audio_path, segments, clf, feature_cols,
                                    normalise_mode="auto", norm_min_segments=4):
        snd_full = parselmouth.Sound(audio_path)
        chance_level = 1.0 / len(clf.classes_)
        classes = np.array(clf.classes_)

        # ---- pass 1: features for every segment ----
        feats_per_seg, meta = [], []
        for sid, seg in enumerate(segments):
            s0, s1 = float(seg["start"]), float(seg["end"])
            try:
                seg_snd = snd_full.extract_part(from_time=s0, to_time=s1, preserve_times=True)
                feats_per_seg.append(clf_features_from_sound(seg_snd))
                meta.append({"segment_id": sid, "start": s0, "end": s1, "ok": True})
            except Exception as e:
                print(f"  segment {sid} ({s0:.2f}-{s1:.2f}s) feature-extract failed: {e}")
                feats_per_seg.append(None)
                meta.append({"segment_id": sid, "start": s0, "end": s1, "ok": False})

        feat_df = pd.DataFrame([f for f in feats_per_seg if f is not None])
        n_ok = len(feat_df)

        if normalise_mode == "off":
            do_norm = False
        elif normalise_mode == "on":
            do_norm = True
        else:
            do_norm = (n_ok >= norm_min_segments)
        do_norm = do_norm and CLF_NORMALISED   # only if the model expects it

        if not feat_df.empty and do_norm:
            _cols = [c for c in feature_cols if c in feat_df.columns]
            # z-score across THIS video's segments: the deployment-time
            # equivalent of the per-speaker normalisation used in training
            _mu = feat_df[_cols].mean()
            _sd = feat_df[_cols].std().replace(0.0, 1.0)
            feat_df[_cols] = ((feat_df[_cols] - _mu) / _sd).fillna(0.0)

        # ---- pass 2: classify, keeping the runner-up for the blend ----
        rows, fi = [], 0
        for m in meta:
            if not m["ok"]:
                rows.append({"segment_id": m["segment_id"], "start": m["start"],
                             "end": m["end"], "pred_emotion": "neutral",
                             "pred_emotion2": None, "p_top": 0.0, "p_second": 0.0,
                             "conf_scale": 0.5, "normalised": False})
                continue
            _vec = (feat_df.iloc[fi][[c for c in feature_cols if c in feat_df.columns]]
                    .to_numpy(dtype=float).reshape(1, -1))
            fi += 1
            proba = clf.predict_proba(_vec)[0]
            order = np.argsort(proba)[::-1]
            pred   = str(classes[order[0]])
            pred2  = str(classes[order[1]]) if len(order) > 1 else None
            p_top  = float(proba[order[0]])
            p_sec  = float(proba[order[1]]) if len(order) > 1 else 0.0
            conf   = float(0.5 + 0.5 * np.clip(
                (p_top - chance_level) / (0.5 - chance_level), 0.0, 1.0))
            rows.append({"segment_id": m["segment_id"], "start": m["start"],
                         "end": m["end"], "pred_emotion": pred, "pred_emotion2": pred2,
                         "p_top": round(p_top, 3), "p_second": round(p_sec, 3),
                         "conf_scale": round(conf, 3), "normalised": bool(do_norm)})

        out = pd.DataFrame(rows)
        print(f"  predicted {len(out)} segment(s) | extractor={CLF_EXTRACTOR} | "
              f"normalised={do_norm} (mode='{normalise_mode}', {n_ok} usable segs)")
        return out

    return (
        NORM_MIN_SEGMENTS,
        SEGMENT_NORM,
        clf_features_from_path,
        predict_segment_emotions_v9,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Part B: WhisperX transcription → word timestamps + per-word prosody
    """)
    return


@app.cell
def _(audio_file, compute_type, device, whisperx):
    # CELL 8 — WHISPERX TRANSCRIPTION
    # asr_model is returned too, so CELL 23 reuses it instead of loading a
    # second copy per video.
    asr_model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = asr_model.transcribe(audio, batch_size=16)
    print("LANGUAGE:", result["language"])
    print("SEGMENTS:", result["segments"])
    return asr_model, audio, result


@app.cell
def _(audio, device, result, whisperx):
    # CELL 9 — WORD-LEVEL ALIGNMENT
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )
    aligned["word_segments"]
    return (aligned,)


@app.cell
def _(call, np, parselmouth, pd):
    # CELL 10 — PER-WORD PROSODY EXTRACTOR (+ f0 slope) — GUARDED
    # Praat's pitch and harmonicity analysis needs a minimum window length
    # (roughly 0.04-0.06s at defaults) and raises rather than returning "no
    # pitch", so every call is guarded per word and falls back to the same zero
    # already used for unvoiced.
    def extract_word_features(audio_path, word_segments):
        snd = parselmouth.Sound(audio_path)
        rows = []

        for i, w in enumerate(word_segments):
            start, end = float(w["start"]), float(w["end"])
            duration = end - start

            # pause length = gap to the NEXT word's start; 0 for the last word
            if i < len(word_segments) - 1:
                pause_after = float(word_segments[i + 1]["start"]) - end
            else:
                pause_after = 0.0

            word_snd = snd.extract_part(from_time=start, to_time=end, preserve_times=True)

            try:
                pitch = word_snd.to_pitch()
                f0 = pitch.selected_array["frequency"]
                f0v = f0[f0 > 0]                      # drop unvoiced frames
                f0_mean = float(f0v.mean()) if len(f0v) else 0.0
                f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0
            except parselmouth.PraatError:
                f0v = np.array([])
                f0_mean = 0.0
                f0_range = 0.0

            # semitones per second, so the slope is perceptually scaled and
            # comparable between a low and a high voice
            if len(f0v) >= 3:
                _t = np.linspace(0.0, max(duration, 1e-6), len(f0v))
                _semitones = 12.0 * np.log2(f0v / f0v[0])
                f0_slope = float(np.polyfit(_t, _semitones, 1)[0])
            else:
                f0_slope = 0.0

            try:
                rms = call(word_snd, "Get root-mean-square", 0, 0)
                rms = 0.0 if rms != rms else rms   # nan guard
            except Exception:
                rms = 0.0

            try:
                harm = word_snd.to_harmonicity()
                hnr_vals = harm.values[harm.values != -200]   # -200 dB sentinel
                hnr = float(hnr_vals.mean()) if len(hnr_vals) else 0.0
            except parselmouth.PraatError:
                hnr = 0.0

            rows.append({
                "word": w["word"], "start": round(start, 3), "end": round(end, 3),
                "duration": round(duration, 3), "pause_after": round(pause_after, 3),
                "f0_mean": round(f0_mean, 1), "f0_range": round(f0_range, 1),
                "f0_slope": round(f0_slope, 2), "rms": round(rms, 4), "hnr": round(hnr, 1),
            })
        return pd.DataFrame(rows)

    return (extract_word_features,)


@app.cell
def _(aligned, audio_file, extract_word_features):
    # CELL 11 — RUN PER-WORD EXTRACTION
    word_df = extract_word_features(audio_file, aligned["word_segments"])
    word_df
    return (word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part C: The expressive budget + all styling dials
    """)
    return


@app.cell
def _():
    # CELL 12 — TUNABLE PARAMETERS

    # ----- budget: unchanged since V0.2 -----
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}
    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5
    USE_CONFIDENCE_SCALING = True

    # ----- TYPOGRAPHY channel: intensity (size + weight) -----
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
    # lift = pitch rises across the word, drop = falls, wobble = wide range.
    # Gated by salience, so a word must be prominent AND its pitch must be
    # doing something before it moves.
    MOTION_SOURCE = "pitch"
    MOTION_MIN_INTENSITY = 0.45
    SLOPE_DEADZONE = 2.0       # |semitones/sec| below this counts as flat
    SLOPE_FULL = 12.0          # |slope| at or above this = full-strength move
    WOBBLE_RANGE_HZ = 90.0     # f0_range above this = wobble instead of lift/drop
    MOTION_STYLE = "scale"     # "scale" (fscy, no reflow) | "tilt" (frz, riskier)
    MOTION_SWELL_PEAK = 30
    MOTION_TILT_DEG = 7
    MOTION_MIN_MS, MOTION_MAX_MS = 200, 700
    MOTION_TEMPO = {"pop": 0.80, "soft": 1.50, "flat": 1.00}

    # ----- CHANNEL 1: SATURATION = INTENSITY -----
    # Hue and saturation are separable, so hue can keep saying WHICH emotion
    # while saturation says HOW STRONGLY. The floor stops a low-intensity word
    # from washing out to grey and losing its emotion entirely.
    SATURATION_INTENSITY = True
    SAT_FLOOR_FRAC = 0.5       # a word never drops below half the emotion's sat

    # ----- CHANNEL 2: UNCERTAINTY HUE BLEND -----
    # When the top two classes are within BLEND_MARGIN, the word colour is
    # interpolated between their hues instead of committing to the argmax.
    #   "blend"    -> weight from the probabilities, leaned by word intensity so
    #                 strong words pull toward the winner and weak ones drift
    #                 toward the runner-up
    #   "gradient" -> weight from word position, so the blend sweeps the line
    BLEND_MODE = "blend"       # "off" | "blend" | "gradient"
    BLEND_MARGIN = 0.20        # p_top - p_second at or below this = ambiguous
    BLEND_PERWORD_SWING = 0.30

    # ----- CHANNEL 3: LETTER SPACING = CALM -----
    # calm = slow AND quiet, both normalised within the clip. Percentile-clipped
    # so one outlier cannot compress the scale.
    TRACKING_CALM = True
    CALM_SPACING_MAX = 6.0     # px of extra \fsp at maximum calm

    # ----- CHANNEL 4: HELD SPACE = SILENCE -----
    # A pause of >= PAUSE_HOLD_THRESH widens the gap BEFORE the next word. The
    # gap scales linearly from 0 at THRESH to MAX_FSP at FULL, then caps -- so a
    # 5s alignment artefact and a 1.5s dramatic pause hold the same maximum
    # beat. The threshold is conservative so only intentional pauses fire, not
    # inter-word micro-gaps. Only affects "sentence" mode; "word" mode shows one
    # word at a time so there is no gap to widen. False is the A/B arm.
    PAUSE_HOLD = True
    PAUSE_HOLD_THRESH = 0.40   # seconds; below this, no hold
    PAUSE_HOLD_FULL = 1.20     # seconds; pause at/above this = max hold
    PAUSE_HOLD_MAX_FSP = 40.0  # px of extra \fsp on the held gap at full

    # ----- COLOUR channel base per emotion (hue fixed; s is now a ceiling) -----
    # hue      : Jonauskaite 37-nation survey (n=8,615), max-weight assignment
    # sat/value: Valdez & Mehrabian (1994) PAD regressions inverted per emotion
    # italic and font family are CATEGORICAL: they identify the emotion, they
    # never signal emphasis.
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
    return (
        BASE_FONT_SIZE,
        BLEND_MARGIN,
        BLEND_MODE,
        BLEND_PERWORD_SWING,
        BOLD_THRESHOLD,
        CALM_SPACING_MAX,
        CAPTION_MODE,
        DIM_ALPHA,
        EMOTION_STYLES,
        FONT_SWING,
        FULL_DRAMA_RATIO,
        HOLD_MAX_TAIL,
        MIN_LINE_DURATION,
        MIN_POINTS,
        MOTION_MAX_MS,
        MOTION_MIN_INTENSITY,
        MOTION_MIN_MS,
        MOTION_SOURCE,
        MOTION_STYLE,
        MOTION_SWELL_PEAK,
        MOTION_TEMPO,
        MOTION_TILT_DEG,
        PAUSE_HOLD,
        PAUSE_HOLD_FULL,
        PAUSE_HOLD_MAX_FSP,
        PAUSE_HOLD_THRESH,
        REVEAL_MODE,
        SALIENCE_WEIGHTS,
        SAT_FLOOR_FRAC,
        SATURATION_INTENSITY,
        SLOPE_DEADZONE,
        SLOPE_FULL,
        SOFTMAX_TEMPERATURE,
        TRACKING_CALM,
        USE_CONFIDENCE_SCALING,
        WOBBLE_RANGE_HZ,
        ZERO_MEANS_MISSING,
    )


@app.cell
def _(colorsys, np):
    # CELL 13 — STYLE HELPERS (colour + blend, calm, pause gap, motion, styling)
    def _emotion_hsv(emotion, styles):
        fam = styles.get(emotion, styles["neutral"])
        return fam["h"], fam["s"], fam["v"]

    def _hsv_to_ass(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(min(1.0, max(0.0, h)),
                                      min(1.0, max(0.0, s)),
                                      min(1.0, max(0.0, v)))
        R, G, B = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
        return f"&H{B:02X}{G:02X}{R:02X}&"     # ASS is &HBBGGRR&, NOT RGB

    def resolve_word_color(emo1, emo2, p1, p2, do_blend, blend_mode, pos_frac,
                           intensity, styles, saturation_intensity, sat_floor_frac,
                           blend_perword_swing):
        """Hue (possibly blended with the runner-up) then saturation scaled by
        intensity. Interpolation happens in RGB rather than around the hue
        circle, because hue interpolation between, say, red and blue would pass
        through green — a colour neither emotion claims."""
        h1, s1, v1 = _emotion_hsv(emo1, styles)
        if do_blend and emo2 is not None and emo1 != emo2:
            h2, s2, v2 = _emotion_hsv(emo2, styles)
            r1, g1, b1 = colorsys.hsv_to_rgb(h1, s1, v1)
            r2, g2, b2 = colorsys.hsv_to_rgb(h2, s2, v2)
            if blend_mode == "gradient":
                t = pos_frac
            else:
                denom = (p1 + p2) if (p1 + p2) > 1e-9 else 1.0
                base_w2 = p2 / denom
                # high-intensity words pull toward the winner, low-intensity
                # ones drift toward the runner-up
                lean = (float(intensity) - 0.5) * blend_perword_swing
                t = min(1.0, max(0.0, base_w2 - lean))
            r = r1 + (r2 - r1) * t
            g = g1 + (g2 - g1) * t
            b = b1 + (b2 - b1) * t
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
        else:
            h, s, v = h1, s1, v1

        if saturation_intensity:
            s = s * (sat_floor_frac + (1.0 - sat_floor_frac) *
                     min(1.0, max(0.0, float(intensity))))
        return _hsv_to_ass(h, s, v)

    def compute_calm(df):
        """calm = slow AND quiet, both measured within this clip. The product
        rather than a sum, so a word has to be both."""
        out = df.copy()
        words = out["word"].astype(str).str.strip().str.len().clip(lower=1)
        dur = out["duration"].astype(float).clip(lower=1e-3)
        rate = (words / dur).to_numpy(dtype=float)     # characters per second
        rms = out["rms"].astype(float).to_numpy()

        def _norm01(x):
            # 5th-95th percentile, so one outlier cannot compress the scale
            x = np.asarray(x, dtype=float)
            lo, hi = np.nanpercentile(x, 5), np.nanpercentile(x, 95)
            if hi - lo < 1e-9:
                return np.zeros_like(x)
            return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

        slow = 1.0 - _norm01(rate)
        quiet = 1.0 - _norm01(rms)
        out["calm"] = slow * quiet
        return out

    def pause_gap(pause_after, thresh, full, gap_max):
        """Extra \\fsp px for the gap AFTER a word, from its pause_after.
        0 below thresh; linear thresh->full; capped at gap_max. Validated in
        test_pause.py (monotonic, bounded, silent below threshold)."""
        pa = float(pause_after or 0.0)
        if pa < thresh:
            return 0.0
        frac = min(1.0, (pa - thresh) / max(full - thresh, 1e-9))
        return round(gap_max * frac, 1)

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
        # a wobble's strength comes from pitch RANGE, not slope
        _wob = df["gesture"] == "wobble"
        df.loc[_wob, "motion_strength"] = (
            (df.loc[_wob, "f0_range"] / (wobble_range_hz * 2.0)).clip(0.3, 1.0)
        )
        df.loc[df["gesture"] == "none", "motion_strength"] = 0.0
        return df

    def assign_styles_v10(words_df, seg_emotion_df, styles,
                          base_font, font_swing, bold_thresh,
                          saturation_intensity, sat_floor_frac,
                          blend_mode, blend_margin, blend_perword_swing,
                          tracking_calm, calm_spacing_max):
        df = words_df.copy()

        emo1 = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df["pred_emotion"]))
        emo2 = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df.get("pred_emotion2")))
        p1   = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df["p_top"]))
        p2   = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df.get("p_second", 0.0)))
        conf = dict(zip(seg_emotion_df["segment_id"], seg_emotion_df["conf_scale"]))

        df["emotion"]    = df["segment_id"].map(emo1).fillna("neutral")
        df["emotion2"]   = df["segment_id"].map(emo2)
        df["p_top"]      = df["segment_id"].map(p1).fillna(0.0)
        df["p_second"]   = df["segment_id"].map(p2).fillna(0.0)
        df["conf_scale"] = df["segment_id"].map(conf).fillna(1.0)

        # each word's OWN segment's confidence, not one number for the file
        df["intensity"] = (df["intensity_raw"].astype(float) * df["conf_scale"]).clip(0.0, 1.0)

        df["font_size"] = (base_font + font_swing * df["intensity"]).round().astype(int)
        df["bold"] = (df["intensity"] >= bold_thresh).astype(int)

        df["font"]   = df["emotion"].apply(lambda e: styles.get(e, styles["neutral"])["font"])
        df["italic"] = df["emotion"].apply(lambda e: int(styles.get(e, styles["neutral"])["i"]))
        df["anim"]   = df["emotion"].apply(lambda e: styles.get(e, styles["neutral"])["anim"])

        if tracking_calm:
            df = compute_calm(df)
            df["tracking"] = (calm_spacing_max * df["calm"]).round(2)
        else:
            df["calm"] = 0.0
            df["tracking"] = 0.0

        # position within the segment, for BLEND_MODE = "gradient"
        _pos = df.groupby("segment_id").cumcount()
        _size = df.groupby("segment_id")["word"].transform("size")
        df["_pos_frac"] = np.where(_size > 1, _pos / (_size - 1).clip(lower=1), 0.0)

        def _is_ambiguous(row):
            e2 = row["emotion2"]
            if blend_mode == "off" or e2 is None:
                return False
            if isinstance(e2, float) and np.isnan(e2):
                return False
            if row["emotion"] == e2:
                return False
            return (float(row["p_top"]) - float(row["p_second"])) <= blend_margin

        def _color(row):
            return resolve_word_color(
                row["emotion"], row["emotion2"], float(row["p_top"]),
                float(row["p_second"]), _is_ambiguous(row), blend_mode,
                float(row["_pos_frac"]), float(row["intensity"]), styles,
                saturation_intensity, sat_floor_frac, blend_perword_swing)

        df["color_ass"] = df.apply(_color, axis=1)
        df["blended"] = df.apply(_is_ambiguous, axis=1)
        df.drop(columns=["_pos_frac"], inplace=True)
        return df

    return assign_styles_v10, attach_motion, pause_gap


@app.cell
def _(EMOTION_STYLES, subprocess):
    # CELL 14 — FONT AVAILABILITY CHECK
    # libass falls back silently on a missing family, so a wrong name never
    # crashes -- it just quietly ignores the font choice.
    try:
        _installed = subprocess.run(["fc-list"], capture_output=True, text=True).stdout.lower()
        for _emo, _fam in EMOTION_STYLES.items():
            _ok = _fam["font"].lower() in _installed
            print(f"{_emo:10s} -> {_fam['font']:24s} {'OK' if _ok else 'MISSING (libass will fall back)'}")
    except FileNotFoundError:
        print("fc-list not found; cannot verify fonts on this machine.")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step C1: Clip-level emotion prediction (legacy single-clip path)
    """)
    return


@app.cell
def _(
    BLEND_MARGIN,
    CLF_NORMALISED,
    USE_CONFIDENCE_SCALING,
    audio_file,
    clf_feature_cols,
    clf_features_from_path,
    clf_full,
    np,
    pd,
):
    # CELL 15 — PREDICT THE CLIP'S EMOTION (top-2, so the blend works here too)
    # This whole-file path cannot speaker-normalise: one vector, nothing to
    # normalise against. Since the loaded model was trained on normalised
    # features, it is less reliable than the per-segment path.
    clip_pred_feats = clf_features_from_path(audio_file)
    clip_pred_vec = pd.DataFrame([clip_pred_feats])[clf_feature_cols].to_numpy()

    _proba = clf_full.predict_proba(clip_pred_vec)[0]
    _classes = np.array(clf_full.classes_)
    _order = np.argsort(_proba)[::-1]
    pred_emotion  = str(_classes[_order[0]])
    pred_emotion2 = str(_classes[_order[1]]) if len(_order) > 1 else None
    p_top    = float(_proba[_order[0]])
    p_second = float(_proba[_order[1]]) if len(_order) > 1 else 0.0

    if USE_CONFIDENCE_SCALING:
        chance_level = 1.0 / len(clf_full.classes_)
        conf_scale = float(0.5 + 0.5 * np.clip(
            (p_top - chance_level) / (0.5 - chance_level), 0.0, 1.0))
    else:
        conf_scale = 1.0

    if CLF_NORMALISED:
        print("note: single-clip path cannot speaker-normalise; use process_any_video "
              "for the reliable per-segment prediction.")
    _margin = p_top - p_second
    print(f"Clip-level prediction: {pred_emotion.upper()} "
          f"(p_top {p_top:.2f}, 2nd {pred_emotion2} {p_second:.2f}, "
          f"margin {_margin:.2f} -> {'BLEND' if _margin <= BLEND_MARGIN else 'single'})")
    proba_table = pd.DataFrame(
        {"emotion": clf_full.classes_, "probability": np.round(_proba, 3)}
    ).sort_values("probability", ascending=False).reset_index(drop=True)
    proba_table
    return conf_scale, p_second, p_top, pred_emotion, pred_emotion2


@app.cell
def _(conf_scale, pred_emotion, true_emotion):
    # CELL 16 — PREDICTION vs LABEL VERDICT
    _verdict = "MATCH" if pred_emotion == true_emotion else "MISMATCH"
    print(f"model predicted : {pred_emotion}")
    print(f"dataset label   : {true_emotion}")
    print(f"confidence scale: {conf_scale:.2f}")
    print(f"--> {_verdict}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step C2: Salience → competitive 100-point budget
    """)
    return


@app.cell
def _(np):
    # CELL 17 — BUDGET MACHINERY (synthetic-test validated, unchanged)
    def assign_words_to_segments(words_df, segments):
        """Segment whose [start, end] window contains the word's midpoint, with
        a nearest-segment fallback for alignment drift."""
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
        """Weighted sum of |z| of each feature against the word's OWN sentence.
        Features in `zero_missing` treat an exact 0 as "no data"."""
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
        """Softmax split of a fixed 100-point budget per segment, with a floor.
        The exponential makes it competitive: because the pool is fixed, one
        word's gain is everyone else's loss."""
        df = words_df.copy()
        df["points"] = 0.0
        seg_ids = df["segment_id"].values
        for sid in np.unique(seg_ids):
            m = seg_ids == sid
            s = df.loc[m, "salience"].values
            n = int(m.sum())
            floor = min(min_points, 100.0 / n)   # can't promise more than an even split
            pool = 100.0 - floor * n
            logits = s / max(temperature, eps)
            logits = logits - logits.max()       # numerical stability for exp()
            w = np.exp(logits)
            w = w / w.sum()
            df.loc[m, "points"] = floor + pool * w
        fair = df.groupby("segment_id")["points"].transform(lambda p: 100.0 / len(p))
        df["share_ratio"] = df["points"] / fair
        return df

    return allocate_points, assign_words_to_segments, compute_salience


@app.cell
def _(
    FULL_DRAMA_RATIO,
    MIN_POINTS,
    SALIENCE_WEIGHTS,
    SOFTMAX_TEMPERATURE,
    ZERO_MEANS_MISSING,
    aligned,
    allocate_points,
    assign_words_to_segments,
    compute_salience,
    np,
    result,
    word_df,
):
    # CELL 18 — RUN THE BUDGET
    # Stores intensity_raw only; confidence is applied per segment in styling.
    seg_list = aligned.get("segments") or result["segments"]

    tagged_word_df = word_df.copy()
    tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

    salient_word_df = compute_salience(
        tagged_word_df, SALIENCE_WEIGHTS, zero_missing=ZERO_MEANS_MISSING)
    budget_df = allocate_points(
        salient_word_df, temperature=SOFTMAX_TEMPERATURE, min_points=MIN_POINTS)
    budget_df["intensity_raw"] = np.clip(
        (budget_df["share_ratio"] - 1.0) / (FULL_DRAMA_RATIO - 1.0), 0.0, 1.0)

    for _sid in np.unique(budget_df["segment_id"].values):
        _m = budget_df["segment_id"] == _sid
        print(f"Segment {_sid}: {int(_m.sum())} words, "
              f"points sum = {budget_df.loc[_m, 'points'].sum():.2f}, "
              f"top word = '{budget_df.loc[_m].sort_values('points').iloc[-1]['word'].strip()}'")
    budget_df[["word", "segment_id", "pause_after", "salience", "share_ratio", "intensity_raw"]].round(2)
    return budget_df, seg_list


@app.cell
def _(mo):
    mo.md("""
    ## Part D: Style mapping — the four channels applied
    """)
    return


@app.cell
def _(
    BASE_FONT_SIZE,
    BLEND_MARGIN,
    BLEND_MODE,
    BLEND_PERWORD_SWING,
    BOLD_THRESHOLD,
    CALM_SPACING_MAX,
    EMOTION_STYLES,
    FONT_SWING,
    MOTION_MIN_INTENSITY,
    MOTION_SOURCE,
    PAUSE_HOLD,
    PAUSE_HOLD_THRESH,
    SAT_FLOOR_FRAC,
    SATURATION_INTENSITY,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    TRACKING_CALM,
    WOBBLE_RANGE_HZ,
    assign_styles_v10,
    attach_motion,
    budget_df,
    conf_scale,
    p_second,
    p_top,
    pd,
    pred_emotion,
    pred_emotion2,
):
    # CELL 19 — LEGACY SINGLE-CLIP STYLING
    # Fakes a one-row-per-segment emotion table from the clip-level prediction,
    # so the single-clip path can go through the same styling function as the
    # per-segment path rather than keeping a second copy of it.
    _seg_ids = sorted(budget_df["segment_id"].unique().tolist())
    seg_emotion_single = pd.DataFrame([{
        "segment_id": sid, "pred_emotion": pred_emotion, "pred_emotion2": pred_emotion2,
        "p_top": p_top, "p_second": p_second, "conf_scale": conf_scale}
        for sid in _seg_ids])

    styled_word_df = assign_styles_v10(
        budget_df, seg_emotion_single, EMOTION_STYLES,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING, bold_thresh=BOLD_THRESHOLD,
        saturation_intensity=SATURATION_INTENSITY, sat_floor_frac=SAT_FLOOR_FRAC,
        blend_mode=BLEND_MODE, blend_margin=BLEND_MARGIN,
        blend_perword_swing=BLEND_PERWORD_SWING,
        tracking_calm=TRACKING_CALM, calm_spacing_max=CALM_SPACING_MAX)
    styled_word_df = attach_motion(
        styled_word_df, motion_source=MOTION_SOURCE,
        motion_min_intensity=MOTION_MIN_INTENSITY, slope_deadzone=SLOPE_DEADZONE,
        slope_full=SLOPE_FULL, wobble_range_hz=WOBBLE_RANGE_HZ)

    _moving = int((styled_word_df["gesture"] != "none").sum())
    _blended = int(styled_word_df["blended"].sum())
    _held = int((styled_word_df["pause_after"] >= PAUSE_HOLD_THRESH).sum()) if PAUSE_HOLD else 0
    print(f"emotion={pred_emotion} | blended={_blended}/{len(styled_word_df)} "
          f"| moving={_moving} | held pauses={_held} "
          f"| calm {styled_word_df['calm'].min():.2f}-{styled_word_df['calm'].max():.2f}")
    styled_word_df[["word", "intensity", "color_ass", "tracking", "pause_after",
                    "gesture", "font_size", "blended"]].round(2)
    return (styled_word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part E: Render .ass + FFmpeg burn-in

    The held pause is implemented by a shared `held_separator()` that chooses
    the gap between two words. On a qualifying pause it emits `{\\fsp<gap>}`
    around the single separating space, so that space stretches. The next word's
    own block re-declares `\\fsp` (its calm value), which bounds the hold to
    exactly that gap and stops it leaking — the same re-declare-every-tag
    discipline every other channel already uses. No extra glyphs, so the caption
    stays word-for-word. Only "sentence" mode uses it.
    """)
    return


@app.cell
def _(
    CAPTION_MODE,
    DIM_ALPHA,
    EMOTION_STYLES,
    HOLD_MAX_TAIL,
    MIN_LINE_DURATION,
    MOTION_MAX_MS,
    MOTION_MIN_MS,
    MOTION_STYLE,
    MOTION_SWELL_PEAK,
    MOTION_TEMPO,
    MOTION_TILT_DEG,
    PAUSE_HOLD,
    PAUSE_HOLD_FULL,
    PAUSE_HOLD_MAX_FSP,
    PAUSE_HOLD_THRESH,
    REVEAL_MODE,
    audio_file,
    os,
    out_tag,
    pause_gap,
    pred_emotion,
    styled_word_df,
    subprocess,
):
    # CELL 20 — SHARED HELD-SEPARATOR + LEGACY RENDER (black screen)
    def held_separator(pause_after, pause_hold, thresh, full, gap_max):
        """Separator between two words: a normal space, unless the pause earns a
        hold, in which case the space is widened via \\fsp."""
        if not pause_hold:
            return " "
        gap = pause_gap(pause_after, thresh, full, gap_max)
        if gap <= 0.0:
            return " "
        return f" {{\\fsp{gap:g}}} "   # next word's block resets \fsp, bounding it

    def build_segment_text(seg_rows, tags_fn, line_start, pause_hold,
                           thresh, full, gap_max):
        """Join a segment's word blocks, inserting held separators BETWEEN words
        only — never after the last word, where pause_after reports the gap to
        the next segment and a hold would be a duplicate line break."""
        rws = list(seg_rows.iterrows())
        chunks = []
        for i, (_, rw) in enumerate(rws):
            chunks.append("{" + tags_fn(rw, line_start) + "}" + str(rw["word"]).strip())
            if i < len(rws) - 1:
                chunks.append(held_separator(rw.get("pause_after", 0.0), pause_hold,
                                             thresh, full, gap_max))
        return "".join(chunks)

    def render_budget_video(audio_path, df, emotion, styles, out_dir="outputs",
                            caption_mode="sentence", reveal_mode="wipe",
                            hold_max_tail=0.6, min_line_dur=1.0, dim_alpha=150,
                            motion_style="scale", swell_peak=30, tilt_deg=7,
                            motion_min_ms=200, motion_max_ms=700, motion_tempo=None,
                            pause_hold=True, pause_thresh=0.40, pause_full=1.20,
                            pause_gap_max=40.0, tag="demo"):
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
            "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {width}\nPlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Cap,Liberation Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)
        lines = []

        def word_tags(rw, line_start):
            # \fsp carries this word's calm tracking; \fscx100\fscy100\frz0 are
            # RESETS, because .ass tags persist along a line and a previous
            # word's swell, tilt or held gap would otherwise leak into this one.
            _fsp = float(rw.get("tracking", 0.0) or 0.0)
            tags = (f"\\fn{fam['font']}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}"
                    f"\\fsp{_fsp:g}\\fscx100\\fscy100\\frz0")
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
                        tags += f"\\t({d0},{mid},\\fscy{int(100 + swell_peak * k)})\\t({mid},{end},\\fscy100)"
                    elif g == "drop":
                        tags += f"\\t({d0},{mid},\\fscy{int(100 - (swell_peak*0.6)*k)})\\t({mid},{end},\\fscy100)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        hi = int(100 + (swell_peak*0.5)*k); lo = int(100 - (swell_peak*0.4)*k)
                        tags += (f"\\t({d0},{d0+q},\\fscy{hi})\\t({d0+q},{d0+2*q},\\fscy{lo})"
                                 f"\\t({d0+2*q},{end},\\fscy100)")
                else:
                    t_ = int(round(tilt_deg * k))
                    if g == "lift":
                        tags += f"\\frz{t_}\\t({d0},{end},\\frz0)"
                    elif g == "drop":
                        tags += f"\\frz-{t_}\\t({d0},{end},\\frz0)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        tags += (f"\\t({d0},{d0+q},\\frz{t_})\\t({d0+q},{d0+2*q},\\frz-{t_})"
                                 f"\\t({d0+2*q},{end},\\frz0)")
            return tags

        if caption_mode == "word":
            for i, rw in rows.iterrows():
                start_t = float(rw["start"])
                if i < len(rows) - 1:
                    end_t = min(float(rows.loc[i + 1, "start"]), float(rw["end"]) + hold_max_tail)
                else:
                    end_t = float(rw["end"]) + 0.35
                text = "{\\an5" + word_tags(rw, start_t) + "}" + str(rw["word"]).strip()
                lines.append(f"Dialogue: 0,{sec_to_ass(start_t)},{sec_to_ass(end_t)},Cap,,0,0,0,,{text}")
        else:
            seg_ids = list(dict.fromkeys(rows["segment_id"].tolist()))
            seg_starts = {s: float(rows.loc[rows["segment_id"] == s, "start"].min()) for s in seg_ids}
            for si, sid in enumerate(seg_ids):
                seg = rows[rows["segment_id"] == sid].sort_values("start")
                s0 = float(seg["start"].min()); last_end = float(seg["end"].max())
                e0 = max(last_end + hold_max_tail, s0 + min_line_dur)
                if si < len(seg_ids) - 1:
                    e0 = min(e0, seg_starts[seg_ids[si + 1]])   # never overlap next line
                e0 = max(e0, s0 + 0.10)
                body = build_segment_text(seg, word_tags, s0, pause_hold,
                                          pause_thresh, pause_full, pause_gap_max)
                lines.append(f"Dialogue: 0,{sec_to_ass(s0)},{sec_to_ass(e0)},Cap,,0,0,0,,"
                             + "{\\fad(120,120)}" + body)

        ass_path = f"{out_dir}/ass/{tag}.ass"
        with open(ass_path, "w") as f:
            f.write(header + "\n".join(lines) + "\n")

        duration = float(rows["end"].max()) + 0.8
        out_path = f"{out_dir}/video/{tag}.mp4"
        cmd = ["ffmpeg", "-y", "-f", "lavfi",
               "-i", f"color=c=black:s={width}x{height}:r=25:d={duration}",
               "-i", audio_path, "-map", "0:v", "-map", "1:a", "-vf", f"ass={ass_path}",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print("FFmpeg return code:", proc.returncode)
        if proc.returncode != 0:
            print(proc.stderr[-2000:])
        print(f"{len(lines)} caption line(s) | motion={motion_style} | wrote {out_path}")
        return out_path, ass_path

    out_video, out_ass = render_budget_video(
        audio_file, styled_word_df, pred_emotion, EMOTION_STYLES,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE, hold_max_tail=HOLD_MAX_TAIL,
        min_line_dur=MIN_LINE_DURATION, dim_alpha=DIM_ALPHA, motion_style=MOTION_STYLE,
        swell_peak=MOTION_SWELL_PEAK, tilt_deg=MOTION_TILT_DEG, motion_min_ms=MOTION_MIN_MS,
        motion_max_ms=MOTION_MAX_MS, motion_tempo=MOTION_TEMPO,
        pause_hold=PAUSE_HOLD, pause_thresh=PAUSE_HOLD_THRESH, pause_full=PAUSE_HOLD_FULL,
        pause_gap_max=PAUSE_HOLD_MAX_FSP, tag=out_tag + "_" + CAPTION_MODE + "_" + MOTION_STYLE)
    print("Wrote:", out_video, "and", out_ass)
    return build_segment_text, held_separator


@app.cell
def _(
    NORM_MIN_SEGMENTS,
    SEGMENT_NORM,
    audio_file,
    clf_feature_cols,
    clf_full,
    predict_segment_emotions_v9,
    seg_list,
):
    # CELL 21 — PER-SEGMENT PREDICTION on the test clip (standalone display)
    # A one-utterance clip usually falls below NORM_MIN_SEGMENTS, so the
    # `normalised` column reading False here is expected.
    seg_emotion_df = predict_segment_emotions_v9(
        audio_file, seg_list, clf_full, clf_feature_cols,
        normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
    seg_emotion_df
    return


@app.cell
def _(build_segment_text, json, os, subprocess):
    # CELL 22 — RENDER onto a REAL video (colour and font change per segment),
    # sharing the same held-separator logic as CELL 20.
    def get_video_info(path):
        """width, height, duration, has_audio -- via ffprobe, no guessing."""
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True)
        info = json.loads(probe.stdout)
        w = info["streams"][0]["width"]; h = info["streams"][0]["height"]
        dur = float(info["format"]["duration"])
        audio_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "json", path],
            capture_output=True, text=True)
        has_audio = len(json.loads(audio_probe.stdout).get("streams", [])) > 0
        return w, h, dur, has_audio

    def render_long_video(audio_path, df, styles, out_dir="outputs",
                          caption_mode="sentence", reveal_mode="wipe",
                          hold_max_tail=0.6, min_line_dur=1.0, dim_alpha=150,
                          motion_style="scale", swell_peak=30, tilt_deg=7,
                          motion_min_ms=200, motion_max_ms=700, motion_tempo=None,
                          pause_hold=True, pause_thresh=0.40, pause_full=1.20,
                          pause_gap_max=40.0, bg_video_path=None, tag="demo_video"):
        """Colour, font and anim are read PER WORD from df rather than from one
        fixed emotion, because a long clip has more than one. bg_video_path=None
        falls back to the black screen."""
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
            _fsp = float(rw.get("tracking", 0.0) or 0.0)
            tags = (f"\\fn{rw['font']}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}"
                    f"\\fsp{_fsp:g}\\fscx100\\fscy100\\frz0")
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
                        tags += f"\\t({d0},{mid},\\fscy{int(100 + swell_peak * k)})\\t({mid},{end},\\fscy100)"
                    elif g == "drop":
                        tags += f"\\t({d0},{mid},\\fscy{int(100 - (swell_peak*0.6)*k)})\\t({mid},{end},\\fscy100)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        hi = int(100 + (swell_peak*0.5)*k); lo = int(100 - (swell_peak*0.4)*k)
                        tags += (f"\\t({d0},{d0+q},\\fscy{hi})\\t({d0+q},{d0+2*q},\\fscy{lo})"
                                 f"\\t({d0+2*q},{end},\\fscy100)")
                else:
                    t_ = int(round(tilt_deg * k))
                    if g == "lift":
                        tags += f"\\frz{t_}\\t({d0},{end},\\frz0)"
                    elif g == "drop":
                        tags += f"\\frz-{t_}\\t({d0},{end},\\frz0)"
                    elif g == "wobble":
                        q = max(1, dur // 3)
                        tags += (f"\\t({d0},{d0+q},\\frz{t_})\\t({d0+q},{d0+2*q},\\frz-{t_})"
                                 f"\\t({d0+2*q},{end},\\frz0)")
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
            body = build_segment_text(seg, word_tags, s0, pause_hold,
                                      pause_thresh, pause_full, pause_gap_max)
            lines.append(f"Dialogue: 0,{sec_to_ass(s0)},{sec_to_ass(e0)},Cap,,0,0,0,,"
                         + "{\\fad(120,120)}" + body)

        ass_path = f"{out_dir}/ass/{tag}.ass"
        with open(ass_path, "w") as f:
            f.write(header + "\n".join(lines) + "\n")

        out_path = f"{out_dir}/video/{tag}.mp4"
        if bg_video_path:
            # duration comes from the VIDEO ITSELF, not the transcript -- real
            # footage can run past the last word and should not be truncated.
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

    return (render_long_video,)


@app.cell
def _(
    BASE_FONT_SIZE,
    BLEND_MARGIN,
    BLEND_MODE,
    BLEND_PERWORD_SWING,
    BOLD_THRESHOLD,
    CALM_SPACING_MAX,
    CAPTION_MODE,
    DIM_ALPHA,
    EMOTION_STYLES,
    FONT_SWING,
    FULL_DRAMA_RATIO,
    HOLD_MAX_TAIL,
    MIN_LINE_DURATION,
    MIN_POINTS,
    MOTION_MAX_MS,
    MOTION_MIN_INTENSITY,
    MOTION_MIN_MS,
    MOTION_SOURCE,
    MOTION_STYLE,
    MOTION_SWELL_PEAK,
    MOTION_TEMPO,
    MOTION_TILT_DEG,
    NORM_MIN_SEGMENTS,
    PAUSE_HOLD,
    PAUSE_HOLD_FULL,
    PAUSE_HOLD_MAX_FSP,
    PAUSE_HOLD_THRESH,
    Path,
    REVEAL_MODE,
    SALIENCE_WEIGHTS,
    SAT_FLOOR_FRAC,
    SATURATION_INTENSITY,
    SEGMENT_NORM,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    SOFTMAX_TEMPERATURE,
    TRACKING_CALM,
    WOBBLE_RANGE_HZ,
    ZERO_MEANS_MISSING,
    allocate_points,
    asr_model,
    assign_styles_v10,
    assign_words_to_segments,
    attach_motion,
    clf_feature_cols,
    clf_full,
    compute_salience,
    device,
    extract_word_features,
    np,
    os,
    predict_segment_emotions_v9,
    render_long_video,
    subprocess,
    whisperx,
):
    # CELL 23 — "INSERT ANY VIDEO": full pipeline, real video out.
    # Needs CELLS 0-7 already run once, so clf_full, clf_feature_cols,
    # asr_model and the predictor exist.
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

        print("[5/6] emotion per segment + styling")
        seg_emotion_df = predict_segment_emotions_v9(
            extracted_audio, seg_list, clf_full, clf_feature_cols,
            normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
        styled_df = assign_styles_v10(
            budget_df, seg_emotion_df, EMOTION_STYLES,
            base_font=BASE_FONT_SIZE, font_swing=FONT_SWING, bold_thresh=BOLD_THRESHOLD,
            saturation_intensity=SATURATION_INTENSITY, sat_floor_frac=SAT_FLOOR_FRAC,
            blend_mode=BLEND_MODE, blend_margin=BLEND_MARGIN,
            blend_perword_swing=BLEND_PERWORD_SWING,
            tracking_calm=TRACKING_CALM, calm_spacing_max=CALM_SPACING_MAX)
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
            pause_hold=PAUSE_HOLD, pause_thresh=PAUSE_HOLD_THRESH, pause_full=PAUSE_HOLD_FULL,
            pause_gap_max=PAUSE_HOLD_MAX_FSP,
            bg_video_path=(video_path if use_bg_video else None), tag=tag)

        _held = int((styled_df["pause_after"] >= PAUSE_HOLD_THRESH).sum()) if PAUSE_HOLD else 0
        print("emotions found:", seg_emotion_df["pred_emotion"].value_counts().to_dict())
        print(f"segments blended: {int(styled_df['blended'].any())} | "
              f"calm spacing: {bool((styled_df['tracking'] > 0).any())} | "
              f"held pauses: {_held}")
        if "normalised" in seg_emotion_df:
            print("normalisation applied:", bool(seg_emotion_df["normalised"].any()))
        print("wrote:", out_path)
        return out_path, ass_path, seg_emotion_df, styled_df

    return (process_any_video,)


@app.cell
def _(iemocap_dir, os, process_any_video):
    # CELL 24 — DEMO: pull a whole conversation off IEMOCAP. A full dialog has
    # enough segments for the normalisation to engage and for more than one
    # emotion to appear.
    def find_iemocap_dialog_video(utt_id, iemocap_dir):
        """Full dialog recordings live at {Session}/dialog/avi/DivX/{dialog}.avi
        -- the same `dialog` derivation CELL 1 uses. None if this copy of the
        release doesn't ship dialog/avi/."""
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
        print("No dialog video at the expected path -- pass any video path straight "
              "into process_any_video('/path/to/clip.mp4').")
    return


@app.cell
def _(os, process_any_video):
    # CELL 25 — RUN ON A NEW VIDEO
    # Not RAVDESS, not IEMOCAP -- a real film clip in the project root.
    new_video_path = "12AngryMenTest.mp4"

    if not os.path.exists(new_video_path):
        print(f"Can't find '{new_video_path}' from {os.getcwd()}. cd to the project "
              "or set new_video_path to the full path.")
    else:
        angry_men_out, angry_men_ass, angry_men_seg_emotions, angry_men_styled_df = \
            process_any_video(new_video_path)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Decision log

    1. **Saturation now carries intensity, alongside typography.** This is a
       relaxation of the one-variable-per-channel rule and worth being explicit
       about. The defence is that colour is not one variable: hue and saturation
       are separately perceptible, so hue can keep saying *which* emotion while
       saturation says *how strongly*, without either contradicting the other.
       `SAT_FLOOR_FRAC = 0.5` means a word never drops below half the emotion's
       saturation, so a low-intensity word still reads as that emotion rather
       than washing out to grey. Set `SATURATION_INTENSITY = False` for the A/B
       arm that keeps the strict rule.
    2. **The classifier's uncertainty is shown rather than hidden.** Argmax
       throws away the fact that the model was nearly undecided. When the top
       two probabilities are within `BLEND_MARGIN`, the word colour interpolates
       between both emotions' hues. A viewer who cannot hear the audio has no
       way to check a wrong colour, so showing a hard hue for a near-coin-flip
       prediction misrepresents the system's own confidence.
    3. **Blending happens in RGB, not around the hue circle.** Interpolating hue
       between red and blue would pass through green — a colour neither emotion
       claims. Converting both to RGB, mixing, and converting back keeps the
       result on the line between the two actual colours.
    4. **The blend leans with intensity.** `BLEND_PERWORD_SWING` pulls
       high-intensity words toward the winning emotion and lets low-intensity
       ones drift toward the runner-up, so within one ambiguous line the
       prosodically strong words carry the model's best guess.
    5. **Letter spacing carries calm, defined as slow AND quiet.** The product
       of the two normalised signals rather than the sum, so a word has to be
       both. Normalisation is 5th-95th percentile clipped, because one outlier
       would otherwise compress everything else into a narrow band. This exists
       because the low-arousal end of the range previously had nothing of its
       own — it was expressed only by the absence of everything else.
    6. **Silence is rendered as held space, not punctuation.** A pause of
       `PAUSE_HOLD_THRESH` or more (0.4s, conservative, so only intentional
       beats fire) widens the gap before the next word via `\\fsp` on the single
       separating space, scaled linearly to `PAUSE_HOLD_FULL` and capped at
       `PAUSE_HOLD_MAX_FSP` so a bad alignment gap cannot blow the line open.
       Dots were rejected: inserting characters the speaker never said desyncs a
       deaf viewer's read from the audio and disrupts the per-word wipe timing,
       which undermines the accessibility case the whole project rests on.
    7. **Held space is a genuinely new kind of channel.** Every other channel
       encodes something about the words; this one encodes the space between
       them. It uses `pause_after`, which has been extracted since the per-word
       prosody was first written and excluded from salience ever since, because
       it is structurally 0 on a segment's last word.
    8. **The hold is bounded by the same discipline as every other tag.** The
       separator emits `\\fsp<gap>` and the next word's block re-declares its own
       `\\fsp`, so the widened space cannot leak down the line. `build_segment_text`
       inserts separators between words only — never after the last word, where
       `pause_after` reports the gap to the next segment and a hold would
       duplicate a line break that already exists.
    9. **Both renderers share one implementation of the hold.** `held_separator`
       and `build_segment_text` are defined once and used by the black-screen
       and real-video paths, so the two cannot drift apart.
    """)
    return


if __name__ == "__main__":
    app.run()

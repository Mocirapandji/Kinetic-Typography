import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # =====================================================================
    # TranscriberV11: V10 + held-space on intentional pauses
    # New: a long pause (>= PAUSE_HOLD_THRESH, default 0.4s) widens the gap
    # BEFORE the next word, proportional to pause length and capped. Uses
    # pause_after, already extracted since V7. No new glyphs are inserted, so
    # the caption stays word-for-word faithful to the audio (accessibility).
    # Only the line assembly in the two renderers changes; budget, motion,
    # colour, saturation, tracking and the classifier path are all V10.
    # =====================================================================

    # =====================================================================
    # CELL 0 — IMPORTS
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
    import opensmile
    import joblib

    mo.md(r"""
    # TranscriberV11: silence as a channel

    V10 spent the better classifier on colour, saturation and tracking. V11
    adds the one expressive dimension nothing else touched: **silence**.

    A pause longer than `PAUSE_HOLD_THRESH` (default 0.4s, so only clearly
    intentional beats) widens the gap before the next word, scaled by how long
    the pause is and capped so one bad alignment gap can't blow the line open.
    The held space reads as anticipation or reluctance — the caption holding
    its breath.

    Deliberately NOT dots: inserting "..." would put characters on screen the
    speaker never said, desyncing a deaf viewer's read from the audio and
    fighting the wipe timing. Widening the gap keeps the caption word-for-word
    faithful while still making the silence visible. It's also a genuinely new
    *temporal* signal — every other channel encodes something about the words,
    not the space between them.
    """)

    # =====================================================================
    # CELL 1 — CONFIG + DATASET SWITCH
    # =====================================================================
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
        _utt = "Ses01F_impro01_F012"
        _utt = _utt.replace(".wav", "")
        _sess = f"Session{int(_utt[3:5])}"
        _dialog = _utt.rsplit("_", 1)[0]
        audio_file = f"{iemocap_dir}/{_sess}/sentences/wav/{_dialog}/{_utt}.wav"

    out_tag = "v11_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
    print(f"dataset={DATASET}\nclip={audio_file}\nfeature cache: {features_csv}")

    # =====================================================================
    # CELL 2 — IEMOCAP CLIP PICKER
    # =====================================================================
    _iem_all = pd.read_csv(iemocap_csv)
    print(f"{len(_iem_all)} labelled IEMOCAP utterances available")
    _iem_all.groupby("emotion").head(3)[["file", "emotion", "actor"]]

    # =====================================================================
    # CELL 3 — GROUND-TRUTH LABEL for the current clip
    # =====================================================================
    _clip_name = audio_file.split("/")[-1].replace(".wav", "")
    if DATASET == "ravdess":
        true_emotion = emotion_map.get(_clip_name.split("-")[2], "unknown")
    else:
        _lookup = pd.read_csv(iemocap_csv)
        _hit = _lookup[_lookup["file"].astype(str).str.contains(_clip_name)]
        true_emotion = _hit["emotion"].iloc[0] if len(_hit) else "NOT IN CSV"
    print(f"dataset label for this clip: {true_emotion}")

    mo.md("""## Part A: Clip-level classifier (load the V2 model)""")

    # =====================================================================
    # CELL 4 — CLIP-LEVEL 14-FEATURE EXTRACTOR (fallback model + 14-feat path)
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
    # CELL 5 — LOAD 14-FEATURE CACHE (only used by the fallback model)
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
    # =====================================================================
    model_bundle_path = "outputs/clf_v2.joblib"

    if os.path.exists(model_bundle_path):
        _bundle = joblib.load(model_bundle_path)
        clf_full         = _bundle["clf"]
        clf_feature_cols = _bundle["feature_cols"]
        CLF_EXTRACTOR    = _bundle["extractor"]
        CLF_NORMALISED   = _bundle["speaker_normalised"]
        print(f"Loaded clf_v2: extractor={CLF_EXTRACTOR}, "
              f"{len(clf_feature_cols)} features, "
              f"speaker_normalised={CLF_NORMALISED}, classes={list(clf_full.classes_)}")
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

    # =====================================================================
    # CELL 7 — eGeMAPS extraction + per-segment prediction (top-2 for blend)
    # =====================================================================
    SEGMENT_NORM = "auto"      # "auto" | "on" | "off"
    NORM_MIN_SEGMENTS = 4

    if CLF_EXTRACTOR == "egemaps":
        _smile_v9 = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        _smile_v9 = None

    def clf_features_from_path(path):
        if CLF_EXTRACTOR == "egemaps":
            return _smile_v9.process_file(str(path)).iloc[0].to_dict()
        return extract_clip_features(path)

    def clf_features_from_sound(seg_snd, tmp_wav="outputs/audio/_seg_tmp.wav"):
        if CLF_EXTRACTOR == "egemaps":
            os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)
            call(seg_snd, "Save as WAV file", tmp_wav)
            return _smile_v9.process_file(tmp_wav).iloc[0].to_dict()
        return extract_clip_features_from_sound(seg_snd)

    def predict_segment_emotions_v9(audio_path, segments, clf, feature_cols,
                                    normalise_mode="auto", norm_min_segments=4):
        snd_full = parselmouth.Sound(audio_path)
        chance_level = 1.0 / len(clf.classes_)
        classes = np.array(clf.classes_)

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
        do_norm = do_norm and CLF_NORMALISED

        if not feat_df.empty and do_norm:
            _cols = [c for c in feature_cols if c in feat_df.columns]
            _mu = feat_df[_cols].mean()
            _sd = feat_df[_cols].std().replace(0.0, 1.0)
            feat_df[_cols] = ((feat_df[_cols] - _mu) / _sd).fillna(0.0)

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

    mo.md("""## Part B: WhisperX transcription → word timestamps + per-word prosody""")

    # =====================================================================
    # CELL 8 — WHISPERX TRANSCRIPTION
    # =====================================================================
    asr_model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = asr_model.transcribe(audio, batch_size=16)
    print("LANGUAGE:", result["language"])
    print("SEGMENTS:", result["segments"])

    # =====================================================================
    # CELL 9 — WORD-LEVEL ALIGNMENT
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
    # CELL 10 — PER-WORD PROSODY EXTRACTOR (+ f0 slope) — GUARDED
    # pause_after (used by V11's held space) is computed here, as it has been
    # since V7: the gap to the NEXT word's start.
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

            if len(f0v) >= 3:
                _t = np.linspace(0.0, max(duration, 1e-6), len(f0v))
                _semitones = 12.0 * np.log2(f0v / f0v[0])
                f0_slope = float(np.polyfit(_t, _semitones, 1)[0])
            else:
                f0_slope = 0.0

            try:
                rms = call(word_snd, "Get root-mean-square", 0, 0)
                rms = 0.0 if rms != rms else rms
            except Exception:
                rms = 0.0

            try:
                harm = word_snd.to_harmonicity()
                hnr_vals = harm.values[harm.values != -200]
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

    # =====================================================================
    # CELL 11 — RUN PER-WORD EXTRACTION
    # =====================================================================
    word_df = extract_word_features(audio_file, aligned["word_segments"])
    word_df

    mo.md("""## Part C: The expressive budget + all styling dials""")

    # =====================================================================
    # CELL 12 — TUNABLE PARAMETERS (V10 dials + V11 pause channel)
    # =====================================================================
    # ----- budget (unchanged since V3) -----
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}
    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5
    USE_CONFIDENCE_SCALING = True

    # ----- TYPOGRAPHY channel: intensity (size + weight) -----
    BASE_FONT_SIZE = 32
    FONT_SWING = 32
    BOLD_THRESHOLD = 0.60

    # ----- OPACITY channel: time only -----
    CAPTION_MODE = "sentence"
    REVEAL_MODE = "wipe"
    DIM_ALPHA = 150
    HOLD_MAX_TAIL = 0.6
    MIN_LINE_DURATION = 1.0

    # ----- MOTION channel: pitch direction -----
    MOTION_SOURCE = "pitch"
    MOTION_MIN_INTENSITY = 0.45
    SLOPE_DEADZONE = 2.0
    SLOPE_FULL = 12.0
    WOBBLE_RANGE_HZ = 90.0
    MOTION_STYLE = "scale"
    MOTION_SWELL_PEAK = 30
    MOTION_TILT_DEG = 7
    MOTION_MIN_MS, MOTION_MAX_MS = 200, 700
    MOTION_TEMPO = {"pop": 0.80, "soft": 1.50, "flat": 1.00}

    # ----- V10 CHANNEL 1: SATURATION = INTENSITY -----
    SATURATION_INTENSITY = True
    SAT_FLOOR_FRAC = 0.5

    # ----- V10 CHANNEL 2: UNCERTAINTY HUE BLEND -----
    BLEND_MODE = "blend"        # "off" | "blend" | "gradient"
    BLEND_MARGIN = 0.20
    BLEND_PERWORD_SWING = 0.30

    # ----- V10 CHANNEL 3: LETTER SPACING = CALM -----
    TRACKING_CALM = True
    CALM_SPACING_MAX = 6.0

    # ===================== V11 CHANNEL 4: HELD SPACE = SILENCE ============
    # A pause of >= PAUSE_HOLD_THRESH seconds widens the gap BEFORE the next
    # word. Gap scales linearly from 0 at THRESH to PAUSE_HOLD_MAX_FSP px at
    # PAUSE_HOLD_FULL seconds, then caps -- so a 5s alignment artefact and a
    # 1.5s dramatic pause hold the same maximum beat. Conservative threshold
    # (0.4s) so only intentional pauses fire, not inter-word micro-gaps.
    # Only affects "sentence" caption mode (words share a line); "word" mode
    # shows one word at a time so there is no gap to widen. Set False for the
    # V10 A/B arm.
    PAUSE_HOLD = True
    PAUSE_HOLD_THRESH = 0.40    # seconds; below this, no hold
    PAUSE_HOLD_FULL = 1.20      # seconds; pause at/above this = max hold
    PAUSE_HOLD_MAX_FSP = 40.0   # px of extra \fsp on the held gap at full

    # ----- COLOUR channel base per emotion (hue fixed; sat a ceiling) -----
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
    # CELL 13 — V10/V11 STYLE HELPERS (colour, calm, PAUSE GAP, motion, styling)
    # =====================================================================
    def _emotion_hsv(emotion, styles):
        fam = styles.get(emotion, styles["neutral"])
        return fam["h"], fam["s"], fam["v"]

    def _hsv_to_ass(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(min(1.0, max(0.0, h)),
                                      min(1.0, max(0.0, s)),
                                      min(1.0, max(0.0, v)))
        R, G, B = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
        return f"&H{B:02X}{G:02X}{R:02X}&"     # ASS is &HBBGGRR&

    def resolve_word_color(emo1, emo2, p1, p2, do_blend, blend_mode, pos_frac,
                           intensity, styles, saturation_intensity, sat_floor_frac,
                           blend_perword_swing):
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
        out = df.copy()
        words = out["word"].astype(str).str.strip().str.len().clip(lower=1)
        dur = out["duration"].astype(float).clip(lower=1e-3)
        rate = (words / dur).to_numpy(dtype=float)
        rms = out["rms"].astype(float).to_numpy()

        def _norm01(x):
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
            do_blend = _is_ambiguous(row)
            return resolve_word_color(
                row["emotion"], row["emotion2"], float(row["p_top"]),
                float(row["p_second"]), do_blend, blend_mode, float(row["_pos_frac"]),
                float(row["intensity"]), styles, saturation_intensity,
                sat_floor_frac, blend_perword_swing)

        df["color_ass"] = df.apply(_color, axis=1)
        df["blended"] = df.apply(_is_ambiguous, axis=1)
        df.drop(columns=["_pos_frac"], inplace=True)
        return df

    # =====================================================================
    # CELL 14 — FONT AVAILABILITY CHECK
    # =====================================================================
    try:
        _installed = subprocess.run(["fc-list"], capture_output=True, text=True).stdout.lower()
        for _emo, _fam in EMOTION_STYLES.items():
            _ok = _fam["font"].lower() in _installed
            print(f"{_emo:10s} -> {_fam['font']:24s} {'OK' if _ok else 'MISSING (libass will fall back)'}")
    except FileNotFoundError:
        print("fc-list not found; cannot verify fonts on this machine.")

    mo.md("""### Step C1: Clip-level emotion prediction (legacy single-clip path)""")

    # =====================================================================
    # CELL 15 — PREDICT THE CLIP'S EMOTION (top-2 for the blend)
    # =====================================================================
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

    # =====================================================================
    # CELL 16 — PREDICTION vs LABEL VERDICT
    # =====================================================================
    _verdict = "MATCH" if pred_emotion == true_emotion else "MISMATCH"
    print(f"model predicted : {pred_emotion}")
    print(f"dataset label   : {true_emotion}")
    print(f"confidence scale: {conf_scale:.2f}")
    print(f"--> {_verdict}")

    mo.md("""### Step C2: Salience → competitive 100-point budget""")

    # =====================================================================
    # CELL 17 — BUDGET MACHINERY (unchanged)
    # =====================================================================
    def assign_words_to_segments(words_df, segments):
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
    # CELL 18 — RUN THE BUDGET (stores intensity_raw; conf applied in styling)
    # =====================================================================
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

    mo.md("""## Part D: Style mapping — V10 channels + V11 pause""")

    # =====================================================================
    # CELL 19 — LEGACY SINGLE-CLIP STYLING via assign_styles_v10
    # =====================================================================
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

    mo.md(r"""
    ## Part E: Render .ass + FFmpeg burn-in

    V11 render change: a shared `held_separator()` chooses the gap between two
    words. On a pause >= `PAUSE_HOLD_THRESH` it emits `{\fsp<gap>} ` so the
    single space between the words is stretched; the next word's block
    re-declares `\fsp` (its calm value), so the hold is bounded to exactly that
    gap and can't leak — the same re-declare-every-tag discipline every other
    channel uses. No extra glyphs, so the caption stays word-for-word. Only
    "sentence" mode uses it; "word" mode shows one word at a time.
    """)

    # =====================================================================
    # CELL 20 — SHARED HELD-SEPARATOR + LEGACY RENDER (black screen)
    # =====================================================================
    def held_separator(pause_after, pause_hold, thresh, full, gap_max):
        """Separator string between two words. A normal space, unless the pause
        earns a hold, in which case the space is widened via \\fsp."""
        if not pause_hold:
            return " "
        gap = pause_gap(pause_after, thresh, full, gap_max)
        if gap <= 0.0:
            return " "
        return f" {{\\fsp{gap:g}}} "   # next word's block resets \fsp, bounding it

    def build_segment_text(seg_rows, tags_fn, line_start, pause_hold,
                           thresh, full, gap_max):
        """Join a segment's word blocks, inserting held separators BETWEEN words
        only (never after the last word)."""
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
                            pause_gap_max=40.0, tag="v11_demo"):
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
                    e0 = min(e0, seg_starts[seg_ids[si + 1]])
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

    v11_video, v11_ass = render_budget_video(
        audio_file, styled_word_df, pred_emotion, EMOTION_STYLES,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE, hold_max_tail=HOLD_MAX_TAIL,
        min_line_dur=MIN_LINE_DURATION, dim_alpha=DIM_ALPHA, motion_style=MOTION_STYLE,
        swell_peak=MOTION_SWELL_PEAK, tilt_deg=MOTION_TILT_DEG, motion_min_ms=MOTION_MIN_MS,
        motion_max_ms=MOTION_MAX_MS, motion_tempo=MOTION_TEMPO,
        pause_hold=PAUSE_HOLD, pause_thresh=PAUSE_HOLD_THRESH, pause_full=PAUSE_HOLD_FULL,
        pause_gap_max=PAUSE_HOLD_MAX_FSP, tag=out_tag + "_" + CAPTION_MODE + "_" + MOTION_STYLE)
    print("Wrote:", v11_video, "and", v11_ass)

    # =====================================================================
    # CELL 21 — PER-SEGMENT PREDICTION on the test clip (standalone display)
    # =====================================================================
    seg_emotion_df = predict_segment_emotions_v9(
        audio_file, seg_list, clf_full, clf_feature_cols,
        normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
    seg_emotion_df

    # =====================================================================
    # CELL 22 — RENDER onto a REAL video (per-segment colour) — held pauses
    # =====================================================================
    def get_video_info(path):
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
                          pause_gap_max=40.0, bg_video_path=None, tag="v11_demo"):
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
    # CELL 23 — "INSERT ANY VIDEO": full pipeline (V11 styling), real video out
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

        print("[5/6] emotion per segment (V9 model) + V10/V11 styling")
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
            pause_gap_max=PAUSE_HOLD_MAX_FSP, bg_video_path=(video_path if use_bg_video else None), tag=tag)

        _held = int((styled_df["pause_after"] >= PAUSE_HOLD_THRESH).sum()) if PAUSE_HOLD else 0
        print("emotions found:", seg_emotion_df["pred_emotion"].value_counts().to_dict())
        print(f"segments blended: {int(styled_df['blended'].any())} | "
              f"calm spacing: {bool((styled_df['tracking'] > 0).any())} | "
              f"held pauses: {_held}")
        if "normalised" in seg_emotion_df:
            print("normalisation applied:", bool(seg_emotion_df["normalised"].any()))
        print("wrote:", out_path)
        return out_path, ass_path, seg_emotion_df, styled_df

    # =====================================================================
    # CELL 24 — DEMO: pull a whole conversation off IEMOCAP
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
        print("No dialog video at the expected path -- pass any video path straight "
              "into process_any_video('/path/to/your/clip.mp4').")

    # =====================================================================
    # CELL 25 — RUN ON A NEW VIDEO: 12AngryMenTest.mp4
    # =====================================================================
    new_video_path = "12AngryMenTest.mp4"

    if not os.path.exists(new_video_path):
        print(f"Can't find '{new_video_path}' from {os.getcwd()}. cd to the project "
              "or set new_video_path to the full path.")
    else:
        angry_men_out, angry_men_ass, angry_men_seg_emotions, angry_men_styled_df = \
            process_any_video(new_video_path)

    mo.md(r"""
    ## Decision log — V11

    24. **Silence is rendered as held space, not punctuation.** A pause of
        `PAUSE_HOLD_THRESH` seconds or more (default 0.4s, conservative, so only
        intentional beats fire) widens the gap before the next word via `\fsp`
        on the single separating space, scaled linearly to `PAUSE_HOLD_FULL` and
        capped at `PAUSE_HOLD_MAX_FSP` so a bad alignment gap cannot blow the
        line open. Dots/ellipses were rejected: inserting characters the speaker
        never said desyncs a deaf viewer's read from the audio and disrupts the
        per-word wipe timing, undermining the accessibility case. Widening the
        gap keeps the caption word-for-word faithful while making the silence
        visible, and is a genuinely new *temporal* channel — the only one
        encoding the space between words rather than the words. Uses
        `pause_after`, extracted since V7. Held space applies only in "sentence"
        caption mode (words share a line); "word" mode shows one word at a time.
        The pause-to-gap mapping is validated for monotonicity, bounds and the
        between-words-only edge case (no trailing gap after a line's last word).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

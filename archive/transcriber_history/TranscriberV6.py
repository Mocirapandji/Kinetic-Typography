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
    import colorsys
    import subprocess
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import parselmouth
    from parselmouth.praat import call
    import whisperx
    from sklearn.ensemble import RandomForestClassifier

    return (
        Path,
        RandomForestClassifier,
        call,
        colorsys,
        mo,
        np,
        os,
        parselmouth,
        pd,
        subprocess,
        whisperx,
    )


@app.cell
def _(mo):
    mo.md("""
    # TranscriberV6: karaoke wipe + gated emphasis + trailing voice

    Everything from V5 (literature-anchored palette, V&M intensity ramp,
    caption plates, per-emotion fonts), with the animation system rebuilt:

    - **Wipe reveal**: every word rests half-transparent and blooms to full
      opacity OVER ITS OWN SPOKEN DURATION — the opacity front crawls across
      the line in sync with the voice (the clipping-mask effect, done at word
      granularity so no pixel/font-metric math is needed).
    - **Gated emphasis**: only budget winners (intensity >= EMPHASIS_TRIGGER)
      animate, and the move is emotion-specific — happy/surprised words
      bounce, angry words stretch wider. Everything else just wipes.
    - **Trailing voice**: words that are quiet relative to their own sentence
      never reach full opacity, so the caption visibly fades where the
      speaker's voice wavers or drops.
    """)
    return


@app.cell
def _(os):
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
    features_csv = "outputs/features.csv"              # RAVDESS-only model
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

    out_tag = "v6_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
    print(f"dataset={DATASET}\nclip={audio_file}\nmodel trained from: {features_csv}")
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
        out_tag,
    )


@app.cell
def _(iemocap_csv, pd):
    # =====================================================================
    # CELL 2 — IEMOCAP CLIP PICKER
    # Browse labelled utterances, copy a "file" value into CELL 1's _utt.
    # =====================================================================
    _iem_all = pd.read_csv(iemocap_csv)
    print(f"{len(_iem_all)} labelled IEMOCAP utterances available")
    _iem_all.groupby("emotion").head(3)[["file", "emotion", "actor"]]
    return


@app.cell
def _(DATASET, audio_file, emotion_map, iemocap_csv, pd):
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
    return (true_emotion,)


@app.cell
def _(mo):
    mo.md("""
    ## Part A: Clip-level classifier (features + training)
    """)
    return


@app.cell
def _(call, np, parselmouth):
    # =====================================================================
    # CELL 4 — CLIP-LEVEL FEATURE EXTRACTOR (identical to train_emotion.py)
    # =====================================================================
    def extract_clip_features(path):
        snd = parselmouth.Sound(str(path))
        dur = snd.get_total_duration()

        # --- pitch (F0), voiced frames only ---
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

        # --- intensity (dB) ---
        try:
            intensity = snd.to_intensity()
            ivals = intensity.values.flatten()
            ivals = ivals[np.isfinite(ivals)]
            int_mean = float(np.mean(ivals)); int_std = float(np.std(ivals))
            int_max = float(np.max(ivals)); int_min = float(np.min(ivals))
        except Exception:
            int_mean = int_std = int_max = int_min = 0.0

        # --- overall RMS energy ---
        rms = float(call(snd, "Get root-mean-square", 0, 0))

        # --- HNR (voice quality) ---
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
    # =====================================================================
    # CELL 5 — LOAD TRAINING FEATURES (from cache; extracts only if missing)
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
    return (clip_df,)


@app.cell
def _(RandomForestClassifier, clip_df):
    # =====================================================================
    # CELL 6 — TRAIN THE CLIP-LEVEL RANDOM FOREST
    # Trains on ALL cached clips; the honest speaker-independent number
    # (46.8% vs 14.3% chance) lives in train_emotion.py and is not re-run here.
    # =====================================================================
    clip_feature_cols = [c for c in clip_df.columns if c not in ("file", "emotion", "actor")]
    X_clip = clip_df[clip_feature_cols].values
    y_clip = clip_df["emotion"].values

    clf_full = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ).fit(X_clip, y_clip)

    print(f"Trained clf_full on {len(clip_df)} clips / {len(clip_feature_cols)} features.")
    print("Classes:", list(clf_full.classes_))
    return clf_full, clip_feature_cols


@app.cell
def _(mo):
    mo.md("""
    ## Part B: WhisperX transcription → word timestamps + per-word prosody
    """)
    return


@app.cell
def _(audio_file, compute_type, device, whisperx):
    # =====================================================================
    # CELL 7 — WHISPERX TRANSCRIPTION
    # =====================================================================
    asr_model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = asr_model.transcribe(audio, batch_size=16)
    print("LANGUAGE:", result["language"])
    print("SEGMENTS:", result["segments"])
    return audio, result


@app.cell
def _(audio, device, result, whisperx):
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
    return (aligned,)


@app.cell
def _(call, parselmouth, pd):
    # =====================================================================
    # CELL 9 — PER-WORD PROSODY EXTRACTOR
    # =====================================================================
    def extract_word_features(audio_path, word_segments):
        snd = parselmouth.Sound(audio_path)
        rows = []

        for i, w in enumerate(word_segments):
            start, end = float(w["start"]), float(w["end"])
            duration = end - start

            # pause length = gap from this word's end to the NEXT word's start
            if i < len(word_segments) - 1:
                pause_after = float(word_segments[i + 1]["start"]) - end
            else:
                pause_after = 0.0  # last word, no following gap

            word_snd = snd.extract_part(from_time=start, to_time=end, preserve_times=True)

            # pitch (F0) - drop unvoiced frames (0 Hz)
            pitch = word_snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0v = f0[f0 > 0]
            f0_mean = float(f0v.mean()) if len(f0v) else 0.0
            f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0

            # intensity / loudness (RMS energy)
            rms = call(word_snd, "Get root-mean-square", 0, 0)

            # voice quality (HNR) - drop undefined frames (-200 dB sentinel)
            harm = word_snd.to_harmonicity()
            hnr_vals = harm.values[harm.values != -200]
            hnr = float(hnr_vals.mean()) if len(hnr_vals) else 0.0

            rows.append({
                "word": w["word"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "pause_after": round(pause_after, 3),
                "f0_mean": round(f0_mean, 1),
                "f0_range": round(f0_range, 1),
                "rms": round(rms, 4),
                "hnr": round(hnr, 1),
            })

        return pd.DataFrame(rows)

    return (extract_word_features,)


@app.cell
def _(aligned, audio_file, extract_word_features):
    # =====================================================================
    # CELL 10 — RUN PER-WORD EXTRACTION
    # =====================================================================
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
    # =====================================================================
    # CELL 11 — TUNABLE PARAMETERS
    # =====================================================================
    # ----- budget (unchanged) -----
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}
    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5
    USE_CONFIDENCE_SCALING = True

    # =====================================================================
    # ONE VARIABLE PER CHANNEL. This is the design rule now.
    #
    #   COLOUR (hue + sat + value) -> EMOTION. Constant across the clip.
    #   TYPOGRAPHY (size + weight) -> INTENSITY. The ONLY intensity channel.
    #   OPACITY                    -> TIME. The karaoke wipe, nothing else.
    #   MOTION                     -> THE EXCEPTION. One word per sentence, max.
    #
    # Previously all four encoded intensity simultaneously, which is why a
    # loud word arrived bigger + bolder + darker + more saturated + bouncing.
    # Five cues for one variable reads as noise, not emphasis.
    # =====================================================================

    # ----- TYPOGRAPHY channel: the sole carrier of intensity -----
    # Range widened, because size no longer competes with colour to say
    # "this word matters" — it is now the only thing saying it.
    BASE_FONT_SIZE = 32
    FONT_SWING = 32            # 32..64 px
    BOLD_THRESHOLD = 0.60

    # ----- OPACITY channel: time only -----
    CAPTION_MODE = "sentence"
    REVEAL_MODE = "wipe"       # "wipe" | "snap" | "none"
    DIM_ALPHA = 150            # unspoken words (0=opaque, 255=invisible)
    HOLD_MAX_TAIL = 0.6
    MIN_LINE_DURATION = 1.0
    # NOTE: the trailing-voice fade is GONE. It was a second thing living on
    # the opacity channel, so a quiet word and an unspoken word looked alike.
    # Loudness already feeds salience, so quiet words are handled by the
    # budget -> typography path instead.

    # ----- MOTION channel: rare by construction -----
    # Fires ONLY on the single highest-scoring word in a sentence, and only if
    # that word also cleared the trigger. So: at most one moving word per
    # sentence, often zero. Motion is punctuation, not decoration.
    EMPHASIS_TRIGGER = 0.75
    EMPHASIS_ANIM = {
        "happy": "bounce", "surprised": "bounce",
        "angry": "stretch",
        "sad": "none", "fearful": "none", "disgust": "none", "neutral": "none",
    }
    BOUNCE_MIN_MS, BOUNCE_MAX_MS = 180, 650
    BOUNCE_TEMPO = {"pop": 0.75, "soft": 1.60, "flat": 1.00}
    BOUNCE_PEAK_BASE, BOUNCE_PEAK_GAIN = 118, 22
    STRETCH_X = 124
    STRETCH_SQUASH = 94

    # ----- COLOUR channel: one fixed colour per emotion, whole clip -----
    # hue      : Jonauskaite 37-nation survey (n=8,615), max-weight assignment
    # sat/value: Valdez & Mehrabian (1994) PAD regressions, inverted per
    #            emotion, then compressed into a legible band and separated
    #            for discriminability (sad vs fear, disgust vs surprised).
    # italic is now a CATEGORICAL marker (disgust is always italic), not an
    # intensity trigger — same reasoning as font: it identifies, not emphasises.
    EMOTION_STYLES = {
        #             hue     sat   val   italic  anim    font
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
        BOLD_THRESHOLD,
        BOUNCE_MAX_MS,
        BOUNCE_MIN_MS,
        BOUNCE_PEAK_BASE,
        BOUNCE_PEAK_GAIN,
        BOUNCE_TEMPO,
        CAPTION_MODE,
        DIM_ALPHA,
        EMOTION_STYLES,
        EMPHASIS_ANIM,
        EMPHASIS_TRIGGER,
        FONT_SWING,
        FULL_DRAMA_RATIO,
        HOLD_MAX_TAIL,
        MIN_LINE_DURATION,
        MIN_POINTS,
        REVEAL_MODE,
        SALIENCE_WEIGHTS,
        SOFTMAX_TEMPERATURE,
        STRETCH_SQUASH,
        STRETCH_X,
        USE_CONFIDENCE_SCALING,
        ZERO_MEANS_MISSING,
    )


@app.cell
def _(EMOTION_STYLES, subprocess):
    # =====================================================================
    # CELL 12 — FONT AVAILABILITY CHECK
    # libass silently falls back to a default when a font is missing, so a
    # wrong name never crashes — it just quietly ignores your font choice.
    # This cell reports which mapped fonts actually resolve on THIS machine.
    # Richer fonts: unzip O'Donovan's gwfonts.zip into ~/.fonts, run
    # `fc-cache -f`, then update the names in CELL 11.
    # =====================================================================
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
    ### Step C1: Clip-level emotion prediction (the "which emotion" decision)
    """)
    return


@app.cell
def _(
    USE_CONFIDENCE_SCALING,
    audio_file,
    clf_full,
    clip_feature_cols,
    extract_clip_features,
    np,
    pd,
):
    # =====================================================================
    # CELL 13 — PREDICT THE CLIP'S EMOTION (pred_emotion + conf_scale)
    # =====================================================================
    clip_pred_feats = extract_clip_features(audio_file)
    clip_pred_vec = pd.DataFrame([clip_pred_feats])[clip_feature_cols]  # enforces column order

    # .to_numpy(): clf_full was fitted on a plain array, so passing a DataFrame
    # would trigger a harmless-but-noisy sklearn feature-names warning.
    pred_emotion = str(clf_full.predict(clip_pred_vec.to_numpy())[0])
    pred_proba = clf_full.predict_proba(clip_pred_vec.to_numpy())[0]
    p_top = float(np.max(pred_proba))

    if USE_CONFIDENCE_SCALING:
        # top-class probability -> 0.5..1.0 multiplier: at chance level the
        # styling runs at half strength, at probability >= 0.5 at full.
        chance_level = 1.0 / len(clf_full.classes_)
        conf_scale = float(0.5 + 0.5 * np.clip(
            (p_top - chance_level) / (0.5 - chance_level), 0.0, 1.0))
    else:
        conf_scale = 1.0

    print(f"Clip-level prediction: {pred_emotion.upper()}  "
          f"(top probability {p_top:.2f}, confidence scale {conf_scale:.2f})")
    proba_table = pd.DataFrame(
        {"emotion": clf_full.classes_, "probability": np.round(pred_proba, 3)}
    ).sort_values("probability", ascending=False).reset_index(drop=True)
    proba_table
    return conf_scale, pred_emotion


@app.cell
def _(conf_scale, pred_emotion, true_emotion):
    # =====================================================================
    # CELL 14 — PREDICTION vs LABEL VERDICT
    # =====================================================================
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
    # =====================================================================
    # CELL 15 — BUDGET MACHINERY (synthetic-test validated, unchanged)
    # =====================================================================
    def assign_words_to_segments(words_df, segments):
        """Tag each word with the id of the sentence/segment it belongs to.

        A word belongs to the segment whose [start, end] window contains the
        word's midpoint, with a nearest-segment fallback for alignment drift.
        """
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
            floor = min(min_points, 100.0 / n)   # can't promise more than an even split
            pool = 100.0 - floor * n             # what's actually competed over
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
    conf_scale,
    np,
    result,
    word_df,
):
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

    # intensity in 0..1: 0 at fair share, 1 at FULL_DRAMA_RATIO x fair share,
    # restrained by classifier confidence.
    budget_df["intensity"] = np.clip(
        (budget_df["share_ratio"] - 1.0) / (FULL_DRAMA_RATIO - 1.0), 0.0, 1.0
    ) * conf_scale

    for _sid in np.unique(budget_df["segment_id"].values):
        _m = budget_df["segment_id"] == _sid
        print(f"Segment {_sid}: {int(_m.sum())} words, "
              f"points sum = {budget_df.loc[_m, 'points'].sum():.2f}, "
              f"top word = '{budget_df.loc[_m].sort_values('points').iloc[-1]['word'].strip()}'")

    budget_df[["word", "segment_id", "salience", "points", "share_ratio", "intensity"]].round(2)
    return (budget_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part D: Style mapping — literature-anchored HSV ramp + settle opacity

    Hue comes from the clip's emotion (survey-anchored). Intensity moves each
    word along the Valdez & Mehrabian arousal gradient: **saturation up, value
    down**. New in V6: each word also gets a **settle opacity** — words that
    are quiet relative to their own sentence never reach fully opaque, so the
    caption fades where the voice trails off.
    """)
    return


@app.cell
def _(
    BASE_FONT_SIZE,
    BOLD_THRESHOLD,
    EMOTION_STYLES,
    EMPHASIS_TRIGGER,
    FONT_SWING,
    budget_df,
    colorsys,
    pred_emotion,
):
    # =====================================================================
    # CELL 17 — STYLE MAPPING v5 (one variable per channel)
    # =====================================================================
    def emotion_color(emotion, styles):
        """One fixed colour for the whole clip. Colour says WHICH emotion —
        it does not also say which word matters. That job belongs to size."""
        fam = styles.get(emotion, styles["neutral"])
        r, g, b = (int(round(c * 255))
                   for c in colorsys.hsv_to_rgb(fam["h"], fam["s"], fam["v"]))
        return f"&H{b:02X}{g:02X}{r:02X}&"   # ASS colour is &HBBGGRR&, NOT RGB

    def assign_styles_v5(words_df, emotion, styles,
                         base_font, font_swing, bold_thresh, emph_trigger):
        df = words_df.copy()
        fam = styles.get(emotion, styles["neutral"])
        clip_color = emotion_color(emotion, styles)

        # COLOUR: identical for every word in the clip.
        df["color_ass"] = clip_color
        # ITALIC: categorical (disgust), not intensity-driven.
        df["italic"] = int(fam["i"])

        # TYPOGRAPHY: the only channel encoding intensity.
        df["font_size"] = (base_font + font_swing * df["intensity"]).round().astype(int)
        df["bold"] = (df["intensity"] >= bold_thresh).astype(int)

        # MOTION: the sentence's single top word, and only if it also cleared
        # the trigger. At most one moving word per sentence.
        top_idx = df.groupby("segment_id")["points"].idxmax()
        df["is_winner"] = False
        df.loc[top_idx, "is_winner"] = True
        df["moves"] = df["is_winner"] & (df["intensity"] >= emph_trigger)

        return df

    styled_word_df = assign_styles_v5(
        budget_df, pred_emotion, EMOTION_STYLES,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING,
        bold_thresh=BOLD_THRESHOLD, emph_trigger=EMPHASIS_TRIGGER,
    )
    _fam = EMOTION_STYLES.get(pred_emotion, EMOTION_STYLES["neutral"])
    print(f"emotion={pred_emotion} | colour={styled_word_df['color_ass'].iloc[0]} "
          f"| font={_fam['font']} | moving words={int(styled_word_df['moves'].sum())}")
    styled_word_df[["word", "intensity", "font_size", "bold", "moves"]]
    return (styled_word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part E: Render — wipe reveal, gated emphasis, trailing fade

    Mechanics worth knowing:

    - **Two styles in the header.** `CapDark` = outline captions straight on
      the video. `CapLight` = `BorderStyle=4` draws one continuous box behind
      the whole line (white plate, used when the emotion's evidence-based
      text colour is dark, i.e. fear). BorderStyle=4 is a libass extension:
      fine for FFmpeg burn-in (FFmpeg *is* libass), but the raw .ass would
      not show plates in a non-libass player.
    - **The wipe** animates each word's alpha over its real spoken interval,
      so the opacity front travels with the voice. A true pixel clipping mask
      (animated \clip) would need per-word x-positions and therefore font
      metrics; word-granular alpha gives the same read with none of that
      fragility.
    - **Emphasis moves** fire only above EMPHASIS_TRIGGER. The bounce is a
      \fscy-only pulse (width never changes, so no sideways reflow). The
      angry stretch uses \fscx and IS the one deliberate exception to the
      no-reflow rule: the line breathes outward for ~400 ms on one word. If
      it reads badly in motion, set EMPHASIS_ANIM["angry"] = "bounce".
    """)
    return


@app.cell
def _(
    BOUNCE_MAX_MS,
    BOUNCE_MIN_MS,
    BOUNCE_PEAK_BASE,
    BOUNCE_PEAK_GAIN,
    BOUNCE_TEMPO,
    CAPTION_MODE,
    DIM_ALPHA,
    EMOTION_STYLES,
    EMPHASIS_ANIM,
    HOLD_MAX_TAIL,
    MIN_LINE_DURATION,
    REVEAL_MODE,
    STRETCH_SQUASH,
    STRETCH_X,
    audio_file,
    os,
    out_tag,
    pred_emotion,
    styled_word_df,
    subprocess,
):
    # =====================================================================
    # CELL 18 — RENDER: wipe (time) + typography (intensity) + rare motion
    # =====================================================================
    def render_budget_video(audio_path, df, emotion, styles, out_dir="outputs",
                            caption_mode="sentence", reveal_mode="wipe",
                            hold_max_tail=0.6, min_line_dur=1.0,
                            dim_alpha=150, emphasis_anim=None,
                            bounce_min_ms=180, bounce_max_ms=650,
                            bounce_tempo=None, peak_base=118, peak_gain=22,
                            stretch_x=124, stretch_squash=94,
                            tag="v6_demo"):
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720
        fam = styles.get(emotion, styles["neutral"])
        tempo = (bounce_tempo or {"pop": 0.75, "soft": 1.6, "flat": 1.0})[fam["anim"]]
        anim_kind = (emphasis_anim or {}).get(emotion, "none")

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
            "Style: CapDark,Liberation Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)
        lines = []

        def word_tags(rw, line_start):
            # COLOUR + TYPOGRAPHY: colour is the clip's emotion; size/weight
            # carry intensity. These are set once and never animated.
            tags = (f"\\fn{fam['font']}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}")

            # OPACITY: time, and only time. Word blooms from dim to fully
            # opaque over its own spoken interval -> the front moves with the
            # voice. Every word ends at alpha 00; nothing else lives here.
            d0 = max(0, int(round((float(rw["start"]) - line_start) * 1000)))
            d1 = max(d0 + 120, int(round((float(rw["end"]) - line_start) * 1000)))
            if reveal_mode == "wipe":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d1},\\alpha&H00&)"
            elif reveal_mode == "snap":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d0 + 90},\\alpha&H00&)"

            # MOTION: only the sentence's winner, only if it cleared the
            # trigger (`moves` was decided in Cell 17). Usually 0-1 per line.
            if bool(rw.get("moves", False)) and anim_kind != "none":
                dur = int(min(max((float(rw["end"]) - float(rw["start"])) * 1000,
                                  bounce_min_ms), bounce_max_ms) * tempo)
                up, down = d0 + int(dur * 0.38), d0 + dur
                if anim_kind == "bounce":
                    peak = int(peak_base + peak_gain * float(rw["intensity"]))
                    tags += f"\\t({d0},{up},\\fscy{peak})\\t({up},{down},\\fscy100)"
                elif anim_kind == "stretch":
                    tags += (f"\\t({d0},{up},\\fscx{stretch_x}\\fscy{stretch_squash})"
                             f"\\t({up},{down},\\fscx100\\fscy100)")
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
                             f"CapDark,,0,0,0,,{text}")
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
                             f"CapDark,,0,0,0,," + "{\\fad(120,120)}" + " ".join(parts))

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
        print(f"{len(lines)} caption line(s) | {int(df['moves'].sum())} moving word(s) "
              f"({caption_mode} / {reveal_mode} / motion={anim_kind})")
        return out_path, ass_path

    v6_video, v6_ass = render_budget_video(
        audio_file, styled_word_df, pred_emotion, EMOTION_STYLES,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE,
        hold_max_tail=HOLD_MAX_TAIL, min_line_dur=MIN_LINE_DURATION,
        dim_alpha=DIM_ALPHA, emphasis_anim=EMPHASIS_ANIM,
        bounce_min_ms=BOUNCE_MIN_MS, bounce_max_ms=BOUNCE_MAX_MS,
        bounce_tempo=BOUNCE_TEMPO, peak_base=BOUNCE_PEAK_BASE,
        peak_gain=BOUNCE_PEAK_GAIN, stretch_x=STRETCH_X,
        stretch_squash=STRETCH_SQUASH,
        tag=out_tag + "_" + CAPTION_MODE + "_" + REVEAL_MODE,
    )
    print("Wrote:", v6_video, "and", v6_ass)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Decision log (V6 additions)

    1. **Wipe reveal replaces universal bounce**: every word blooms from
       half-transparent to opaque over its own spoken interval, so the
       opacity front moves with the voice. A literal pixel clipping mask
       (animated \clip) would require per-word x-positions and font metrics;
       word-granular alpha produces the same read robustly.
    2. **Emphasis animations are gated and emotion-specific**: only words
       with intensity >= 0.75 (fair-share-normalised, i.e. the "score above
       75-80" idea on a sentence-length-independent scale) animate.
       Happy/surprised winners bounce (\fscy pulse, no reflow); angry
       winners stretch wider (\fscx) — the one documented exception to the
       no-reflow rule, ~400 ms on a single word, swappable to bounce via one
       dict value if it reads badly in motion.
    3. **Trailing-voice fade**: per-word rms is normalised within its own
       sentence; words in the bottom 35% of the range settle short of full
       opacity (down to alpha 0x78 for the quietest). The caption visibly
       fades where the speaker's voice drops or wavers — an expressive cue no
       prior kinetic-caption system renders, and a direct differentiator from
       Google Expressive Captions' loudness-only formatting.
    4. **Everything upstream unchanged**: classifier, budget, prosody
       extraction, palette, plates, and fonts are exactly as in V5; V6 only
       changes how emotion + intensity + loudness map to motion and opacity.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    ## Decision log (V6)

    ### The governing rule: one variable per channel

    The system has four visual channels. Earlier versions had **all four
    encoding intensity at once** — a loud word arrived bigger, bolder, darker,
    more saturated AND bouncing. Five cues for one variable reads as noise, not
    emphasis, and it is the single reason the early renders looked bad. Each
    channel now carries exactly one thing:

    | Channel | Encodes | Varies |
    |---|---|---|
    | Colour (hue, saturation, value) | **which emotion** | per clip |
    | Typography (size, weight) | **word intensity** | per word |
    | Opacity | **time** (the karaoke wipe) | per word, continuously |
    | Motion | **the exception** | at most one word per sentence |

    Font family and italics are *categorical* markers, like colour: they
    identify the emotion, they never signal emphasis.

    ### Individual decisions

    1. **Colour is constant across a clip.** Valdez & Mehrabian measured how
       *a colour* makes a viewer feel, which maps to the *emotion* — not to
       word-level emphasis. Their PAD regressions are inverted per emotion to
       fix each one's saturation and value, then compressed into a legible
       band and separated for discriminability (sad vs fearful, disgust vs
       surprised). Using the same equations as a per-word ramp, as V5 did, was
       a misapplication of the source.
    2. **Hue palette from survey data.** Jonauskaite et al. 37-nation survey
       (n=8,615), assigned by maximum-weight bipartite matching: anger-red
       (54%), joy-yellow (54%), disgust-brown (37%). Two documented overrides:
       sadness-blue (survey rank 3, but rank 1 across 23 studies in the
       128-year systematic review, and it keeps sad and fearful
       discriminable in the mute test) and surprised-orange (surprise is
       absent from the Geneva Emotion Wheel, so it rests on review counts
       alone — the weakest anchor in the palette).
    3. **Legibility constraints are documented, not hidden.** Fear's
       evidence-based colour is black (48%), which is unreadable burned into
       video. A white caption plate was trialled (BorderStyle=4) and rejected
       as visually intrusive; fear now takes a desaturated cold blue-violet.
       The palette is therefore a *constrained* empirical selection, not a
       naive argmax, and the constraint is legibility.
    4. **Typography is the sole intensity channel**, so its range was widened
       (32-64 px) once it stopped competing with colour to say the same thing.
    5. **Opacity encodes time and nothing else.** Every word rests
       half-transparent and blooms to *fully* opaque over its own spoken
       interval, so the opacity front travels across the line with the voice.
       A literal pixel clipping mask (animated \clip) would need per-word
       x-positions and font metrics; word-granular alpha gives the same read
       without that fragility.
    6. **The trailing-voice fade was built, then removed.** Quiet words
       settling short of opacity was expressive, but it put a second variable
       on the opacity channel — so an unspoken word and a quiet word looked
       alike, which is exactly the confusion this whole redesign exists to
       eliminate. Loudness already feeds salience, so quiet words are handled
       through the budget -> typography path instead. Worth revisiting on a
       channel that is genuinely free.
    7. **Motion is rare by construction**: only the single top-scoring word in
       a sentence animates, and only if it also clears intensity 0.75. Often
       zero words per line. Motion is punctuation, not decoration.
       Happy/surprised bounce (\fscy pulse, no reflow); angry stretches wider
       (\fscx) — the one deliberate exception to the no-reflow rule, ~400 ms
       on one word.
    8. **Everything upstream is untouched**: classifier, budget, and prosody
       extraction are unchanged. Every version since V3 has only re-mapped
       emotion + intensity onto visuals.

    ### Known weakness (next iteration)

    Motion currently fires on an intensity threshold, which means it is the
    typography channel again, just jumpier. It should encode something size
    cannot: **pitch direction**. See V7.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

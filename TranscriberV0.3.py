import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # CELL 0 — IMPORTS (all imports live here, marimo convention)
    import marimo as mo
    import os
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
    # V0.3: dataset switch + bottom-aligned subtitles

    Same pipeline as V0.2 (clip-level emotion + per-word expressive budget), with
    two additions:

    - a **dataset switch**, so the same code captions either a RAVDESS clip
      (acted studio speech, what the model was trained on) or an IEMOCAP clip
      (conversational speech the model has never seen)
    - a **caption layout switch**: V0.2's one-word-at-a-time centre display, or a
      normal bottom-of-screen subtitle showing the whole sentence at once with
      per-word styling preserved across the line.

    The file is long enough now that cells are numbered, so the decision log and
    the markdown can point at specific ones.

    Note: very short IEMOCAP utterances ("Yeah.", "Okay.") can give WhisperX
    nothing to align. If word_segments comes back empty, pick a longer clip.
    """)
    return


@app.cell
def _(os):
    # CELL 1 — CONFIG + DATASET SWITCH

    # "ravdess" -> acted studio speech (what the model was trained on)
    # "iemocap" -> conversational speech (the harder, realistic test)
    DATASET = "iemocap"

    ravdess_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    iemocap_dir = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"
    data_dir = ravdess_dir  # CELL 5's glob still uses this if the cache is missing

    emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_emotions = {"calm"}

    # which cached features the clip-level classifier trains on
    features_csv = "outputs/features.csv"              # RAVDESS-only model
    # features_csv = "outputs/features_combined.csv"   # RAVDESS + IEMOCAP model

    # IEMOCAP labels CSV (built by extract_iemocap.py). Checks local outputs/
    # first, falls back to the copy on the T7.
    iemocap_csv = "outputs/features_iemocap.csv"
    if not os.path.exists(iemocap_csv):
        iemocap_csv = "/run/media/s5812886/T7 Shield/kinetic_outputs/features_iemocap.csv"

    device = "cpu"
    compute_type = "int8"

    if DATASET == "ravdess":
        audio_file = f"{ravdess_dir}/Actor_01/03-01-06-01-02-01-01.wav"
    else:
        _utt = "Ses01F_impro01_F012"   # paste any "file" value from CELL 2's table
        _utt = _utt.replace(".wav", "")
        _sess = f"Session{int(_utt[3:5])}"   # "Ses01F..." -> "Session1"
        _dialog = _utt.rsplit("_", 1)[0]     # "Ses01F_impro01_F012" -> "Ses01F_impro01"
        audio_file = f"{iemocap_dir}/{_sess}/sentences/wav/{_dialog}/{_utt}.wav"

    # names the output .ass/.mp4 so runs never overwrite each other
    out_tag = "v0_3_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
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
    # CELL 2 — IEMOCAP CLIP PICKER
    # Browse labelled utterances, copy a "file" value into CELL 1's _utt.
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
    ## Part A: Clip-level classifier (features + training)
    """)
    return


@app.cell
def _(call, np, parselmouth):
    # CELL 4 — CLIP-LEVEL FEATURE EXTRACTOR (identical to train_emotion.py, so
    # the fitted feature vector matches the one built at prediction time)
    def extract_clip_features(path):
        snd = parselmouth.Sound(str(path))
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
    # CELL 5 — LOAD TRAINING FEATURES (from cache; extracts only if missing)
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
    # CELL 6 — TRAIN THE CLIP-LEVEL RANDOM FOREST
    # Trains on ALL cached clips. The speaker-independent evaluation (46.8% vs
    # 14.3% chance) lives in train_emotion.py and is NOT re-run here, so this
    # notebook stays fast to open.
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
    # CELL 7 — WHISPERX TRANSCRIPTION
    asr_model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = asr_model.transcribe(audio, batch_size=16)
    print("LANGUAGE:", result["language"])
    print("SEGMENTS:", result["segments"])
    return audio, result


@app.cell
def _(audio, device, result, whisperx):
    # CELL 8 — WORD-LEVEL ALIGNMENT
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
    # CELL 9 — PER-WORD PROSODY EXTRACTOR
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

            # drop unvoiced frames (0 Hz)
            pitch = word_snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0v = f0[f0 > 0]
            f0_mean = float(f0v.mean()) if len(f0v) else 0.0
            f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0

            rms = call(word_snd, "Get root-mean-square", 0, 0)

            # drop undefined HNR frames (-200 dB sentinel)
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
    # CELL 10 — RUN PER-WORD EXTRACTION
    word_df = extract_word_features(audio_file, aligned["word_segments"])
    word_df
    return (word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part C: The expressive budget

    Each sentence gets a fixed budget of **100 points**. Words compete for those
    points based on how much their prosody stands out from the *other words in
    the same sentence*. One word taking a big share automatically leaves less
    for everyone else.
    """)
    return


@app.cell
def _():
    # CELL 11 — TUNABLE PARAMETERS

    # Pitch level, pitch movement and loudness are the classic acoustic cues of
    # spoken emphasis -> full weight. HNR is a real but noisier cue at word scale
    # -> half. Raw duration confounds emphasis with word length ("sitting" is
    # longer than "by" however it is said) -> half.
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}

    # For these, a value of exactly 0 means "no voiced frames found" -- absence of
    # information, not evidence of extreme pitch. Such words are excluded from the
    # segment statistics and contribute z = 0.
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}

    # LOWER = closer to winner-takes-all, HIGHER = flatter.
    SOFTMAX_TEMPERATURE = 1.5

    # Guaranteed floor per word, because a caption word can never visually
    # vanish. Auto-shrinks for long sentences (see allocate_points).
    MIN_POINTS = 2.0

    # Full intensity at this multiple of fair share (fair share = 100 / n words).
    FULL_DRAMA_RATIO = 2.5

    # Every word is tinted at least this far towards the emotion colour, so the
    # budget modulates how strongly a word expresses the emotion, never whether
    # the sentence reads as that emotion at all.
    BASE_EMOTION_TINT = 0.35

    BOLD_THRESHOLD = 0.55
    ANIM_TRIGGER = 0.55

    # size = BASE_FONT_SIZE + intensity * FONT_SWING.
    # Sized for SENTENCE mode (34..60 px), where a whole line shares the screen.
    # The one-word display in V0.2 could afford 48..92, but at that scale a full
    # sentence wraps onto three lines and looks broken.
    BASE_FONT_SIZE = 34
    FONT_SWING = 26

    # "sentence" = whole segment on one line at the BOTTOM of frame, like a real
    #              subtitle. Per-word colour/size/bold/italic still vary along it.
    # "word"     = one word at a time, centred (the V0.2 behaviour).
    CAPTION_MODE = "sentence"

    # Sentence mode only: do words light up as they are spoken?
    # "reveal" = unspoken words sit dimmed, snap to full opacity on their cue.
    # "none"   = the whole line appears fully styled at once (most subtitle-like).
    REVEAL_MODE = "reveal"

    # How long a caption lingers past its last word, and the shortest time any
    # caption stays up (so one-word utterances don't flash past).
    HOLD_MAX_TAIL = 0.6
    MIN_LINE_DURATION = 1.0

    # Styling is restrained when the classifier is unsure: top-class probability
    # at chance -> half strength, >= 0.5 -> full.
    USE_CONFIDENCE_SCALING = True

    # Hues reuse the palette from the earlier RF/MLP label-demo videos, for
    # continuity across deliverables.
    # anim: "pop"  = energetic scale-in on high-intensity words
    #       "soft" = slow fade, longer for higher intensity
    #       "flat" = plain quick fade
    # These animation families only apply in CAPTION_MODE = "word". In sentence
    # mode \\fad is a LINE-level .ass tag, so per-word fades and scale-pops are
    # impossible once all words share one Dialogue line.
    EMOTION_STYLES = {
        "angry":     {"rgb": (255,  60,  60), "anim": "pop"},
        "happy":     {"rgb": ( 60, 220,  60), "anim": "pop"},
        "surprised": {"rgb": ( 60, 230, 230), "anim": "pop"},
        "sad":       {"rgb": ( 80, 120, 255), "anim": "soft"},
        "fearful":   {"rgb": (200,  80, 220), "anim": "soft"},
        "disgust":   {"rgb": (255, 165,   0), "anim": "flat"},
        "neutral":   {"rgb": (255, 255, 255), "anim": "flat"},
    }
    return (
        ANIM_TRIGGER,
        BASE_EMOTION_TINT,
        BASE_FONT_SIZE,
        BOLD_THRESHOLD,
        CAPTION_MODE,
        EMOTION_STYLES,
        FONT_SWING,
        FULL_DRAMA_RATIO,
        HOLD_MAX_TAIL,
        MIN_LINE_DURATION,
        MIN_POINTS,
        REVEAL_MODE,
        SALIENCE_WEIGHTS,
        SOFTMAX_TEMPERATURE,
        USE_CONFIDENCE_SCALING,
        ZERO_MEANS_MISSING,
    )


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
    # CELL 12 — PREDICT THE CLIP'S EMOTION (pred_emotion + conf_scale)
    # Same extractor the classifier was trained with, on the same whole clip.
    # This is the only place emotion is classified; words never get their own
    # category, because no word-level ground truth exists to justify one.
    clip_pred_feats = extract_clip_features(audio_file)
    clip_pred_vec = pd.DataFrame([clip_pred_feats])[clip_feature_cols]  # enforces column order

    # .to_numpy(): clf_full was fitted on a plain array, so a DataFrame would
    # trigger a harmless but noisy sklearn feature-names warning.
    pred_emotion = str(clf_full.predict(clip_pred_vec.to_numpy())[0])
    pred_proba = clf_full.predict_proba(clip_pred_vec.to_numpy())[0]
    p_top = float(np.max(pred_proba))

    if USE_CONFIDENCE_SCALING:
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
    # CELL 13 — PREDICTION vs LABEL VERDICT
    _verdict = "MATCH" if pred_emotion == true_emotion else "MISMATCH"
    print(f"model predicted : {pred_emotion}")
    print(f"dataset label   : {true_emotion}")
    print(f"confidence scale: {conf_scale:.2f}")
    print(f"--> {_verdict}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Step C2: Salience → competitive 100-point budget (the "how strongly" decision)
    """)
    return


@app.cell
def _(np):
    # CELL 14 — BUDGET MACHINERY (synthetic-test validated)
    def assign_words_to_segments(words_df, segments):
        """Tag each word with the id of the segment it belongs to.

        A word belongs to the segment whose [start, end] window contains its
        midpoint. If alignment drift leaves a word just outside every window it
        falls back to the NEAREST segment rather than crashing. For a
        single-sentence clip this collapses to one segment, but it generalises
        unchanged to longer multi-sentence audio.
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
        """How much a word's prosody stands out from ITS OWN sentence.

        Each feature is z-scored against the mean and standard deviation of the
        words in the same segment, not against global dataset statistics, then
        the absolute z-scores are summed with the given weights. |z| is used
        because standing out in either direction reads as emphasis. A feature
        with near-zero variance inside a segment carries no emphasis information
        there, so its z stays 0 rather than dividing by ~0. Features in
        `zero_missing` treat a raw 0 as "no data" and get z = 0.
        """
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
        """Split a fixed 100-point budget across each segment's words.

        A softmax over (salience / temperature) makes the split COMPETITIVE: the
        exponential means a word only moderately more salient than its
        neighbours still takes a disproportionately larger slice, and because
        the pool is fixed its gain is everyone else's loss.

        Every word is guaranteed a floor, which auto-shrinks for long sentences
        so it can never exceed an even split. Segment totals are always exactly
        100, and a single-word segment trivially takes all of it.
        """
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
        # share_ratio > 1 means the word out-competed its neighbours.
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
    # CELL 15 — RUN THE BUDGET on this clip's words
    # Prefer the aligned segment boundaries (refined timings); fall back to the
    # raw transcription segments if alignment returned none.
    seg_list = aligned.get("segments") or result["segments"]

    tagged_word_df = word_df.copy()
    tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

    salient_word_df = compute_salience(
        tagged_word_df, SALIENCE_WEIGHTS, zero_missing=ZERO_MEANS_MISSING
    )
    budget_df = allocate_points(
        salient_word_df, temperature=SOFTMAX_TEMPERATURE, min_points=MIN_POINTS
    )

    # 0 at or below fair share, 1 at FULL_DRAMA_RATIO x fair share, then
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
    ## Part D: Style mapping (emotion picks the family, budget picks the intensity)
    """)
    return


@app.cell
def _(
    ANIM_TRIGGER,
    BASE_EMOTION_TINT,
    BASE_FONT_SIZE,
    BOLD_THRESHOLD,
    EMOTION_STYLES,
    FONT_SWING,
    budget_df,
    conf_scale,
    pred_emotion,
):
    # CELL 16 — STYLE MAPPING (now emits an explicit `italic` column)
    def assign_styles_v2(words_df, emotion, styles, base_tint, confidence,
                         base_font, font_swing, bold_thresh, anim_trigger):
        """Map (clip emotion, per-word intensity) -> concrete .ass styling.

        The emotion picks the family (hue plus animation flavour); each word's
        intensity picks how far it moves from a plain white caption towards that
        family's full look. Low-intensity words stay near the tinted baseline,
        the budget winner gets max size, saturated colour, bold, and in word mode
        the family's animation.
        """
        df = words_df.copy()
        fam = styles.get(emotion, styles["neutral"])
        emo_rgb = fam["rgb"]
        anim_family = fam["anim"]

        def lerp(a, b, t):
            return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))

        def rgb_to_ass(r, g, b):
            return f"&H{b:02X}{g:02X}{r:02X}&"  # ASS colour is &HBBGGRR&, NOT RGB

        # Low classifier confidence pulls the whole sentence back towards white.
        tint_floor = base_tint * confidence
        sizes, colors, bolds, italics, anims = [], [], [], [], []
        for _, row in df.iterrows():
            inten = float(row["intensity"])
            t = tint_floor + (1.0 - tint_floor) * inten
            colors.append(rgb_to_ass(*lerp((255, 255, 255), emo_rgb, t)))
            sizes.append(int(round(base_font + font_swing * inten)))
            bolds.append(1 if inten >= bold_thresh else 0)

            # italic gets its OWN column, because it survives sentence mode
            # whereas the fade/scale tags below do not (they are line-level)
            italics.append(1 if (emotion == "disgust" and inten >= anim_trigger) else 0)

            # anim_ass is only consumed by CAPTION_MODE = "word"
            if anim_family == "pop" and inten >= anim_trigger:
                # lands at 55% size and springs to 100% in 140 ms
                anims.append("\\fscx55\\fscy55\\t(0,140,\\fscx100\\fscy100)")
            elif anim_family == "soft":
                anims.append(f"\\fad({int(150 + 300 * inten)},0)")
            else:
                anims.append("\\fad(80,0)")

        df["font_size"] = sizes
        df["color_ass"] = colors
        df["bold"] = bolds
        df["italic"] = italics
        df["anim_ass"] = anims
        return df

    styled_word_df = assign_styles_v2(
        budget_df, pred_emotion, EMOTION_STYLES,
        base_tint=BASE_EMOTION_TINT, confidence=conf_scale,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING,
        bold_thresh=BOLD_THRESHOLD, anim_trigger=ANIM_TRIGGER,
    )
    styled_word_df[["word", "points", "intensity", "font_size", "color_ass",
                    "bold", "italic", "anim_ass"]]
    return (styled_word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part E: Render .ass + FFmpeg burn-in

    In sentence mode the whole segment becomes ONE .ass Dialogue line, pinned to
    the bottom of frame (Alignment 2). Per-word colour, size, weight and italics
    still vary along that line, because .ass override tags can appear mid-line.

    What does NOT survive the move: `\\fad` is a LINE-level tag, so per-word
    fades and scale-pops are impossible once all the words share one line. The
    speech-synced dimension is kept instead by REVEAL_MODE = "reveal", which dims
    each word until its spoken moment. Alpha is animated rather than size,
    because changing size mid-line reflows the text and makes the whole subtitle
    jitter sideways.
    """)
    return


@app.cell
def _(
    CAPTION_MODE,
    HOLD_MAX_TAIL,
    MIN_LINE_DURATION,
    REVEAL_MODE,
    audio_file,
    os,
    out_tag,
    styled_word_df,
    subprocess,
):
    # CELL 17 — RENDER: bottom-aligned, one line per sentence
    def render_budget_video(audio_path, df, out_dir="outputs",
                            caption_mode="sentence", reveal_mode="reveal",
                            hold_max_tail=0.6, min_line_dur=1.0,
                            dim_alpha="&H80&", tag="budget_demo"):
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720

        def sec_to_ass(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            cs = int(round((t - int(t)) * 100))
            if cs == 100: cs = 0; s += 1
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        # Alignment 2 = bottom-centre (real subtitle position).
        # Alignment 5 = middle-centre (the one-word-at-a-time look).
        # WrapStyle 0 lets libass break a long line automatically.
        align = 2 if caption_mode == "sentence" else 5
        margin_v = 50 if caption_mode == "sentence" else 10

        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            f"PlayResX: {width}\n"
            f"PlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,{align},40,40,{margin_v},1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)
        lines = []

        if caption_mode == "word":
            for i, row in rows.iterrows():
                start_t = float(row["start"])
                if i < len(rows) - 1:
                    # hold until the next word appears, capped at hold_max_tail
                    end_t = min(float(rows.loc[i + 1, "start"]),
                                float(row["end"]) + hold_max_tail)
                else:
                    end_t = float(row["end"]) + 0.35  # small tail on the last word
                override = ("{" + str(row["anim_ass"])
                            + f"\\fs{int(row['font_size'])}\\c{row['color_ass']}"
                            + f"\\b{int(row['bold'])}\\i{int(row['italic'])}" + "}")
                lines.append(
                    f"Dialogue: 0,{sec_to_ass(start_t)},{sec_to_ass(end_t)},"
                    f"Default,,0,0,0,,{override}{str(row['word']).strip()}"
                )
        else:
            seg_ids = list(dict.fromkeys(rows["segment_id"].tolist()))
            seg_starts = {s: float(rows.loc[rows["segment_id"] == s, "start"].min())
                          for s in seg_ids}

            for si, sid in enumerate(seg_ids):
                seg = rows[rows["segment_id"] == sid].sort_values("start")
                start_t = float(seg["start"].min())
                last_end = float(seg["end"].max())

                # linger past the last word, stay up long enough to read, but
                # never overlap the next caption (that would stack two lines)
                end_t = max(last_end + hold_max_tail, start_t + min_line_dur)
                if si < len(seg_ids) - 1:
                    end_t = min(end_t, seg_starts[seg_ids[si + 1]])
                end_t = max(end_t, start_t + 0.10)  # never zero-length

                parts = []
                for _, row in seg.iterrows():
                    # every word re-declares ALL of its tags, because .ass
                    # override tags persist along the line until overridden
                    tags = (f"\\fs{int(row['font_size'])}\\c{row['color_ass']}"
                            f"\\b{int(row['bold'])}\\i{int(row['italic'])}")
                    if reveal_mode == "reveal":
                        # \t() times are milliseconds from the LINE start, not
                        # from the start of the clip
                        d0 = max(0, int(round((float(row["start"]) - start_t) * 1000)))
                        parts.append("{" + tags + f"\\alpha{dim_alpha}"
                                     + f"\\t({d0},{d0 + 120},\\alpha&H00&)" + "}"
                                     + str(row["word"]).strip())
                    else:
                        parts.append("{" + tags + "}" + str(row["word"]).strip())

                # \fad is line-level: the caption as a whole fades in and out
                text = "{\\fad(120,120)}" + " ".join(parts)
                lines.append(
                    f"Dialogue: 0,{sec_to_ass(start_t)},{sec_to_ass(end_t)},"
                    f"Default,,0,0,0,,{text}"
                )

        ass_content = header + "\n".join(lines) + "\n"
        ass_path = f"{out_dir}/ass/{tag}.ass"
        with open(ass_path, "w") as f:
            f.write(ass_content)

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
        print(f"{len(lines)} caption line(s) written ({caption_mode} mode)")
        return out_path, ass_path

    out_video, out_ass = render_budget_video(
        audio_file, styled_word_df,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE,
        hold_max_tail=HOLD_MAX_TAIL, min_line_dur=MIN_LINE_DURATION,
        tag=out_tag + "_" + CAPTION_MODE,
    )
    print("Wrote:", out_video, "and", out_ass)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Decision log

    1. **Separate notebook, V0.2 untouched:** V0.2 stays as the validated
       single-dataset record; this adds the dataset switch so RAVDESS and IEMOCAP
       runs are reproducible from one file without editing code between runs.
    2. **Clip choice via labelled CSV only:** the IEMOCAP picker reads
       features_iemocap.csv, so every selectable clip already survived the label
       mapping (excited->happy; frustrated/xxx/oth dropped). Prevents picking a
       clip with no usable ground truth.
    3. **Honesty caveat for RAVDESS runs:** clf_full is trained on ALL RAVDESS
       clips, so a correct prediction on a RAVDESS demo clip is in-training-set
       and proves nothing. The honest RAVDESS number remains the 46.8%
       speaker-independent CV figure from train_emotion.py. IEMOCAP clips under
       the RAVDESS-only model are genuinely unseen cross-corpus data, which is
       the point of adding the switch.
    4. **Bottom-aligned sentence captions (Alignment 2, MarginV 50):** the
       one-word-at-a-time centre display was legible for single-word RAVDESS
       demos but is not a caption in any conventional sense. Real captions place
       a whole phrase at the bottom of frame, so the accessibility framing of
       this project requires that layout as the default. Per-word styling is
       retained WITHIN the line.
    5. **Per-word animation cannot survive sentence mode:** `\\fad` and `\\fscx`
       pops are line-level .ass effects. Once all words share one Dialogue line,
       only tags that can appear mid-line (colour, size, weight, italic, alpha)
       remain per-word. This is a format constraint, not a design preference.
    6. **Alpha reveal instead of scale pop:** to keep a speech-synced dimension
       in sentence mode, each word is dimmed until its spoken moment (\\alpha
       plus an inline \\t). Alpha is animated rather than size because changing
       size mid-line reflows the text and makes the caption jitter sideways,
       which harms readability more than the lost motion gains.
    7. **Font sizes rescaled (34..60, from 48..92):** a 92 px word is fine alone
       on screen but forces a full sentence onto three wrapped lines. Sentence
       mode needs a narrower dynamic range to stay legible.
    8. **Two caption modes retained as an evaluation condition:** word-centre vs
       sentence-bottom is a legitimate A/B for the user study, testing
       expressiveness against readability directly.
    """)
    return


if __name__ == "__main__":
    app.run()

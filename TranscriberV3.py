import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
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
def _():
    data_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    emotion_map = {"01":"neutral","02":"calm","03":"happy","04":"sad",
                   "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
    drop_emotions = {"calm"}  # 7-class scheme drops calm
    features_csv = "outputs/features.csv"

    device = "cpu"
    compute_type = "int8"
    audio_file = "data/raw/Actor_01/03-01-06-01-02-01-01.wav"
    return (
        audio_file,
        compute_type,
        data_dir,
        device,
        drop_emotions,
        emotion_map,
        features_csv,
    )


@app.cell
def _(mo):
    mo.md("""
    # TranscriberV3: clip-level emotion + per-word expressive budget

    Combines the two existing pipelines:

    - **Part A** (from `train_emotion.py`): the clip-level Random Forest, trained on
      RAVDESS ground truth. It decides **which** emotion the sentence is expressing.
    - **Part B** (from `TranscriberV2`): WhisperX word timestamps + per-word prosody.
      These decide **how strongly** each individual word carries that emotion, via a
      competitive 100-point budget per sentence.

    No word ever gets its own emotion category (no ground truth exists at word level).
    Words only compete for *intensity* of the sentence's validated emotion.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part A: Clip-level classifier (features + training)
    """)
    return


@app.cell
def _(call, np, parselmouth):
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
    # cache: extract once, reuse afterwards (extraction is the slow part)
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
    # Train the final model on ALL clips. The proper speaker-independent
    # evaluation (46.8% vs 14.3% chance) lives in train_emotion.py and is NOT
    # re-run here, so this notebook stays fast to open and never duplicates the
    # slow 5-fold cross-validation.
    clip_feature_cols = [c for c in clip_df.columns if c not in ("file", "emotion", "actor")]
    X_clip = clip_df[clip_feature_cols].values
    y_clip = clip_df["emotion"].values

    clf_full = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ).fit(X_clip, y_clip)

    print(f"Trained clf_full on {len(clip_df)} clips / {len(clip_feature_cols)} features.")
    print("Classes:", list(clf_full.classes_))
    print("(Evaluation lives in train_emotion.py: 46.8% speaker-independent vs 14.3% chance.)")
    return clf_full, clip_feature_cols


@app.cell
def _(mo):
    mo.md("""
    ## Part B: WhisperX transcription → word-level timestamps + per-word prosody
    """)
    return


@app.cell
def _(audio_file, compute_type, device, whisperx):
    asr_model = whisperx.load_model("base", device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_file)
    result = asr_model.transcribe(audio, batch_size=16)
    print("LANGUAGE:", result["language"])
    print("SEGMENTS:", result["segments"])
    return audio, result


@app.cell
def _(audio, device, result, whisperx):
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
    word_df = extract_word_features(audio_file, aligned["word_segments"])
    word_df
    return (word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part C: The expressive budget

    Each sentence gets a fixed budget of **100 points**. Words compete for those
    points based on how much their prosody stands out from the *other words in the
    same sentence*. Because the budget is fixed, one word taking a big share
    automatically leaves less for everyone else, which is exactly the intended
    effect: in a sad sentence, everything stays sad, but only the prosodically
    extreme word(s) get the full dramatic version of the sad styling.
    """)
    return


@app.cell
def _():
    # ============================ TUNABLE PARAMETERS ============================
    # How much each per-word feature contributes to "standing out".
    # Pitch level, pitch movement, and loudness are the classic acoustic cues of
    # spoken emphasis, so they get full weight. HNR (voice quality) is a real but
    # noisier cue at word scale -> half weight. Duration is also a classic cue,
    # BUT raw word duration confounds emphasis with word length ("sitting" is
    # longer than "by" no matter how it's said) -> half weight.
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}

    # For these features a value of exactly 0 means "no voiced frames found",
    # i.e. absence of information, NOT evidence of extreme pitch. Such words are
    # excluded from the segment statistics and contribute z = 0 for that feature.
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}

    # Softmax temperature: LOWER = closer to winner-takes-all, HIGHER = flatter.
    # ~0.8 makes a clear winner take nearly the whole pool; ~2.0-2.5 lets a
    # runner-up word keep a visible share too.
    SOFTMAX_TEMPERATURE = 1.5

    # Guaranteed minimum points per word, because a caption word can never
    # visually vanish. Auto-shrinks for very long sentences (see allocate_points).
    MIN_POINTS = 2.0

    # A word reaches FULL styling intensity when it has taken this many times its
    # "fair share" (fair share = 100 / number of words in the sentence). Using the
    # ratio instead of raw points makes intensity independent of sentence length.
    FULL_DRAMA_RATIO = 2.5

    # Even a zero-intensity word is tinted this far towards the emotion colour,
    # so the whole sentence still visibly "reads" as the clip's emotion.
    BASE_EMOTION_TINT = 0.35

    # Intensity thresholds for the discrete style switches.
    BOLD_THRESHOLD = 0.55
    ANIM_TRIGGER = 0.55

    # Font geometry: size = BASE_FONT_SIZE + intensity * FONT_SWING  (48..92 px).
    BASE_FONT_SIZE = 48
    FONT_SWING = 44

    # Each word stays on screen until the next word appears (kills the black
    # flicker between words), but never longer than this many seconds past its
    # own end (so a word doesn't freeze awkwardly across a long pause).
    HOLD_MAX_TAIL = 0.6

    # If True, styling is restrained when the clip classifier is unsure:
    # top-class probability at chance level -> half strength, >= 0.5 -> full.
    USE_CONFIDENCE_SCALING = True

    # Style family per emotion. Hues follow the palette already used in the
    # earlier rf/mlp label-demo videos, for visual continuity across deliverables.
    # anim families: "pop"  = energetic scale-in on high-intensity words
    #                "soft" = slow fade-in, longer for higher intensity
    #                "flat" = plain quick fade (disgust adds italics when intense)
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
        EMOTION_STYLES,
        FONT_SWING,
        FULL_DRAMA_RATIO,
        HOLD_MAX_TAIL,
        MIN_POINTS,
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
    # Reuses the SAME feature extractor the classifier was trained with, on the
    # same whole clip. This is the only place emotion is classified; words never
    # get their own category (no word-level ground truth exists to justify one).
    clip_pred_feats = extract_clip_features(audio_file)
    clip_pred_vec = pd.DataFrame([clip_pred_feats])[clip_feature_cols]  # enforces column order

    # .to_numpy(): clf_full was fitted on a plain array, so passing a DataFrame
    # would trigger a harmless-but-noisy sklearn feature-names warning.
    pred_emotion = str(clf_full.predict(clip_pred_vec.to_numpy())[0])
    pred_proba = clf_full.predict_proba(clip_pred_vec.to_numpy())[0]
    p_top = float(np.max(pred_proba))

    if USE_CONFIDENCE_SCALING:
        # Map top-class probability to a 0.5..1.0 multiplier: at chance level the
        # styling runs at half strength, at probability >= 0.5 it runs at full.
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
def _(mo):
    mo.md("""
    ### Step C2: Salience → competitive 100-point budget (the "how strongly" decision)
    """)
    return


@app.cell
def _(np):
    def assign_words_to_segments(words_df, segments):
        """Tag each word with the id of the sentence/segment it belongs to.

        A word belongs to the segment whose [start, end] window contains the
        word's midpoint. If alignment drift leaves a word just outside every
        window, it falls back to the NEAREST segment instead of crashing.
        For single-sentence RAVDESS clips this collapses to one segment, but it
        generalises unchanged to longer multi-sentence audio.
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
        """Raw salience = how much a word's prosody stands out from ITS OWN sentence.

        Every feature in `weights` is z-scored against the mean/std of the words
        in the same segment (not global RAVDESS statistics), then the absolute
        z-scores are summed with the given weights. |z| is used because standing
        out in EITHER direction (unusually high OR unusually low pitch) reads as
        emphasis. If a feature has ~zero variance within a segment it carries no
        emphasis information there, so its z stays 0 instead of dividing by ~0.
        Features listed in `zero_missing` treat a raw value of exactly 0 as
        "no data" (e.g. an unvoiced word has no pitch, which is not the same as
        extreme pitch): such words are excluded from the segment stats and get
        z = 0 for that feature.
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

        A softmax over (salience / temperature) makes the split COMPETITIVE:
        the exponential means a word only moderately more salient than its
        neighbours still takes a disproportionately bigger slice, and because
        the pool is fixed, its gain is literally everyone else's loss.
        Lower temperature -> closer to winner-takes-all; higher -> flatter.

        Every word is guaranteed a small floor (a caption word can't visually
        vanish). The floor auto-shrinks for very long sentences (never more
        than an even split), so the segment total is always exactly 100.
        Single-word segments trivially take all 100.
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
        # "fair share" = what a word would get if the budget were split evenly.
        # share_ratio > 1 means the word out-competed its neighbours; using the
        # ratio (not raw points) keeps intensity independent of sentence length.
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
    # Prefer the aligned segment boundaries (refined timings); fall back to the
    # raw transcription segments if alignment didn't return any.
    seg_list = aligned.get("segments") or result["segments"]

    tagged_word_df = word_df.copy()
    tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

    salient_word_df = compute_salience(
        tagged_word_df, SALIENCE_WEIGHTS, zero_missing=ZERO_MEANS_MISSING
    )
    budget_df = allocate_points(
        salient_word_df, temperature=SOFTMAX_TEMPERATURE, min_points=MIN_POINTS
    )

    # intensity in 0..1: 0 at (or below) fair share, 1 at FULL_DRAMA_RATIO x fair
    # share, then restrained by classifier confidence.
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
    ## Part D: Style mapping v2 (emotion picks the family, budget picks the intensity)
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
    def assign_styles_v2(words_df, emotion, styles, base_tint, confidence,
                         base_font, font_swing, bold_thresh, anim_trigger):
        """Map (clip emotion, per-word intensity) -> concrete .ass styling.

        The clip-level emotion picks the style FAMILY (hue + animation flavour);
        each word's intensity (0..1) picks how far that word moves from a plain
        white caption towards the family's full dramatic look. Low-intensity
        words stay near the tinted baseline; the budget winner(s) get the full
        version: max size, saturated colour, bold, and the family's animation.
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
        sizes, colors, bolds, anims = [], [], [], []
        for _, row in df.iterrows():
            inten = float(row["intensity"])
            # colour: tint floor at zero intensity, family hue fully saturated at 1
            t = tint_floor + (1.0 - tint_floor) * inten
            colors.append(rgb_to_ass(*lerp((255, 255, 255), emo_rgb, t)))
            sizes.append(int(round(base_font + font_swing * inten)))
            bolds.append(1 if inten >= bold_thresh else 0)
            # animation flavour by family
            if anim_family == "pop" and inten >= anim_trigger:
                # scale-pop: word lands at 55% size and springs to 100% in 140 ms
                anims.append("\\fscx55\\fscy55\\t(0,140,\\fscx100\\fscy100)")
            elif anim_family == "soft":
                # gentle fade-in, longer for more intense words (150..450 ms)
                anims.append(f"\\fad({int(150 + 300 * inten)},0)")
            elif emotion == "disgust" and inten >= anim_trigger:
                anims.append("\\i1\\fad(80,0)")  # italic flavour for intense disgust
            else:
                anims.append("\\fad(80,0)")      # default quick fade
        df["font_size"] = sizes
        df["color_ass"] = colors
        df["bold"] = bolds
        df["anim_ass"] = anims
        return df

    styled_word_df = assign_styles_v2(
        budget_df, pred_emotion, EMOTION_STYLES,
        base_tint=BASE_EMOTION_TINT, confidence=conf_scale,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING,
        bold_thresh=BOLD_THRESHOLD, anim_trigger=ANIM_TRIGGER,
    )
    styled_word_df[["word", "points", "intensity", "font_size", "color_ass", "bold", "anim_ass"]]
    return (styled_word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part E: Render .ass + FFmpeg burn-in
    """)
    return


@app.cell
def _(HOLD_MAX_TAIL, audio_file, os, styled_word_df, subprocess):
    def render_budget_video(audio_path, df, out_dir="outputs", hold_max_tail=0.6):
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720

        def sec_to_ass(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
            cs = int(round((t - int(t)) * 100))
            if cs == 100: cs = 0; s += 1
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        header = (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            f"PlayResX: {width}\n"
            f"PlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,5,10,10,10,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)
        lines = []
        for i, row in rows.iterrows():
            start_t = float(row["start"])
            if i < len(rows) - 1:
                # hold each word until the next appears (no black flicker between
                # words), but never more than hold_max_tail past its own end
                end_t = min(float(rows.loc[i + 1, "start"]),
                            float(row["end"]) + hold_max_tail)
            else:
                end_t = float(row["end"]) + 0.35  # small tail on the last word

            override = ("{" + str(row["anim_ass"])
                        + f"\\fs{int(row['font_size'])}\\c{row['color_ass']}\\b{int(row['bold'])}"
                        + "}")
            text = override + str(row["word"]).strip()
            lines.append(
                f"Dialogue: 0,{sec_to_ass(start_t)},{sec_to_ass(end_t)},Default,,0,0,0,,{text}"
            )

        ass_content = header + "\n".join(lines) + "\n"
        ass_path = f"{out_dir}/ass/v3_budget_demo.ass"
        with open(ass_path, "w") as f:
            f.write(ass_content)

        duration = float(rows["end"].max()) + 0.5
        out_path = f"{out_dir}/video/v3_budget_demo.mp4"
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
        return out_path, ass_path

    v3_video, v3_ass = render_budget_video(
        audio_file, styled_word_df, hold_max_tail=HOLD_MAX_TAIL
    )
    print("Wrote:", v3_video, "and", v3_ass)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Decision log (for the methodology chapter)

    1. **Deliverable shape:** new combined notebook rather than editing TranscriberV2
       in place, because the pipeline needs both the trained classifier and the
       WhisperX machinery in one runnable file; V2 is preserved untouched as the
       proof-of-concept record. Training loads from the cached `features.csv` in
       seconds; the slow 5-fold evaluation is deliberately not duplicated here.
    2. **Salience features and weights:** pitch level (`f0_mean`), pitch movement
       (`f0_range`) and loudness (`rms`) at weight 1.0, as the primary acoustic
       correlates of spoken emphasis; `hnr` at 0.5 (a real but noisier voice-quality
       cue at word scale); `duration` at 0.5 because raw word duration confounds
       emphasis with word length (more phonemes = longer regardless of delivery).
    3. **`pause_after` excluded** from salience: it is structurally 0 for the last
       word of every segment, which would systematically mark sentence-final words
       as un-salient — the opposite of English nuclear stress, which tends to fall
       late in the sentence.
    4. **Zero-as-missing for f0/hnr:** a word with no voiced frames has its pitch
       recorded as 0. That is absence of pitch information, not evidence of extreme
       pitch, so such words are excluded from segment statistics and contribute
       z = 0 for those features (prevents short unvoiced fragments from spuriously
       winning the budget).
    5. **Allocation:** softmax over (salience / temperature), default temperature
       1.5; a 2-point-per-word floor that auto-shrinks for long sentences so the
       segment total is always exactly 100. Verified: budgets sum to 100 in all
       tested cases, single-word segments take 100, zero-variance segments split
       evenly, and higher temperature monotonically flattens the winner's share.
    6. **Intensity = share relative to fair share (100/n),** reaching full drama at
       2.5x fair share. Using the ratio rather than raw points makes visual
       intensity independent of sentence length (a 20-word sentence does not dilute
       its winner just because each fair share is smaller).
    7. **Base emotion tint (0.35):** every word is tinted at least this far towards
       the clip emotion's colour, so the budget modulates *how strongly* each word
       expresses the emotion, never *whether* the sentence reads as that emotion.
    8. **Confidence restraint (optional, on):** styling strength is scaled by the
       classifier's top-class probability (chance level → half strength, ≥ 0.5 →
       full), so the system visually hedges when its own emotion call is weak.
    9. **Hold-until-next display (capped at +0.6 s):** presentation-only change that
       removes the black flicker between words; capped so a word never freezes
       across a long pause.
    10. **Palette continuity:** emotion hues reuse the palette from the earlier
        RF/MLP label-demo videos; energetic emotions (angry/happy/surprised) use a
        scale-pop animation, low-arousal ones (sad/fearful) use slow fades, disgust
        is flavoured with italics.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

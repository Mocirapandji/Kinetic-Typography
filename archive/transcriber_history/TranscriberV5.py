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
    # TranscriberV5: literature-anchored styling + karaoke bounce

    Everything from V4 (dataset switch, sentence-mode bottom captions), plus:

    - **Literature-anchored palette**: hue per emotion from the Jonauskaite
      37-nation survey (8,615 participants) solved as a maximum-weight
      assignment; saturation/value ramp from Valdez & Mehrabian (1994)
      (intensity -> saturation UP, value DOWN, the arousal gradient).
    - **Caption plates**: emotions whose evidence-based colour is dark
      (fear -> black) render on a white plate (BorderStyle=4, libass);
      the tint baseline flips so low-intensity words stay legible.
    - **Per-emotion fonts** via inline \fn (provisional map; see font cell).
    - **Karaoke reveal with bounce**: each word sits dimmed, then lights up
      and does a perky vertical hop on its spoken cue. Bounce duration
      follows the word's real spoken duration (slow word = slow bounce),
      scaled by an emotion tempo dial.
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

    out_tag = "v5_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
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
    # =====================================================================
    # CELL 4 — CLIP-LEVEL FEATURE EXTRACTOR (identical to train_emotion.py)
    # =====================================================================
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
    # (46.8% vs 14.3% chance) lives in train_emotion.py.
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

            if i < len(word_segments) - 1:
                pause_after = float(word_segments[i + 1]["start"]) - end
            else:
                pause_after = 0.0

            word_snd = snd.extract_part(from_time=start, to_time=end, preserve_times=True)

            pitch = word_snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0v = f0[f0 > 0]
            f0_mean = float(f0v.mean()) if len(f0v) else 0.0
            f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0

            rms = call(word_snd, "Get root-mean-square", 0, 0)

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
    # ----- budget (unchanged from V3/V4) -----
    SALIENCE_WEIGHTS = {"f0_mean": 1.0, "f0_range": 1.0, "rms": 1.0,
                        "hnr": 0.5, "duration": 0.5}
    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr"}
    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5
    BOLD_THRESHOLD = 0.55
    ITALIC_TRIGGER = 0.55          # disgust adds italics above this intensity
    BASE_FONT_SIZE = 34
    FONT_SWING = 26
    USE_CONFIDENCE_SCALING = True

    # ----- caption layout -----
    CAPTION_MODE = "sentence"      # "sentence" (bottom line) or "word" (old centre)
    # "karaoke" = dim -> light + BOUNCE on each word's spoken cue
    # "reveal"  = dim -> light only (V4 behaviour)
    # "none"    = whole line appears fully styled at once
    REVEAL_MODE = "karaoke"
    HOLD_MAX_TAIL = 0.6
    MIN_LINE_DURATION = 1.0
    DIM_ALPHA = "&H96&"            # how faded an unspoken word sits (00=opaque, FF=invisible)

    # ----- karaoke bounce -----
    # Bounce duration follows the word's REAL spoken duration (a drawled word
    # bounces slowly, a snapped word pops), clamped to sane bounds, then scaled
    # by the emotion family's tempo. Default tempi follow the arousal
    # literature (angry/happy/surprised = quick, sad/fearful = slow). If you
    # want the literal "angrier = slower bounce", just raise BOUNCE_TEMPO["pop"].
    BOUNCE_MIN_MS, BOUNCE_MAX_MS = 180, 650
    BOUNCE_TEMPO = {"pop": 0.75, "soft": 1.60, "flat": 1.00}
    # hop height: fscy peak = BASE + GAIN * intensity  (118..140 => perky)
    BOUNCE_PEAK_BASE, BOUNCE_PEAK_GAIN = 118, 22

    # ----- literature-anchored palette -----
    # Hue per emotion: Jonauskaite et al. 37-nation survey (n=8,615), solved as
    # a maximum-weight assignment with two documented overrides:
    #   sadness -> blue (survey rank 3; systematic-review rank 1 across 23
    #   studies; keeps sad/fear categorically discriminable in the mute test)
    #   surprised -> orange (GEW has no surprise; review evidence, weak anchor)
    # Saturation/value RAMP: Valdez & Mehrabian (1994) arousal gradient
    # (Arousal = -.31 Brightness + .60 Saturation): intensity -> S UP, V DOWN.
    # plate: "light" puts a white box behind the line (for dark text colours).
    EMOTION_STYLES = {
        #            hue(0-1) s_max  achro  plate    anim    font
        "angry":     {"h": 0.000, "s_max": 0.95, "achro": 0, "plate": "dark",  "anim": "pop",  "font": "DejaVu Sans Condensed"},
        "happy":     {"h": 0.140, "s_max": 0.95, "achro": 0, "plate": "dark",  "anim": "pop",  "font": "DejaVu Sans"},
        "surprised": {"h": 0.083, "s_max": 0.95, "achro": 0, "plate": "dark",  "anim": "pop",  "font": "DejaVu Sans"},
        "sad":       {"h": 0.610, "s_max": 0.85, "achro": 0, "plate": "dark",  "anim": "soft", "font": "Liberation Serif"},
        "fearful":   {"h": 0.000, "s_max": 0.00, "achro": 1, "plate": "light", "anim": "soft", "font": "DejaVu Serif"},
        "disgust":   {"h": 0.075, "s_max": 0.80, "achro": 0, "plate": "dark",  "anim": "flat", "font": "Liberation Mono"},
        "neutral":   {"h": 0.000, "s_max": 0.00, "achro": 1, "plate": "dark",  "anim": "flat", "font": "Liberation Sans"},
    }
    # per-plate (S, V) ramp endpoints. On BOTH plates V goes DOWN with
    # intensity (V&M) — on the white plate that also means MORE contrast.
    RAMP = {
        "dark":  {"s_lo": 0.12, "v_hi": 0.96, "v_lo": 0.58},
        "light": {"s_lo": 0.12, "v_hi": 0.38, "v_lo": 0.06},
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
        FONT_SWING,
        FULL_DRAMA_RATIO,
        HOLD_MAX_TAIL,
        ITALIC_TRIGGER,
        MIN_LINE_DURATION,
        MIN_POINTS,
        RAMP,
        REVEAL_MODE,
        SALIENCE_WEIGHTS,
        SOFTMAX_TEMPERATURE,
        USE_CONFIDENCE_SCALING,
        ZERO_MEANS_MISSING,
    )


@app.cell
def _(EMOTION_STYLES, subprocess):
    # =====================================================================
    # CELL 12 — FONT AVAILABILITY CHECK
    # libass silently falls back to a default when a font is missing, so a
    # wrong name never crashes — it just quietly ignores your font choice.
    # This cell tells you which mapped fonts actually resolve on THIS machine.
    # To use richer fonts: unzip O'Donovan's gwfonts.zip into ~/.fonts and
    # run `fc-cache -f`, then update the font names in CELL 11.
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
    clip_pred_vec = pd.DataFrame([clip_pred_feats])[clip_feature_cols]

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
    ## Part D: Style mapping — literature-anchored HSV ramp

    Hue comes from the clip's emotion (survey-anchored). Intensity moves each
    word along the Valdez & Mehrabian arousal gradient: **saturation up, value
    down**. On the dark plate a weak word is a pale wash and the budget winner
    is a deep saturated hue; on the white plate (fear) weak words are mid-grey
    and the winner is near-black — more contrast in both cases.
    """)
    return


@app.cell
def _(
    BASE_FONT_SIZE,
    BOLD_THRESHOLD,
    EMOTION_STYLES,
    FONT_SWING,
    ITALIC_TRIGGER,
    RAMP,
    budget_df,
    colorsys,
    pred_emotion,
):
    # =====================================================================
    # CELL 17 — STYLE MAPPING v3 (HSV ramp; plates decided by emotion)
    # =====================================================================
    def hsv_ramp_color(emotion, inten, styles, ramp):
        """intensity (0..1) -> ASS colour along the V&M arousal gradient."""
        fam = styles.get(emotion, styles["neutral"])
        r_ = ramp["light" if fam["plate"] == "light" else "dark"]
        s = 0.0 if fam["achro"] else r_["s_lo"] + (fam["s_max"] - r_["s_lo"]) * inten
        v = r_["v_hi"] - (r_["v_hi"] - r_["v_lo"]) * inten
        r, g, b = (int(round(c * 255)) for c in colorsys.hsv_to_rgb(fam["h"], max(s, 0.0), v))
        return f"&H{b:02X}{g:02X}{r:02X}&"   # ASS colour is &HBBGGRR&, NOT RGB

    def assign_styles_v3(words_df, emotion, styles, ramp,
                         base_font, font_swing, bold_thresh, italic_trigger):
        df = words_df.copy()
        sizes, colors, bolds, italics = [], [], [], []
        for _, row in df.iterrows():
            inten = float(row["intensity"])
            colors.append(hsv_ramp_color(emotion, inten, styles, ramp))
            sizes.append(int(round(base_font + font_swing * inten)))
            bolds.append(1 if inten >= bold_thresh else 0)
            italics.append(1 if (emotion == "disgust" and inten >= italic_trigger) else 0)
        df["font_size"] = sizes
        df["color_ass"] = colors
        df["bold"] = bolds
        df["italic"] = italics
        return df

    styled_word_df = assign_styles_v3(
        budget_df, pred_emotion, EMOTION_STYLES, RAMP,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING,
        bold_thresh=BOLD_THRESHOLD, italic_trigger=ITALIC_TRIGGER,
    )
    print(f"emotion={pred_emotion} | plate={EMOTION_STYLES.get(pred_emotion, EMOTION_STYLES['neutral'])['plate']} "
          f"| font={EMOTION_STYLES.get(pred_emotion, EMOTION_STYLES['neutral'])['font']}")
    styled_word_df[["word", "points", "intensity", "font_size", "color_ass", "bold", "italic"]]
    return (styled_word_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Part E: Render — plates, fonts, karaoke bounce

    Mechanics worth knowing:

    - **Two styles in the header.** `CapDark` = outline captions straight on
      the video. `CapLight` = `BorderStyle=4` draws one continuous box behind
      the whole line (white plate) — used when the emotion's evidence-based
      text colour is dark (fear). `BorderStyle=4` is a libass extension: fine
      for FFmpeg burn-in (FFmpeg *is* libass), but the raw .ass would not show
      plates in a non-libass player.
    - **The bounce is a vertical-scale hop** (`\fscy` pulse). True per-word
      position animation is impossible mid-line in .ass (position tags are
      line-level). Scaling Y only means width never changes, so the line can
      never jitter sideways; the word stretches up off the baseline and
      settles. Peak height rises with intensity; duration follows the word's
      spoken duration times the emotion tempo.
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
    HOLD_MAX_TAIL,
    MIN_LINE_DURATION,
    REVEAL_MODE,
    audio_file,
    os,
    out_tag,
    pred_emotion,
    styled_word_df,
    subprocess,
):
    # =====================================================================
    # CELL 18 — RENDER: plates + fonts + karaoke bounce
    # =====================================================================
    def render_budget_video(audio_path, df, emotion, styles, out_dir="outputs",
                            caption_mode="sentence", reveal_mode="karaoke",
                            hold_max_tail=0.6, min_line_dur=1.0,
                            dim_alpha="&H96&",
                            bounce_min_ms=180, bounce_max_ms=650,
                            bounce_tempo=None, peak_base=118, peak_gain=22,
                            tag="v5_demo"):
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720
        fam = styles.get(emotion, styles["neutral"])
        tempo = (bounce_tempo or {"pop": 0.75, "soft": 1.6, "flat": 1.0})[fam["anim"]]
        style_name = "CapLight" if fam["plate"] == "light" else "CapDark"

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
            # light text straight on video (outline + soft shadow)
            "Style: CapDark,Liberation Sans,48,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1\n"
            # dark text on a continuous white plate (BorderStyle=4, libass)
            "Style: CapLight,Liberation Sans,48,&H00101010,&H000000FF,&H00F8F8F8,&H00F8F8F8,0,0,0,0,100,100,0,0,4,6,0,2,40,40,50,1\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        rows = df.sort_values("start").reset_index(drop=True)
        lines = []

        def word_tags(rw, line_start):
            """All override tags for one word, incl. karaoke reveal + bounce."""
            tags = (f"\\fn{fam['font']}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}")
            if reveal_mode in ("karaoke", "reveal"):
                d0 = max(0, int(round((float(rw["start"]) - line_start) * 1000)))
                # word sits dimmed, snaps to full opacity on its spoken cue
                tags += f"\\alpha{dim_alpha}\\t({d0},{d0 + 90},\\alpha&H00&)"
                if reveal_mode == "karaoke":
                    # bounce duration = real spoken duration, clamped, x tempo
                    dur_ms = (float(rw["end"]) - float(rw["start"])) * 1000.0
                    b = int(min(max(dur_ms, bounce_min_ms), bounce_max_ms) * tempo)
                    up = d0 + int(b * 0.38)      # quick rise...
                    down = d0 + b                # ...slower settle = perky
                    peak = int(peak_base + peak_gain * float(rw["intensity"]))
                    tags += f"\\t({d0},{up},\\fscy{peak})\\t({up},{down},\\fscy100)"
            return tags

        if caption_mode == "word":
            # one word at a time, centred — \an5 override on each line
            for i, rw in rows.iterrows():
                start_t = float(rw["start"])
                if i < len(rows) - 1:
                    end_t = min(float(rows.loc[i + 1, "start"]),
                                float(rw["end"]) + hold_max_tail)
                else:
                    end_t = float(rw["end"]) + 0.35
                text = "{\\an5" + word_tags(rw, start_t) + "}" + str(rw["word"]).strip()
                lines.append(f"Dialogue: 0,{sec_to_ass(start_t)},{sec_to_ass(end_t)},"
                             f"{style_name},,0,0,0,,{text}")
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
                             f"{style_name},,0,0,0,," + "{\\fad(120,120)}" + " ".join(parts))

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
        print(f"{len(lines)} caption line(s) written "
              f"({caption_mode} / {reveal_mode} / plate={fam['plate']} / font={fam['font']})")
        return out_path, ass_path

    v5_video, v5_ass = render_budget_video(
        audio_file, styled_word_df, pred_emotion, EMOTION_STYLES,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE,
        hold_max_tail=HOLD_MAX_TAIL, min_line_dur=MIN_LINE_DURATION,
        dim_alpha=DIM_ALPHA,
        bounce_min_ms=BOUNCE_MIN_MS, bounce_max_ms=BOUNCE_MAX_MS,
        bounce_tempo=BOUNCE_TEMPO, peak_base=BOUNCE_PEAK_BASE,
        peak_gain=BOUNCE_PEAK_GAIN,
        tag=out_tag + "_" + CAPTION_MODE + "_" + REVEAL_MODE,
    )
    print("Wrote:", v5_video, "and", v5_ass)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Decision log (V5 additions)

    1. **Literature-anchored hue palette**: hues from the Jonauskaite et al.
       37-nation survey (n=8,615), assigned by maximum-weight bipartite
       matching (anger-red 54%, joy-yellow 54%, fear-black 48%,
       disgust-brown 37%). Two documented overrides: sadness->blue
       (survey 28%, but rank 1 in the 128-year systematic review, 23 studies;
       preserves sad/fear discriminability in the mute test) and
       surprised->orange (surprise is absent from the Geneva Emotion Wheel;
       anchored only by review counts — flagged as the weakest mapping).
    2. **Caption plates**: fear's evidence-based colour (black) is illegible on
       video, so its lines render dark-on-white via BorderStyle=4 (one
       continuous box; libass extension — fine for FFmpeg burn-in, not
       portable to non-libass players). The tint baseline flips with the
       plate so low-intensity words stay visible on white.
    3. **Valdez & Mehrabian intensity ramp**: colour intensity now moves along
       the published arousal gradient (Arousal = -.31B + .60S): saturation up,
       value down. Replaces the old white->hue lerp, which only implemented
       the saturation half. On the white plate, value-down also means
       contrast-up — psychology and legibility agree.
    4. **Per-emotion fonts** via inline \fn, mapped to families installed on
       typical Linux images (condensed sans = aggressive, serif = sombre,
       mono = flat/clinical). Provisional heuristic pending integration of
       the O'Donovan crowdsourced font-attribute dataset; a check cell
       reports which fonts resolve, since libass falls back silently.
    5. **Karaoke bounce** (REVEAL_MODE="karaoke"): per-word \fscy pulse on the
       spoken cue. Y-scale only — width never changes, so no sideways reflow
       (the failure mode that ruled out size animation in V4). Bounce
       duration = the word's real spoken duration (clamped) x an emotion
       tempo dial; hop height rises with budget intensity. Default tempi
       follow arousal (pop fast, soft slow); flipping to "angrier = slower"
       is a one-value change in BOUNCE_TEMPO.
    6. **Pipeline unchanged upstream**: classifier, budget, and per-word
       prosody are untouched; V5 only re-maps emotion+intensity to visuals.
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


if __name__ == "__main__":
    app.run()

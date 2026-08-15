import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    # =====================================================================
    # TranscriberV15: V12 revised against the literature audit,
    # then extended with the dynamics/sequence work (V13.x -> V15)
    # ("V12 Design Choices: Literature Audit and Revision Brief")
    #
    #   PALETTE (audit 2.2, 2.4, 2.6, 2.10, Part 8 item 6) — hues are now
    #     chosen by CIEDE2000 search rather than by eye. Surprised moves
    #     27 -> 190 deg (cyan; Jonauskaite offers no anchor for surprise, and
    #     turquoise is one of their four lowest-variance terms), disgust
    #     moves 40 -> 94 deg (bile green, inside the 88-112 constraint).
    #     Fearful's saturation/value are corrected so its Valdez & Mehrabian
    #     implied arousal goes from -0.10 (below sad!) to +0.18, and it
    #     leaves the slow "soft" anim group for a new fast "tremor" group.
    #     A colour-science cell (12c) validates dE2000 against the Sharma
    #     et al. (2005) reference pairs, and a palette-audit cell (12d)
    #     reports pairwise dE under normal vision, protanopia, deuteranopia
    #     and achromatopsia (Machado et al. 2009 simulations).
    #
    #   ---- ADDED SINCE THE AUDIT WORK (V15) ---------------------------
    #
    #   SEGMENT AROUSAL FLOOR — the 100-point budget is relative by
    #     construction, so a line where EVERY word is shouted gives every
    #     word fair share, every share_ratio ~1.0, and therefore every
    #     intensity ~0. A screamed line rendered at minimum styling. Each
    #     segment's absolute arousal now sets a floor the whole line renders
    #     at, with the within-line competition riding on top of it.
    #
    #   SHOUTING -> CAPITALS — DCMP's Captioning Key reserves capitals for
    #     screaming or shouting and forbids them for general emphasis. That
    #     rule is now applied from a measurement (spectral tilt / alpha
    #     ratio, gated against loudness AND an absolute dB margin) rather
    #     than by ear. Punctuation may corroborate, never trigger.
    #
    #   EMOTION SMOOTHING — segments were classified independently, which
    #     assumed emotion is redrawn every sentence. A first-order HMM over
    #     the segment sequence (Viterbi, self_bias swept to 0.55) decodes
    #     the most likely PATH instead, so colour stops strobing on
    #     posterior differences smaller than the classifier's own error.
    #
    #   CELL COMPACTION — 52 cells -> 32, verified content-preserving
    #     (identical public definitions, zero code lines lost or added).
    #
    #   CONFIDENCE (audit 5.4) — the old curve divided by (0.5 - chance), so
    #     it saturated at p=0.5 and mapped a 0.43 posterior to ~90% styling
    #     strength. The calibrated curve spans the full [chance, 1] range
    #     with a gamma, landing 0.43 at ~60%.
    #
    #   CHANNEL RULE (audit Part 1) — resolved as an explicit switch.
    #     CHANNEL_MODE="hue_only" enforces one-variable-per-channel
    #     (Position A); "redundant" keeps font/italic/tempo as designed
    #     redundancy (Position B). The audit cell settles the argument
    #     empirically: under achromatopsia no hue assignment exceeds a
    #     minimum pairwise dE of ~4.8, so hue alone cannot serve those
    #     viewers, which is the accessibility case for Position B.
    #
    #   MOTION (audit 4.3) — the \fscy swell is now volume-conserving
    #     squash-and-stretch: each \fscy is paired with an inverse \fscx
    #     (Lee et al. 2002, after Lasseter). The absolute layout reserves
    #     horizontal headroom so a widening word cannot collide.
    #
    #   Plus: reading-rate enforcement (Part 8 item 3), outline/shadow as
    #     documented parameters (item 2), loud font-substitution failure
    #     (item 8), styling-coverage reporting (2.7), EmoLex stimulus
    #     screening for the mute test (7.1), and an optional adjective/
    #     adverb prior tier (5.2, default OFF).
    #
    # V12's three fixes (motion isolation, directional salience, ASR model
    # size) are all carried forward unchanged.
    # =====================================================================

    # =====================================================================
    # CELL 0 — IMPORTS
    # =====================================================================
    import marimo as mo
    import os
    import re
    import colorsys
    import functools
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
    import matplotlib
    import matplotlib.patheffects
    import matplotlib.pyplot as plt
    import gc
    # torch is imported here as well as inside the device-detection cell, so
    # the VRAM helpers in CELL 7c can reach it. Guarded, because a CPU-only
    # install is a valid configuration for this notebook.
    try:
        import torch
    except Exception:
        torch = None
    return (
        Path,
        RandomForestClassifier,
        call,
        colorsys,
        gc,
        joblib,
        json,
        matplotlib,
        mo,
        np,
        opensmile,
        os,
        parselmouth,
        pd,
        plt,
        re,
        subprocess,
        torch,
        whisperx,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # TranscriberV15: the audited palette, with dynamics and sequence

    V13 is V12 revised against a section-by-section literature audit. The
    headline changes: the palette is now chosen by **perceptual measurement**
    (CIEDE2000 search under colour-vision-deficiency simulation) rather than
    by eye; fearful's arousal encoding is corrected in both colour and tempo;
    the confidence curve no longer overstates the classifier; the channel
    rule is an explicit, evidenced switch; and the swell is a proper
    volume-conserving squash-and-stretch. New audit cells report pairwise
    colour distances, styling coverage, and lexical neutrality of stimuli.

    V12's three fixes below are all retained.

    **The whole caption jumped.** Motion was `\fscy` applied to one word inside
    a single `Dialogue` event that held the entire sentence. libass sizes a line
    box from its tallest run, so growing one word grew the box and shifted every
    other word's baseline. V12 measures each word with the real font file and
    emits one `\pos`'d event per word. A word can now swell, tilt or wobble
    without touching its neighbours.

    **The wrong words were emphasised.** Salience was `w * |z|`. The absolute
    value meant a word that was unusually *quiet*, *short* and *low* scored
    exactly like one that was unusually loud, long and high — which is why
    "OR", "the" and "a" kept blowing up. Emphasis is now positive-only and
    signed: louder, higher, longer-than-expected earns points; the opposite
    earns nothing.

    **Connectives outranked content.** V12 adds a word-class prior. Nouns,
    verbs, adjectives and negation carry full weight; determiners, prepositions,
    conjunctions and auxiliaries are damped. The damping is *conditional* — a
    function word carrying genuine contrastive stress ("this OR that") clears
    the override threshold and recovers full weight, because sometimes the
    speaker really is stressing "or".
    """)
    return


@app.cell
def _(os, torch):
    # =====================================================================
    # CELL 1 — CONFIG + DATASET SWITCH
    # =====================================================================
    DATASET = "iemocap"                       # "ravdess" | "iemocap"

    ravdess_dir = "/run/media/s5812886/T7 Shield/RAVDESS"
    iemocap_dir = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"
    data_dir = ravdess_dir

    emotion_map = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                   "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
    drop_emotions = {"calm"}

    # Every output filename is built from this one string. It was three
    # separate hardcoded literals before, which is why the corpus renders
    # were still coming out labelled v12 two versions later, and why any
    # video run was tagged "anyvideo". A stale filename on a stimulus is
    # not cosmetic: it is how the wrong render ends up in the study.
    VERSION_TAG = "test"

    features_csv = "outputs/features.csv"     # 14-feature cache (fallback model)

    iemocap_csv = "outputs/features_iemocap.csv"
    if not os.path.exists(iemocap_csv):
        iemocap_csv = "/run/media/s5812886/T7 Shield/kinetic_outputs/features_iemocap.csv"

    # ---------- FIX 3 / V15: device is PROBED, not just queried --------
    # torch comes from CELL 0 (marimo forbids defining a name in two
    # cells), so it is None on a CPU-only install rather than raising.
    #
    # torch.cuda.is_available() only reports that a driver and a device
    # exist. It does NOT tell you a CUDA context can be created, and
    # creating one costs a few hundred MB of VRAM. On a shared machine
    # whose card is already full, is_available() says True and then the
    # first real CUDA call dies with 'CUDA error: out of memory'. That
    # sent the whole notebook down a GPU path it could never walk.
    #
    # So: actually try to allocate. A tiny tensor forces context creation
    # and a real allocation, which is the only honest test. If it fails we
    # fall back to CPU and say why, because a slow render today beats a
    # fast one that never happens.
    FORCE_DEVICE = None        # None = probe. Or pin: "cuda" | "cpu"

    def _probe_cuda():
        """-> (device, compute_type, message)."""
        if torch is None:
            return "cpu", "int8", "torch not installed"
        try:
            if not torch.cuda.is_available():
                return "cpu", "int8", "no CUDA device visible"
            _t = torch.zeros(256, 256, device="cuda")   # forces a context
            del _t
            torch.cuda.synchronize()
            _f, _tot = torch.cuda.mem_get_info()
            if _f < 2 * 2**30:
                return ("cpu", "int8",
                        f"CUDA usable but only {_f/2**30:.2f} GiB free of "
                        f"{_tot/2**30:.2f} GiB — too little for whisper; "
                        "free the card or set FORCE_DEVICE='cuda' with a "
                        "smaller ASR_MODEL_SIZE to override")
            return "cuda", "float16", f"CUDA ok, {_f/2**30:.2f} GiB free"
        except Exception as _e:
            return ("cpu", "int8",
                    f"CUDA present but unusable ({type(_e).__name__}: "
                    f"{str(_e).splitlines()[0][:90]}) — the card is "
                    "probably full. Run nvidia-smi to see whose.")

    if FORCE_DEVICE == "cpu":
        device, compute_type, _dev_msg = "cpu", "int8", "pinned to CPU"
    elif FORCE_DEVICE == "cuda":
        device, compute_type, _dev_msg = "cuda", "float16", "pinned to CUDA"
    else:
        device, compute_type, _dev_msg = _probe_cuda()

    if DATASET == "ravdess":
        audio_file = f"{ravdess_dir}/Actor_01/03-01-06-01-02-01-01.wav"
    else:
        _utt = "Ses01F_impro01_F012"
        _utt = _utt.replace(".wav", "")
        _sess = f"Session{int(_utt[3:5])}"
        _dialog = _utt.rsplit("_", 1)[0]
        audio_file = f"{iemocap_dir}/{_sess}/sentences/wav/{_dialog}/{_utt}.wav"

    out_tag = (VERSION_TAG + "_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", ""))
    print(f"dataset={DATASET}\nclip={audio_file}\ndevice={device} ({compute_type})  [{_dev_msg}]")

    # =====================================================================
    # CELL 1b — ASR QUALITY DIALS  (FIX 3)
    # ---------------------------------------------------------------------
    # V11 used "base", which is the second-smallest Whisper checkpoint and is
    # simply not good enough for conversational audio, crosstalk or film
    # dialogue. Model size is by far the biggest lever on WER; everything
    # else here is a second-order guard against known failure modes.
    #
    #   "auto"  -> large-v3 on CUDA, medium on CPU
    #   Force a size by replacing "auto" with "small" | "medium" | "large-v3".
    #   large-v3 is ~3GB and slow on CPU; medium is the sane CPU default.
    # =====================================================================
    ASR_MODEL_SIZE = "auto"

    # Pinning the language stops Whisper from mis-detecting on the first
    # 30s of a clip (a very common cause of garbage output). None = detect.
    ASR_LANGUAGE = "en"

    ASR_BEAM_SIZE = 5          # 1 = greedy. 5 is the usual accuracy/speed knee.
    ASR_BATCH_SIZE = 8         # lower this if you run out of memory
    ASR_CHUNK_SIZE = 30        # seconds per VAD chunk

    # Temperature fallback: if a chunk decodes with bad log-prob or a
    # suspicious compression ratio, retry it hotter instead of keeping junk.
    ASR_TEMPERATURE_FALLBACK = True

    # condition_on_previous_text=False is important. Leaving it True is the
    # main cause of Whisper's runaway repetition loops on noisy audio.
    ASR_CONDITION_ON_PREV = False

    ASR_LOGPROB_THRESHOLD = -1.0
    ASR_NO_SPEECH_THRESHOLD = 0.6
    ASR_COMPRESSION_RATIO_THRESHOLD = 2.4

    # Domain priming. A short prompt in the register of the audio measurably
    # helps with proper nouns and punctuation. Keep it short; long prompts
    # get echoed into the transcript.
    ASR_INITIAL_PROMPT = None
    # e.g. ASR_INITIAL_PROMPT = "A jury room argument. Twelve men, tense, interrupting."

    # Post-ASR screening for hallucination loops (same phrase repeated).
    ASR_DROP_LOOPS = True
    ASR_LOOP_MIN_REPEATS = 4
    return (
        ASR_BATCH_SIZE,
        ASR_BEAM_SIZE,
        ASR_CHUNK_SIZE,
        ASR_COMPRESSION_RATIO_THRESHOLD,
        ASR_CONDITION_ON_PREV,
        ASR_DROP_LOOPS,
        ASR_INITIAL_PROMPT,
        ASR_LANGUAGE,
        ASR_LOGPROB_THRESHOLD,
        ASR_LOOP_MIN_REPEATS,
        ASR_MODEL_SIZE,
        ASR_NO_SPEECH_THRESHOLD,
        ASR_TEMPERATURE_FALLBACK,
        DATASET,
        VERSION_TAG,
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
def _(DATASET, audio_file, emotion_map, iemocap_csv, pd):
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
    return (true_emotion,)


@app.cell
def _(mo):
    mo.md("""
    ## Part A: Clip-level classifier (load the V2 model)
    """)
    return


@app.cell
def _(
    Path,
    RandomForestClassifier,
    call,
    data_dir,
    drop_emotions,
    emotion_map,
    features_csv,
    joblib,
    np,
    os,
    parselmouth,
    pd,
):
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
    # CELL 6 — LOAD THE MODEL BUNDLE  (V18: clf_v3 preferred)
    # ---------------------------------------------------------------------
    # V18 change. V17 hardcoded clf_v2.joblib and read exactly 4 keys, which
    # meant the file could not say what it was trained on. That is how the
    # "60.6%" figure quoted all over this notebook became untraceable: no
    # bundle recorded the corpus, the protocol, or the class count it was
    # measured under, so the number outlived the model it described.
    #
    # clf_v3 (train_emotionsV3) writes provenance keys alongside the 4 the
    # loader needs. This cell now reads and PRINTS them, so every run states
    # its own pedigree in the log. Keys are read defensively with .get() --
    # a clf_v2 bundle has none of them and must still load.
    #
    # The search order is a list, not one path: clf_v3 first, clf_v2 second,
    # 14-feature retrain last. A missing clf_v3 falls back rather than
    # crashing, but it says loudly which one it got, because a silent
    # fallback to the old model is the failure mode most likely to waste a
    # render batch.
    # =====================================================================
    MODEL_BUNDLE_CANDIDATES = [
        "outputs/clf_v3.joblib",      # V18: IEMOCAP-primary, RAVDESS support
        "outputs/clf_v2.joblib",      # V17 legacy, kept so old runs reproduce
    ]

    model_bundle_path = next(
        (p for p in MODEL_BUNDLE_CANDIDATES if os.path.exists(p)), None)

    # Bundle provenance, defaulted for the clf_v2 / fallback paths that
    # cannot supply it. Downstream cells read CLF_* only; the CLF_META dict
    # exists so the audit cells can print what was actually loaded.
    CLF_META = {}

    if model_bundle_path is not None:
        _bundle = joblib.load(model_bundle_path)
        clf_full         = _bundle["clf"]
        clf_feature_cols = _bundle["feature_cols"]
        CLF_EXTRACTOR    = _bundle["extractor"]
        CLF_NORMALISED   = _bundle["speaker_normalised"]

        CLF_META = {
            "bundle_path":     model_bundle_path,
            "trained_by":      _bundle.get("trained_by", "<unrecorded>"),
            "trained_at":      _bundle.get("trained_at", "<unrecorded>"),
            "label_space":     _bundle.get("label_space_source", "<unrecorded>"),
            "cv_protocol":     _bundle.get("cv_protocol", "<unrecorded>"),
            "cv_accuracy":     _bundle.get("cv_pooled_accuracy"),
            "cv_macro_f1":     _bundle.get("cv_pooled_macro_f1"),
            "norm_scope":      _bundle.get("norm_scope", "<unrecorded>"),
            "ravdess_weight":  _bundle.get("ravdess_weight"),
            "n_rows_iemocap":  _bundle.get("n_rows_iemocap"),
            "n_rows_ravdess":  _bundle.get("n_rows_ravdess"),
            "merged_exc_hap":  _bundle.get("merge_excited_into_happy"),
            "dropped_classes": _bundle.get("dropped_ravdess_classes", []),
        }

        print(f"Loaded {model_bundle_path}")
        print(f"  extractor={CLF_EXTRACTOR}  features={len(clf_feature_cols)}  "
              f"speaker_normalised={CLF_NORMALISED}")
        print(f"  classes ({len(clf_full.classes_)}): {list(clf_full.classes_)}")
        if CLF_META["trained_by"] != "<unrecorded>":
            print(f"  trained by {CLF_META['trained_by']} "
                  f"at {CLF_META['trained_at']}")
            print(f"  label space from: {CLF_META['label_space']}  |  "
                  f"norm scope: {CLF_META['norm_scope']}")
            if CLF_META["n_rows_iemocap"] is not None:
                print(f"  rows: {CLF_META['n_rows_iemocap']} iemocap + "
                      f"{CLF_META['n_rows_ravdess']} ravdess "
                      f"@ w={CLF_META['ravdess_weight']}")
            print(f"  cv: {CLF_META['cv_protocol']}")
            _acc = CLF_META["cv_accuracy"]
            _f1  = CLF_META["cv_macro_f1"]
            print(f"      accuracy={_acc if _acc is None else round(_acc, 4)}  "
                  f"macro_f1={_f1 if _f1 is None else round(_f1, 4)}")
        else:
            print("  WARNING: this bundle records no provenance (pre-V18 "
                  "format). Any accuracy figure quoted for it in comments "
                  "below is unverifiable from the file itself.")

        # -------- validation, not assumption -----------------------------
        # A feature-count mismatch against the extractor is silent garbage
        # rather than an error: CELL 7 builds its inference vector from
        # CLF_EXTRACTOR, so a bundle claiming "egemaps" with the wrong
        # width would be fed 88 columns it was never fitted on.
        if CLF_EXTRACTOR == "egemaps" and len(clf_feature_cols) != 88:
            print(f"  WARNING: extractor='egemaps' but the bundle carries "
                  f"{len(clf_feature_cols)} feature columns, not the 88 of "
                  f"eGeMAPSv02/Functionals. Either the training run dropped "
                  f"columns (train_emotionsV3 CELL 7 drops any feature "
                  f"missing from one corpus) or the extractor label is "
                  f"wrong. Check before trusting a render.")
        if CLF_EXTRACTOR == "praat14" and len(clf_feature_cols) != 14:
            print(f"  WARNING: extractor='praat14' with "
                  f"{len(clf_feature_cols)} columns, expected 14.")
    else:
        print("No model bundle found on any of:")
        for _p in MODEL_BUNDLE_CANDIDATES:
            print(f"  {_p}")
        print("-> falling back to the 14-feature model trained here, in-notebook.")
        print("   This is NOT the shipping model. Its label space comes from "
              "RAVDESS, so it can emit 'calm' and cannot emit "
              "'frustrated'/'excited'.")
        _cols14 = [c for c in clip_df.columns if c not in ("file", "emotion", "actor")]
        clf_full = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        ).fit(clip_df[_cols14].to_numpy(), clip_df["emotion"].to_numpy())
        clf_feature_cols = _cols14
        CLF_EXTRACTOR  = "praat14"
        CLF_NORMALISED = False
        CLF_META = {"bundle_path": "<none: in-notebook 14-feature fallback>"}
        print(f"Trained 14-feature clf_full on {len(clip_df)} clips.")

    CLF_N_CLASSES = int(len(clf_full.classes_))
    CLF_CLASSES = [str(c) for c in clf_full.classes_]
    print(f"\nchance level for this model: 1/{CLF_N_CLASSES} = "
          f"{1.0 / CLF_N_CLASSES:.3f}")
    return (
        CLF_CLASSES,
        CLF_EXTRACTOR,
        CLF_META,
        CLF_NORMALISED,
        CLF_N_CLASSES,
        clf_feature_cols,
        clf_full,
        extract_clip_features,
        extract_clip_features_from_sound,
    )


@app.cell
def _(clf_full):
    print(list(clf_full.classes_))
    return


@app.cell
def _(
    CLF_EXTRACTOR,
    CLF_NORMALISED,
    CONF_CURVE,
    CONF_GAMMA,
    call,
    extract_clip_features,
    extract_clip_features_from_sound,
    np,
    opensmile,
    os,
    parselmouth,
    pd,
):
    # =====================================================================
    # CELL 12b — CONFIDENCE CURVE  (V13, audit 5.4)
    # ---------------------------------------------------------------------
    # One shared function for both prediction paths (cells 7 and 15), so the
    # clip-level and per-segment pipelines cannot drift apart again.
    #
    # The legacy formula divided by (0.5 - chance), which saturates at
    # p = 0.5: any posterior above one-half rendered at full strength, and a
    # 0.43 posterior at ~90%. The calibrated curve spans [chance, 1.0] and
    # applies a gamma, so styling strength now tracks the classifier's
    # relative preference without overstating its certainty. The benchmark
    # for that certainty is human agreement on emotion labelling — EmoLex
    # micro-averaged kappa 0.29 — not the near-ceiling agreement of tasks
    # like transcription. Output stays in [0.5, 1.0] as before: uncertainty
    # halves the expressive range, it never blanks the caption.
    # =====================================================================
    def confidence_scale(p_top, n_classes, curve=CONF_CURVE, gamma=CONF_GAMMA):
        _chance = 1.0 / max(int(n_classes), 2)
        _p = float(np.clip(p_top, 0.0, 1.0))
        if curve == "legacy":
            _t = np.clip((_p - _chance) / (0.5 - _chance), 0.0, 1.0)
        else:
            _t = np.clip((_p - _chance) / (1.0 - _chance), 0.0, 1.0) ** float(gamma)
        return float(0.5 + 0.5 * _t)

    # self-check: the two curves at the flagged operating point
    _c43 = confidence_scale(0.43, 7)
    _l43 = confidence_scale(0.43, 7, curve="legacy")
    print(f"confidence at p_top=0.43 (7 classes): calibrated {_c43:.2f} "
          f"| legacy {_l43:.2f}")
    assert confidence_scale(1.0 / 7, 7) == 0.5 and confidence_scale(1.0, 7) == 1.0

    # =====================================================================
    # CELL 7 — eGeMAPS extraction + per-segment prediction (top-2 for blend)
    # =====================================================================
    SEGMENT_NORM = "auto"      # "auto" | "on" | "off"
    NORM_MIN_SEGMENTS = 4

    if CLF_EXTRACTOR == "egemaps":
        smile_v9 = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        smile_v9 = None

    def clf_features_from_path(path):
        if CLF_EXTRACTOR == "egemaps":
            return smile_v9.process_file(str(path)).iloc[0].to_dict()
        return extract_clip_features(path)

    def clf_features_from_sound(seg_snd, tmp_wav="outputs/audio/_seg_tmp.wav"):
        if CLF_EXTRACTOR == "egemaps":
            os.makedirs(os.path.dirname(tmp_wav), exist_ok=True)
            call(seg_snd, "Save as WAV file", tmp_wav)
            return smile_v9.process_file(tmp_wav).iloc[0].to_dict()
        return extract_clip_features_from_sound(seg_snd)

    def enforce_min_dwell(path, seg_df, min_dwell_s, S, speaker_ids=None):
        """V16 -- post-Viterbi hygiene pass.

        self_bias is a PER-STEP cost: it penalises switching once, at the
        moment it happens, but it cannot see how long the run that follows
        actually lasts. A single noisy segment can still win a two-segment
        tug-of-war against self_bias and flash on screen for under a
        second. Duration is a different, complementary signal -- a colour
        state that only lasted a fraction of a second is almost certainly
        classifier noise regardless of what self_bias decided when it let
        the switch through.

        This scans the already-decoded Viterbi (or EMA) path for runs
        shorter than `min_dwell_s` (measured in wall-clock seconds from the
        segment start/end times) and merges each one into whichever
        neighbouring run has the stronger total emission support (summed
        posterior mass for that label across the short run), repeating
        until no run is left under the threshold. It cannot invent a label
        that was never in the path, and a run with no neighbour (the whole
        clip is one run) is left alone.

        V16.1: `speaker_ids`, if given (a per-segment array aligned with
        `path`), makes a speaker change a hard wall a short run can never
        be merged across. A brief run that IS a different speaker's entire
        turn is real information -- who is talking and how -- not noise,
        and merging it into a neighbour would misattribute one person's
        colour to someone else's line. If both neighbours are blocked by
        a speaker boundary, the run is left exactly as decoded.
        """
        if min_dwell_s <= 0.0 or len(path) < 3:
            return path.copy()
        path = path.copy()
        durations = (seg_df["end"].to_numpy(dtype=float)
                     - seg_df["start"].to_numpy(dtype=float))
        _spk = np.asarray(speaker_ids, dtype=object) if speaker_ids is not None else None
        n = len(path)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < n:
                j = i
                while j + 1 < n and path[j + 1] == path[i]:
                    j += 1
                run_dur = float(durations[i:j + 1].sum())
                if run_dur < min_dwell_s and (i > 0 or j < n - 1):
                    _left_blocked = (_spk is not None and i > 0
                                      and _spk[i - 1] != _spk[i])
                    _right_blocked = (_spk is not None and j + 1 < n
                                       and _spk[j] != _spk[j + 1])
                    left_lab = path[i - 1] if (i > 0 and not _left_blocked) else None
                    right_lab = path[j + 1] if (j + 1 < n and not _right_blocked) else None
                    if left_lab is None and right_lab is None:
                        # both neighbours blocked by a speaker boundary (or
                        # there is no neighbour at all) -- a lone different
                        # speaker's short turn stays as decoded
                        i = j + 1
                        continue
                    left_score = (float(S[i:j + 1, left_lab].sum())
                                  if left_lab is not None else -np.inf)
                    right_score = (float(S[i:j + 1, right_lab].sum())
                                   if right_lab is not None else -np.inf)
                    path[i:j + 1] = left_lab if left_score >= right_score else right_lab
                    changed = True
                i = j + 1
        return path

    def smooth_segment_emotions(seg_df, classes, mode="viterbi",
                                self_bias=0.72, min_dwell_s=0.0,
                                speaker_ids=None,
                                confidence_scale_fn=None, eps=1e-12):
        """Smooth emotion over TIME instead of deciding each segment alone.

        Every segment is currently classified independently, which encodes an
        assumption nobody chose: that a speaker's emotion is redrawn from
        scratch each sentence. It is not. Emotion persists across a scene,
        and five angry sentences in a row are five samples of one state, not
        five unrelated draws. Independent argmax throws that away, so a run
        of segments the classifier scores 0.31/0.29/0.28 flips colour on
        differences far smaller than the model's own error, and the caption
        strobes through the palette while the actor's delivery has not
        changed at all.

        The fix is a first-order hidden Markov model over the segment
        sequence (Rabiner 1989). Emissions are the classifier posteriors,
        already computed; the transition matrix puts `self_bias` on staying
        put and spreads the rest evenly over switching. Viterbi then returns
        the most likely PATH, so a switch has to be paid for in likelihood.

        Two properties make this the right tool rather than a blur:

        * It is confidence-weighted for free. A segment with a sharp
          posterior overrules its neighbours; a flat one is carried by them.
          No separate rule needed, and it is exactly the behaviour you want
          from a classifier that sits at 60.6 percent.
        * It cannot invent a label. Viterbi picks from what the classifier
          actually proposed, so a colour can never appear on screen that no
          segment supported.

        mode="ema" is the cheaper alternative: an exponential moving average
        over posteriors, then argmax. It is smoother but it can average two
        emotions into a third that neither segment voted for, which is why
        viterbi is the default.

        V16: after the path is decoded, enforce_min_dwell runs as a second,
        duration-based pass -- self_bias controls the PER-STEP cost of
        switching, `min_dwell_s` controls the MINIMUM ON-SCREEN duration of
        whatever the path decided. They catch different failure modes:
        self_bias alone can still let a single noisy segment win a close
        call; the dwell pass mops that up afterward without touching
        genuinely sustained runs.

        V16.1: `speaker_ids`, a dict {segment_id: speaker_label} from
        diarization (or None), removes the single biggest source of wrong
        persistence: self_bias and min_dwell_s both assume one continuous
        voice, which is false on fast multi-speaker dialogue -- there, a
        real, isolated angry outburst looks identical to classifier noise,
        and both features fixed above will happily erase it. When speaker
        change is known, the Viterbi transition (and the EMA blend) resets
        to unbiased at every speaker boundary instead of applying self_bias
        uniformly, and enforce_min_dwell refuses to merge a short run
        across a speaker change at all. speaker_ids=None reproduces plain
        V16 behaviour exactly (uniform persistence, no boundaries).

        Returns a copy with pred_emotion / pred_emotion2 / p_top / p_second /
        conf_scale rewritten, plus pred_emotion_raw and switched_by_smoother
        so the change is auditable rather than invisible.
        """
        out = seg_df.copy().sort_values("start").reset_index(drop=True)
        k = len(classes)
        if mode == "off" or len(out) < 2 or k < 2:
            out["pred_emotion_raw"] = out["pred_emotion"]
            out["switched_by_smoother"] = False
            return out

        cols = [f"p_{c}" for c in classes]
        if not all(c in out.columns for c in cols):
            print("SMOOTH: no posterior columns — these predictions predate "
                  "V15. Re-run prediction; returning unsmoothed.")
            out["pred_emotion_raw"] = out["pred_emotion"]
            out["switched_by_smoother"] = False
            return out

        P = out[cols].to_numpy(dtype=float)
        P = np.clip(P, eps, None)
        P = P / P.sum(axis=1, keepdims=True)
        n = len(P)

        # V16.1: resolve speaker_ids (a {segment_id: label} dict) against
        # THIS function's own post-sort row order, not the caller's -- out
        # is already sorted by "start" above, and segment_id order is not
        # guaranteed to match that. .map() keeps them correctly aligned
        # regardless of what order the caller built the dict in.
        _spk = (out["segment_id"].map(speaker_ids).to_numpy(dtype=object)
                if speaker_ids is not None else None)

        if mode == "ema":
            a = float(np.clip(1.0 - self_bias, 0.05, 1.0))
            S = P.copy()
            for i in range(1, n):
                if _spk is not None and _spk[i] != _spk[i - 1]:
                    S[i] = P[i]                           # reset: new speaker,
                else:                                       # no history carried in
                    S[i] = a * P[i] + (1.0 - a) * S[i - 1]
            for i in range(n - 2, -1, -1):        # backward pass, so the
                if _spk is not None and _spk[i] != _spk[i + 1]:
                    pass                                   # do not blend
                else:                                       # across a speaker change
                    S[i] = a * S[i] + (1.0 - a) * S[i + 1]   # smoothing is not
            S = S / S.sum(axis=1, keepdims=True)  # lagged in one direction
            path = np.argmax(S, axis=1)
        else:
            # ---- Viterbi over log-probabilities -------------------------
            sb = float(np.clip(self_bias, 1.0 / k + 1e-6, 0.999))
            log_stay = np.log(sb)
            log_move = np.log((1.0 - sb) / (k - 1))
            logP = np.log(P)

            # Full k-by-k transition matrix in logs: log_stay on the
            # diagonal, log_move everywhere else.
            logT = np.full((k, k), log_move)
            np.fill_diagonal(logT, log_stay)
            # V16.1: the reset matrix used at a speaker boundary instead --
            # uniform, so switching costs exactly what staying costs, and
            # the current segment's OWN emission decides the label,
            # undiluted by a bias meant for one continuous voice.
            logT_reset = np.full((k, k), -np.log(k))

            delta = logP[0].copy()               # uniform prior over states
            psi = np.zeros((n, k), dtype=int)
            for i in range(1, n):
                # cand[a, b] = score of being in a at i-1 and moving to b
                _use_reset = _spk is not None and _spk[i] != _spk[i - 1]
                cand = delta[:, None] + (logT_reset if _use_reset else logT)
                psi[i] = np.argmax(cand, axis=0)
                delta = cand[psi[i], np.arange(k)] + logP[i]
            path = np.zeros(n, dtype=int)
            path[-1] = int(np.argmax(delta))
            for i in range(n - 1, 0, -1):
                path[i - 1] = psi[i, path[i]]
            S = P

        # V16: duration-based cleanup pass, independent of self_bias.
        # V16.1: speaker-aware -- a short run is never merged across a
        # speaker boundary (see enforce_min_dwell docstring).
        path = enforce_min_dwell(path, out, min_dwell_s, S, speaker_ids=_spk)

        classes = np.asarray(classes)
        out["pred_emotion_raw"] = out["pred_emotion"]
        new_lab = classes[path]
        out["switched_by_smoother"] = new_lab != out["pred_emotion"].to_numpy()

        # Rewrite the downstream fields from the SMOOTHED posterior, so the
        # confidence the styling uses matches the label the styling shows.
        # Leaving p_top pointing at the old winner would render a colour at a
        # strength that was measured for a different colour.
        _rows = []
        for i in range(len(out)):
            probs = S[i]
            order = np.argsort(probs)[::-1]
            chosen = int(path[i])
            second = int(order[0]) if order[0] != chosen else int(order[1])
            _rows.append((str(classes[chosen]), str(classes[second]),
                          float(probs[chosen]), float(probs[second])))
        out["pred_emotion"] = [r[0] for r in _rows]
        out["pred_emotion2"] = [r[1] for r in _rows]
        out["p_top"] = [round(r[2], 3) for r in _rows]
        out["p_second"] = [round(r[3], 3) for r in _rows]
        if confidence_scale_fn is not None:
            out["conf_scale"] = [round(float(confidence_scale_fn(r[2], k)), 3)
                                 for r in _rows]

        _n_sw = int(out["switched_by_smoother"].sum())
        _runs_before = int((out["pred_emotion_raw"] !=
                            out["pred_emotion_raw"].shift()).sum())
        _runs_after = int((out["pred_emotion"] !=
                           out["pred_emotion"].shift()).sum())
        print(f"SMOOTH ({mode}, self_bias={self_bias}, min_dwell_s={min_dwell_s}, "
              f"speaker_aware={speaker_ids is not None}): "
              f"{_n_sw}/{len(out)} segment labels changed | colour changes "
              f"{_runs_before} -> {_runs_after}")
        if _runs_after <= 1 and len(out) > 4:
            print("  WARNING: the whole clip collapsed to ONE emotion. That "
                  "may be correct for a single-mood scene, but check it is "
                  "not self_bias or min_dwell_s set so high that the "
                  "classifier no longer matters.")
        return out

    def predict_segment_emotions_v9(audio_path, segments, clf, feature_cols,
                                    normalise_mode="auto", norm_min_segments=4):
        snd_full = parselmouth.Sound(audio_path)
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
                _r = {"segment_id": m["segment_id"], "start": m["start"],
                      "end": m["end"], "pred_emotion": "neutral",
                      "pred_emotion2": None, "p_top": 0.0, "p_second": 0.0,
                      "conf_scale": 0.5, "normalised": False}
                # a failed segment gets a FLAT posterior, not a confident
                # "neutral". Flat contributes no evidence to the smoother, so
                # it inherits its neighbours instead of punching a hole in them.
                for _cn in classes:
                    _r[f"p_{_cn}"] = round(1.0 / len(classes), 4)
                rows.append(_r)
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
            # V13: shared calibrated curve (cell 12b) — the inline formula
            # here and the one in cell 15 had to be kept in sync by hand
            conf   = confidence_scale(p_top, len(clf.classes_))
            _row = {"segment_id": m["segment_id"], "start": m["start"],
                    "end": m["end"], "pred_emotion": pred, "pred_emotion2": pred2,
                    "p_top": round(p_top, 3), "p_second": round(p_sec, 3),
                    "conf_scale": round(conf, 3), "normalised": bool(do_norm)}
            # V15: keep the WHOLE posterior, not just the top two. Smoothing
            # has to combine evidence across segments, and a label plus two
            # scalars has already thrown away the thing you would combine.
            for _ci, _cn in enumerate(classes):
                _row[f"p_{_cn}"] = round(float(proba[_ci]), 4)
            rows.append(_row)

        out = pd.DataFrame(rows)
        print(f"  predicted {len(out)} segment(s) | extractor={CLF_EXTRACTOR} | "
              f"normalised={do_norm} (mode='{normalise_mode}', {n_ok} usable segs)")
        return out

    return (
        NORM_MIN_SEGMENTS,
        SEGMENT_NORM,
        clf_features_from_path,
        confidence_scale,
        predict_segment_emotions_v9,
        smooth_segment_emotions,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Part B: WhisperX transcription → word timestamps + per-word prosody
    """)
    return


@app.cell
def _(
    ASR_BEAM_SIZE,
    ASR_COMPRESSION_RATIO_THRESHOLD,
    ASR_CONDITION_ON_PREV,
    ASR_INITIAL_PROMPT,
    ASR_LOGPROB_THRESHOLD,
    ASR_NO_SPEECH_THRESHOLD,
    ASR_TEMPERATURE_FALLBACK,
    gc,
    re,
    torch,
    whisperx,
):
    # =====================================================================
    # CELL 7c — VRAM HYGIENE  (V15)
    # ---------------------------------------------------------------------
    # WhisperX runs Whisper through CTranslate2, which allocates GPU memory
    # from the CUDA driver DIRECTLY, not from PyTorch's caching allocator.
    # PyTorch never hands freed blocks back to the driver on its own; it
    # keeps them cached for reuse. So a notebook that has loaded an
    # alignment model, dropped it, and then asks CTranslate2 for memory can
    # fail with "CUDA failed with error out of memory" while torch still
    # reports gigabytes free. The two allocators cannot see each other.
    #
    # free_vram() is therefore not a nicety. It is the specific thing that
    # makes a second transcription possible inside one long-lived reactive
    # session. Call it before any load_model or .transcribe on CUDA.
    # =====================================================================
    # A diagnostic must never be the thing that breaks the notebook. Every
    # CUDA call below is wrapped, because on a full card even mem_get_info()
    # raises: it needs a context, and creating one needs memory.
    def cuda_report(tag=""):
        """Print allocator state. driver-free is what CTranslate2 sees."""
        if torch is None:
            print("torch not installed")
            return None
        try:
            if not torch.cuda.is_available():
                print("no CUDA device")
                return None
            _free, _total = torch.cuda.mem_get_info()
            _alloc = torch.cuda.memory_allocated()
            _resv = torch.cuda.memory_reserved()
            print(f"VRAM {tag}: driver-free {_free/2**30:5.2f} / "
                  f"{_total/2**30:5.2f} GiB | torch allocated "
                  f"{_alloc/2**30:5.2f}, reserved {_resv/2**30:5.2f} GiB "
                  f"(reserved minus allocated is cache CTranslate2 CANNOT use)")
            return _free
        except Exception as _e:
            # Failing HERE is itself the finding: the card is full enough
            # that a context cannot be created, which means nothing your
            # notebook does to its own memory will help.
            print(f"VRAM {tag}: CUDA unusable ({type(_e).__name__}: "
                  f"{str(_e).splitlines()[0][:90]}).")
            print("  The GPU was already full before this notebook ran. "
                  "Check `nvidia-smi` for whose processes hold it; if they "
                  "are stale ones of yours, kill them. Otherwise set "
                  "FORCE_DEVICE = 'cpu' in the config cell and carry on.")
            return None

    def free_vram(*objs):
        """Drop references, collect, then RETURN torch's cache to the driver."""
        for _o in objs:
            del _o
        gc.collect()
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass          # nothing to give back if CUDA never came up
        gc.collect()
        return True

    cuda_report("at start")

    # =====================================================================
    # CELL 8a — ASR LOADER + LOOP SCREEN  (FIX 3)
    # ---------------------------------------------------------------------
    # whisperx's load_model signature and the set of keys it accepts inside
    # asr_options both drift between releases, so this walks a ladder from
    # the richest call down to the barest one and takes the first that binds.
    # =====================================================================
    def resolve_model_size(size, device):
        if size != "auto":
            return size
        return "large-v3" if device == "cuda" else "medium"

    def build_asr_options():
        temps = ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                 if ASR_TEMPERATURE_FALLBACK else [0.0])
        return {
            "beam_size": ASR_BEAM_SIZE,
            "best_of": ASR_BEAM_SIZE,
            "patience": 1.0,
            "temperatures": temps,
            "condition_on_previous_text": ASR_CONDITION_ON_PREV,
            "compression_ratio_threshold": ASR_COMPRESSION_RATIO_THRESHOLD,
            "log_prob_threshold": ASR_LOGPROB_THRESHOLD,
            "no_speech_threshold": ASR_NO_SPEECH_THRESHOLD,
            "initial_prompt": ASR_INITIAL_PROMPT,
        }

    def load_asr_model(size, device, compute_type, language):
        size = resolve_model_size(size, device)
        full = build_asr_options()
        minimal = {k: full[k] for k in ("beam_size", "temperatures",
                                        "condition_on_previous_text")}
        attempts = [
            dict(asr_options=full, language=language, vad_method="silero"),
            dict(asr_options=full, language=language),
            dict(asr_options=minimal, language=language),
            dict(language=language),
            dict(),
        ]
        last = None
        for kw in attempts:
            try:
                m = whisperx.load_model(size, device, compute_type=compute_type, **kw)
                print(f"  loaded whisper '{size}' on {device}/{compute_type} "
                      f"(opts: {sorted(kw.keys()) or 'defaults'})")
                return m, size
            except TypeError as e:
                last = e
                continue
            except ValueError as e:
                last = e
                continue
        raise RuntimeError(f"could not load whisper '{size}': {last}")

    LOOP_WORD_RE = re.compile(r"[A-Za-z']+")

    def looks_like_loop(text, min_repeats=4):
        """Whisper's classic failure: one phrase emitted N times in a row.
        Catches unigram and bigram runs; leaves genuine repetition alone
        below the threshold."""
        toks = LOOP_WORD_RE.findall(str(text).lower())
        if len(toks) < min_repeats:
            return False
        for size in (1, 2, 3):
            if len(toks) < size * min_repeats:
                continue
            grams = [tuple(toks[i:i + size]) for i in range(0, len(toks) - size + 1, size)]
            run, best = 1, 1
            for a, b in zip(grams, grams[1:]):
                run = run + 1 if a == b else 1
                best = max(best, run)
            if best >= min_repeats:
                return True
        return False

    def screen_segments(segments, drop_loops=True, min_repeats=4):
        if not drop_loops:
            return segments, 0
        kept, dropped = [], 0
        for s in segments:
            if looks_like_loop(s.get("text", ""), min_repeats):
                dropped += 1
                continue
            kept.append(s)
        return kept, dropped

    return free_vram, load_asr_model, screen_segments


@app.cell
def _(
    ASR_BATCH_SIZE,
    ASR_CHUNK_SIZE,
    ASR_DROP_LOOPS,
    ASR_LANGUAGE,
    ASR_LOOP_MIN_REPEATS,
    ASR_MODEL_SIZE,
    DIARIZE_ENABLE,
    DIARIZE_MAX_SPEAKERS,
    DIARIZE_MIN_SPEAKERS,
    HF_TOKEN,
    audio_file,
    compute_type,
    device,
    free_vram,
    load_asr_model,
    screen_segments,
    whisperx,
):
    # =====================================================================
    # CELL 8b — WHISPERX TRANSCRIPTION
    # =====================================================================
    # V15: hand torch's cache back to the driver BEFORE CTranslate2 asks
    # it for a ~3GB large-v3 encoder. Without this, a re-run of this cell in
    # a live session fails with "CUDA failed with error out of memory".
    free_vram()
    asr_model, asr_size_used = load_asr_model(
        ASR_MODEL_SIZE, device, compute_type, ASR_LANGUAGE)

    audio = whisperx.load_audio(audio_file)
    try:
        result = asr_model.transcribe(audio, batch_size=ASR_BATCH_SIZE,
                                      chunk_size=ASR_CHUNK_SIZE)
    except TypeError:
        result = asr_model.transcribe(audio, batch_size=ASR_BATCH_SIZE)

    _before = len(result["segments"])
    result["segments"], _dropped = screen_segments(
        result["segments"], ASR_DROP_LOOPS, ASR_LOOP_MIN_REPEATS)

    print(f"LANGUAGE: {result['language']} | model: {asr_size_used}")
    print(f"segments: {_before} kept {len(result['segments'])} "
          f"(dropped {_dropped} as repetition loops)")
    for _s in result["segments"][:12]:
        print(f"  [{_s['start']:7.2f}-{_s['end']:7.2f}] {_s['text'].strip()}")

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
    print(f"aligned {len(aligned['word_segments'])} words")
    aligned["word_segments"]

    # =====================================================================
    # CELL 9d — SPEAKER DIARIZATION (V16.1)
    # ---------------------------------------------------------------------
    # Stamps a "speaker" field onto every segment and word in `aligned`,
    # in place. Downstream, smooth_segment_emotions reads that field to
    # decide where a scene's persistence assumption should NOT apply.
    # Failure here is loud, not fatal -- a missing/invalid token or an
    # unaccepted model licence should not block the rest of the notebook,
    # it should just mean no speaker-aware smoothing this run.
    # =====================================================================
    if DIARIZE_ENABLE:
        if not HF_TOKEN:
            print("DIARIZE: HF_TOKEN not set (see CELL 12 for setup steps) "
                  "-- skipping. Smoothing will run without speaker "
                  "awareness, same as before diarization was added.")
        else:
            try:
                try:
                    _diarize_pipe = whisperx.DiarizationPipeline(
                        use_auth_token=HF_TOKEN, device=device)
                except AttributeError:
                    # older/newer whisperx releases have moved this in and
                    # out of the top-level namespace at different times
                    _diarize_pipe = whisperx.diarize.DiarizationPipeline(
                        use_auth_token=HF_TOKEN, device=device)
                _diarize_segments = _diarize_pipe(
                    audio_file, min_speakers=DIARIZE_MIN_SPEAKERS,
                    max_speakers=DIARIZE_MAX_SPEAKERS)
                try:
                    aligned = whisperx.assign_word_speakers(_diarize_segments, aligned)
                except AttributeError:
                    aligned = whisperx.diarize.assign_word_speakers(_diarize_segments, aligned)
                _n_spk = len({_s.get("speaker") for _s in aligned["segments"]
                              if _s.get("speaker")})
                _n_labelled = sum(1 for _s in aligned["segments"] if _s.get("speaker"))
                print(f"DIARIZE: {_n_spk} distinct speaker(s) found, "
                      f"{_n_labelled}/{len(aligned['segments'])} segments labelled")
            except Exception as _e:
                print(f"DIARIZE failed ({type(_e).__name__}: {str(_e)[:200]}) "
                      "-- continuing without speaker labels. Common causes: "
                      "HF_TOKEN invalid/expired, pyannote.audio not "
                      "installed (pip install pyannote.audio), or the "
                      "gated model terms not yet accepted at "
                      "huggingface.co/pyannote/speaker-diarization-3.1 and "
                      "huggingface.co/pyannote/segmentation-3.0.")
    return aligned, asr_model, result


@app.cell
def _(ADJ_ADV_TIER, CONTENT_MOD_PRIOR, np, re):
    # =====================================================================
    # CELL 9b — WORD CLASS LEXICON  (FIX 2, part 1)
    # ---------------------------------------------------------------------
    # The direct answer to "filter connecting words so nouns and verbs get
    # emphasised". This is a closed-class lexicon rather than a POS tagger:
    # function words are a *finite* list in English (a few hundred items),
    # so a lookup is exact, instant, dependency-free and deterministic.
    # Anything not in the list is treated as content — which is the right
    # default, because open classes (nouns, verbs, adjectives, adverbs) are
    # exactly the ones that carry lexical stress.
    #
    # Optional: set POS_SOURCE="spacy" if you have spaCy + en_core_web_sm
    # installed and would rather tag properly. It falls back silently.
    # =====================================================================
    POS_SOURCE = "lexicon"       # "lexicon" | "spacy"

    _DET = """a an the this that these those my your his her its our their
              some any no each every either neither both all half such what
              which whose another other others"""
    _PREP = """of in on at by for with about against between into through
               during before after above below to from up down out off over
               under again further then once upon within without across
               behind beyond near onto toward towards among around along
               beside besides despite except inside outside past per than
               unto via"""
    _CONJ = """and or but nor so yet because as if while whereas although
               though unless until whether since when where whenever wherever
               plus minus versus"""
    _AUX = """am is are was were be been being have has had having do does
              did doing get gets got will would shall going gonna wanna
              let lets"""
    _MODAL = """can could may might must should ought need dare"""
    _PRON = """i me you he him she it we us they them myself yourself himself
               herself itself ourselves yourselves themselves who whom
               someone somebody something anyone anybody anything everyone
               everybody everything nobody nothing one ones"""
    _NEG = """not no never none nothing cannot cant dont doesnt didnt wont
              wouldnt shouldnt couldnt isnt arent wasnt werent havent hasnt
              hadnt aint"""
    _FILLER = """uh um er ah eh mm mhm hmm uhh umm like well okay ok yeah
                 yep nope oh"""
    _DEGREE = """very really so too quite rather just even only also still
                 almost nearly barely hardly"""

    def _mk(s):
        return frozenset(w.strip() for w in s.split() if w.strip())

    WORD_CLASSES = {}
    for _cls, _src in (("det", _DET), ("prep", _PREP), ("conj", _CONJ),
                       ("aux", _AUX), ("modal", _MODAL), ("pron", _PRON),
                       ("filler", _FILLER), ("degree", _DEGREE)):
        for _w in _mk(_src):
            WORD_CLASSES.setdefault(_w, _cls)
    # negation wins over any earlier assignment ("no" is in _DET too)
    for _w in _mk(_NEG):
        WORD_CLASSES[_w] = "neg"

    # ---------------------------------------------------------------------
    # The prior. 1.0 = full weight, 0.0 = never emphasised.
    #
    #   neg     1.15  negation is very often THE stressed word ("I did NOT")
    #   content 1.00  nouns, verbs, adjectives, adverbs, numbers, names
    #   modal   0.85  "you MUST" is common and real
    #   degree  0.80  intensifiers legitimately carry stress
    #   pron    0.55  contrastive pronouns happen ("*I* said that")
    #   aux     0.45
    #   prep    0.40
    #   det     0.35
    #   conj    0.30  <-- "or", "and", "but": the reported false positive
    #   filler  0.25
    # ---------------------------------------------------------------------
    CLASS_PRIOR = {"neg": 1.15, "content": 1.00, "modal": 0.85, "degree": 0.80,
                   "pron": 0.55, "aux": 0.45, "prep": 0.40, "det": 0.35,
                   "conj": 0.30, "filler": 0.25,
                   # V13 (audit 5.2): suffix-marked adjectives/adverbs. The
                   # EmoLex POS breakdown (adjectives 68% / adverbs 67%
                   # emotion-associated vs nouns 46%) supports a SLIGHT tier
                   # above plain content. Applied only when ADJ_ADV_TIER is
                   # on; the actual value comes from CONTENT_MOD_PRIOR.
                   "content_mod": 1.00}

    # ------------------------------------------------------------------
    # V13: suffix heuristic for adjective/adverb detection (audit 5.2).
    # Deliberately conservative: derivational suffixes with stem-length and
    # vowel guards plus an exclusion list of the classic traps ("table" is
    # not -able, "family" is not -ly). Suffix-less adjectives (big, sad,
    # good) are OUT OF SCOPE and keep the plain content prior — this tier
    # only claims the morphologically marked cases. Measured in the
    # self-test cell below on an 81-word labelled sample.
    # ------------------------------------------------------------------
    ADJ_ADV_EXCLUDE = frozenset("""family assembly supply reply apply multiply
        imply comply rely ally rally tally folly belly jelly bully lily fly
        butterfly monopoly anomaly italy july
        handful spoonful cupful mouthful armful fistful
        olive motive arrive survive derive revive forgive receive deprive
        strive thrive contrive
        vegetable timetable turntable roundtable parable constable
        cable""".split())

    ADJ_VOWELS = frozenset("aeiou")

    def looks_adj_adv(token):
        """True when a (normalised) token is a morphologically marked
        adjective or adverb. Conservative by design; misses are cheap
        (the word keeps prior 1.00), false positives are not."""
        _w = str(token)
        if len(_w) < 4 or _w in ADJ_ADV_EXCLUDE:
            return False
        if _w.endswith("ly"):
            _stem = _w[:-2]
            if not _stem or not any(_ch in ADJ_VOWELS for _ch in _w):
                return False
            if _stem[-1] not in ADJ_VOWELS:       # quickly, early, only, ugly
                return True
            if _w.endswith("ely") and len(_w) >= 5:   # likely, lovely, safely
                return True
            if _w.endswith("ily") and len(_w) >= 6:   # happily, easily
                return True
            return False
        for _suf, _minstem in (("ful", 2), ("ous", 2), ("less", 2),
                               ("ive", 2), ("able", 3), ("ible", 3)):
            if _w.endswith(_suf):
                _stem = _w[: -len(_suf)]
                if (len(_stem) >= _minstem
                        and any(_ch in ADJ_VOWELS for _ch in _stem)):
                    return True
        return False

    TOKEN_CLEAN_RE = re.compile(r"[^a-z']")

    def normalise_token(w):
        return TOKEN_CLEAN_RE.sub("", str(w).strip().lower()).strip("'")

    def _spacy_tagger():
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            return None
        open_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM", "INTJ"}
        pos_to_cls = {"DET": "det", "ADP": "prep", "CCONJ": "conj",
                      "SCONJ": "conj", "AUX": "aux", "PRON": "pron",
                      "PART": "aux"}
        def tag(words):
            doc = spacy.tokens.Doc(nlp.vocab, words=[w or "x" for w in words])
            for _, proc in nlp.pipeline:
                doc = proc(doc)
            out = []
            for t in doc:
                if t.pos_ in open_pos:
                    out.append("content")
                else:
                    out.append(pos_to_cls.get(t.pos_, "content"))
            return out
        return tag

    SPACY_TAG = _spacy_tagger() if POS_SOURCE == "spacy" else None
    if POS_SOURCE == "spacy" and SPACY_TAG is None:
        print("spaCy unavailable -> using the built-in lexicon.")

    def classify_words(word_series, adj_adv_tier=ADJ_ADV_TIER,
                       content_mod_prior=CONTENT_MOD_PRIOR):
        """-> (list_of_class, ndarray_of_prior)"""
        toks = [normalise_token(w) for w in word_series]
        if SPACY_TAG is not None:
            try:
                classes = SPACY_TAG(toks)
            except Exception:
                classes = [WORD_CLASSES.get(t, "content") for t in toks]
        else:
            classes = [WORD_CLASSES.get(t, "content") for t in toks]
        # a token that survives cleaning as empty is punctuation-only
        classes = [c if t else "filler" for c, t in zip(classes, toks)]
        # V13: promote suffix-marked adjectives/adverbs to their own tier.
        # Only ever applied to words the lexicon already calls content, so
        # a conjunction like "unless" can never sneak in via its -less.
        if adj_adv_tier:
            classes = ["content_mod" if (c == "content" and looks_adj_adv(t))
                       else c for c, t in zip(classes, toks)]
        priors = np.array([content_mod_prior if c == "content_mod"
                           else CLASS_PRIOR.get(c, 1.0)
                           for c in classes], dtype=float)
        return classes, priors

    VOWEL_RUN_RE = re.compile(r"[aeiouy]+")

    def count_syllables(word):
        """Cheap English syllable estimate. Only needs to be monotonic in
        length for the duration model to work, not phonetically correct."""
        w = re.sub(r"[^a-z]", "", str(word).lower())
        if not w:
            return 1
        n = len(VOWEL_RUN_RE.findall(w))
        if w.endswith("e") and not w.endswith(("le", "ee", "ye", "oe", "ie")) and n > 1:
            n -= 1
        if w.endswith(("ed",)) and n > 1 and not w.endswith(("ted", "ded")):
            n -= 1
        return max(1, n)

    # =====================================================================
    # CELL 9c — ADJ/ADV HEURISTIC SELF-TEST  (V13, audit 5.2)
    # ---------------------------------------------------------------------
    # The first draft of this heuristic failed its own examples: "table"
    # parsed as t+able and "family" as fami+ly. Measure, don't assume.
    # Scope note: the positive set is DERIVATIONALLY SUFFIXED adj/adv only —
    # suffix-less adjectives (big, sad, good) are out of the heuristic's
    # scope by design and keep the plain content prior, so they are not
    # counted as misses here.
    # =====================================================================
    _SHOULD_MATCH = """quickly really suddenly angrily happily easily likely
        lovely lonely early only ugly silly deadly daily weekly friendly
        badly sadly fully barely slowly loudly softly beautiful careful
        wonderful awful painful famous nervous dangerous gorgeous serious
        obvious furious anxious massive active expensive positive negative
        creative effective aggressive careless endless harmless hopeless
        comfortable terrible horrible possible visible incredible
        reasonable valuable miserable""".split()
    _SHOULD_NOT = """table cable stable vegetable syllable timetable family
        italy supply reply apply rally belly jelly ally assembly butterfly
        monopoly olive motive arrive survive forgive receive five drive
        handful spoonful mouthful music magic finish publish polish window
        teacher water""".split()

    _hits = [_w_m1 for _w_m1 in _SHOULD_MATCH if looks_adj_adv(_w_m1)]
    _false = [_w_m1 for _w_m1 in _SHOULD_NOT if looks_adj_adv(_w_m1)]
    _recall = len(_hits) / len(_SHOULD_MATCH)
    print(f"adj/adv heuristic on {len(_SHOULD_MATCH) + len(_SHOULD_NOT)} "
          f"labelled words: recall {100 * _recall:.0f}% "
          f"({len(_hits)}/{len(_SHOULD_MATCH)}), "
          f"false positives {len(_false)}/{len(_SHOULD_NOT)}")
    _missed = sorted(set(_SHOULD_MATCH) - set(_hits))
    if _missed:
        print(f"  missed (cheap — they keep prior 1.00): {_missed}")
    assert not _false, f"adj/adv heuristic false positives: {_false}"
    assert _recall >= 0.85, "adj/adv heuristic recall regressed below 85%"
    return classify_words, count_syllables, normalise_token


@app.cell
def _(aligned, audio_file, call, count_syllables, np, parselmouth, pd):
    # =====================================================================
    # CELL 10 — PER-WORD PROSODY EXTRACTOR — ROBUST  (FIX 2, part 2)
    # ---------------------------------------------------------------------
    # Four changes from V11, all aimed at the false-positive problem:
    #
    #  1. Pitch is tracked ONCE over the whole clip with a speaker-adapted
    #     floor/ceiling, then sliced per word. V11 re-ran to_pitch() on each
    #     ~150ms word, where Praat has almost no context and octave errors
    #     are rife. An octave halving on one frame made f0_range explode,
    #     which is precisely how a throwaway "or" scored as a shout.
    #
    #  2. Octave outliers are dropped against the word's own median, and
    #     f0_range is the 10th-90th percentile spread rather than max-min,
    #     so one bad frame can no longer define the range.
    #
    #  3. A word needs MIN_VOICED_FRAMES voiced frames before its pitch
    #     features count at all; below that they are marked missing (0.0)
    #     and excluded from the segment's z-statistics.
    #
    #  4. Loudness is reported in dB (log domain, where z-scores behave)
    #     alongside the legacy linear rms.
    # =====================================================================
    MIN_VOICED_FRAMES = 4
    OCTAVE_TOL = 1.6          # drop frames >1.6x or <1/1.6x the word median
    PITCH_TIME_STEP = 0.005

    def estimate_pitch_range(snd):
        """ABLATION (no two-pass): De Looze & Hirst's speaker-adapted
        calibration is disabled. This always returns Praat's plain generic
        window (75-600Hz) regardless of who is speaking, so the rest of the
        pipeline (single whole-clip tracking pass, octave-outlier cleaning,
        MIN_VOICED_FRAMES gate, percentile-based f0_range) is exercised
        exactly as in V18, with only this one variable changed. This isolates
        the two-pass calibration's effect for comparison against the
        speaker-adapted version."""
        return 75.0, 600.0

    def clean_f0(f0v, tol=OCTAVE_TOL):
        if len(f0v) == 0:
            return f0v
        med = float(np.median(f0v))
        if med <= 0:
            return np.array([])
        return f0v[(f0v > med / tol) & (f0v < med * tol)]

    def extract_word_features(audio_path, word_segments, count_syllables_fn):
        _ALPHA_WARNED = []   # report a failed spectrum once, not per word
        snd = parselmouth.Sound(audio_path)
        floor, ceil = estimate_pitch_range(snd)

        # --- track once, slice per word -----------------------------------
        try:
            pitch = snd.to_pitch(time_step=PITCH_TIME_STEP,
                                 pitch_floor=floor, pitch_ceiling=ceil)
            p_t = np.asarray(pitch.xs(), dtype=float)
            p_f = np.asarray(pitch.selected_array["frequency"], dtype=float)
        except Exception:
            p_t = np.array([]); p_f = np.array([])

        try:
            inten = snd.to_intensity(minimum_pitch=max(floor, 50.0), time_step=0.01)
            i_t = np.asarray(inten.xs(), dtype=float)
            i_v = np.asarray(inten.values.flatten(), dtype=float)
        except Exception:
            i_t = np.array([]); i_v = np.array([])

        try:
            harm = snd.to_harmonicity()
            h_t = np.asarray(harm.xs(), dtype=float)
            h_v = np.asarray(harm.values.flatten(), dtype=float)
        except Exception:
            h_t = np.array([]); h_v = np.array([])

        rows = []
        for i, w in enumerate(word_segments):
            # WhisperX occasionally emits a word with no timing at all
            if w.get("start") is None or w.get("end") is None:
                continue
            start, end = float(w["start"]), float(w["end"])
            duration = max(end - start, 1e-4)

            if i < len(word_segments) - 1:
                nxt = word_segments[i + 1].get("start")
                pause_after = float(nxt) - end if nxt is not None else 0.0
            else:
                pause_after = 0.0
            pause_after = max(0.0, pause_after)

            # ---- pitch ----
            if len(p_t):
                m = (p_t >= start) & (p_t < end)
                f0v = p_f[m]
                f0v = f0v[f0v > 0]
            else:
                f0v = np.array([])
            f0v = clean_f0(f0v)
            n_voiced = int(len(f0v))

            if n_voiced >= MIN_VOICED_FRAMES:
                f0_mean = float(np.median(f0v))
                f0_range = float(np.percentile(f0v, 90) - np.percentile(f0v, 10))
            else:
                f0_mean = 0.0
                f0_range = 0.0

            # ---- slope, in semitones/sec, on the cleaned contour ----
            if n_voiced >= max(5, MIN_VOICED_FRAMES + 1):
                _t = np.linspace(0.0, duration, n_voiced)
                _st = 12.0 * np.log2(np.clip(f0v, 1e-6, None) / max(f0v[0], 1e-6))
                f0_slope = float(np.polyfit(_t, _st, 1)[0])
                f0_slope = float(np.clip(f0_slope, -60.0, 60.0))
            else:
                f0_slope = 0.0

            # ---- loudness ----
            if len(i_t):
                mi = (i_t >= start) & (i_t < end)
                iv = i_v[mi]
                iv = iv[np.isfinite(iv)]
                intensity_db = float(np.mean(iv)) if len(iv) else 0.0
            else:
                intensity_db = 0.0
            intensity_db = max(0.0, intensity_db)

            try:
                word_snd = snd.extract_part(from_time=start, to_time=end,
                                            preserve_times=True)
                rms = float(call(word_snd, "Get root-mean-square", 0, 0))
                rms = 0.0 if rms != rms else rms
            except Exception:
                rms = 0.0

            # ---- spectral tilt (alpha ratio), V15 ----------------------
            # Loudness alone cannot tell shouting from a speaker who is
            # simply close to the microphone, or from a clip whose gain was
            # pushed. Vocal EFFORT is different: pushing the voice raises
            # subglottal pressure, which puts proportionally more energy in
            # the upper spectrum and flattens the spectral tilt. The alpha
            # ratio (energy above 1 kHz over energy below it, in dB) is the
            # standard cheap measure of that, and it is a ratio, so a gain
            # change cancels out of it. This is what lets "yelling" be a
            # measurement rather than a synonym for "loud".
            alpha_ratio = 0.0
            try:
                if duration >= 0.06:
                    _spec = call(word_snd, "To Spectrum", "yes")
                    _lo = float(call(_spec, "Get band energy...", 50, 1000))
                    _hi = float(call(_spec, "Get band energy...", 1000, 5000))
                    if _lo > 0.0 and _hi > 0.0:
                        alpha_ratio = float(10.0 * np.log10(_hi / _lo))
                        alpha_ratio = float(np.clip(alpha_ratio, -60.0, 30.0))
            except Exception as _e:
                # Swallowing this silently is how a feature ends up zero for
                # every word while the detector quietly falls back to
                # loudness. Report once, then stop shouting about it.
                alpha_ratio = 0.0
                if not _ALPHA_WARNED:
                    print(f"alpha_ratio unavailable ({type(_e).__name__}: "
                          f"{str(_e)[:70]}) — shouting detection will fall "
                          "back to level only, which cannot tell a shout "
                          "from a loud recording.")
                    _ALPHA_WARNED.append(True)

            # ---- harmonicity ----
            if len(h_t):
                mh = (h_t >= start) & (h_t < end)
                hv = h_v[mh]
                hv = hv[(hv != -200) & np.isfinite(hv)]
                hnr = float(np.mean(hv)) if len(hv) else 0.0
            else:
                hnr = 0.0

            rows.append({
                "word": w["word"], "start": round(start, 3), "end": round(end, 3),
                "duration": round(duration, 3), "pause_after": round(pause_after, 3),
                "syllables": count_syllables_fn(w["word"]),
                "n_voiced": n_voiced,
                "f0_mean": round(f0_mean, 1), "f0_range": round(f0_range, 1),
                "f0_slope": round(f0_slope, 2),
                "rms": round(rms, 4), "intensity_db": round(intensity_db, 2),
                "hnr": round(hnr, 1),
                "alpha_ratio": round(alpha_ratio, 2),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # =================================================================
        # DURATION RESIDUAL  (the other half of the "OR" fix)
        # -----------------------------------------------------------------
        # V11 fed raw duration into salience, which conflates two different
        # things: a word can be long because it has four syllables, or long
        # because the speaker leaned on it. Only the second is emphasis.
        # Fit duration ~ a + b*syllables across the clip and score the
        # RESIDUAL. "Extraordinarily" no longer earns points for being long,
        # and a drawn-out "or" still can.
        # =================================================================
        syl = df["syllables"].to_numpy(dtype=float)
        dur = df["duration"].to_numpy(dtype=float)
        if len(df) >= 6 and syl.std() > 1e-9:
            b, a = np.polyfit(syl, dur, 1)
            expected = np.clip(a + b * syl, 0.03, None)
        else:
            expected = np.full_like(dur, max(float(np.median(dur)), 0.03))
        df["dur_expected"] = np.round(expected, 4)
        # log ratio: symmetric, scale-free, immune to speaking-rate drift
        df["dur_resid"] = np.round(np.log(np.clip(dur / expected, 1e-3, 1e3)), 4)
        return df

    # =====================================================================
    # CELL 11 — RUN PER-WORD EXTRACTION
    # =====================================================================
    word_df = extract_word_features(audio_file, aligned["word_segments"],
                                    count_syllables)
    word_df
    return extract_word_features, word_df


@app.cell
def _(mo):
    mo.md("""
    ## Part C: The expressive budget + all styling dials
    """)
    return


@app.cell
def _(os):
    # =====================================================================
    # CELL 12 — TUNABLE PARAMETERS
    # =====================================================================
    # ----- budget -----
    # intensity_db replaces rms (log domain), dur_resid replaces duration
    # (syllable-corrected). Weights favour loudness and lengthening, which
    # are the two most reliable acoustic correlates of English stress.
    SALIENCE_WEIGHTS = {"f0_mean": 0.9, "f0_range": 0.7,
                        "intensity_db": 1.2, "dur_resid": 1.0, "hnr": 0.3}

    # THE HEADLINE FIX. Features here score only when the word is ABOVE the
    # segment norm. V11 used |z| on everything, so being unusually quiet and
    # short scored the same as being loud and long. hnr stays two-sided:
    # both creak and breathiness are marked, in opposite directions.
    POSITIVE_ONLY_FEATURES = {"f0_mean", "f0_range", "intensity_db", "dur_resid"}

    ZERO_MEANS_MISSING = {"f0_mean", "f0_range", "hnr", "intensity_db"}

    ROBUST_STATS = True        # median/MAD instead of mean/std
    # Minimum spread, as a fraction of the feature's own centre, before a
    # feature is allowed to produce z-scores for a segment. Guards the
    # degenerate case where MAD collapses to 0 and a 0.1Hz difference across
    # an otherwise monotone line manufactures a z of 3. Raise to be stricter.
    SCALE_REL_FLOOR = 0.02
    MIN_WORDS_FOR_SALIENCE = 4 # below this a segment renders flat
    SALIENCE_SHRINK_K = 5.0    # short segments get flatter distributions

    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5
    USE_CONFIDENCE_SCALING = True

    # ----- V15: SEGMENT AROUSAL FLOOR --------------------------------
    # The salience budget is RELATIVE BY CONSTRUCTION, which is its whole
    # point, but it has a consequence nobody had measured. Every segment
    # gets exactly 100 points, and intensity is derived from a word's SHARE
    # of them. So a line where every word is shouted has every word at fair
    # share, every share_ratio near 1.0, and therefore every intensity near
    # ZERO. A screamed line and a flat line render identically, at minimum
    # styling. Anger is the class this hurts most, because sustained high
    # arousal across a whole utterance is exactly what anger sounds like.
    #
    # The fix is a floor, not a gain: multiplying an intensity of 0 by any
    # number leaves 0. A segment's ABSOLUTE arousal, measured against the
    # rest of the clip, sets the baseline the whole line renders at, and
    # the within-line competition then rides on top of it:
    #
    #     intensity = floor + (1 - floor) * intensity_raw
    #
    # which is monotone in intensity_raw, so relative emphasis is preserved
    # exactly, and bounded by 1, so nothing can blow out. This restores the
    # between-line dynamics the per-segment z-scoring discards. It is
    # emotion-agnostic on purpose: it fires on loud speech whatever the
    # classifier called it, so it cannot bias one class in the evaluation.
    SEGMENT_AROUSAL_FLOOR = 0.45     # 0.0 = V15 behaviour (no floor)
    AROUSAL_FEATURES = {"intensity_db": 1.0, "f0_mean": 0.6}
    AROUSAL_SPREAD = 1.5             # z at which the floor is ~76% of max

    # ----- V15: PER-EMOTION FLOOR BONUS (declare this if you use it) --
    # Additive bonus on the floor for a given predicted emotion. This is a
    # TASTE dial, not a measured one, and it is the exact move the audit
    # warned about, so it ships at zero. If an A/B shows anger genuinely
    # under-reads even after the arousal floor, raise it and say plainly in
    # the write-up that one class was hand-weighted, because a mute test
    # cannot otherwise separate "the mapping works" from "anger got more
    # ink than the other six".
    EMOTION_FLOOR_BONUS = {"angry": 0.00, "disgust": 0.00, "fearful": 0.00,
                           "happy": 0.00, "neutral": 0.00, "sad": 0.00,
                           "surprised": 0.00}

    # ----- V15: SHOUTING DETECTION AND CASE --------------------------
    # This is not a styling invention. DCMP's Captioning Key already
    # prescribes it: mixed case is preferred for readability, capitals are
    # used for screaming or shouting, and capitals must NOT be used for
    # general emphasis. Human captioners apply that rule by listening. The
    # contribution here is applying it from a measurement instead, which
    # means the gate has to be vocal effort and NOT salience, NOT loudness
    # on its own, and NOT the predicted emotion. Anger and shouting are
    # different things: people shout when happy and are quietly furious.
    #
    # Because the standard restricts capitals to shouting, over-firing is
    # the expensive error. Capitals cost readability (they strip the
    # ascender and descender cues that word-shape recognition uses), and
    # DCMP separately requires fonts that HAVE descenders, so a caption in
    # permanent caps would violate the same guidance it is citing. The
    # threshold is therefore set high and the coverage is reported.
    # ----- V15: TEMPORAL SMOOTHING OF EMOTION -------------------------
    # Segments were classified independently, which quietly assumed a
    # speaker's emotion is redrawn from scratch every sentence. Five angry
    # sentences are five samples of one state, not five unrelated draws, and
    # independent argmax makes the caption strobe through the palette on
    # posterior differences (0.34 / 0.29 / 0.36) far smaller than the
    # classifier's own error at 60.6 percent.
    #
    # "viterbi" runs a first-order HMM over the segment sequence: emissions
    # are the posteriors, the transition matrix favours staying put, and the
    # most likely PATH is decoded. It is confidence-weighted for free (a
    # sharp segment overrules its neighbours, a flat one is carried by them)
    # and it can only choose labels the classifier actually proposed.
    #
    # SELF_BIAS IS MEASURED, NOT GUESSED. Swept on four synthetic cases:
    #   0.40-0.60  wobble collapses to one colour AND a confident (p=0.81)
    #              one-segment outburst survives  <- the usable window
    #   >=0.65     the same outburst is erased; the path pays two
    #              transitions to visit it and the emission cannot cover it
    #   all values a genuine sustained change (angry -> sad) survives
    # 0.55 sits mid-window. A weak (p=0.45) outburst is erased at every
    # value, which is the intended behaviour: at that confidence the
    # classifier is barely distinguishing it from its neighbours anyway.
    #
    # Report this in the write-up. It changes what appears on screen, and a
    # reader is entitled to know the caption shows a decoded state sequence
    # rather than seven independent decisions.
    # V16.1: switched off. On fast multi-speaker dialogue (12 Angry Men)
    # the smoother's "favour staying" assumption actively works against
    # you -- it erased every single-segment angry outburst in a 51-segment
    # test clip (6 segments where the RAW classifier picked angry as an
    # outright winner, all overwritten to happy/disgust/surprised by the
    # Viterbi path), because it cannot tell "one noisy segment" apart from
    # "a different person just started talking, angrily." The notebook's
    # own limitation note below was right: without diarisation, persistence
    # assumptions don't hold across a speaker change. Re-enable ("viterbi")
    # once/if speaker-aware smoothing (reset stay-bias at speaker
    # boundaries) is in place -- see the diarization discussion.
    EMOTION_SMOOTH = "off"   # "viterbi" | "ema" | "off"
    # V16: raised 0.55 -> 0.68. The swept "usable window" above (0.40-0.65)
    # was chosen to protect a confident one-segment outburst from being
    # erased -- but 0.55 sits close to a coin flip on the stay/switch
    # decision, and with a 60%-accurate classifier that is not enough
    # persistence to stop ordinary noise from flipping colour. Pushing
    # self_bias alone past ~0.65 starts erasing genuine short outbursts
    # (per the sweep above), so V16 pairs a modest bump here with a
    # separate, duration-based guard (MIN_DWELL_S) that catches short
    # noisy runs self_bias would otherwise let through.
    EMOTION_SELF_BIAS = 0.68     # P(stay in the same emotion) per segment
    # V16 NEW: minimum on-screen duration for a smoothed colour run.
    # self_bias is a PER-STEP cost -- it cannot distinguish a genuine
    # two-segment outburst from one noisy segment sitting inside a long
    # run. Duration is a different, complementary signal: a colour state
    # that only lasted a fraction of a second is almost certainly
    # classifier noise regardless of what self_bias decided when it let
    # the switch through. enforce_min_dwell() (CELL 7) merges any run
    # shorter than this into whichever neighbour has stronger emission
    # support. Set to 0.0 to disable and get pure self_bias behaviour.
    MIN_DWELL_S = 1.2
    # Known limit: there is no diarisation, so the model treats a scene as
    # one speaker. In a fast argument between two people the smoother will
    # blend across a speaker change. For 12 Angry Men that is the main
    # thing to watch, and it is a limitation to state rather than hide.

    # ----- V16.1: SPEAKER DIARIZATION -----------------------------------
    # Directly targets the limitation named above. EMOTION_SELF_BIAS and
    # MIN_DWELL_S have no way to tell "one noisy segment" apart from "a
    # different person just started talking" -- both look identical to a
    # sequence model with no speaker signal. Diarization supplies that
    # signal: whisperx.DiarizationPipeline (a pyannote.audio wrapper)
    # labels each word/segment with a speaker id, and CELL 9's alignment
    # step folds that into `aligned`. smooth_segment_emotions then resets
    # its stay-bias at every speaker change instead of applying it
    # uniformly, so a genuine outburst that arrives with a new speaker is
    # no longer suppressed by persistence logic meant for one continuous
    # voice.
    #
    # pyannote's diarization model is GATED on HuggingFace:
    #   1. Create a free account at https://huggingface.co
    #   2. Visit and click "Agree and access repository" on BOTH:
    #        https://huggingface.co/pyannote/speaker-diarization-3.1
    #        https://huggingface.co/pyannote/segmentation-3.0
    #      (the diarization pipeline depends on both models)
    #   3. Generate a token at https://huggingface.co/settings/tokens
    #      ("Read" access is enough)
    #   4. Set it as an environment variable before launching marimo --
    #      NOT as a literal string in this file, so it never ends up
    #      committed to git or visible in a shared notebook:
    #        export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
    #
    # Degrades loudly rather than fatally if the token is missing or the
    # model download fails: CELL 9 prints exactly why and continues
    # without speaker labels, which reproduces the pre-diarization
    # behaviour (speaker_ids=None everywhere -> no resets, same as V16).
    DIARIZE_ENABLE = True
    HF_TOKEN = os.environ.get("HF_TOKEN")
    DIARIZE_MIN_SPEAKERS = None   # int, or None to let pyannote decide
    DIARIZE_MAX_SPEAKERS = None

    YELL_DETECT = True
    YELL_CASE = "upper"        # "upper" | "off" (detect but do not recase)
    # ----- normalisation reference (V15) ------------------------------
    # "robust_z" scores each segment against the clip's CENTRE (median and
    # MAD). "peak_ref" scores it against the clip's RANGE, rescaling the
    # 5th-95th percentile span to 0-1, so 1.0 is the loudest thing in the
    # clip and YELL_PEAK_FRAC asks "how close to that are we".
    #
    # peak_ref exists because centre-referencing has a specific blind spot:
    # on a clip that is loud throughout, a shout sits barely above a high
    # median and its z stays small, so nothing fires however hard someone
    # yells. Range-referencing does not care where the centre is.
    #
    # It has the opposite blind spot, and it is worse, so read this before
    # switching: EVERY clip has a maximum. On calm narration the quietest
    # possible "peak" is still 1.0, so peak_ref alone would capitalise
    # ordinary speech in a recording where nobody raises their voice. It is
    # therefore a RANKING signal, never a licence. The tilt gate and the
    # absolute dB margin below are what decide whether shouting happened at
    # all, and they apply identically in both modes. Percentiles rather than
    # true min/max, so one door slam cannot define the top of the scale.
    #
    # This mirrors what compute_calm already does for the spacing channel,
    # which normalises rate and loudness across the 5th-95th percentile
    # range. The yell detector was the inconsistent one.
    YELL_NORM = "peak_ref"     # "peak_ref" | "robust_z"
    YELL_PEAK_FRAC = 0.80      # peak_ref only: fire at/above this share of
                               # the clip's loud-to-quiet range
    YELL_Z = 1.30              # robust_z only: z of the effort score
    YELL_FEATURES = {"intensity_db": 1.0, "alpha_ratio": 0.9, "f0_mean": 0.5}
    YELL_TILT_MIN_Z = 0.75     # the spectral-tilt term must ALSO clear this
                               # on its own. Without it the weighted sum is
                               # dominated by loudness, and a speaker close
                               # to the mic (or a clip with the gain pushed)
                               # gets capitalised for volume they never
                               # produced. A conjunction, not a sum: loud
                               # AND effortful. Set to None to disable the
                               # gate, e.g. if alpha_ratio is unavailable.
    YELL_MIN_DB_OVER_MEDIAN = 4.0
                               # ABSOLUTE margin, in dB, over the clip's
                               # median segment loudness. z-scores are
                               # relative, so on a clip delivered at an even
                               # level the robust spread collapses and a 1 dB
                               # difference becomes a z of 2. That is the
                               # same degenerate-MAD failure SCALE_REL_FLOOR
                               # exists to catch in the salience budget, and
                               # it matters more here: a spurious flag does
                               # not merely over-emphasise a word, it tells a
                               # deaf viewer someone shouted when nobody did.
                               # Relative evidence says "loud for this clip";
                               # this says "loud full stop". Both must hold.
    YELL_MAX_FRAC = 0.25       # refuse to recase more than this share of
                               # the clip; above it the detector is miscal-
                               # ibrated and capitals stop being a marked
                               # form, so only the strongest lines are kept

    # Whisper's "!" is a LANGUAGE-MODEL guess about plausible text, not an
    # acoustic measurement. Letting it trigger capitals on its own would
    # quietly turn part of this project into a text-to-visual system, which
    # is the thing it claims not to be. So punctuation may only corroborate
    # a decision the acoustics already support: it lowers the threshold a
    # little, it can never cross it alone.
    YELL_PUNCT_ASSIST = 0.25   # z reduction when the line ends in "!"
    YELL_READING_PENALTY = 0.9 # all-caps reads slower, so shrink the cps
                               # ceiling on recased lines by this factor

    # ----- V16.2: EMOTION-DRIVEN SHOUT MARKERS ------------------------
    # The acoustic yell detector above answers "was this SPOKEN loudly,
    # with raised vocal effort" -- a measurement, deliberately blind to
    # the classified emotion (people shout when happy and are quietly
    # furious; see its docstring). This is a SECOND, independent trigger,
    # requested explicitly: when a segment's own emotion lands on anger,
    # or the classifier's top-2 are torn between anger and something in
    # EXCLAIM_EMOTIONS (the "close to mad" case -- read from the SAME
    # blend evidence already used for colour, so no separate threshold
    # to keep in sync), the line is capitalised and gets exclamation
    # marks appended, scaled by that segment's own intensity.
    #
    # This is a real departure from the acoustic detector's design
    # principle above ("the gate has to be vocal effort and NOT... the
    # predicted emotion"). That principle was there to stop a quietly
    # furious line reading as calm and a boisterous happy line reading
    # as angry; this dial deliberately overrides it because the ask
    # here is "make mad/happy READ as shouted," not "measure whether
    # shouting occurred." Worth stating plainly in any write-up, since
    # it changes what "capitals mean shouting" is actually keyed on for
    # this pipeline. Set EXCLAIM_ENABLE = False to fall back to pure
    # acoustic yelling (the pre-existing V15/V16 behaviour).
    EXCLAIM_ENABLE = True

    # V18 — EMOTION SET WIDENED TO THE NEW LABEL SPACE
    # ---------------------------------------------------------------------
    # {"angry", "happy"} was written when the model could only emit RAVDESS
    # classes. clf_v3's label space is IEMOCAP's, and "mad" now has TWO
    # words in it: `angry` (flaring) and `frustrated` (sustained). Leaving
    # frustrated out is not a neutral choice -- frustrated is one of
    # IEMOCAP's LARGEST classes on conversational speech, so under the old
    # set the trigger was blind to most of the footage that is, in plain
    # terms, someone being mad. Likewise `excited` is a real class now, not
    # a synonym for happy.
    #
    # This is the biggest single lever on how often the trigger fires --
    # considerably bigger than any threshold change below, because it
    # changes how many segments are even ELIGIBLE. If you want the
    # capitals back to their old rarity, shrink this set first and touch
    # SHOUT_SENSITIVITY second.
    EXCLAIM_EMOTIONS = {"angry", "frustrated", "excited", "happy"}
    EXCLAIM_MIN_MARKS = 1       # marks on the mildest qualifying segment
    EXCLAIM_MAX_MARKS = 3       # marks on the most intense one

    # V18 — SHOUT SENSITIVITY, AS A NAMED PRESET
    # ---------------------------------------------------------------------
    # This dial was previously two bare numbers (0.40 / 0.30) sitting in the
    # body of apply_emotion_shout, which is how "we tuned this once and then
    # lost which way we tuned it" happens. The V17 comment above them even
    # described a 0.50 floor while the code held 0.40 -- the comment and the
    # constant had already drifted apart. Naming the state makes it
    # recoverable: you can say "we shipped trigger_happy" in a write-up and
    # the notebook can prove what that meant.
    #
    # Floors are multiples of CHANCE (1/n_classes), not absolute posteriors,
    # so a label-space change does not silently re-tune the trigger. See
    # apply_emotion_shout for why absolute floors broke on the model swap.
    #
    #   conservative  reproduces V17's exact 0.40/0.30 at 7 classes.
    #                 On clf_v3 (8 classes, flatter posteriors) this fires
    #                 rarely to never -- it is the "off by accident" state.
    #   balanced      fires on segments the model is meaningfully confident
    #                 about, without needing a near-ceiling posterior.
    #   trigger_happy the pre-tuning behaviour: capitals whenever the
    #                 segment leans mad/excited at all. Deliberately loose.
    #
    # At 8 classes (chance 0.125) these resolve to:
    #   conservative   p_top >= 0.350   p_second >= 0.263
    #   balanced       p_top >= 0.250   p_second >= 0.200
    #   trigger_happy  p_top >= 0.169   p_second >= 0.138
    #
    # Read those against what the model actually produces. clf_v3 sits at
    # ~36% accuracy on 8 classes, so a "confident" segment often peaks
    # around 0.25-0.35, not 0.6. trigger_happy at 0.169 is only ~1.35x
    # chance, which means the capitals will land on plenty of segments the
    # classifier is close to guessing on. That is the trade you are asking
    # for -- expressive over accurate -- and it is a legitimate one for a
    # stylistic render. It is worth stating in a write-up rather than
    # leaving implicit, because it changes what the capitals MEAN: at this
    # setting they mark "the model leaned mad here", not "this was shouted".
    SHOUT_SENSITIVITY = "trigger_happy"   # "conservative"|"balanced"|"trigger_happy"

    SHOUT_PRESETS = {
        "conservative":  {"p_floor_mult": 2.80, "p_second_mult": 2.10},
        "balanced":      {"p_floor_mult": 2.00, "p_second_mult": 1.60},
        "trigger_happy": {"p_floor_mult": 1.35, "p_second_mult": 1.10},
    }
    if SHOUT_SENSITIVITY not in SHOUT_PRESETS:
        raise ValueError(
            f"SHOUT_SENSITIVITY={SHOUT_SENSITIVITY!r} is not one of "
            f"{sorted(SHOUT_PRESETS)}")
    SHOUT_P_FLOOR_MULT = SHOUT_PRESETS[SHOUT_SENSITIVITY]["p_floor_mult"]
    SHOUT_P_SECOND_MULT = SHOUT_PRESETS[SHOUT_SENSITIVITY]["p_second_mult"]
    print(f"shout sensitivity: {SHOUT_SENSITIVITY} "
          f"(p_floor={SHOUT_P_FLOOR_MULT}x chance, "
          f"p_second={SHOUT_P_SECOND_MULT}x chance) on "
          f"{sorted(EXCLAIM_EMOTIONS)}")

    # ----- word-class filter (FIX 2, part 3) -----
    USE_WORD_CLASS_PRIOR = True
    # ...but let genuine contrastive stress through.
    #
    # The override is keyed on RELATIVE prominence, not an absolute sigma.
    # An absolute threshold cannot work here. For SD-based z the exact bound
    # is |z| <= (n-1)/sqrt(n)  (about 2.67 for a 9-word line, not sqrt(n-1)),
    # and the outlier inflates the very scale estimate it is measured
    # against; with ROBUST_STATS=True the score is MAD-based, for which that
    # algebraic bound does not even hold — MAD compresses small samples
    # further still (and can collapse to 0, which is what SCALE_REL_FLOOR
    # guards). Either way, the most prominent word in a 9-word line
    # typically peaks around z=1.3, so a 1.75-sigma gate would never fire
    # and the filter would silently suppress every stressed connective —
    # the exact failure this feature exists to avoid. Cell 18b reports the
    # observed distribution of per-line peak z, which is the empirical
    # version of this argument (audit 5.3).
    #
    # Two conditions must both hold:
    #   CLASS_OVERRIDE_Z    absolute floor — the word must stand out at all,
    #                       so a flat line cannot promote its least-flat word
    #   CLASS_OVERRIDE_REL  the word must reach this fraction of the most
    #                       prominent word in its own line
    # Full recovery to content-word parity at CLASS_OVERRIDE_FULL.
    CLASS_OVERRIDE_Z = 0.90
    CLASS_OVERRIDE_REL = 0.75
    CLASS_OVERRIDE_FULL = 1.00

    # ----- emphasis quota -----
    # Caps how much of a line can be loud at once. Even with correct scoring,
    # emphasising half a sentence emphasises nothing.
    EMPHASIS_QUOTA_FRAC = 0.34
    QUOTA_DAMP = 0.45

    # ----- TYPOGRAPHY channel: intensity (size + weight) -----
    BASE_FONT_SIZE = 40
    FONT_SWING = 24        # was 32 on a base of 32 (a 2x jump); this is calmer
    FONT_GAMMA = 1.6       # >1 keeps mid intensities near base size
    BOLD_THRESHOLD = 0.62

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
    # "tremor" (V13): a fast group for fear. Lee et al. (2002) put trembling
    # with the HIGH-arousal states; V12 had fearful in "soft" at 1.50x, the
    # slowest setting, grouped with sad — the same arousal inversion as the
    # colour bug fixed in EMOTION_STYLES below (audit 2.4).
    MOTION_TEMPO = {"pop": 0.80, "soft": 1.50, "flat": 1.00, "tremor": 0.70}
    # words in a "tremor" segment wobble more readily: their wobble gate is
    # WOBBLE_RANGE_HZ * this factor (the V7 wobble was built for exactly
    # these nervous, unsteady words — fear is the class it was made for)
    TREMOR_WOBBLE_FACTOR = 0.6

    # ============ V12 FIX 1: LAYOUT MODE ================================
    # "absolute" — measure the text, emit one \pos'd Dialogue per word.
    #              A word's \fscy swell is then purely local: nothing else
    #              on the line moves. This is the fix for "the whole text
    #              jumps".
    # "flow"     — the V11 behaviour: one Dialogue per sentence, libass
    #              flows it. Kept for A/B comparison.
    LAYOUT_MODE = "absolute"
    LAYOUT_MARGIN_X = 60        # px from each edge before wrapping
    LAYOUT_MARGIN_V = 60        # px from the bottom to the last baseline
    LAYOUT_LINE_GAP = 0.22      # extra leading, as a fraction of row height
    LAYOUT_SPACE_SCALE = 1.0    # multiplier on the inter-word space
    MOTION_ANCHOR = "baseline"  # "baseline" (grows upward) | "center"
    # =====================================================================

    # ----- V10 CHANNEL 1: SATURATION = INTENSITY -----
    SATURATION_INTENSITY = True
    SAT_FLOOR_FRAC = 0.5

    # ----- V10 CHANNEL 2: UNCERTAINTY HUE BLEND -----
    BLEND_MODE = "blend"        # "off" | "blend" | "gradient"
    # V16: tightened 0.20 -> 0.08. Blending between the top-2 emotion
    # colours now only fires when the classifier is genuinely close to a
    # coin flip, not merely "somewhat unsure" -- most segments now render
    # as one clean, decisive colour.
    BLEND_MARGIN = 0.08
    # V16: lowered 0.30 -> 0.0. Every word inside an ambiguous segment now
    # gets the SAME blend ratio (p2 / (p1 + p2)), instead of each word's
    # own emphasis intensity nudging it independently -- that per-word
    # nudging was what produced a visible colour gradient across a single
    # sentence even when the segment's classification never changed.
    BLEND_PERWORD_SWING = 0.0

    # ----- V10 CHANNEL 3: LETTER SPACING = CALM -----
    TRACKING_CALM = True
    CALM_SPACING_MAX = 6.0

    # ----- V11 CHANNEL 4: HELD SPACE = SILENCE -----
    PAUSE_HOLD = True
    PAUSE_HOLD_THRESH = 0.40
    PAUSE_HOLD_FULL = 1.20
    PAUSE_HOLD_MAX_FSP = 40.0

    # ----- V13: CHANNEL RULE, made explicit (audit Part 1) --------------
    # "redundant" (Position B): emotion is carried by hue AND font family
    #   AND italic AND gesture tempo. Justified as designed redundancy for
    #   accessibility: the audit cell (12d) shows that under achromatopsia
    #   NO hue assignment achieves a minimum pairwise dE2000 above ~4.8, so
    #   a hue-only palette cannot serve viewers without colour vision at
    #   all. Cost: an A/B can no longer attribute an effect to one channel.
    # "hue_only" (Position A): one variable per channel, strictly. Font is
    #   uniform, italic off, tempo flat; emotion lives in hue alone.
    #   Matches AffType's finding that stacked channels underperform
    #   single ones. Use this condition to isolate the colour channel in
    #   the evaluation.
    CHANNEL_MODE = "redundant"       # "redundant" | "hue_only"

    # ----- V13: CONFIDENCE CURVE (audit 5.4) ----------------------------
    # "legacy" divided by (0.5 - chance): saturated at p=0.5, so a 0.43
    # posterior rendered at ~90% strength — more certainty than either the
    # classifier or human annotators possess (EmoLex Table 8: micro-avg
    # kappa 0.29 on this task). "calibrated" spans the full [chance, 1]
    # range with a gamma; 0.43 now lands at ~60%. Monotone either way.
    CONF_CURVE = "calibrated"        # "calibrated" | "legacy"
    CONF_GAMMA = 1.5

    # ----- V13: SQUASH AND STRETCH (audit 4.3) --------------------------
    # Pair every \fscy with an inverse \fscx so the swell conserves
    # apparent area and reads as deformation, not scale change (Lee et al.
    # 2002 after Lasseter/Thomas & Johnston). 1.0 = exact area
    # conservation; 0.45 reproduces the audit's example pairing
    # (\fscy110 with \fscx~96). Only active in absolute layout, where the
    # per-word \pos events make the horizontal squeeze provably local; the
    # layout reserves the widening a drop/wobble needs (see 13c).
    SQUASH_STRETCH = True
    SQUASH_CONSERVATION = 0.45

    # ----- V13: READING RATE (audit Part 8 item 3) ----------------------
    # Broadcast caption practice works to a characters-per-second ceiling
    # (BBC guidance is commonly summarised around 160-180 wpm; ~15-20 cps
    # is the usual band — verify the current BBC/Ofcom documents before
    # citing, per audit 2.9). When a line's natural duration implies a
    # faster rate, its end time is extended up to the next segment's start;
    # lines that still exceed the ceiling are reported loudly.
    READING_RATE_ENFORCE = True
    READING_RATE_MAX_CPS = 17.0

    # ----- V13: CONTRAST OVER ARBITRARY VIDEO (audit Part 8 item 2) -----
    # These were hardcoded in the ASS header; they are load-bearing for
    # legibility (AffType's original per-emotion colour scheme died on
    # readability) so they are dials now. Outline/shadow in px at PlayRes.
    CAPTION_OUTLINE_PX = 3
    CAPTION_SHADOW_PX = 1
    CAPTION_OUTLINE_COLOUR = "&H00000000"   # black outline (ASS &HAABBGGRR)
    CAPTION_BACK_COLOUR = "&H64000000"      # translucent shadow

    # ----- V13: FONT SUBSTITUTION (audit Part 8 item 8) -----------------
    # In absolute layout a silent substitution desyncs measurement from
    # rendering (words measured in one face, drawn in another), so it has
    # to be visible. It does NOT have to be fatal, and shipping it fatal
    # was a mistake: a missing font on a lab machine is an environment
    # problem, not a defect in the work, and halting the notebook over it
    # blocks every downstream cell. Default is now loud-and-continue.
    #   False = print exactly which families were substituted, carry on.
    #   True  = refuse to render at all. Turn this on for the renders
    #           that go in the dissertation and the evaluation, once
    #           CELL 14 reports every family resolving to itself, so the
    #           measured geometry provably matches the burned frames.
    FONT_STRICT = False

    # ----- V13: ADJ/ADV PRIOR TIER (audit 5.2) — default OFF ------------
    # EmoLex Table 4: adjectives 68% / adverbs 67% associated with at
    # least one emotion vs nouns 46%, so suffix-marked adj/adv get a
    # slight boost over other content words. Heuristic (suffix + guards,
    # measured in the self-test cell); OFF until A/B'd, per the audit.
    ADJ_ADV_TIER = False
    CONTENT_MOD_PRIOR = 1.08

    # ----- V13: EmoLex lexicon for stimulus screening (audit 7.1) -------
    # Download "NRC Word-Emotion Association Lexicon" from Saif Mohammad's
    # NRC page (free for research) and point this at the word-level file.
    EMOLEX_PATH = "resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

    # ----- COLOUR channel base per emotion (V13 palette) ----------------
    # Chosen by CIEDE2000 maximin search under Machado et al. (2009)
    # protanopia/deuteranopia simulation (cell 12d re-derives the numbers).
    # Evidence tiers per the audit (2.3):
    #   Tier 1, anchored (Jonauskaite p>=.4): angry=red, happy=yellow.
    #   Tier 2, anchored-with-stated-deviation: sad=blue (their anchors are
    #     black/gray — both illegible over video; blue is the idiom).
    #   Tier 3, unanchored, assigned on DISCRIMINABILITY: surprised=cyan
    #     190deg (no anchor exists for surprise; turquoise is one of their
    #     four lowest cross-national-variance terms; the search plateau is
    #     188-195deg), disgust=bile green 94deg (constraint 88-112deg; the
    #     plateau is flat to ~0.7 dE across the range — 94 favours the
    #     deuteranope case, the more common deficiency, and sits mid-bile),
    #     fearful=violet 252deg kept BUT s/v corrected: V12's (0.28, 0.88)
    #     gave Valdez & Mehrabian arousal -0.10, BELOW sad (+0.08) — fear
    #     was the lowest-arousal chromatic class. (0.70, 0.78) gives +0.18.
    #     Note purple is Jonauskaite's least stable term cross-nationally
    #     (.659): record first language + upbringing country at intake.
    #   Neutral: near-white, kept deliberately — a false positive of
    #     "relief" (white's anchor) harms less than mid-gray's "sadness".
    # Worst-case pairwise dE2000, V12 -> V13 (from cell 12d):
    #   normal 17.5 -> 24.0 | protan 7.5 -> 13.7 | deutan 10.2 -> 16.7
    #   achromatopsia 1.8 -> 4.8 (and <=4.8 for ANY hues: see CHANNEL_MODE)
    # V18 — TWO NEW CLASSES: frustrated, excited
    # ---------------------------------------------------------------------
    # clf_v3's label space comes from IEMOCAP, not RAVDESS, and IEMOCAP has
    # two categories RAVDESS never had. V17 had no entry for either, and the
    # lookup is styles.get(emotion, styles["neutral"]) in five places -- so
    # under V17 a frustrated segment rendered as near-white flat neutral
    # with no warning at all. On IEMOCAP, frustrated is one of the LARGEST
    # classes, so that silent fallback was not an edge case: it was a large
    # fraction of conversational speech being drawn as "no emotion".
    #
    # These two placements are PROPOSED, not audited. The V13 hue values
    # above came out of a documented search over CVD-simulated dE2000
    # (cell 12d) across 7 classes. Adding classes to a palette that was
    # optimised for 7 necessarily lowers worst-case pairwise separation --
    # there is less room on the wheel. Re-run CELL 12d before using these
    # for anything that matters; the numbers quoted above (normal 24.0 /
    # protan 13.7 / deutan 16.7 / achromatopsia 4.8) describe the OLD
    # 7-class palette and will drop.
    #
    #   frustrated -> 20deg rust. Frustration is anger's sustained, banked
    #     sibling, so it sits next to angry in hue (semantic grouping) and
    #     is separated from it on the other two axes instead: lower value
    #     (0.62 vs 0.80) and lower saturation (0.70 vs 0.85) read as
    #     smouldering rather than flaring. anim "flat" not "pop" for the
    #     same reason -- frustration does not spike. Shares angry's
    #     condensed family so the pair reads as related on screen.
    #     RISK: 20deg from angry is the tightest hue gap in this palette
    #     and the pair will be the worst case under protanopia, where the
    #     red-orange axis compresses. If cell 12d shows that pair below
    #     your floor, push frustrated toward 30-35deg and re-audit.
    #   excited -> 315deg magenta. Placed in the largest genuine gap left
    #     on the wheel (fearful 252 -> angry 360), not by semantics.
    #     Magenta reads high-arousal and is far from every existing hue,
    #     which is what the discriminability constraint wants. Shares
    #     happy's family and "pop" animation because excited/happy are the
    #     pair annotators confuse most.
    #     If this reads wrong on screen, the cleaner fix is upstream: set
    #     MERGE_EXCITED_INTO_HAPPY=True in train_emotionsV3 CELL 1 and
    #     retrain. That folds excited into happy, returns the palette to a
    #     size the V13 audit actually covers, and is what most published
    #     IEMOCAP work does anyway.
    #
    # "disgust" is kept below even though clf_v3 will usually NOT emit it:
    # IEMOCAP has only a handful of disgust utterances, so MIN_CLASS_COUNT
    # =40 drops the class at training time. An unused style entry is inert;
    # a missing one is a silent neutral render. Keep it.
    EMOTION_STYLES = {
        #             hue     sat   val  italic  anim      font
        "angry":      {"h": 0.0000, "s": 0.85, "v": 0.80, "i": 0, "anim": "pop",    "font": "DejaVu Sans Condensed"},
        "frustrated": {"h": 0.0556, "s": 0.70, "v": 0.62, "i": 0, "anim": "flat",   "font": "DejaVu Sans Condensed"},  # V18: 20deg rust
        "happy":      {"h": 0.1400, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},
        "excited":    {"h": 0.8750, "s": 0.85, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},             # V18: 315deg magenta
        "surprised":  {"h": 0.5278, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},   # 190deg cyan (was 27deg)
        "sad":        {"h": 0.6100, "s": 0.55, "v": 0.82, "i": 0, "anim": "soft",   "font": "Liberation Serif"},
        "fearful":    {"h": 0.7000, "s": 0.70, "v": 0.78, "i": 0, "anim": "tremor", "font": "DejaVu Serif"},  # arousal fix (was s.28 v.88, anim soft)
        "disgust":    {"h": 0.2611, "s": 0.55, "v": 0.72, "i": 1, "anim": "flat",   "font": "Liberation Mono"},  # 94deg bile (was 40deg)
        "neutral":    {"h": 0.0000, "s": 0.00, "v": 0.95, "i": 0, "anim": "flat",   "font": "Liberation Sans"},
    }
    return (
        ADJ_ADV_TIER,
        AROUSAL_FEATURES,
        AROUSAL_SPREAD,
        BASE_FONT_SIZE,
        BLEND_MARGIN,
        BLEND_MODE,
        BLEND_PERWORD_SWING,
        BOLD_THRESHOLD,
        CALM_SPACING_MAX,
        CAPTION_BACK_COLOUR,
        CAPTION_MODE,
        CAPTION_OUTLINE_COLOUR,
        CAPTION_OUTLINE_PX,
        CAPTION_SHADOW_PX,
        CHANNEL_MODE,
        CLASS_OVERRIDE_FULL,
        CLASS_OVERRIDE_REL,
        CLASS_OVERRIDE_Z,
        CONF_CURVE,
        CONF_GAMMA,
        CONTENT_MOD_PRIOR,
        DIARIZE_ENABLE,
        DIARIZE_MAX_SPEAKERS,
        DIARIZE_MIN_SPEAKERS,
        DIM_ALPHA,
        EMOLEX_PATH,
        EMOTION_FLOOR_BONUS,
        EMOTION_SELF_BIAS,
        EMOTION_SMOOTH,
        EMOTION_STYLES,
        EMPHASIS_QUOTA_FRAC,
        EXCLAIM_EMOTIONS,
        EXCLAIM_ENABLE,
        EXCLAIM_MAX_MARKS,
        EXCLAIM_MIN_MARKS,
        FONT_GAMMA,
        FONT_STRICT,
        FONT_SWING,
        FULL_DRAMA_RATIO,
        HF_TOKEN,
        HOLD_MAX_TAIL,
        LAYOUT_LINE_GAP,
        LAYOUT_MARGIN_V,
        LAYOUT_MARGIN_X,
        LAYOUT_MODE,
        LAYOUT_SPACE_SCALE,
        MIN_DWELL_S,
        MIN_LINE_DURATION,
        MIN_POINTS,
        MIN_WORDS_FOR_SALIENCE,
        MOTION_ANCHOR,
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
        POSITIVE_ONLY_FEATURES,
        QUOTA_DAMP,
        READING_RATE_ENFORCE,
        READING_RATE_MAX_CPS,
        REVEAL_MODE,
        ROBUST_STATS,
        SALIENCE_SHRINK_K,
        SALIENCE_WEIGHTS,
        SATURATION_INTENSITY,
        SAT_FLOOR_FRAC,
        SCALE_REL_FLOOR,
        SEGMENT_AROUSAL_FLOOR,
        SHOUT_P_FLOOR_MULT,
        SHOUT_P_SECOND_MULT,
        SLOPE_DEADZONE,
        SLOPE_FULL,
        SOFTMAX_TEMPERATURE,
        SQUASH_CONSERVATION,
        SQUASH_STRETCH,
        TRACKING_CALM,
        TREMOR_WOBBLE_FACTOR,
        USE_CONFIDENCE_SCALING,
        USE_WORD_CLASS_PRIOR,
        WOBBLE_RANGE_HZ,
        YELL_CASE,
        YELL_DETECT,
        YELL_FEATURES,
        YELL_MAX_FRAC,
        YELL_MIN_DB_OVER_MEDIAN,
        YELL_NORM,
        YELL_PEAK_FRAC,
        YELL_PUNCT_ASSIST,
        YELL_TILT_MIN_Z,
        YELL_Z,
        ZERO_MEANS_MISSING,
    )


@app.cell
def _(CLF_CLASSES, CLF_META, EMOTION_STYLES, EXCLAIM_EMOTIONS):
    # =====================================================================
    # CELL 6b — LABEL-SPACE CONTRACT CHECK  (V18, new)
    # ---------------------------------------------------------------------
    # The single most dangerous coupling in this notebook: the classifier's
    # label space and EMOTION_STYLES are set in two different cells, by two
    # different people's decisions, and NOTHING checked that they agree.
    #
    # The style lookup is styles.get(emotion, styles["neutral"]) in five
    # places. That default is deliberate -- rendering must not crash
    # mid-batch -- but it means a class the model emits and the palette
    # lacks renders as near-white flat neutral, silently, forever. Under
    # V17 + clf_v3 that would have hit `frustrated`, one of IEMOCAP's
    # largest classes: a substantial share of conversational speech drawn
    # as "no emotion detected", with no log line saying so.
    #
    # This cell fails LOUDLY at load time instead. It is cheap, it runs
    # before any render, and it is the difference between "the palette is
    # wrong" and "the study used the wrong palette for three weeks".
    #
    # It checks three directions, because they fail differently:
    #   1. model -> palette   a class with no style = silent neutral. FATAL.
    #   2. palette -> model   a style no class emits = dead colour. Inert,
    #                         but usually means the palette is stale.
    #   3. EXCLAIM_EMOTIONS   a shout trigger naming a class the model
    #                         cannot emit never fires. Silently inert.
    # =====================================================================
    STRICT_LABEL_SPACE = True     # False downgrades the fatal case to a warning

    _model = set(CLF_CLASSES)
    _palette = set(EMOTION_STYLES)

    _missing_style = sorted(_model - _palette)      # 1. fatal
    _dead_style = sorted(_palette - _model)         # 2. inert
    _dead_shout = sorted(set(EXCLAIM_EMOTIONS) - _model)   # 3. inert

    print(f"label-space contract check  ({CLF_META.get('bundle_path', '?')})")
    print(f"  model classes  ({len(_model)}): {sorted(_model)}")
    print(f"  palette styles ({len(_palette)}): {sorted(_palette)}")

    if _dead_style:
        print(f"\n  note: {len(_dead_style)} style(s) the model never emits: "
              f"{_dead_style}")
        print("    Harmless (an unused dict entry costs nothing), but check "
              "it is stale rather than a sign the model dropped a class you "
              "wanted. clf_v3 drops any IEMOCAP class under MIN_CLASS_COUNT "
              "-- 'disgust' goes this way, IEMOCAP has very few of them.")

    if _dead_shout:
        print(f"\n  WARNING: EXCLAIM_EMOTIONS names {_dead_shout}, which this "
              f"model cannot emit. Those entries can never fire. If you "
              f"expected shouting on {_dead_shout}, the trigger is silently "
              f"dead -- consider whether 'frustrated' should be in "
              f"EXCLAIM_EMOTIONS now that the model can emit it.")

    if _missing_style:
        _msg = (
            f"LABEL-SPACE MISMATCH — {len(_missing_style)} class(es) the "
            f"model emits have no EMOTION_STYLES entry:\n"
            f"    {_missing_style}\n"
            f"  Every segment predicted as one of these renders as NEUTRAL "
            f"(near-white, flat, no animation) via the styles.get(...) "
            f"default, with no per-segment warning.\n"
            f"  Fix by adding entries to EMOTION_STYLES in CELL 12, or -- if "
            f"the class is one you do not want -- retrain with a label space "
            f"that excludes it (train_emotionsV3: raise MIN_CLASS_COUNT, or "
            f"set MERGE_EXCITED_INTO_HAPPY=True) so the model cannot emit it "
            f"at all. Do NOT rely on the neutral fallback."
        )
        if STRICT_LABEL_SPACE:
            raise RuntimeError(_msg)
        print("\n  WARNING: " + _msg)
    else:
        print("\n  OK: every class the model can emit has its own style.")

    LABEL_SPACE_OK = not _missing_style
    return


@app.cell
def _(EMOTION_STYLES, SAT_FLOOR_FRAC, colorsys, matplotlib, np, os, pd, plt):
    # =====================================================================
    # CELL 12c — COLOUR SCIENCE  (V13, audit Part 8 item 6 + 2.10)
    # ---------------------------------------------------------------------
    # Hue degrees are a poor proxy for how far apart two styles LOOK,
    # because perceptual distance is not uniform around the hue circle.
    # This cell provides the real measure: CIEDE2000 in CIELAB, plus
    # colour-vision-deficiency simulation so the palette can be audited for
    # the ~8% of male viewers with a red-green deficiency.
    #
    # Everything here self-checks on load:
    #   * dE2000 is the Sharma, Wu & Dalal (2005) formulation, asserted
    #     against six of their published reference pairs (and verified
    #     off-line against the colour-science library to 0 error on the
    #     full 34-pair set and 400 random pairs).
    #   * CVD uses the PRECOMPUTED severity-1.0 matrices of Machado,
    #     Oliveira & Fernandes (2009), applied in linear RGB — digit-exact
    #     against the reference implementation. Their rows sum to 1, so
    #     neutrals are preserved exactly (asserted): the failure mode that
    #     sank the hand-derived tritanopia attempt.
    #   * Tritanopia proper is NOT simulated: Machado's own model treats
    #     tritan as a shift-paradigm approximation and explicitly refrains
    #     from claiming to model true tritanopia, so the "tritanopia" key
    #     here is severe tritanomaly, carries that caveat, and is excluded
    #     from the headline audit (it is also ~1000x rarer than red-green).
    # =====================================================================
    def srgb_to_linear(_c):
        _c = np.asarray(_c, dtype=float)
        return np.where(_c <= 0.04045, _c / 12.92, ((_c + 0.055) / 1.055) ** 2.4)

    def linear_to_srgb(_c):
        _c = np.clip(np.asarray(_c, dtype=float), 0.0, 1.0)
        return np.where(_c <= 0.0031308, 12.92 * _c,
                        1.055 * (_c ** (1 / 2.4)) - 0.055)

    RGB2XYZ_D65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                            [0.2126729, 0.7151522, 0.0721750],
                            [0.0193339, 0.1191920, 0.9503041]])
    XYZ_WHITE_D65 = RGB2XYZ_D65 @ np.ones(3)

    def srgb_to_lab(rgb):
        """Gamma-encoded sRGB in [0,1] -> CIELAB (D65)."""
        _xyz = RGB2XYZ_D65 @ srgb_to_linear(np.asarray(rgb, dtype=float))
        _xr, _yr, _zr = _xyz / XYZ_WHITE_D65
        _eps, _kap = 216.0 / 24389.0, 24389.0 / 27.0

        def _f(_t):
            return np.cbrt(_t) if _t > _eps else (_kap * _t + 16.0) / 116.0

        _fx, _fy, _fz = _f(_xr), _f(_yr), _f(_zr)
        return np.array([116.0 * _fy - 16.0, 500.0 * (_fx - _fy),
                         200.0 * (_fy - _fz)])

    def delta_e_2000(lab1, lab2):
        """CIEDE2000 per Sharma, Wu & Dalal (2005)."""
        _L1, _a1, _b1 = [float(_v) for _v in lab1]
        _L2, _a2, _b2 = [float(_v) for _v in lab2]
        _C1 = np.hypot(_a1, _b1); _C2 = np.hypot(_a2, _b2)
        _Cbar = 0.5 * (_C1 + _C2)
        _G = 0.5 * (1.0 - np.sqrt(_Cbar**7 / (_Cbar**7 + 25.0**7)))
        _a1p, _a2p = (1.0 + _G) * _a1, (1.0 + _G) * _a2
        _C1p = np.hypot(_a1p, _b1); _C2p = np.hypot(_a2p, _b2)
        _h1p = np.degrees(np.arctan2(_b1, _a1p)) % 360.0 if (_a1p or _b1) else 0.0
        _h2p = np.degrees(np.arctan2(_b2, _a2p)) % 360.0 if (_a2p or _b2) else 0.0
        _dLp = _L2 - _L1
        _dCp = _C2p - _C1p
        if _C1p * _C2p == 0.0:
            _dhp = 0.0
        else:
            _dhp = _h2p - _h1p
            if _dhp > 180.0:
                _dhp -= 360.0
            elif _dhp < -180.0:
                _dhp += 360.0
        _dHp = 2.0 * np.sqrt(_C1p * _C2p) * np.sin(np.radians(_dhp) / 2.0)
        _Lbp = 0.5 * (_L1 + _L2)
        _Cbp = 0.5 * (_C1p + _C2p)
        if _C1p * _C2p == 0.0:
            _hbp = _h1p + _h2p
        else:
            _hsum = _h1p + _h2p
            if abs(_h1p - _h2p) <= 180.0:
                _hbp = 0.5 * _hsum
            elif _hsum < 360.0:
                _hbp = 0.5 * (_hsum + 360.0)
            else:
                _hbp = 0.5 * (_hsum - 360.0)
        _T = (1.0 - 0.17 * np.cos(np.radians(_hbp - 30.0))
              + 0.24 * np.cos(np.radians(2.0 * _hbp))
              + 0.32 * np.cos(np.radians(3.0 * _hbp + 6.0))
              - 0.20 * np.cos(np.radians(4.0 * _hbp - 63.0)))
        _dtheta = 30.0 * np.exp(-(((_hbp - 275.0) / 25.0) ** 2))
        _RC = 2.0 * np.sqrt(_Cbp**7 / (_Cbp**7 + 25.0**7))
        _SL = 1.0 + (0.015 * (_Lbp - 50.0) ** 2) / np.sqrt(20.0 + (_Lbp - 50.0) ** 2)
        _SC = 1.0 + 0.045 * _Cbp
        _SH = 1.0 + 0.015 * _Cbp * _T
        _RT = -np.sin(np.radians(2.0 * _dtheta)) * _RC
        return float(np.sqrt((_dLp / _SL) ** 2 + (_dCp / _SC) ** 2
                             + (_dHp / _SH) ** 2
                             + _RT * (_dCp / _SC) * (_dHp / _SH)))

    # Machado, Oliveira & Fernandes (2009), severity 1.0, linear RGB.
    MACHADO_CVD = {
        "protanopia":   np.array([[ 0.152286,  1.052583, -0.204868],
                                  [ 0.114503,  0.786281,  0.099216],
                                  [-0.003882, -0.048116,  1.051998]]),
        "deuteranopia": np.array([[ 0.367322,  0.860646, -0.227968],
                                  [ 0.280085,  0.672501,  0.047413],
                                  [-0.011820,  0.042940,  0.968881]]),
        # severe tritanomaly ONLY — see the cell comment. Not in the
        # headline audit; kept for exploratory use with the caveat.
        "tritanopia":   np.array([[ 1.255528, -0.076749, -0.178779],
                                  [-0.078411,  0.930809,  0.147602],
                                  [ 0.004733,  0.691367,  0.303900]]),
    }
    LUMA_REC709 = np.array([0.2126, 0.7152, 0.0722])

    def simulate_cvd(rgb, condition):
        """Gamma sRGB [0,1] -> simulated gamma sRGB [0,1]."""
        _lin = srgb_to_linear(np.asarray(rgb, dtype=float))
        if condition == "normal":
            _out = _lin
        elif condition == "achromatopsia":
            _y = float(LUMA_REC709 @ _lin)
            _out = np.array([_y, _y, _y])
        else:
            _out = MACHADO_CVD[condition] @ _lin
        return linear_to_srgb(np.clip(_out, 0.0, 1.0))

    def emotion_rgb01(styles, emotion, sat_frac=1.0):
        _fam = styles.get(emotion, styles["neutral"])
        return np.array(colorsys.hsv_to_rgb(_fam["h"] % 1.0,
                                            min(1.0, _fam["s"] * sat_frac),
                                            _fam["v"]))

    # ---- self-checks (fail loudly rather than audit with broken maths) --
    _SHARMA_REF = [  # six pairs from Sharma et al. (2005), Table 1
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((50.0, 2.5, 0.0), (50.0, 3.2592, 0.3350), 1.0000),
    ]
    for _lab1, _lab2, _want in _SHARMA_REF:
        _got = delta_e_2000(_lab1, _lab2)
        assert abs(_got - _want) < 1e-3, f"dE2000 self-check: {_got} != {_want}"

    for _cond in ("protanopia", "deuteranopia", "tritanopia", "achromatopsia"):
        _g = simulate_cvd(np.array([0.5, 0.5, 0.5]), _cond)
        assert float(np.max(np.abs(_g - 0.5))) < 1e-6, \
            f"{_cond} does not preserve neutral grey: {_g}"

    def _sim_de(_rgb1, _rgb2, _cond):
        return delta_e_2000(srgb_to_lab(simulate_cvd(_rgb1, _cond)),
                            srgb_to_lab(simulate_cvd(_rgb2, _cond)))

    # red/green must COLLAPSE under red-green deficiency and SURVIVE
    # blue/yellow, and vice versa is not required. Ratio-based thresholds
    # (with margin around the measured values 0.49 / 0.22 / 0.84 / 0.91)
    # rather than the absolute cutoff that was left unverified last time.
    _rg_norm = _sim_de((1, 0, 0), (0, 1, 0), "normal")
    _by_norm = _sim_de((0, 0, 1), (1, 1, 0), "normal")
    assert _sim_de((1, 0, 0), (0, 1, 0), "protanopia") / _rg_norm < 0.65
    assert _sim_de((1, 0, 0), (0, 1, 0), "deuteranopia") / _rg_norm < 0.45
    assert _sim_de((0, 0, 1), (1, 1, 0), "protanopia") / _by_norm > 0.75
    assert _sim_de((0, 0, 1), (1, 1, 0), "deuteranopia") / _by_norm > 0.75
    print("colour science self-checks passed "
          "(dE2000 vs Sharma refs, grey preservation, red-green collapse)")

    # =====================================================================
    # CELL 12d — PALETTE AUDIT  (V13; audit 2.2/2.4/2.6/2.10, Part 8 item 6)
    # ---------------------------------------------------------------------
    # Re-derives, on every run, the numbers the V13 palette was chosen by,
    # so a future palette edit is audited automatically. Reports:
    #   * pairwise dE2000 between all 7 styles under normal vision,
    #     protanopia, deuteranopia and achromatopsia
    #   * the same at the saturation floor (intensity=0 words render at
    #     s * SAT_FLOOR_FRAC, the worst case for discriminability)
    #   * the Valdez & Mehrabian implied arousal ordering — the check that
    #     caught V12's fear bug (arousal = -0.31*V + 0.60*S applied to the
    #     palette's own S/V; fear must land ABOVE sad)
    # =====================================================================
    AUDIT_CONDITIONS = ("normal", "protanopia", "deuteranopia", "achromatopsia")

    def audit_palette(styles, conditions=AUDIT_CONDITIONS, sat_frac=1.0):
        """-> long DataFrame: condition, emotion_a, emotion_b, delta_e."""
        _names = list(styles)
        _rows = []
        for _cond_m1 in conditions:
            _labs = {_n: srgb_to_lab(simulate_cvd(
                emotion_rgb01(styles, _n, sat_frac), _cond_m1)) for _n in _names}
            for _i, _a in enumerate(_names):
                for _b in _names[_i + 1:]:
                    _rows.append({"condition": _cond_m1, "emotion_a": _a,
                                  "emotion_b": _b,
                                  "delta_e": round(delta_e_2000(_labs[_a],
                                                                _labs[_b]), 1)})
        return pd.DataFrame(_rows)

    palette_audit_df = audit_palette(EMOTION_STYLES)

    print("pairwise dE2000 by condition (full saturation):")
    for _cond_m1 in AUDIT_CONDITIONS:
        _sub = palette_audit_df[palette_audit_df["condition"] == _cond_m1]
        _worst = _sub.nsmallest(3, "delta_e")
        _n_low = int((_sub["delta_e"] < 15.0).sum())
        print(f"  {_cond_m1:13s} min={_sub['delta_e'].min():5.1f}  "
              f"pairs<15: {_n_low:2d}/21  worst: "
              + ", ".join(f"{_r.emotion_a}/{_r.emotion_b}={_r.delta_e}"
                          for _r in _worst.itertuples()))

    _floor_df = audit_palette(EMOTION_STYLES, sat_frac=SAT_FLOOR_FRAC)
    for _cond_m1 in ("normal", "deuteranopia"):
        _sub = _floor_df[_floor_df["condition"] == _cond_m1]
        _w = _sub.nsmallest(1, "delta_e").iloc[0]
        print(f"  at the saturation floor (s x{SAT_FLOOR_FRAC}), {_cond_m1}: "
              f"min={_w['delta_e']} ({_w['emotion_a']}/{_w['emotion_b']}) — "
              f"SAT_FLOOR_FRAC trades intensity range against baseline "
              f"discriminability")

    print("achromatopsia note: a hue-space search puts the best achievable "
          "minimum at ~4.8 dE for ANY assignment — hue cannot serve viewers "
          "without colour vision, which is the empirical case for "
          "CHANNEL_MODE='redundant' (audit Part 1 / 2.10).")

    # Valdez & Mehrabian implied PAD, applied to the palette's own S/V.
    # (Their equations map colour to FELT emotion; using them in reverse to
    # pick colours is an inversion the write-up must flag — audit 2.4.)
    def vm_pad(_s, _v):
        return {"pleasure": 0.69 * _v + 0.22 * _s,
                "arousal": -0.31 * _v + 0.60 * _s,
                "dominance": -0.76 * _v + 0.32 * _s}

    _arousal = {_n: vm_pad(_f_m1["s"], _f_m1["v"])["arousal"]
                for _n, _f_m1 in EMOTION_STYLES.items()}
    print("implied arousal ordering:",
          "  ".join(f"{_n}{_arousal[_n]:+.2f}" for _n in
                    sorted(_arousal, key=_arousal.get, reverse=True)))
    assert _arousal["fearful"] > _arousal["sad"], (
        "fear must sit above sad on implied arousal — this assert is the "
        "V12 bug (fear at -0.10 vs sad +0.08) made permanent")
    # value stays fixed per emotion while only saturation carries intensity,
    # so the ramp does not fight the arousal equation (audit 2.1's check)
    palette_audit_df

    # =====================================================================
    # CELL 12e — CVD FIGURE  (V13, audit 2.10 — "highest value-per-hour")
    # ---------------------------------------------------------------------
    # Renders every emotion label in its palette colour, outlined the way
    # libass outlines it, on a video-grey ground, under each simulated
    # condition. Saved to outputs/figures/cvd_palette.png for the
    # dissertation. Tritanopia is deliberately absent: see cell 12c.
    # =====================================================================
    _FIG_CONDS = ("normal", "protanopia", "deuteranopia", "achromatopsia")
    _emos = list(EMOTION_STYLES)
    cvd_fig, _axes = plt.subplots(len(_FIG_CONDS), 1,
                                  figsize=(9.5, 1.05 * len(_FIG_CONDS)))
    for _ax, _cond_m2 in zip(np.atleast_1d(_axes), _FIG_CONDS):
        _ax.set_facecolor("#3f3f3f")
        _ax.set_xlim(0, len(_emos))
        _ax.set_ylim(0, 1)
        _ax.set_xticks([]); _ax.set_yticks([])
        _sub_m2 = palette_audit_df[palette_audit_df["condition"] == _cond_m2]
        _ax.set_ylabel(f"{_cond_m2}\nmin dE {_sub_m2['delta_e'].min():.1f}",
                       rotation=0, ha="right", va="center", fontsize=9)
        for _j, _e in enumerate(_emos):
            _rgb = simulate_cvd(emotion_rgb01(EMOTION_STYLES, _e), _cond_m2)
            _ax.text(_j + 0.5, 0.5, _e, color=tuple(_rgb), fontsize=13,
                     fontweight="bold", ha="center", va="center",
                     path_effects=[matplotlib.patheffects.withStroke(
                         linewidth=3, foreground="black")])
    cvd_fig.suptitle("V13 palette under colour-vision-deficiency simulation "
                     "(Machado et al. 2009, severity 1.0)", fontsize=11)
    cvd_fig.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    cvd_fig.savefig("outputs/figures/cvd_palette.png", dpi=200,
                    bbox_inches="tight")
    print("wrote outputs/figures/cvd_palette.png")
    cvd_fig
    return


@app.cell
def _(colorsys, np, re):
    # =====================================================================
    # CELL 13 — STYLE HELPERS (colour, calm, pause gap, motion, styling)
    # =====================================================================
    def emotion_hsv(emotion, styles):
        fam = styles.get(emotion, styles["neutral"])
        return fam["h"], fam["s"], fam["v"]

    def hsv_to_ass(h, s, v):
        r, g, b = colorsys.hsv_to_rgb(min(1.0, max(0.0, h)),
                                      min(1.0, max(0.0, s)),
                                      min(1.0, max(0.0, v)))
        R, G, B = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
        return f"&H{B:02X}{G:02X}{R:02X}&"     # ASS is &HBBGGRR&

    def resolve_word_color(emo1, emo2, p1, p2, do_blend, blend_mode, pos_frac,
                           intensity, styles, saturation_intensity, sat_floor_frac,
                           blend_perword_swing):
        h1, s1, v1 = emotion_hsv(emo1, styles)
        if do_blend and emo2 is not None and emo1 != emo2:
            h2, s2, v2 = emotion_hsv(emo2, styles)
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
        return hsv_to_ass(h, s, v)

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
        """Extra px for the gap AFTER a word, from its pause_after.
        0 below thresh; linear thresh->full; capped at gap_max."""
        pa = float(pause_after or 0.0)
        if pa < thresh:
            return 0.0
        frac = min(1.0, (pa - thresh) / max(full - thresh, 1e-9))
        return round(gap_max * frac, 1)

    def attach_motion(words_df, motion_source, motion_min_intensity,
                      slope_deadzone, slope_full, wobble_range_hz,
                      min_voiced=4, tremor_wobble_factor=1.0):
        df = words_df.copy()

        def _row_wobble_gate(r):
            # V13 (audit 2.4): fear's segments sit in the "tremor" anim
            # group, and their wobble gate is lowered so the V7 wobble —
            # built for nervous, unsteady words — actually fires on the
            # class it was built for. Everyone else keeps the full gate.
            if str(r.get("anim", "")) == "tremor":
                return wobble_range_hz * tremor_wobble_factor
            return wobble_range_hz

        def _gesture(r):
            if motion_source != "pitch":
                return "none"
            if float(r["intensity"]) < motion_min_intensity:
                return "none"
            # V12: no gesture without a trustworthy pitch contour. V11 let a
            # 2-frame polyfit invent a "lift" on a word with no pitch at all.
            if int(r.get("n_voiced", 99)) < min_voiced:
                return "none"
            if float(r["f0_range"]) >= _row_wobble_gate(r):
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
        if _wob.any():
            _gate = df.loc[_wob].apply(_row_wobble_gate, axis=1).astype(float)
            df.loc[_wob, "motion_strength"] = (
                (df.loc[_wob, "f0_range"] / (_gate * 2.0)).clip(0.3, 1.0)
            )
        df.loc[df["gesture"] == "none", "motion_strength"] = 0.0
        return df

    def assign_styles(words_df, seg_emotion_df, styles,
                      base_font, font_swing, font_gamma, bold_thresh,
                      saturation_intensity, sat_floor_frac,
                      blend_mode, blend_margin, blend_perword_swing,
                      tracking_calm, calm_spacing_max,
                      channel_mode="redundant", emotion_floor_bonus=None):
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

        # V15: the segment's absolute arousal sets a FLOOR, then the
        # within-line competition rides on top of it. Monotone in
        # intensity_raw, so word ordering inside a line is untouched.
        _floor = df["arousal_floor"].astype(float).values if \
            "arousal_floor" in df.columns else np.zeros(len(df))
        if emotion_floor_bonus:
            _floor = _floor + df["emotion"].map(
                lambda _e: float(emotion_floor_bonus.get(_e, 0.0))).values
        _floor = np.clip(_floor, 0.0, 0.9)
        df["arousal_floor_used"] = _floor
        _raw = df["intensity_raw"].astype(float).clip(0.0, 1.0).values
        df["intensity"] = np.clip(
            (_floor + (1.0 - _floor) * _raw) * df["conf_scale"].astype(float).values,
            0.0, 1.0)

        # gamma keeps mid-range words near the base size, so only genuinely
        # salient words read as large
        _shaped = np.power(df["intensity"].astype(float).clip(0.0, 1.0), font_gamma)
        df["font_size"] = (base_font + font_swing * _shaped).round().astype(int)
        df["bold"] = (df["intensity"] >= bold_thresh).astype(int)

        if channel_mode == "hue_only":
            # V13 Position A (audit Part 1): one variable per channel,
            # strictly. Emotion lives in hue alone; every other emotion
            # carrier is flattened, so an A/B against "redundant" isolates
            # exactly what the extra channels buy (or cost).
            df["font"]   = styles["neutral"]["font"]
            df["italic"] = 0
            df["anim"]   = "flat"
        else:
            # Position B: designed redundancy for accessibility — see the
            # CHANNEL_MODE comment in cell 12 and the achromatopsia numbers
            # in cell 12d for why this is the default.
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

    _EXCLAIM_STRIP_RE = re.compile(r"[.!?]+$")

    def apply_emotion_shout(styled_df, emotions, blend_margin,
                            max_marks=3, min_marks=2,
                            n_classes=7, p_floor_mult=2.8, p_second_mult=2.1):
        """V16.2 — a SECOND, emotion-driven shout trigger, independent of
        the acoustic yell detector (CELL 17/18).

        Capitalises every word in a segment whose own emotion (or, when
        the classifier was torn, its runner-up) falls in `emotions`, and
        appends 1-N '!' to the segment's LAST word, scaled by that
        segment's mean intensity. "Close to mad" reads off the SAME
        blend evidence already used for colour (emotion2 + the
        p_top - p_second margin against blend_margin), so a segment the
        classifier called happy but was genuinely torn between happy and
        angry gets the same treatment as an outright angry call.

        Must run AFTER assign_styles (needs emotion/emotion2/p_top/
        p_second/intensity) and BEFORE any width measurement or render
        step, for the same reason apply_yell_case must: word_display has
        to be final before layout measures it, or the burned-in text and
        the measured geometry desync.
        """
        out = styled_df.copy()
        if not emotions:
            out["is_emotion_shout"] = False
            return out

        # V18: both floors are CHANCE-RELATIVE, driven by the
        # SHOUT_SENSITIVITY preset in CELL 12 rather than by constants
        # buried here.
        #
        # V17 hardcoded 0.40 and 0.30, reasoned against "7 classes, ~60.6%
        # accuracy". clf_v3 is 8 classes at ~0.36 pooled accuracy on
        # leave-one-IEMOCAP-session-out, so BOTH halves of that reasoning
        # moved: chance dropped 0.143 -> 0.125, and a less accurate model
        # produces FLATTER posteriors. A fixed 0.40 floor against a flatter
        # distribution is not the same bar it was -- it is a considerably
        # higher one, and that clause fired rarely to never on the new
        # model. Anyone reading only the constant would have concluded the
        # feature was working and quietly shipped a render with almost no
        # capitals in it.
        #
        # Expressing the floor as a multiple of chance keeps the INTENT
        # stable across a label-space change instead of silently re-tuning
        # the trigger every time the class count moves.
        _chance = 1.0 / max(int(n_classes), 2)
        _p_floor = _chance * float(p_floor_mult)
        _p2_floor = _chance * float(p_second_mult)

        def _is_shout_emotion(row):
                # 1. Direct match: the winning emotion is in the set and
                # clears the floor. At trigger_happy (1.35x chance) this is
                # close to "the model leaned this way at all", which is the
                # intended behaviour, not an accident -- see the
                # SHOUT_SENSITIVITY note in CELL 12 for what that costs.
                if row["emotion"] in emotions and float(row["p_top"]) >= _p_floor:
                    return True

                # 2. Torn match ("close to mad"): the runner-up must ALSO
                # clear its own floor, not just be close to the winner --
                # this is what filters out "everyone's uncertain" (e.g. a
                # 0.28/0.25/0.16 three-way muddle) from "two genuinely
                # confident candidates were neck-and-neck." At
                # trigger_happy the floor is barely above chance, so this
                # clause is doing most of the widening.
                e2 = row.get("emotion2")
                if (e2 in emotions
                        and (float(row["p_top"]) - float(row["p_second"])) <= blend_margin
                        and float(row["p_second"]) >= _p2_floor):
                    return True

                return False

        out["is_emotion_shout"] = out.apply(_is_shout_emotion, axis=1)

        # capitalise: OR together with the acoustic yell decision, never
        # overwrite it -- a word already capitalised for being genuinely
        # loud stays capitalised even if this trigger doesn't also fire
        _shout = out["is_emotion_shout"] | out["is_yell"].astype(bool)
        out["word_display"] = np.where(
            _shout, out["word_display"].astype(str).str.upper(),
            out["word_display"])

        # marks: one decision per qualifying SEGMENT, applied to its
        # last word only -- "OUT!!!" once, not on every word in the line
        for sid, seg in out[out["is_emotion_shout"]].groupby("segment_id"):
            _intensity = float(seg["intensity"].mean())
            _n = int(np.clip(round(min_marks + (max_marks - min_marks) * _intensity),
                             min_marks, max_marks))
            _last_idx = seg.sort_values("start").index[-1]
            _base = _EXCLAIM_STRIP_RE.sub(
                "", str(out.loc[_last_idx, "word_display"]))
            out.loc[_last_idx, "word_display"] = _base + ("!" * _n)

        # V18: report the FIRE RATE. "Too trigger happy" and "never fires"
        # are both statements about a number nobody was printing -- V17 ran
        # this trigger silently, so the only way to find out how often it
        # fired was to watch the render. Segment-level, not word-level,
        # because the decision is per segment; word counts just track
        # segment length.
        _segs = out.groupby("segment_id")["is_emotion_shout"].any()
        _n_seg = int(len(_segs))
        _n_fired = int(_segs.sum())
        _frac = _n_fired / max(_n_seg, 1)
        _also_yell = int((out["is_emotion_shout"] & out["is_yell"].astype(bool)).any()
                         if len(out) else 0)
        print(f"SHOUT: {_n_fired}/{_n_seg} segments capitalised "
              f"({_frac:.0%}) | floors p_top>={_p_floor:.3f} "
              f"p_second>={_p2_floor:.3f} (chance {_chance:.3f}, "
              f"{n_classes} classes) | set={sorted(emotions)}")
        if _n_fired == 0 and _n_seg:
            print("  nothing fired. Either the footage has no mad/excited "
                  "segments, or the floors are above what this model "
                  "produces -- check p_top's actual max in seg_emotion_df "
                  "before raising sensitivity further.")
        elif _frac >= 0.80 and _n_seg >= 5:
            print(f"  {_frac:.0%} of segments are in capitals. At that rate "
                  f"the capitals no longer contrast with anything, which "
                  f"defeats the point of the emphasis -- consider "
                  f"SHOUT_SENSITIVITY='balanced' or trimming "
                  f"EXCLAIM_EMOTIONS.")

        return out

    # =====================================================================
    # CELL 13b — TEXT MEASUREMENT  (FIX 1, part 1)
    # ---------------------------------------------------------------------
    # To position words individually we have to know how wide they are. We
    # ask fontconfig for the exact file libass will pick, then measure with
    # FreeType through Pillow. Because PlayResX is set to the video width,
    # one ASS unit is one pixel, so \fs32 and a 32px PIL font agree.
    # =====================================================================
    try:
        from PIL import ImageFont
        HAVE_PIL = True
    except Exception:
        ImageFont = None
        HAVE_PIL = False
        print("Pillow not installed -> LAYOUT_MODE='absolute' will fall back "
              "to 'flow'. pip install Pillow to enable isolated motion.")
    return (
        HAVE_PIL,
        ImageFont,
        apply_emotion_shout,
        assign_styles,
        attach_motion,
        pause_gap,
    )


@app.cell
def _(subprocess):
    def font_file_for(family, bold=0, italic=0):
        q = family
        if bold:
            q += ":bold"
        if italic:
            q += ":italic"
        try:
            r = subprocess.run(["fc-match", "-f", "%{file}", q],
                               capture_output=True, text=True, timeout=10)
            p = r.stdout.strip()
            return p or None
        except Exception:
            return None

    return (font_file_for,)


@app.cell
def _(HAVE_PIL, ImageFont, font_file_for):
    def load_font_at(family, size, bold, italic):
        """Raw Pillow load at an em-square size. Callers should go through
        libass_scale() rather than using this size directly."""
        if not HAVE_PIL:
            return None
        path = font_file_for(family, bold, italic)
        if not path:
            try:
                return ImageFont.load_default()
            except Exception:
                return None
        try:
            return ImageFont.truetype(path, int(max(1, size)))
        except Exception:
            try:
                return ImageFont.load_default()
            except Exception:
                return None

    LIBASS_REF_SIZE = 240   # measure once at a large size, then scale — PIL only accepts
                 # integer sizes, and rounding at 40px costs ~2.5% accuracy
    return LIBASS_REF_SIZE, load_font_at


@app.cell
def _(LIBASS_REF_SIZE, load_font_at):
    def ref_font_metrics(family, bold, italic):
        """(font_at_REF, ascent+descent at REF). The span is what libass
        equates to \\fs, so em_libass = size * REF / span."""
        f = load_font_at(family, LIBASS_REF_SIZE, bold, italic)
        if f is None:
            return None, 0.0
        try:
            a, d = f.getmetrics()
            return f, float(a) + abs(float(d))
        except Exception:
            return f, float(LIBASS_REF_SIZE)

    def libass_scale(family, size, bold, italic):
        """Pixels-per-reference-pixel for a given \\fs value.

        This correction is not optional. libass inherits VSFilter's
        convention that \\fsN sizes a font so that ascent+descent equals N
        pixels — NOT so that the em square equals N. FreeType (and therefore
        Pillow) sizes by the em square. For Liberation Sans the two differ
        by hhea(1854+434)/upem(2048) = 1.117, so an uncorrected measurement
        over-predicts every width by ~11%, which in absolute layout shows up
        as visibly loose word spacing and premature line wrapping.

        Measured against libass: 0.72% mean error after correction, 11.2%
        before. See tests/test_measure2.py."""
        _, span = ref_font_metrics(family, bold, italic)
        if span <= 0.0:
            return float(size) / LIBASS_REF_SIZE
        return float(size) / span

    return libass_scale, ref_font_metrics


@app.cell
def _(
    EMOTION_STYLES,
    FONT_STRICT,
    LIBASS_REF_SIZE,
    libass_scale,
    pause_gap,
    ref_font_metrics,
    subprocess,
):
    def ref_text_length(text, family, bold, italic):
        f, _ = ref_font_metrics(family, bold, italic)
        if f is None:
            return 0.58 * LIBASS_REF_SIZE * max(len(text), 1)
        try:
            return float(f.getlength(text))
        except AttributeError:
            return float(f.getsize(text)[0])
        except Exception:
            return 0.58 * LIBASS_REF_SIZE * max(len(text), 1)

    def text_width(text, family, size, bold=0, italic=0, tracking=0.0):
        """Advance width in px at \\fs<size>, including \\fsp letter spacing."""
        if not text:
            return 0.0
        k = libass_scale(family, float(size), int(bold), int(italic))
        w = ref_text_length(text, family, int(bold), int(italic)) * k
        return w + float(tracking) * len(text)

    def font_vmetrics(family, size, bold=0, italic=0):
        """(ascent, descent) in px at \\fs<size>. By the libass convention
        these sum to approximately `size`."""
        f, span = ref_font_metrics(family, int(bold), int(italic))
        if f is None or span <= 0:
            return 0.80 * size, 0.20 * size
        try:
            a, d = f.getmetrics()
        except Exception:
            return 0.80 * size, 0.20 * size
        k = float(size) / span
        return float(a) * k, float(abs(d)) * k

    def space_width(family, size, bold=0, italic=0):
        w = text_width(" ", family, size, bold, italic, 0.0)
        return w if w > 0.5 else 0.30 * size

    def verify_fonts(styles, strict=True):
        """V13 (audit Part 8 item 8): substitution is a FAILURE, not a
        fallback. In absolute layout every word is measured against the
        file fontconfig reports and positioned from that measurement; if
        libass then resolves the family to a different face, measurement
        and rendering desync and the burned-in output is only observed
        after the render. strict=True refuses to continue; strict=False
        shouts and carries on. Returns the list of (family, resolved)
        mismatches."""
        _missing = []
        for _emo, _fam in styles.items():
            _want = str(_fam["font"])
            try:
                _r = subprocess.run(["fc-match", "-f", "%{family}", _want],
                                    capture_output=True, text=True, timeout=10)
                _got = _r.stdout.strip()
            except Exception:
                _got = ""
            # fc-match may report a family list ("DejaVu Sans,DejaVu Sans
            # Condensed"); a match anywhere in it counts
            _ok = _want.lower() in _got.lower() if _got else False
            if not _ok:
                _missing.append((_want, _got or "<fc-match unavailable>"))
        if _missing:
            _msg = ("FONT SUBSTITUTION DETECTED — measurement and rendering "
                    "would desync:\n" +
                    "\n".join(f"  requested '{_w}' -> fontconfig resolves "
                               f"'{_g}'" for _w, _g in _missing) +
                    "\n  install the missing families or edit "
                    "EMOTION_STYLES before rendering.")
            if strict:
                raise RuntimeError(_msg)
            print("WARNING: " + _msg)
        return _missing

    # =====================================================================
    # CELL 13c — ABSOLUTE LINE LAYOUT  (FIX 1, part 2)
    # ---------------------------------------------------------------------
    # Lays a segment out by hand: measure every word, wrap to the frame,
    # stack rows from the bottom margin upward, return a concrete (x, y) per
    # word. \pos'd words are laid out independently by libass, which is
    # exactly the property we need — one word's \fscy can no longer resize
    # the line box its neighbours live in.
    #
    # Words are anchored \an1 (bottom-left) at (x, baseline + descent), so
    # every word in a row sits on a shared baseline regardless of its font
    # size, and a \fscy swell grows straight up out of that baseline.
    # =====================================================================
    # ---- V13 squash-and-stretch geometry (audit 4.3) --------------------
    # The gesture amplitude fractions live HERE, in one place, because two
    # cells must agree on them: the tagger (20a) uses them to write the \t
    # transforms, and this layout uses them to reserve the horizontal room
    # a squashed word will occupy. V12 had 0.6/0.5/0.4 as magic numbers in
    # the tagger only; splitting them across cells invites a silent desync.
    GESTURE_DROP_FRAC = 0.6      # drop bottoms out at 100 - 0.6*peak*k
    GESTURE_WOBBLE_HI = 0.5      # wobble tops out at   100 + 0.5*peak*k
    GESTURE_WOBBLE_LO = 0.4      # wobble bottoms at    100 - 0.4*peak*k

    def squash_x(fscy, conservation):
        """The \fscx that pairs with a given \fscy for (partial) volume
        conservation: 100 * (100/fscy)^conservation. At 1.0 the glyph's
        apparent area is exactly constant; the default 0.45 reproduces the
        audit's example pairing (\fscy110 with \fscx~96) — a squash that
        reads as deformation without visibly crushing the letterforms
        (Lee et al. 2002, squash-and-stretch after Lasseter)."""
        _fy = max(1.0, float(fscy))
        return 100.0 * (100.0 / _fy) ** float(conservation)

    def swell_headroom_x(gesture, strength, swell_peak, squash_enabled,
                         conservation):
        """Widening factor the layout must reserve for a word's gesture.
        With squash on, a DROP (and the low phase of a wobble) pairs
        fscy<100 with fscx>100, and under \an1 anchoring that growth
        extends rightward — into the next word's box unless we reserve it
        here. Lifts get narrower and need nothing. This is the horizontal
        twin of the asc_reserved vertical headroom below."""
        if not squash_enabled or gesture in ("none", "lift"):
            return 1.0
        _k = float(strength)
        if gesture == "drop":
            _lo = 100.0 - (float(swell_peak) * GESTURE_DROP_FRAC) * _k
        else:                                     # wobble low phase
            _lo = 100.0 - (float(swell_peak) * GESTURE_WOBBLE_LO) * _k
        return squash_x(max(_lo, 30.0), conservation) / 100.0

    def layout_segment(seg_rows, width, height, margin_x, margin_v,
                       line_gap_frac, space_scale,
                       pause_hold, pause_thresh, pause_full, pause_gap_max,
                       swell_peak=30.0, squash_enabled=False,
                       squash_conservation=0.45):
        items = []
        for _, rw in seg_rows.iterrows():
            fam  = str(rw["font"])
            size = int(rw["font_size"])
            bold = int(rw["bold"])
            ital = int(rw["italic"])
            trk  = float(rw.get("tracking", 0.0) or 0.0)
            txt  = str(rw.get("word_display") or rw["word"]).strip()
            if not txt:
                continue
            w = text_width(txt, fam, size, bold, ital, trk)
            asc, desc = font_vmetrics(fam, size, bold, ital)

            # Reserve the room a swell will need. Leading is our
            # responsibility now, and \fscy grows a word upward out of its
            # baseline — without this a lifting word on the lower row can
            # climb into the row above it.
            _gesture_m2 = str(rw.get("gesture", "none"))
            if _gesture_m2 != "none":
                k = float(rw.get("motion_strength", 0.0))
                asc_reserved = asc * (1.0 + (swell_peak / 100.0) * k)
            else:
                k = 0.0
                asc_reserved = asc

            # V13: horizontal headroom for the squash (see swell_headroom_x)
            w_adv = w * swell_headroom_x(_gesture_m2, k, swell_peak,
                                         squash_enabled, squash_conservation)

            gap = space_width(fam, size, bold, ital) * space_scale
            if pause_hold:
                gap += pause_gap(rw.get("pause_after", 0.0), pause_thresh,
                                 pause_full, pause_gap_max)
            items.append({"row": rw, "text": txt, "w": w, "w_adv": w_adv,
                          "asc": asc, "asc_res": asc_reserved, "desc": desc,
                          "gap": gap})

        if not items:
            return []

        # ---- wrap (on the RESERVED width, so a squash cannot overflow) --
        max_w = max(80.0, width - 2.0 * margin_x)
        rows, cur, cur_w = [], [], 0.0
        for it in items:
            add = it["w_adv"] if not cur else cur[-1]["gap"] + it["w_adv"]
            if cur and (cur_w + add) > max_w:
                rows.append(cur)
                cur, cur_w = [it], it["w_adv"]
            else:
                cur.append(it)
                cur_w += add
        if cur:
            rows.append(cur)

        # ---- vertical: stack upward from the bottom margin ----
        placed = []
        y_bottom = float(height - margin_v)
        for r in reversed(rows):
            r_asc = max(i["asc"] for i in r)          # ink height at rest
            r_asc_res = max(i["asc_res"] for i in r)  # incl. swell headroom
            r_desc = max(i["desc"] for i in r)
            baseline = y_bottom - r_desc

            row_w = sum(i["w_adv"] for i in r) + sum(i["gap"] for i in r[:-1])
            x = (width - row_w) / 2.0

            for i in r:
                placed.append({
                    "row": i["row"], "text": i["text"],
                    "x": round(x, 1),
                    "y": round(baseline + i["desc"], 1),   # \an1 bottom edge
                    "cx": round(x + i["w"] / 2.0, 1),
                    "baseline": round(baseline, 1),
                    "w": i["w"],
                })
                x += i["w_adv"] + i["gap"]

            y_bottom = baseline - r_asc_res - (r_asc + r_desc) * line_gap_frac

        # restore reading order
        placed.sort(key=lambda p: (-p["baseline"], p["x"]))
        placed.sort(key=lambda p: float(p["row"]["start"]))
        return placed

    # =====================================================================
    # CELL 14 — FONT AVAILABILITY CHECK
    # =====================================================================
    # V13 (audit Part 8 item 8): substitution is a hard failure, not a
    # note. verify_fonts() asks fc-match — the SAME resolver libass uses —
    # what each family actually resolves to, and with FONT_STRICT raises
    # before any measurement can silently happen in the wrong face. The
    # renderers call it again at render time, so a font removed between
    # notebook start and render is still caught.
    try:
        for _emo_m3, _fam_m3 in EMOTION_STYLES.items():
            print(f"{_emo_m3:10s} -> {_fam_m3['font']}")
        _subs = verify_fonts(EMOTION_STYLES, strict=FONT_STRICT)
        if not _subs:
            print("all families resolve to themselves — measurement and "
                  "rendering agree; safe to set FONT_STRICT = True")
        else:
            print(f"{len(_subs)} family(ies) substituted. Renders will "
                  "still work, but measured geometry and burned frames "
                  "may disagree slightly. Either install the families "
                  "below, or edit EMOTION_STYLES to use ones you have "
                  "(fc-list : family | sort -u lists them):")
            for _want_m3, _got_m3 in _subs:
                print(f"    {_want_m3}  ->  {_got_m3}")
    except FileNotFoundError:
        print("fc-match not found; cannot verify fonts on this machine.")
    return (
        GESTURE_DROP_FRAC,
        GESTURE_WOBBLE_HI,
        GESTURE_WOBBLE_LO,
        layout_segment,
        squash_x,
        verify_fonts,
    )


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
    confidence_scale,
    np,
    pd,
    true_emotion,
):
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
        # V13: calibrated curve from cell 12b (0.43 -> ~0.60, not ~0.90)
        conf_scale = confidence_scale(p_top, len(clf_full.classes_))
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
    return conf_scale, p_second, p_top, pred_emotion, pred_emotion2


@app.cell
def _(mo):
    mo.md("""
    ### Step C2: Salience → competitive 100-point budget
    """)
    return


@app.cell
def _(classify_words, np, pd):
    # =====================================================================
    # CELL 17 — BUDGET MACHINERY  (FIX 2, part 4)
    # =====================================================================
    def assign_words_to_segments(words_df, segments):
        bounds = [(float(s["start"]), float(s["end"])) for s in segments]
        if not bounds:
            return [0] * len(words_df)

        def locate(mid):
            for si, (s0, s1) in enumerate(bounds):
                if s0 <= mid <= s1:
                    return si
            return min(range(len(bounds)),
                       key=lambda i: min(abs(mid - bounds[i][0]), abs(mid - bounds[i][1])))

        mids = (words_df["start"].astype(float) + words_df["end"].astype(float)) / 2.0
        return [locate(m) for m in mids]

    def center_scale(vals, robust, rel_floor=0.02, abs_floor=1e-3):
        """Location and spread, or (centre, 0.0) when the spread is too small
        to mean anything.

        Robust = median / MAD, so one shouted word no longer inflates the
        std and flattens everyone else out of contention.

        The floor matters more than it looks. MAD collapses to exactly 0
        whenever a majority of values are identical — common with rounded
        features — and the std fallback then manufactures a z of 3+ out of a
        0.1 Hz difference. Without this guard a perfectly monotone line will
        promote whichever word happens to differ in the last decimal place.
        Returning scale 0 leaves z at 0, i.e. the feature is treated as
        carrying no information for this segment, which is the truth."""
        c = float(np.median(vals)) if robust else float(np.mean(vals))
        if robust:
            mad = float(np.median(np.abs(vals - c)))
            s = 1.4826 * mad
            if s < 1e-9:                      # degenerate: fall back to std
                s = float(np.std(vals))
        else:
            s = float(np.std(vals))
        if s <= max(abs_floor, rel_floor * abs(c)):
            return c, 0.0
        return c, s

    def compute_salience(words_df, weights, zero_missing=frozenset(),
                         positive_only=frozenset(), robust=True,
                         min_words=4, shrink_k=5.0, scale_rel_floor=0.02,
                         eps=1e-9):
        """Per-segment z-scores, summed with weights.

        The V11 bug was `salience += w * abs(z)`. Emphasis is directional:
        a word is emphasised by being LOUDER, HIGHER and LONGER than its
        neighbours, not merely different from them. Features listed in
        `positive_only` are clipped at zero so that being quiet, low and
        rushed — the signature of an unstressed function word — earns
        nothing instead of earning full marks."""
        df = words_df.copy()
        n = len(df)
        salience = np.zeros(n)
        zmax_pos = np.zeros(n)
        seg_ids = df["segment_id"].values

        for feat, w in weights.items():
            if feat not in df.columns:
                continue
            vals = df[feat].astype(float).values
            z = np.zeros(n)
            missing = (vals == 0.0) if feat in zero_missing else np.zeros(n, dtype=bool)

            for sid in np.unique(seg_ids):
                m = (seg_ids == sid) & ~missing
                if m.sum() >= 2:
                    c, s = center_scale(vals[m], robust,
                                         rel_floor=scale_rel_floor)
                    if s > eps:
                        z[m] = (vals[m] - c) / s

            if feat in positive_only:
                zc = np.maximum(z, 0.0)
                zmax_pos = np.maximum(zmax_pos, zc)
            else:
                zc = np.abs(z)

            salience += w * zc

        # shrink tiny segments: a z-score over 3 words is noise, and a
        # uniform multiplier on the logits flattens the later softmax
        counts = df.groupby("segment_id")["word"].transform("size").to_numpy(dtype=float)
        salience = salience * (counts / (counts + shrink_k))
        salience[counts < min_words] *= 0.35

        df["salience_raw"] = salience
        df["zmax_pos"] = zmax_pos
        return df

    def apply_word_class_prior(df, override_z=0.90, override_rel=0.75,
                               override_full=1.00, enabled=True):
        """Damp connectives, articles, prepositions and auxiliaries — unless
        the prosody says the speaker really did lean on them.

        A flat, throwaway "or" keeps its 0.30 prior and stays small. An "or"
        that is among the most prominent words in its own line climbs back
        toward 1.0 and is allowed to grow, because contrastive stress on a
        conjunction is real ("with milk OR without").

        Prominence is measured RELATIVE to the line's own peak. See the note
        beside CLASS_OVERRIDE_Z: an absolute sigma gate is unreachable,
        because the stressed word inflates the scale it is scored against."""
        out = df.copy()
        classes, priors = classify_words(out["word"])
        out["word_class"] = classes
        out["class_prior"] = priors

        if not enabled:
            out["class_weight"] = 1.0
            out["salience"] = out["salience_raw"]
            out["prominence"] = 0.0
            return out

        z = out["zmax_pos"].to_numpy(dtype=float)
        peak = out.groupby("segment_id")["zmax_pos"].transform("max").to_numpy(dtype=float)
        rel = np.divide(z, peak, out=np.zeros_like(z), where=peak > 1e-9)

        # absolute floor, ramped over half a sigma so it is not a cliff
        gate = np.clip((z - override_z) / 0.5, 0.0, 1.0)
        t = np.clip((rel - override_rel) / max(override_full - override_rel, 1e-9),
                    0.0, 1.0) * gate

        # priors above 1.0 (negation) are a boost and must not be eroded
        eff = np.where(priors >= 1.0, priors, priors + (1.0 - priors) * t)

        out["prominence"] = np.round(rel, 3)
        out["class_weight"] = np.round(eff, 3)
        out["salience"] = out["salience_raw"] * eff
        return out

    def _seg_means(words_df, feat, seg_ids, uniq):
        """Per-segment mean of a feature, ignoring zero (= not measured)."""
        if feat not in words_df.columns:
            return None
        v = words_df[feat].astype(float).values
        m = np.array([np.mean(v[(seg_ids == sid) & (v != 0.0)])
                      if ((seg_ids == sid) & (v != 0.0)).any() else np.nan
                      for sid in uniq])
        return None if np.all(np.isnan(m)) else m

    def _norm_ref(m, mode, robust, eps=1e-9):
        """Segment means -> a comparable score, by centre or by range."""
        ok = ~np.isnan(m)
        if mode == "peak_ref":
            lo, hi = np.nanpercentile(m[ok], 5), np.nanpercentile(m[ok], 95)
            if hi - lo < eps:
                return None
            return np.where(ok, np.clip((m - lo) / (hi - lo), 0.0, 1.0), 0.0)
        _c, _s = center_scale(m[ok], robust, rel_floor=0.02)
        if _s <= eps:
            return None
        return np.where(ok, (m - _c) / _s, 0.0)

    def detect_yelling(words_df, features, z_thresh, max_frac=0.25,
                       punct_assist=0.0, tilt_min_z=None, tilt_feature="alpha_ratio",
                       min_db_over_median=0.0, level_feature="intensity_db",
                       norm="robust_z", peak_frac=0.8,
                       robust=True, eps=1e-9):
        """Flag segments spoken with raised vocal effort.

        Scores each segment on loudness, spectral tilt and pitch, z-scored
        ACROSS segments so the comparison is "loud for this speaker in this
        clip" rather than against an absolute dB value that depends on the
        recording chain. Returns (mask, score) aligned with words_df rows.

        max_frac is a calibration guard, not a style choice. Capitals only
        carry meaning while they stay rare, so if the detector wants to
        shout more than max_frac of the clip, it keeps only the strongest
        segments and says so.
        """
        n = len(words_df)
        mask = np.zeros(n, dtype=bool)
        score = np.zeros(n)
        if n == 0:
            return mask, score
        seg_ids = words_df["segment_id"].values
        uniq = np.unique(seg_ids)
        if len(uniq) < 3:
            return mask, score

        sc = np.zeros(len(uniq))
        used = 0.0
        tilt_z = None
        for feat, w in features.items():
            means = _seg_means(words_df, feat, seg_ids, uniq)
            if means is None:
                continue
            _v = _norm_ref(means, norm, robust, eps)
            if _v is None:
                continue
            if feat == tilt_feature:
                # the tilt GATE always uses centre-referencing, whatever the
                # score uses. Its job is "was effort raised, yes or no",
                # which is an absolute question, and range-referencing would
                # answer it relative to whatever the flattest voice in the
                # clip happened to be.
                _t = _norm_ref(means, "robust_z", robust, eps)
                tilt_z = _t if _t is not None else None
            sc += w * _v
            used += abs(w)
        if used <= eps:
            return mask, score
        sc = sc / used

        # punctuation may only lower the bar, never clear it by itself
        _base = float(peak_frac) if norm == "peak_ref" else float(z_thresh)
        thresh = np.full(len(uniq), _base)
        if punct_assist:
            for _i, sid in enumerate(uniq):
                _txt = " ".join(words_df.loc[seg_ids == sid, "word"].astype(str))
                if _txt.rstrip().endswith("!"):
                    # peak_ref lives on a 0-1 scale, robust_z on a z scale,
                    # so the same nudge means very different things. Scale it.
                    thresh[_i] -= (float(punct_assist) * 0.1
                                   if norm == "peak_ref" else float(punct_assist))

        hot = sc > thresh
        # CONJUNCTION, not a sum. A 20 dB jump produces a large weighted
        # score all by itself, so without this a close mic reads as a
        # scream. Requiring the tilt term to clear its own bar is what
        # makes this a measure of vocal effort rather than of level.
        if tilt_min_z is not None:
            if tilt_z is None:
                print(f"YELL: '{tilt_feature}' not available — falling back to "
                      "level-only detection, which cannot tell shouting from "
                      "a loud recording. Treat these flags as weak.")
            else:
                hot = hot & (tilt_z > float(tilt_min_z))

        # Absolute check, in the units the microphone actually recorded.
        # Everything above this point is relative, and relative evidence on
        # a flat clip is how you end up capitalising an ordinary sentence.
        if min_db_over_median and level_feature in words_df.columns:
            _lv = words_df[level_feature].astype(float).values
            _seg_db = np.array([
                np.mean(_lv[(seg_ids == sid) & (_lv != 0.0)])
                if ((seg_ids == sid) & (_lv != 0.0)).any() else 0.0
                for sid in uniq])
            _med = float(np.median(_seg_db[_seg_db > 0.0])) if (_seg_db > 0).any() else 0.0
            hot = hot & (_seg_db >= _med + float(min_db_over_median))
        if hot.sum() > max(1, int(np.ceil(len(uniq) * max_frac))):
            _keep = max(1, int(np.ceil(len(uniq) * max_frac)))
            _order = np.argsort(-sc)
            hot = np.zeros(len(uniq), dtype=bool)
            hot[_order[:_keep]] = True
            print(f"YELL: detector fired on {int((sc > thresh).sum())}/"
                  f"{len(uniq)} segments, above the {max_frac:.0%} guard — "
                  f"kept the {_keep} strongest. Raise YELL_Z if this recurs.")

        _hot = dict(zip(uniq, hot))
        _sco = dict(zip(uniq, sc))
        mask = np.array([_hot.get(sid, False) for sid in seg_ids])
        score = np.array([_sco.get(sid, 0.0) for sid in seg_ids])
        if mask.any():
            print(f"YELL: {int(hot.sum())}/{len(uniq)} segments recased "
                  f"({mask.mean():.0%} of words)")
        return mask, score

    def yell_report(words_df, features, z_thresh, tilt_min_z=None,
                    min_db_over_median=0.0, norm="robust_z", peak_frac=0.8,
                    top=8, robust=True, eps=1e-9):
        """Say why nothing was recased, per segment, gate by gate.

        "No capitals appeared" has at least four causes that look identical
        from the outside: the tilt feature never computed, the combined
        score fell short, the tilt gate blocked, or the absolute dB margin
        blocked. Guessing between them by nudging thresholds is how a
        conservative detector gets loosened into a broken one, so this
        prints the evidence instead.
        """
        if "alpha_ratio" not in words_df.columns:
            print("YELL DIAG: no alpha_ratio column at all — these words were "
                  "extracted before V15. Re-run the feature extraction.")
        elif float(np.abs(words_df["alpha_ratio"].astype(float)).max()) == 0.0:
            print("YELL DIAG: alpha_ratio is zero for every word — the Praat "
                  "spectrum step failed on this audio. Detection is running "
                  "on level alone and the tilt gate can never pass.")

        seg_ids = words_df["segment_id"].values
        uniq = np.unique(seg_ids)
        if len(uniq) < 3:
            print(f"YELL DIAG: only {len(uniq)} segment(s); at least 3 are "
                  "needed to say what is loud FOR THIS CLIP.")
            return None

        def _segz(feat, mode=None):
            m = _seg_means(words_df, feat, seg_ids, uniq)
            if m is None:
                return None
            v = _norm_ref(m, mode or norm, robust, eps)
            return None if v is None else (v, m)

        sc, used = np.zeros(len(uniq)), 0.0
        tilt_z = None
        for feat, w in features.items():
            r = _segz(feat)
            if r is None:
                continue
            _z, _ = r
            if feat == "alpha_ratio":
                _t = _segz(feat, "robust_z")
                tilt_z = _t[0] if _t else None
            sc += w * _z
            used += abs(w)
        sc = sc / used if used > eps else sc

        lv = words_df["intensity_db"].astype(float).values
        seg_db = np.array([np.mean(lv[(seg_ids == sid) & (lv != 0.0)])
                           if ((seg_ids == sid) & (lv != 0.0)).any() else 0.0
                           for sid in uniq])
        med = float(np.median(seg_db[seg_db > 0])) if (seg_db > 0).any() else 0.0

        rep = pd.DataFrame({
            "segment_id": uniq,
            "score_z": np.round(sc, 2),
            "tilt_z": np.round(tilt_z, 2) if tilt_z is not None else np.nan,
            "dB_over_med": np.round(seg_db - med, 1),
        }).sort_values("score_z", ascending=False).head(top)

        _base = float(peak_frac) if norm == "peak_ref" else float(z_thresh)
        rep["score_ok"] = rep["score_z"] > _base
        rep["tilt_ok"] = True if tilt_min_z is None else (rep["tilt_z"] > tilt_min_z)
        rep["dB_ok"] = rep["dB_over_med"] >= min_db_over_median
        rep["would_fire"] = rep["score_ok"] & rep["tilt_ok"] & rep["dB_ok"]

        print(f"YELL DIAG: norm={norm}, score>{_base}, tilt>{tilt_min_z}, "
              f"dB>=+{min_db_over_median}. Loudest segments:")
        print(rep.to_string(index=False))
        if not rep["would_fire"].any():
            _blocked = []
            if not rep["score_ok"].any():
                _blocked.append("combined score (lower YELL_PEAK_FRAC)"
                                if norm == "peak_ref"
                                else "combined score (lower YELL_Z, or set "
                                     "YELL_NORM='peak_ref' if the whole clip "
                                     "is loud)")
            if not rep["tilt_ok"].any():
                _blocked.append("spectral tilt (lower YELL_TILT_MIN_Z, or the "
                                "speaker is loud without raised effort)")
            if not rep["dB_ok"].any():
                _blocked.append("absolute dB margin (lower "
                                "YELL_MIN_DB_OVER_MEDIAN)")
            print("  nothing fires. Blocked by: " + "; ".join(_blocked))
            print("  NOTE: if the whole clip is shouted, no segment can stand "
                  "out from it — every gate here is relative to this clip. "
                  "That is a real limit, not a bug: capitals only mean "
                  "anything while they stay rare.")
        return rep

    def apply_yell_case(words_df, mode="upper"):
        """Set word_display BEFORE measurement, never at render time.

        The absolute layout measures each word and positions it. Uppercasing
        after measurement would size the box from the lowercase form and
        draw the uppercase one, which is the same desync class as a silent
        font substitution. One column, written once, read by both.
        """
        out = words_df.copy()
        out["word_display"] = out["word"].astype(str)
        if mode == "upper" and "is_yell" in out.columns:
            _m = out["is_yell"].astype(bool)
            out.loc[_m, "word_display"] = out.loc[_m, "word_display"].str.upper()
        return out

    def segment_arousal_floor(words_df, features, floor_max, spread=1.5,
                              robust=True, eps=1e-9):
        """Per-word styling floor from the segment's ABSOLUTE arousal.

        compute_salience z-scores WITHIN a segment, which deliberately
        throws away how loud that segment was compared with the rest of the
        clip. That discarded quantity is precisely what makes a shouted
        line feel shouted, so it is recovered here: each segment's mean
        loudness and pitch are z-scored ACROSS segments, combined, and
        squashed through a tanh into [0, floor_max].

        Returns a float array aligned with words_df rows. All-zero when
        there are too few segments to compare, which is the honest answer
        rather than a guess.
        """
        n = len(words_df)
        out = np.zeros(n)
        if floor_max <= 0.0 or n == 0:
            return out
        seg_ids = words_df["segment_id"].values
        uniq = np.unique(seg_ids)
        if len(uniq) < 3:
            # Two segments cannot establish what "loud for this clip" means.
            return out

        score = np.zeros(len(uniq))
        used = 0.0
        for feat, w in features.items():
            if feat not in words_df.columns:
                continue
            vals = words_df[feat].astype(float).values
            # zero means "unvoiced / not measured" for these features, so
            # averaging it in would read a pause as calm
            means = np.array([
                np.mean(vals[(seg_ids == sid) & (vals != 0.0)])
                if ((seg_ids == sid) & (vals != 0.0)).any() else np.nan
                for sid in uniq])
            if np.all(np.isnan(means)):
                continue
            _c, _s = center_scale(means[~np.isnan(means)], robust,
                                  rel_floor=0.02)
            if _s <= eps:
                continue
            _z = np.where(np.isnan(means), 0.0, (means - _c) / _s)
            score += w * _z
            used += abs(w)
        if used <= eps:
            return out

        score = score / used
        gain = 0.5 * (1.0 + np.tanh(score / max(spread, eps)))   # -> [0, 1]
        per_seg = dict(zip(uniq, floor_max * gain))
        return np.array([per_seg.get(sid, 0.0) for sid in seg_ids])

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

    def apply_emphasis_quota(df, quota_frac=0.34, damp=0.45):
        """Cap how many words per line may run hot. Emphasising a third of a
        sentence is the same as emphasising none of it."""
        out = df.copy()
        rank = out.groupby("segment_id")["intensity_raw"].rank(
            ascending=False, method="first")
        size = out.groupby("segment_id")["word"].transform("size")
        allowed = np.maximum(1, np.ceil(size.to_numpy(dtype=float) * quota_frac))
        over = rank.to_numpy(dtype=float) > allowed
        out.loc[over, "intensity_raw"] = out.loc[over, "intensity_raw"] * damp
        out["over_quota"] = over
        return out

    return (
        allocate_points,
        apply_emphasis_quota,
        apply_word_class_prior,
        apply_yell_case,
        assign_words_to_segments,
        compute_salience,
        detect_yelling,
        segment_arousal_floor,
        yell_report,
    )


@app.cell
def _(
    AROUSAL_FEATURES,
    AROUSAL_SPREAD,
    CLASS_OVERRIDE_FULL,
    CLASS_OVERRIDE_REL,
    CLASS_OVERRIDE_Z,
    EMPHASIS_QUOTA_FRAC,
    FULL_DRAMA_RATIO,
    MIN_POINTS,
    MIN_WORDS_FOR_SALIENCE,
    POSITIVE_ONLY_FEATURES,
    QUOTA_DAMP,
    ROBUST_STATS,
    SALIENCE_SHRINK_K,
    SALIENCE_WEIGHTS,
    SCALE_REL_FLOOR,
    SEGMENT_AROUSAL_FLOOR,
    SOFTMAX_TEMPERATURE,
    USE_WORD_CLASS_PRIOR,
    YELL_CASE,
    YELL_DETECT,
    YELL_FEATURES,
    YELL_MAX_FRAC,
    YELL_MIN_DB_OVER_MEDIAN,
    YELL_NORM,
    YELL_PEAK_FRAC,
    YELL_PUNCT_ASSIST,
    YELL_TILT_MIN_Z,
    YELL_Z,
    ZERO_MEANS_MISSING,
    aligned,
    allocate_points,
    apply_emphasis_quota,
    apply_word_class_prior,
    apply_yell_case,
    assign_words_to_segments,
    compute_salience,
    detect_yelling,
    np,
    result,
    segment_arousal_floor,
    word_df,
    yell_report,
):
    # =====================================================================
    # CELL 18 — RUN THE BUDGET
    # =====================================================================
    seg_list = aligned.get("segments") or result["segments"]

    tagged_word_df = word_df.copy()
    tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

    salient_word_df = compute_salience(
        tagged_word_df, SALIENCE_WEIGHTS,
        zero_missing=ZERO_MEANS_MISSING,
        positive_only=POSITIVE_ONLY_FEATURES,
        robust=ROBUST_STATS,
        min_words=MIN_WORDS_FOR_SALIENCE,
        shrink_k=SALIENCE_SHRINK_K,
        scale_rel_floor=SCALE_REL_FLOOR)

    salient_word_df = apply_word_class_prior(
        salient_word_df, override_z=CLASS_OVERRIDE_Z,
        override_rel=CLASS_OVERRIDE_REL, override_full=CLASS_OVERRIDE_FULL,
        enabled=USE_WORD_CLASS_PRIOR)

    budget_df = allocate_points(
        salient_word_df, temperature=SOFTMAX_TEMPERATURE, min_points=MIN_POINTS)
    budget_df["intensity_raw"] = np.clip(
        (budget_df["share_ratio"] - 1.0) / (FULL_DRAMA_RATIO - 1.0), 0.0, 1.0)
    budget_df = apply_emphasis_quota(budget_df, EMPHASIS_QUOTA_FRAC, QUOTA_DAMP)
    # V15: recover the between-segment dynamics the per-segment z-scoring
    # discards, so a shouted line no longer renders like a flat one.
    budget_df["arousal_floor"] = segment_arousal_floor(
        budget_df, AROUSAL_FEATURES, SEGMENT_AROUSAL_FLOOR,
        spread=AROUSAL_SPREAD, robust=ROBUST_STATS)
    if YELL_DETECT:
        budget_df["is_yell"], budget_df["yell_z"] = detect_yelling(
            budget_df, YELL_FEATURES, YELL_Z, max_frac=YELL_MAX_FRAC,
            punct_assist=YELL_PUNCT_ASSIST, tilt_min_z=YELL_TILT_MIN_Z,
            min_db_over_median=YELL_MIN_DB_OVER_MEDIAN,
            norm=YELL_NORM, peak_frac=YELL_PEAK_FRAC, robust=ROBUST_STATS)
    else:
        budget_df["is_yell"], budget_df["yell_z"] = False, 0.0
    if YELL_DETECT and not np.asarray(budget_df["is_yell"]).any():
        yell_report(budget_df, YELL_FEATURES, YELL_Z,
                    tilt_min_z=YELL_TILT_MIN_Z,
                    min_db_over_median=YELL_MIN_DB_OVER_MEDIAN,
                    norm=YELL_NORM, peak_frac=YELL_PEAK_FRAC)
    budget_df = apply_yell_case(budget_df, YELL_CASE)

    for _sid in np.unique(budget_df["segment_id"].values):
        _m = budget_df["segment_id"] == _sid
        _top = budget_df.loc[_m].sort_values("intensity_raw").iloc[-1]
        print(f"Segment {_sid}: {int(_m.sum())} words | "
              f"loudest = '{str(_top['word']).strip()}' "
              f"({_top['word_class']}, intensity {_top['intensity_raw']:.2f})")

    budget_df[["word", "word_class", "class_weight", "segment_id", "dur_resid",
               "intensity_db", "salience_raw", "salience", "intensity_raw"]].round(2)

    # =====================================================================
    # CELL 18b — EMPHASIS AUDIT
    # ---------------------------------------------------------------------
    # Read this when a word looks wrong on screen. It shows what the word
    # was scored on and whether the class prior held it back or an override
    # let it through, so a false positive can be traced to a dial.
    # =====================================================================
    _aud = budget_df.copy()
    _aud["word"] = _aud["word"].astype(str).str.strip()
    _aud = _aud.sort_values("intensity_raw", ascending=False)
    print("TOP 15 BY FINAL INTENSITY")
    print(_aud[["word", "word_class", "zmax_pos", "prominence", "class_weight",
                "salience_raw", "salience", "intensity_raw"]]
          .head(15).round(2).to_string(index=False))

    # V13 (audit 5.3): the EMPIRICAL version of the z-bound argument. The
    # per-line peak of zmax_pos is what the class override is measured
    # against; its observed distribution replaces the analytic bound that
    # was (a) misstated and (b) derived for SD while the code uses MAD.
    _peaks = _aud.groupby("segment_id")["zmax_pos"].max()
    if len(_peaks):
        print(f"\nPER-LINE PEAK z (n={len(_peaks)} lines): "
              f"median {_peaks.median():.2f}, "
              f"p10-p90 {_peaks.quantile(0.1):.2f}-{_peaks.quantile(0.9):.2f}, "
              f"max {_peaks.max():.2f} — an absolute 1.75-sigma gate would "
              f"fire on {(100.0 * (_peaks > 1.75).mean()):.0f}% of lines")

    _fn = _aud[_aud["word_class"] != "content"]
    if len(_fn):
        print("\nFUNCTION WORDS THAT STILL GOT THROUGH (intensity > 0.5)")
        _hot = _fn[_fn["intensity_raw"] > 0.5]
        if len(_hot):
            print(_hot[["word", "word_class", "zmax_pos", "prominence",
                        "class_weight", "intensity_raw"]].round(2).to_string(index=False))
            print("These cleared the override: they were among the most prominent "
                  "words in their own line. Raise CLASS_OVERRIDE_REL to be stricter.")
        else:
            print("  none — every function word was held down by its prior.")
    return budget_df, seg_list


@app.cell
def _(mo):
    mo.md("""
    ## Part D: Style mapping
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
    CHANNEL_MODE,
    CLF_N_CLASSES,
    EMOTION_FLOOR_BONUS,
    EMOTION_STYLES,
    EXCLAIM_EMOTIONS,
    EXCLAIM_ENABLE,
    EXCLAIM_MAX_MARKS,
    EXCLAIM_MIN_MARKS,
    FONT_GAMMA,
    FONT_SWING,
    MOTION_MIN_INTENSITY,
    MOTION_SOURCE,
    PAUSE_HOLD,
    PAUSE_HOLD_THRESH,
    SATURATION_INTENSITY,
    SAT_FLOOR_FRAC,
    SHOUT_P_FLOOR_MULT,
    SHOUT_P_SECOND_MULT,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    TRACKING_CALM,
    TREMOR_WOBBLE_FACTOR,
    WOBBLE_RANGE_HZ,
    apply_emotion_shout,
    assign_styles,
    attach_motion,
    budget_df,
    conf_scale,
    p_second,
    p_top,
    pd,
    pred_emotion,
    pred_emotion2,
):
    # =====================================================================
    # CELL 19 — LEGACY SINGLE-CLIP STYLING
    # =====================================================================
    _seg_ids = sorted(budget_df["segment_id"].unique().tolist())
    seg_emotion_single = pd.DataFrame([{
        "segment_id": sid, "pred_emotion": pred_emotion, "pred_emotion2": pred_emotion2,
        "p_top": p_top, "p_second": p_second, "conf_scale": conf_scale}
        for sid in _seg_ids])

    styled_word_df = assign_styles(
        budget_df, seg_emotion_single, EMOTION_STYLES,
        base_font=BASE_FONT_SIZE, font_swing=FONT_SWING, font_gamma=FONT_GAMMA,
        bold_thresh=BOLD_THRESHOLD,
        saturation_intensity=SATURATION_INTENSITY, sat_floor_frac=SAT_FLOOR_FRAC,
        blend_mode=BLEND_MODE, blend_margin=BLEND_MARGIN,
        blend_perword_swing=BLEND_PERWORD_SWING,
        tracking_calm=TRACKING_CALM, calm_spacing_max=CALM_SPACING_MAX,
        channel_mode=CHANNEL_MODE,
            emotion_floor_bonus=EMOTION_FLOOR_BONUS)
    # V16.2: emotion-driven caps + exclamation marks, layered on top of
    # the acoustic yell decision assign_styles just carried through.
    # Must run before attach_motion/measurement -- word_display has to
    # be final before anything measures its width.
    if EXCLAIM_ENABLE:
        styled_word_df = apply_emotion_shout(
            styled_word_df, EXCLAIM_EMOTIONS, BLEND_MARGIN,
            max_marks=EXCLAIM_MAX_MARKS, min_marks=EXCLAIM_MIN_MARKS,
            n_classes=CLF_N_CLASSES,          # V18: chance-relative floors
            p_floor_mult=SHOUT_P_FLOOR_MULT,
            p_second_mult=SHOUT_P_SECOND_MULT)
    styled_word_df = attach_motion(
        styled_word_df, motion_source=MOTION_SOURCE,
        motion_min_intensity=MOTION_MIN_INTENSITY, slope_deadzone=SLOPE_DEADZONE,
        slope_full=SLOPE_FULL, wobble_range_hz=WOBBLE_RANGE_HZ,
        tremor_wobble_factor=TREMOR_WOBBLE_FACTOR)

    _moving = int((styled_word_df["gesture"] != "none").sum())
    _blended = int(styled_word_df["blended"].sum())
    _held = int((styled_word_df["pause_after"] >= PAUSE_HOLD_THRESH).sum()) if PAUSE_HOLD else 0
    print(f"emotion={pred_emotion} | blended={_blended}/{len(styled_word_df)} "
          f"| moving={_moving} | held pauses={_held} "
          f"| font {styled_word_df['font_size'].min()}-{styled_word_df['font_size'].max()}px")
    styled_word_df[["word", "word_class", "intensity", "font_size", "tracking",
                    "pause_after", "gesture"]].round(2)

    # =====================================================================
    # CELL 19b — STYLING COVERAGE  (V13, audit 2.7 / Part 8 item 4)
    # ---------------------------------------------------------------------
    # "Emotion-responsive" needs a number attached to it. If ~8% of words
    # style non-neutrally, the phrase is doing a lot of work for a small
    # effect; if ~60% do, that is the over-styling AffType's all-features
    # condition predicts will hurt. Either way the number belongs in the
    # Results, and it is nearly free to produce — so it prints on every
    # run, here and inside process_any_video.
    # =====================================================================
    def styling_coverage(styled_df, label="clip"):
        _n = len(styled_df)
        if _n == 0:
            print(f"styling coverage [{label}]: no words")
            return None
        _stats = {
            "words": _n,
            "non_neutral_frac": float((styled_df["emotion"] != "neutral").mean()),
            "intensity>0.3": float((styled_df["intensity"] > 0.3).mean()),
            "intensity>0.6": float((styled_df["intensity"] > 0.6).mean()),
            "bold_frac": float((styled_df["bold"] > 0).mean()),
            "gesture_frac": float((styled_df.get("gesture", pd.Series(dtype=str))
                                   .ne("none").mean())
                                  if "gesture" in styled_df else 0.0),
            "blended_frac": float(styled_df["blended"].mean())
            if "blended" in styled_df else 0.0,
        }
        print(f"styling coverage [{label}]: "
              f"{100 * _stats['non_neutral_frac']:.0f}% of {_n} words carry a "
              f"non-neutral emotion style | intensity>0.3: "
              f"{100 * _stats['intensity>0.3']:.0f}% | >0.6: "
              f"{100 * _stats['intensity>0.6']:.0f}% | bold: "
              f"{100 * _stats['bold_frac']:.0f}% | moving: "
              f"{100 * _stats['gesture_frac']:.0f}% | blended: "
              f"{100 * _stats['blended_frac']:.0f}%")
        return _stats

    styling_coverage(styled_word_df,
                     label="current clip" + ("" if PAUSE_HOLD else ""))
    _ = PAUSE_HOLD_THRESH  # coverage reads no thresholds; keep cell honest
    return styled_word_df, styling_coverage


@app.cell
def _(mo):
    mo.md(r"""
    ## Part E: Render .ass + FFmpeg burn-in

    `LAYOUT_MODE="absolute"` is the fix for the jumping caption. Each word
    becomes its own `\pos`'d `Dialogue` event at a coordinate we computed from
    real font metrics, so libass never has to flow them together and one word's
    `\fscy` cannot resize the box its neighbours sit in. Words are anchored
    `\an1` on a shared baseline, so a swell grows upward out of the baseline
    rather than shoving the line around.

    Held pauses still work — the silence is now literal extra advance width in
    the layout rather than an `\fsp` hack, which is both more predictable and
    easier to cap.

    Set `LAYOUT_MODE="flow"` to get the V11 renderer back for comparison.
    """)
    return


@app.cell
def _(
    CAPTION_BACK_COLOUR,
    CAPTION_MODE,
    CAPTION_OUTLINE_COLOUR,
    CAPTION_OUTLINE_PX,
    CAPTION_SHADOW_PX,
    DIM_ALPHA,
    EMOTION_STYLES,
    FONT_STRICT,
    GESTURE_DROP_FRAC,
    GESTURE_WOBBLE_HI,
    GESTURE_WOBBLE_LO,
    HAVE_PIL,
    HOLD_MAX_TAIL,
    LAYOUT_LINE_GAP,
    LAYOUT_MARGIN_V,
    LAYOUT_MARGIN_X,
    LAYOUT_MODE,
    LAYOUT_SPACE_SCALE,
    MIN_LINE_DURATION,
    MOTION_ANCHOR,
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
    READING_RATE_ENFORCE,
    READING_RATE_MAX_CPS,
    REVEAL_MODE,
    SQUASH_CONSERVATION,
    SQUASH_STRETCH,
    audio_file,
    layout_segment,
    os,
    out_tag,
    pause_gap,
    pred_emotion,
    squash_x,
    styled_word_df,
    subprocess,
    verify_fonts,
):
    # =====================================================================
    # CELL 20a — SHARED ASS BUILDER (both layout modes)
    # =====================================================================
    def sec_to_ass(t):
        t = max(0.0, float(t))
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        cs = int(round((t - int(t)) * 100))
        if cs == 100:
            cs = 0; s += 1
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def ass_header(width, height, wrap_style=0, outline_px=3, shadow_px=1,
                   outline_colour="&H00000000", back_colour="&H64000000"):
        # WrapStyle 0 = libass wraps for us (flow mode needs this, or a long
        # sentence runs straight off the frame). WrapStyle 2 = no automatic
        # wrapping, which is what absolute mode wants since it wraps itself.
        # V13 (audit Part 8 item 2): outline and shadow are parameters now —
        # they are what keeps value-1.0 yellow legible over a bright sky,
        # the exact readability failure that made AffType abandon
        # per-emotion colour. ScaledBorderAndShadow keeps them proportional
        # across resolutions.
        return (
            "[Script Info]\nScriptType: v4.00+\n"
            f"WrapStyle: {int(wrap_style)}\nScaledBorderAndShadow: yes\n"
            f"PlayResX: {width}\nPlayResY: {height}\n\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
            "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
            "MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Cap,Liberation Sans,48,&H00FFFFFF,&H000000FF,"
            f"{outline_colour},{back_colour},"
            f"0,0,0,0,100,100,0,0,1,{int(outline_px)},{int(shadow_px)},2,40,40,50,1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n"
        )

    def make_word_tagger(font_of, reveal_mode, dim_alpha, motion_style,
                         swell_peak, tilt_deg, motion_min_ms, motion_max_ms,
                         motion_tempo, isolated,
                         squash_enabled=False, squash_conservation=0.45):
        """Builds the per-word override block.

        `isolated=True` means the word is its own \\pos'd event, so \\fscy is
        safe: it changes nothing but this word. `isolated=False` is the V11
        flow path, kept verbatim for A/B fidelity — squash is only applied
        when isolated, because only the per-word \\pos architecture
        guarantees the horizontal squeeze is local (and only there does the
        layout reserve the room it needs).

        V13 (audit 4.3): with squash on, every \\fscy carries an inverse
        \\fscx from squash_x(), so the swell conserves apparent area and
        reads as squash-and-stretch (Lee et al. 2002) rather than a scale
        change. Amplitude fractions come from the shared GESTURE_* constants
        so this cell and the layout can never disagree about how far a
        gesture reaches."""

        def _pair(_fy):
            """fscy value -> the '\\fscyN\\fscxM' fragment for one keyframe."""
            if squash_enabled and isolated:
                return (f"\\fscy{int(round(_fy))}"
                        f"\\fscx{int(round(squash_x(_fy, squash_conservation)))}")
            return f"\\fscy{int(round(_fy))}"

        def word_tags(rw, line_start):
            fam = font_of(rw)
            tempo = motion_tempo.get(str(rw.get("anim", "flat")), 1.0)
            fsp = float(rw.get("tracking", 0.0) or 0.0)
            tags = (f"\\fn{fam}\\fs{int(rw['font_size'])}\\c{rw['color_ass']}"
                    f"\\b{int(rw['bold'])}\\i{int(rw['italic'])}"
                    f"\\fsp{fsp:g}\\fscx100\\fscy100\\frz0")

            d0 = max(0, int(round((float(rw["start"]) - line_start) * 1000)))
            d1 = max(d0 + 120, int(round((float(rw["end"]) - line_start) * 1000)))
            if reveal_mode == "wipe":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d1},\\alpha&H00&)"
            elif reveal_mode == "snap":
                tags += f"\\alpha&H{dim_alpha:02X}&\\t({d0},{d0 + 90},\\alpha&H00&)"

            g = str(rw.get("gesture", "none"))
            k = float(rw.get("motion_strength", 0.0))
            if g == "none" or k <= 0.0:
                return tags

            dur = int(min(max((float(rw["end"]) - float(rw["start"])) * 1000,
                              motion_min_ms), motion_max_ms) * tempo)
            dur = max(dur, 60)
            mid = d0 + int(dur * 0.45)
            end = d0 + dur

            peak = swell_peak if isolated else swell_peak * 0.45

            if motion_style == "scale":
                if g == "lift":
                    hi = 100 + peak * k
                    tags += (f"\\t({d0},{mid},{_pair(hi)})"
                             f"\\t({mid},{end},{_pair(100)})")
                elif g == "drop":
                    lo = 100 - (peak * GESTURE_DROP_FRAC) * k
                    tags += (f"\\t({d0},{mid},{_pair(lo)})"
                             f"\\t({mid},{end},{_pair(100)})")
                elif g == "wobble":
                    q = max(1, dur // 3)
                    hi = 100 + (peak * GESTURE_WOBBLE_HI) * k
                    lo = 100 - (peak * GESTURE_WOBBLE_LO) * k
                    tags += (f"\\t({d0},{d0+q},{_pair(hi)})"
                             f"\\t({d0+q},{d0+2*q},{_pair(lo)})"
                             f"\\t({d0+2*q},{end},{_pair(100)})")
            else:
                t_ = max(1, int(round(tilt_deg * k)))
                if g == "lift":
                    tags += f"\\frz{t_}\\t({d0},{end},\\frz0)"
                elif g == "drop":
                    tags += f"\\frz-{t_}\\t({d0},{end},\\frz0)"
                elif g == "wobble":
                    q = max(1, dur // 3)
                    tags += (f"\\t({d0},{d0+q},\\frz{t_})\\t({d0+q},{d0+2*q},\\frz-{t_})"
                             f"\\t({d0+2*q},{end},\\frz0)")
            return tags

        return word_tags

    # ---------- FLOW path (V11 behaviour, kept for A/B) -------------------
    def held_separator(pause_after, pause_hold, thresh, full, gap_max):
        if not pause_hold:
            return " "
        gap = pause_gap(pause_after, thresh, full, gap_max)
        if gap <= 0.0:
            return " "
        return f" {{\\fsp{gap:g}}} "

    def build_segment_text(seg_rows, tags_fn, line_start, pause_hold,
                           thresh, full, gap_max):
        rws = list(seg_rows.iterrows())
        chunks = []
        for i, (_, rw) in enumerate(rws):
            chunks.append("{" + tags_fn(rw, line_start) + "}"
                          + str(rw.get("word_display") or rw["word"]).strip())
            if i < len(rws) - 1:
                chunks.append(held_separator(rw.get("pause_after", 0.0), pause_hold,
                                             thresh, full, gap_max))
        return "".join(chunks)

    # ---------- the dispatcher --------------------------------------------
    def build_ass_events(rows, width, height, tags_fn, layout_mode,
                         hold_max_tail, min_line_dur,
                         pause_hold, pause_thresh, pause_full, pause_gap_max,
                         margin_x, margin_v, line_gap, space_scale,
                         motion_anchor, caption_mode="sentence", swell_peak=30.0,
                         squash_enabled=False, squash_conservation=0.45,
                         rate_enforce=False, rate_max_cps=17.0):
        lines = []
        rate_violations = []

        if caption_mode == "word":
            rr = rows.reset_index(drop=True)
            for i, rw in rr.iterrows():
                s_t = float(rw["start"])
                if i < len(rr) - 1:
                    e_t = min(float(rr.loc[i + 1, "start"]), float(rw["end"]) + hold_max_tail)
                else:
                    e_t = float(rw["end"]) + 0.35
                e_t = max(e_t, s_t + 0.10)
                txt = ("{\\an5" + tags_fn(rw, s_t) + "}"
                       + str(rw.get("word_display") or rw["word"]).strip())
                lines.append(f"Dialogue: 0,{sec_to_ass(s_t)},{sec_to_ass(e_t)},Cap,,0,0,0,,{txt}")
            return lines

        seg_ids = list(dict.fromkeys(rows["segment_id"].tolist()))
        seg_starts = {s: float(rows.loc[rows["segment_id"] == s, "start"].min())
                      for s in seg_ids}

        use_absolute = (layout_mode == "absolute") and HAVE_PIL

        for si, sid in enumerate(seg_ids):
            seg = rows[rows["segment_id"] == sid].sort_values("start")
            s0 = float(seg["start"].min())
            e0 = max(float(seg["end"].max()) + hold_max_tail, s0 + min_line_dur)
            # V13 reading rate (audit Part 8 item 3): when the natural
            # duration implies a faster read than READING_RATE_MAX_CPS,
            # extend the line's end. Extension is bounded by the next
            # segment's start — a caption cannot borrow time that the next
            # one owns — so a genuinely dense passage is REPORTED instead
            # of silently sped past. Held pauses add display time and the
            # per-word layout changes line geometry, so this interacts with
            # both; hence a check, not an assumption.
            n_chars = int(seg.get("word_display", seg["word"]).astype(str)
                          .str.strip().str.len().sum()
                          + max(len(seg) - 1, 0))
            if rate_enforce:
                e0 = max(e0, s0 + n_chars / max(rate_max_cps, 1e-6))
            if si < len(seg_ids) - 1:
                e0 = min(e0, seg_starts[seg_ids[si + 1]])
            e0 = max(e0, s0 + 0.10)
            if rate_enforce:
                cps = n_chars / max(e0 - s0, 1e-9)
                if cps > rate_max_cps + 0.05:
                    rate_violations.append((sid, round(cps, 1), n_chars))

            if not use_absolute:
                body = build_segment_text(seg, tags_fn, s0, pause_hold,
                                          pause_thresh, pause_full, pause_gap_max)
                lines.append(f"Dialogue: 0,{sec_to_ass(s0)},{sec_to_ass(e0)},Cap,,0,0,0,,"
                             + "{\\fad(120,120)}" + body)
                continue

            # ---- ABSOLUTE: one \pos'd event per word ----
            placed = layout_segment(seg, width, height, margin_x, margin_v,
                                    line_gap, space_scale, pause_hold,
                                    pause_thresh, pause_full, pause_gap_max,
                                    swell_peak=swell_peak,
                                    squash_enabled=squash_enabled,
                                    squash_conservation=squash_conservation)
            for p in placed:
                rw = p["row"]
                if motion_anchor == "center":
                    an, px, py = 5, p["cx"], p["baseline"] - 0.35 * float(rw["font_size"])
                else:
                    an, px, py = 1, p["x"], p["y"]
                head = f"\\an{an}\\pos({px:g},{py:g})\\fad(120,120)"
                # rotate about the word's own centre, not its anchor corner
                if str(rw.get("gesture", "none")) != "none":
                    head += f"\\org({p['cx']:g},{p['baseline']:g})"
                txt = "{" + head + tags_fn(rw, s0) + "}" + p["text"]
                lines.append(f"Dialogue: 0,{sec_to_ass(s0)},{sec_to_ass(e0)},Cap,,0,0,0,,{txt}")

        if rate_violations:
            print(f"READING RATE: {len(rate_violations)} line(s) exceed "
                  f"{rate_max_cps:g} cps even after extension "
                  f"(segment, cps, chars): {rate_violations[:8]}"
                  + (" ..." if len(rate_violations) > 8 else ""))
        return lines

    def write_ass(path, width, height, lines, wrap_style=0, outline_px=3,
                  shadow_px=1, outline_colour="&H00000000",
                  back_colour="&H64000000"):
        with open(path, "w") as f:
            f.write(ass_header(width, height, wrap_style, outline_px,
                               shadow_px, outline_colour, back_colour)
                    + "\n".join(lines) + "\n")
        return path

    # =====================================================================
    # CELL 20b — LEGACY RENDER (black screen)
    # =====================================================================
    def render_budget_video(audio_path, df, emotion, styles, out_dir="outputs",
                            caption_mode="sentence", reveal_mode="wipe",
                            hold_max_tail=0.6, min_line_dur=1.0, dim_alpha=150,
                            motion_style="scale", swell_peak=30, tilt_deg=7,
                            motion_min_ms=200, motion_max_ms=700, motion_tempo=None,
                            pause_hold=True, pause_thresh=0.40, pause_full=1.20,
                            pause_gap_max=40.0, layout_mode="absolute",
                            margin_x=60, margin_v=60, line_gap=0.22,
                            space_scale=1.0, motion_anchor="baseline",
                            squash_enabled=False, squash_conservation=0.45,
                            rate_enforce=False, rate_max_cps=17.0,
                            outline_px=3, shadow_px=1,
                            outline_colour="&H00000000",
                            back_colour="&H64000000", font_strict=True,
                            tag=None):
        tag = tag or "demo"
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        width, height = 1280, 720
        fam = styles.get(emotion, styles["neutral"])
        tempo_map = motion_tempo or {"pop": 0.8, "soft": 1.5, "flat": 1.0,
                                     "tremor": 0.7}

        # V13: refuse to measure in one face and render in another
        verify_fonts(styles, strict=font_strict)

        rows = df.sort_values("start").reset_index(drop=True)
        isolated = (layout_mode == "absolute") and HAVE_PIL

        tagger = make_word_tagger(
            font_of=lambda rw: fam["font"],
            reveal_mode=reveal_mode, dim_alpha=dim_alpha,
            motion_style=motion_style, swell_peak=swell_peak, tilt_deg=tilt_deg,
            motion_min_ms=motion_min_ms, motion_max_ms=motion_max_ms,
            motion_tempo={k: v for k, v in tempo_map.items()}, isolated=isolated,
            squash_enabled=squash_enabled,
            squash_conservation=squash_conservation)

        # the single-clip path forces one font family for the whole clip
        rows = rows.copy()
        rows["font"] = fam["font"]
        rows["anim"] = fam["anim"]

        lines = build_ass_events(
            rows, width, height, tagger, layout_mode, hold_max_tail, min_line_dur,
            pause_hold, pause_thresh, pause_full, pause_gap_max,
            margin_x, margin_v, line_gap, space_scale, motion_anchor, caption_mode,
            swell_peak=swell_peak, squash_enabled=squash_enabled,
            squash_conservation=squash_conservation,
            rate_enforce=rate_enforce, rate_max_cps=rate_max_cps)

        ass_path = write_ass(f"{out_dir}/ass/{tag}.ass", width, height, lines,
                             wrap_style=(2 if isolated else 0),
                             outline_px=outline_px, shadow_px=shadow_px,
                             outline_colour=outline_colour,
                             back_colour=back_colour)

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
        print(f"{len(lines)} event(s) | layout={'absolute' if isolated else 'flow'} "
              f"| motion={motion_style} | wrote {out_path}")
        return out_path, ass_path

    v16_video, v16_ass = render_budget_video(
        audio_file, styled_word_df, pred_emotion, EMOTION_STYLES,
        caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE, hold_max_tail=HOLD_MAX_TAIL,
        min_line_dur=MIN_LINE_DURATION, dim_alpha=DIM_ALPHA, motion_style=MOTION_STYLE,
        swell_peak=MOTION_SWELL_PEAK, tilt_deg=MOTION_TILT_DEG, motion_min_ms=MOTION_MIN_MS,
        motion_max_ms=MOTION_MAX_MS, motion_tempo=MOTION_TEMPO,
        pause_hold=PAUSE_HOLD, pause_thresh=PAUSE_HOLD_THRESH, pause_full=PAUSE_HOLD_FULL,
        pause_gap_max=PAUSE_HOLD_MAX_FSP, layout_mode=LAYOUT_MODE,
        margin_x=LAYOUT_MARGIN_X, margin_v=LAYOUT_MARGIN_V, line_gap=LAYOUT_LINE_GAP,
        space_scale=LAYOUT_SPACE_SCALE, motion_anchor=MOTION_ANCHOR,
        squash_enabled=SQUASH_STRETCH, squash_conservation=SQUASH_CONSERVATION,
        rate_enforce=READING_RATE_ENFORCE, rate_max_cps=READING_RATE_MAX_CPS,
        outline_px=CAPTION_OUTLINE_PX, shadow_px=CAPTION_SHADOW_PX,
        outline_colour=CAPTION_OUTLINE_COLOUR, back_colour=CAPTION_BACK_COLOUR,
        font_strict=FONT_STRICT,
        tag=out_tag + "_" + CAPTION_MODE + "_" + MOTION_STYLE)
    print("Wrote:", v16_video, "and", v16_ass)
    return build_ass_events, make_word_tagger, write_ass


@app.cell
def _(
    EMOTION_SELF_BIAS,
    EMOTION_SMOOTH,
    HAVE_PIL,
    MIN_DWELL_S,
    NORM_MIN_SEGMENTS,
    SEGMENT_NORM,
    audio_file,
    build_ass_events,
    clf_feature_cols,
    clf_full,
    confidence_scale,
    json,
    make_word_tagger,
    os,
    predict_segment_emotions_v9,
    seg_list,
    smooth_segment_emotions,
    subprocess,
    verify_fonts,
    write_ass,
):
    # =====================================================================
    # CELL 21 — PER-SEGMENT PREDICTION on the test clip
    # =====================================================================
    seg_emotion_df = predict_segment_emotions_v9(
        audio_file, seg_list, clf_full, clf_feature_cols,
        normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
    # V16.1: {segment_id: speaker_label} from diarization, or all-None if
    # diarization was skipped/failed -- either way smooth_segment_emotions
    # handles it correctly (None-vs-None is never treated as a boundary).
    _speaker_ids_21 = {_i: _s.get("speaker") for _i, _s in enumerate(seg_list)}
    # V16: min_dwell_s wired through -- the second guard alongside the
    # raised self_bias (see CELL 12).
    seg_emotion_df = smooth_segment_emotions(
        seg_emotion_df, list(clf_full.classes_), mode=EMOTION_SMOOTH,
        self_bias=EMOTION_SELF_BIAS, min_dwell_s=MIN_DWELL_S,
        speaker_ids=_speaker_ids_21,
        confidence_scale_fn=confidence_scale)
    seg_emotion_df

    # =====================================================================
    # CELL 22 — RENDER onto a REAL video (per-segment colour)
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
                          pause_gap_max=40.0, layout_mode="absolute",
                          margin_x=60, margin_v=60, line_gap=0.22,
                          space_scale=1.0, motion_anchor="baseline",
                          squash_enabled=False, squash_conservation=0.45,
                          rate_enforce=False, rate_max_cps=17.0,
                          outline_px=3, shadow_px=1,
                          outline_colour="&H00000000",
                          back_colour="&H64000000", font_strict=True,
                          bg_video_path=None, tag=None):
        tag = tag or "demo"
        os.makedirs(f"{out_dir}/ass", exist_ok=True)
        os.makedirs(f"{out_dir}/video", exist_ok=True)
        tempo_map = motion_tempo or {"pop": 0.8, "soft": 1.5, "flat": 1.0,
                                     "tremor": 0.7}

        # V13: substitution desyncs measurement from rendering — fail here,
        # not after a five-minute burn
        verify_fonts(styles, strict=font_strict)

        if bg_video_path:
            width, height, video_dur, bg_has_audio = get_video_info(bg_video_path)
        else:
            width, height = 1280, 720
            bg_has_audio = False

        rows = df.sort_values("start").reset_index(drop=True)
        isolated = (layout_mode == "absolute") and HAVE_PIL

        tagger = make_word_tagger(
            font_of=lambda rw: str(rw["font"]),
            reveal_mode=reveal_mode, dim_alpha=dim_alpha,
            motion_style=motion_style, swell_peak=swell_peak, tilt_deg=tilt_deg,
            motion_min_ms=motion_min_ms, motion_max_ms=motion_max_ms,
            motion_tempo=tempo_map, isolated=isolated,
            squash_enabled=squash_enabled,
            squash_conservation=squash_conservation)

        # margins scale with the frame so 4K and 720p look the same
        _s = width / 1280.0
        lines = build_ass_events(
            rows, width, height, tagger, layout_mode, hold_max_tail, min_line_dur,
            pause_hold, pause_thresh, pause_full, pause_gap_max,
            margin_x * _s, margin_v * _s, line_gap, space_scale,
            motion_anchor, caption_mode, swell_peak=swell_peak,
            squash_enabled=squash_enabled,
            squash_conservation=squash_conservation,
            rate_enforce=rate_enforce, rate_max_cps=rate_max_cps)

        ass_path = write_ass(f"{out_dir}/ass/{tag}.ass", width, height, lines,
                             wrap_style=(2 if isolated else 0),
                             outline_px=outline_px, shadow_px=shadow_px,
                             outline_colour=outline_colour,
                             back_colour=back_colour)

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
        print(f"{len(lines)} event(s)  |  layout={'absolute' if isolated else 'flow'}"
              f"  |  background: {'real video' if bg_video_path else 'black'}")
        return out_path, ass_path

    return (render_long_video,)


@app.cell
def _(
    AROUSAL_FEATURES,
    AROUSAL_SPREAD,
    ASR_BATCH_SIZE,
    ASR_CHUNK_SIZE,
    ASR_DROP_LOOPS,
    ASR_LOOP_MIN_REPEATS,
    BASE_FONT_SIZE,
    BLEND_MARGIN,
    BLEND_MODE,
    BLEND_PERWORD_SWING,
    BOLD_THRESHOLD,
    CALM_SPACING_MAX,
    CAPTION_BACK_COLOUR,
    CAPTION_MODE,
    CAPTION_OUTLINE_COLOUR,
    CAPTION_OUTLINE_PX,
    CAPTION_SHADOW_PX,
    CHANNEL_MODE,
    CLASS_OVERRIDE_FULL,
    CLASS_OVERRIDE_REL,
    CLASS_OVERRIDE_Z,
    CLF_N_CLASSES,
    DIM_ALPHA,
    EMOTION_FLOOR_BONUS,
    EMOTION_SELF_BIAS,
    EMOTION_SMOOTH,
    EMOTION_STYLES,
    EMPHASIS_QUOTA_FRAC,
    EXCLAIM_EMOTIONS,
    EXCLAIM_ENABLE,
    EXCLAIM_MAX_MARKS,
    EXCLAIM_MIN_MARKS,
    FONT_GAMMA,
    FONT_STRICT,
    FONT_SWING,
    FULL_DRAMA_RATIO,
    HOLD_MAX_TAIL,
    LAYOUT_LINE_GAP,
    LAYOUT_MARGIN_V,
    LAYOUT_MARGIN_X,
    LAYOUT_MODE,
    LAYOUT_SPACE_SCALE,
    MIN_DWELL_S,
    MIN_LINE_DURATION,
    MIN_POINTS,
    MIN_WORDS_FOR_SALIENCE,
    MOTION_ANCHOR,
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
    POSITIVE_ONLY_FEATURES,
    Path,
    QUOTA_DAMP,
    READING_RATE_ENFORCE,
    READING_RATE_MAX_CPS,
    REVEAL_MODE,
    ROBUST_STATS,
    SALIENCE_SHRINK_K,
    SALIENCE_WEIGHTS,
    SATURATION_INTENSITY,
    SAT_FLOOR_FRAC,
    SCALE_REL_FLOOR,
    SEGMENT_AROUSAL_FLOOR,
    SEGMENT_NORM,
    SHOUT_P_FLOOR_MULT,
    SHOUT_P_SECOND_MULT,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    SOFTMAX_TEMPERATURE,
    SQUASH_CONSERVATION,
    SQUASH_STRETCH,
    TRACKING_CALM,
    TREMOR_WOBBLE_FACTOR,
    USE_WORD_CLASS_PRIOR,
    VERSION_TAG,
    WOBBLE_RANGE_HZ,
    YELL_CASE,
    YELL_DETECT,
    YELL_FEATURES,
    YELL_MAX_FRAC,
    YELL_MIN_DB_OVER_MEDIAN,
    YELL_NORM,
    YELL_PEAK_FRAC,
    YELL_PUNCT_ASSIST,
    YELL_TILT_MIN_Z,
    YELL_Z,
    ZERO_MEANS_MISSING,
    allocate_points,
    apply_emotion_shout,
    apply_emphasis_quota,
    apply_word_class_prior,
    apply_yell_case,
    asr_model,
    assign_styles,
    assign_words_to_segments,
    attach_motion,
    clf_feature_cols,
    clf_full,
    compute_salience,
    confidence_scale,
    count_syllables,
    detect_yelling,
    device,
    extract_word_features,
    free_vram,
    np,
    os,
    predict_segment_emotions_v9,
    render_long_video,
    screen_segments,
    segment_arousal_floor,
    smooth_segment_emotions,
    styling_coverage,
    subprocess,
    whisperx,
    yell_report,
):
    # =====================================================================
    # CELL 23 — "INSERT ANY VIDEO": full pipeline, real video out
    # =====================================================================
    def process_any_video(video_path, out_tag=None, use_bg_video=True,
                          out_dir="outputs"):
        os.makedirs("outputs/audio", exist_ok=True)
        stem = Path(video_path).stem
        extracted_audio = f"outputs/audio/{stem}.wav"

        print(f"[1/6] extracting audio <- {video_path}")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                        extracted_audio], capture_output=True, text=True, check=True)

        print("[2/6] transcribing (whisperx)")
        free_vram()          # V15 — see CELL 7c
        audio = whisperx.load_audio(extracted_audio)
        try:
            result = asr_model.transcribe(audio, batch_size=ASR_BATCH_SIZE,
                                          chunk_size=ASR_CHUNK_SIZE)
        except TypeError:
            result = asr_model.transcribe(audio, batch_size=ASR_BATCH_SIZE)

        result["segments"], _dropped = screen_segments(
            result["segments"], ASR_DROP_LOOPS, ASR_LOOP_MIN_REPEATS)
        if _dropped:
            print(f"        dropped {_dropped} repetition-loop segment(s)")

        align_model, align_meta = whisperx.load_align_model(
            language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], align_model, align_meta, audio, device,
                                 return_char_alignments=False)
        # V15: this function loads a SECOND alignment model on top of the
        # one cell 9 already holds. Dropping the local name is not enough,
        # because torch keeps the freed blocks in its own cache; free_vram
        # returns them to the driver so the next call can transcribe.
        align_model = None
        free_vram()
        seg_list = aligned.get("segments") or result["segments"]
        print(f"        {len(seg_list)} segment(s), {len(aligned['word_segments'])} word(s)")

        print("[3/6] per-word prosody")
        word_df = extract_word_features(extracted_audio, aligned["word_segments"],
                                        count_syllables)
        tagged_word_df = word_df.copy()
        tagged_word_df["segment_id"] = assign_words_to_segments(tagged_word_df, seg_list)

        print("[4/6] salience budget (positive-only z + word-class prior)")
        salient_word_df = compute_salience(
            tagged_word_df, SALIENCE_WEIGHTS,
            zero_missing=ZERO_MEANS_MISSING,
            positive_only=POSITIVE_ONLY_FEATURES,
            robust=ROBUST_STATS, min_words=MIN_WORDS_FOR_SALIENCE,
            shrink_k=SALIENCE_SHRINK_K, scale_rel_floor=SCALE_REL_FLOOR)
        salient_word_df = apply_word_class_prior(
            salient_word_df, override_z=CLASS_OVERRIDE_Z,
            override_rel=CLASS_OVERRIDE_REL, override_full=CLASS_OVERRIDE_FULL,
            enabled=USE_WORD_CLASS_PRIOR)
        budget_df = allocate_points(salient_word_df, temperature=SOFTMAX_TEMPERATURE,
                                    min_points=MIN_POINTS)
        budget_df["intensity_raw"] = np.clip(
            (budget_df["share_ratio"] - 1.0) / (FULL_DRAMA_RATIO - 1.0), 0.0, 1.0)
        budget_df = apply_emphasis_quota(budget_df, EMPHASIS_QUOTA_FRAC, QUOTA_DAMP)
        budget_df["arousal_floor"] = segment_arousal_floor(
            budget_df, AROUSAL_FEATURES, SEGMENT_AROUSAL_FLOOR,
            spread=AROUSAL_SPREAD, robust=ROBUST_STATS)
        if YELL_DETECT:
            budget_df["is_yell"], budget_df["yell_z"] = detect_yelling(
                budget_df, YELL_FEATURES, YELL_Z, max_frac=YELL_MAX_FRAC,
                punct_assist=YELL_PUNCT_ASSIST, tilt_min_z=YELL_TILT_MIN_Z,
                min_db_over_median=YELL_MIN_DB_OVER_MEDIAN,
                norm=YELL_NORM, peak_frac=YELL_PEAK_FRAC,
                robust=ROBUST_STATS)
        else:
            budget_df["is_yell"], budget_df["yell_z"] = False, 0.0
        if YELL_DETECT and not np.asarray(budget_df["is_yell"]).any():
            yell_report(budget_df, YELL_FEATURES, YELL_Z,
                        tilt_min_z=YELL_TILT_MIN_Z,
                        min_db_over_median=YELL_MIN_DB_OVER_MEDIAN,
                    norm=YELL_NORM, peak_frac=YELL_PEAK_FRAC)
        budget_df = apply_yell_case(budget_df, YELL_CASE)

        print("[5/6] emotion per segment + styling")
        seg_emotion_df = predict_segment_emotions_v9(
            extracted_audio, seg_list, clf_full, clf_feature_cols,
            normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
        # V16.1: same speaker-aware wiring as the single-clip test path.
        _speaker_ids_pav = {_i: _s.get("speaker") for _i, _s in enumerate(seg_list)}
        # V16: min_dwell_s wired through here too, so the demo/production
        # path gets the same stability fix as the single-clip test path.
        seg_emotion_df = smooth_segment_emotions(
            seg_emotion_df, list(clf_full.classes_), mode=EMOTION_SMOOTH,
            self_bias=EMOTION_SELF_BIAS, min_dwell_s=MIN_DWELL_S,
            speaker_ids=_speaker_ids_pav,
            confidence_scale_fn=confidence_scale)
        styled_df = assign_styles(
            budget_df, seg_emotion_df, EMOTION_STYLES,
            base_font=BASE_FONT_SIZE, font_swing=FONT_SWING, font_gamma=FONT_GAMMA,
            bold_thresh=BOLD_THRESHOLD,
            saturation_intensity=SATURATION_INTENSITY, sat_floor_frac=SAT_FLOOR_FRAC,
            blend_mode=BLEND_MODE, blend_margin=BLEND_MARGIN,
            blend_perword_swing=BLEND_PERWORD_SWING,
            tracking_calm=TRACKING_CALM, calm_spacing_max=CALM_SPACING_MAX,
            channel_mode=CHANNEL_MODE,
            emotion_floor_bonus=EMOTION_FLOOR_BONUS)
        # V16.2: same emotion-driven caps + exclamation marks as the
        # single-clip test path -- must run before attach_motion.
        if EXCLAIM_ENABLE:
            styled_df = apply_emotion_shout(
                styled_df, EXCLAIM_EMOTIONS, BLEND_MARGIN,
                max_marks=EXCLAIM_MAX_MARKS, min_marks=EXCLAIM_MIN_MARKS,
                n_classes=CLF_N_CLASSES,      # V18: chance-relative floors
                p_floor_mult=SHOUT_P_FLOOR_MULT,
                p_second_mult=SHOUT_P_SECOND_MULT)
        styled_df = attach_motion(styled_df, motion_source=MOTION_SOURCE,
                                  motion_min_intensity=MOTION_MIN_INTENSITY,
                                  slope_deadzone=SLOPE_DEADZONE, slope_full=SLOPE_FULL,
                                  wobble_range_hz=WOBBLE_RANGE_HZ,
                                  tremor_wobble_factor=TREMOR_WOBBLE_FACTOR)

        print("[6/6] rendering + burning onto the original video")
        tag = out_tag or (VERSION_TAG + "_" + stem)
        out_path, ass_path = render_long_video(
            extracted_audio, styled_df, EMOTION_STYLES, out_dir=out_dir,
            caption_mode=CAPTION_MODE, reveal_mode=REVEAL_MODE, hold_max_tail=HOLD_MAX_TAIL,
            min_line_dur=MIN_LINE_DURATION, dim_alpha=DIM_ALPHA, motion_style=MOTION_STYLE,
            swell_peak=MOTION_SWELL_PEAK, tilt_deg=MOTION_TILT_DEG, motion_min_ms=MOTION_MIN_MS,
            motion_max_ms=MOTION_MAX_MS, motion_tempo=MOTION_TEMPO,
            pause_hold=PAUSE_HOLD, pause_thresh=PAUSE_HOLD_THRESH, pause_full=PAUSE_HOLD_FULL,
            pause_gap_max=PAUSE_HOLD_MAX_FSP, layout_mode=LAYOUT_MODE,
            margin_x=LAYOUT_MARGIN_X, margin_v=LAYOUT_MARGIN_V, line_gap=LAYOUT_LINE_GAP,
            space_scale=LAYOUT_SPACE_SCALE, motion_anchor=MOTION_ANCHOR,
            squash_enabled=SQUASH_STRETCH, squash_conservation=SQUASH_CONSERVATION,
            rate_enforce=READING_RATE_ENFORCE, rate_max_cps=READING_RATE_MAX_CPS,
            outline_px=CAPTION_OUTLINE_PX, shadow_px=CAPTION_SHADOW_PX,
            outline_colour=CAPTION_OUTLINE_COLOUR, back_colour=CAPTION_BACK_COLOUR,
            font_strict=FONT_STRICT,
            bg_video_path=(video_path if use_bg_video else None), tag=tag)

        _hot = styled_df[styled_df["intensity"] > 0.6]
        print("emotions found:", seg_emotion_df["pred_emotion"].value_counts().to_dict())
        print("emphasised words:",
              [str(w).strip() for w in _hot.sort_values('intensity', ascending=False)['word'].head(12)])
        print("emphasised by class:", _hot["word_class"].value_counts().to_dict())
        # V13 (audit 2.7): the styled-proportion number the write-up needs
        styling_coverage(styled_df, label=stem)
        print("wrote:", out_path)
        return out_path, ass_path, seg_emotion_df, styled_df, extracted_audio

    return (process_any_video,)


@app.cell
def _(iemocap_dir, os, process_any_video):
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
        (long_out, long_ass, long_seg_emotions, long_styled_df,
         long_audio) = process_any_video(_demo_video)
    else:
        print("No dialog video at the expected path -- pass any video path straight "
              "into process_any_video('/path/to/your/clip.mp4').")
    return


@app.cell
def _(Path, os, process_any_video, subprocess):
    # =====================================================================
    # CELL 24b — CUT A TEST CLIP  (V15)
    # ---------------------------------------------------------------------
    # Iterating on a feature-length film is the wrong loop. A two-minute
    # extract transcribes in a fraction of the time, so a styling change can
    # be judged in minutes instead of an hour, and it is also what a demo
    # clip and an evaluation stimulus need to be anyway.
    #
    # Honest note: this is NOT the fix for a CUDA out-of-memory error.
    # WhisperX processes audio in fixed-size batches, so peak VRAM is set by
    # ASR_BATCH_SIZE and the model size, not by how long the file is. A
    # shorter clip fails just as fast on a card that is already full. It
    # does cut total runtime and the size of every intermediate frame.
    #
    # copy=True stream-copies, which is instant but can only cut on a
    # keyframe, so the real start may be a little before `start`. That is
    # harmless at start=0 and can shift by a second or two mid-film. Pass
    # copy=False to re-encode for a frame-accurate cut, which is slower but
    # exact — use it whenever the clip is a study stimulus, because a
    # participant's timing should not depend on keyframe placement.
    # =====================================================================
    def cut_clip(src, start=0.0, duration=120.0, out_dir="outputs/clips",
                 copy=True, suffix=None):
        """Write a `duration`-second extract of `src` starting at `start`."""
        if not os.path.exists(src):
            raise FileNotFoundError(f"{src} (cwd is {os.getcwd()})")
        os.makedirs(out_dir, exist_ok=True)
        _stem = Path(src).stem
        _tag = suffix or f"{int(start)}s_{int(duration)}s"
        _out = f"{out_dir}/{_stem}_{_tag}{Path(src).suffix}"

        # -ss BEFORE -i seeks by index and is fast; -t after sets duration.
        _cmd = ["ffmpeg", "-y", "-ss", f"{start:g}", "-i", src,
                "-t", f"{duration:g}"]
        if copy:
            _cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        else:
            _cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                     "-c:a", "aac", "-b:a", "192k"]
        _cmd.append(_out)

        _r = subprocess.run(_cmd, capture_output=True, text=True)
        if _r.returncode != 0 or not os.path.exists(_out):
            raise RuntimeError("ffmpeg failed:\n" + _r.stderr[-1500:])

        # Report what actually landed on disk, not what was asked for: with
        # copy=True those two can differ, and a silently longer clip would
        # quietly reintroduce the runtime you were trying to avoid.
        _p = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", _out],
            capture_output=True, text=True)
        try:
            _dur = float(_p.stdout.strip())
        except ValueError:
            _dur = float("nan")
        _mb = os.path.getsize(_out) / 2**20
        print(f"wrote {_out}  ({_dur:.1f}s, {_mb:.1f} MB, "
              f"{'stream copy' if copy else 're-encoded'})")
        return _out

    # =====================================================================
    # CELL 25 — RUN ON A NEW VIDEO: 12AngryMenTest.mp4
    # =====================================================================
    new_video_path = "12AngryMenTest.mp4"

    if not os.path.exists(new_video_path):
        print(f"Can't find '{new_video_path}' from {os.getcwd()}. cd to the project "
              "or set new_video_path to the full path.")
    else:
        (angry_men_out, angry_men_ass, angry_men_seg_emotions,
         angry_men_styled_df, angry_men_audio) = process_any_video(new_video_path)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part F: Evaluation infrastructure — stimulus screening (V13)

    The mute test measures nothing unless the words themselves are emotionally
    neutral: Lee et al. are explicit that kinetic typography *reinforces or
    tempers* emotive content, it cannot override text that already gives the
    answer away — which is exactly why RAVDESS holds its two carrier sentences
    lexically null. This cell screens any candidate transcript with EmoLex so
    stimulus neutrality is **verified, not assumed** (audit 7.1).
    """)
    return


@app.cell
def _(EMOLEX_PATH, normalise_token, os, pd, word_df):
    # =====================================================================
    # CELL 26 — EMOLEX STIMULUS SCREENING  (V13, audit 7.1)
    # ---------------------------------------------------------------------
    # The NRC Word-Emotion Association Lexicon is free for research; place
    # the word-level file at EMOLEX_PATH. Screening is a rough gate, not a
    # verdict: it reports WHICH words carry core-emotion associations so
    # they can be judged in context (EmoLex annotates words in isolation —
    # its own authors flag that as a limit, and so should the write-up).
    # =====================================================================
    def load_emolex(path):
        """word<TAB>emotion<TAB>flag lines -> {word: set(emotions)}."""
        _lex = {}
        with open(path, encoding="utf-8") as _f:
            for _line in _f:
                _parts = _line.rstrip("\n").split("\t")
                if len(_parts) == 3 and _parts[2] == "1":
                    _lex.setdefault(_parts[0].lower(), set()).add(_parts[1])
        return _lex

    def screen_words_emolex(words, lexicon):
        """-> (per-emotion count DataFrame, loaded fraction, hit list).
        'Loaded' counts words with at least one of the eight core Plutchik
        emotions; bare positive/negative polarity is reported but does not
        count a word as loaded."""
        _toks = [normalise_token(_w) for _w in words]
        _toks = [_t for _t in _toks if _t]
        _counts, _hits, _loaded = {}, [], 0
        for _t in _toks:
            _assoc = lexicon.get(_t, set())
            _core = _assoc - {"positive", "negative"}
            if _core:
                _loaded += 1
                _hits.append((_t, ",".join(sorted(_core))))
            for _e in _assoc:
                _counts[_e] = _counts.get(_e, 0) + 1
        _frac = _loaded / max(len(_toks), 1)
        _summary = pd.DataFrame(sorted(_counts.items()),
                                columns=["emotion", "word_count"])
        return _summary, _frac, _hits

    if os.path.exists(EMOLEX_PATH):
        emolex = load_emolex(EMOLEX_PATH)
        emolex_summary, _frac, _hits = screen_words_emolex(word_df["word"],
                                                           emolex)
        print(f"EmoLex screening of the current transcript "
              f"({len(emolex)} lexicon words loaded):")
        print(f"  {100 * _frac:.0f}% of words carry a core-emotion "
              f"association — as a rough gate, mute-test stimuli want this "
              f"LOW (RAVDESS's carrier sentences are the model)")
        if _hits:
            print("  loaded words (judge these in context): "
                  + ", ".join(f"{_w}[{_e}]" for _w, _e in _hits[:20])
                  + (" ..." if len(_hits) > 20 else ""))
        emolex_summary
    else:
        emolex = None
        emolex_summary = None
        print(f"EmoLex file not found at '{EMOLEX_PATH}'.\n"
              "  Download the NRC Word-Emotion Association Lexicon "
              "(free for research, from Saif Mohammad's NRC page), put the "
              "word-level file at that path, and re-run this cell before "
              "selecting mute-test stimuli.")
    return


@app.cell
def _(
    BASE_FONT_SIZE,
    CAPTION_MODE,
    EMOTION_STYLES,
    HOLD_MAX_TAIL,
    LAYOUT_LINE_GAP,
    LAYOUT_MARGIN_V,
    LAYOUT_MARGIN_X,
    LAYOUT_MODE,
    LAYOUT_SPACE_SCALE,
    MIN_LINE_DURATION,
    Path,
    READING_RATE_ENFORCE,
    READING_RATE_MAX_CPS,
    VERSION_TAG,
    json,
    os,
    pd,
    process_any_video,
    re,
    render_long_video,
    styling_coverage,
    subprocess,
):
    # =====================================================================
    # CELL 27 — BATCH EVALUATION RUNNER  (V18, new)
    # ---------------------------------------------------------------------
    # Renders every video in a source folder TWICE -- once with the full
    # kinetic styling, once as plain baseline subtitles -- into
    # `evaluationoutput/`. Directory-driven, so adding a stimulus is
    # dropping a file in the folder, not editing this cell.
    #
    # WHY ONE PASS, TWO RENDERS (this is the important part):
    # The expensive stages are ASR, alignment, per-word prosody, and
    # per-segment opensmile+RF. The render is just libass and ffmpeg. So
    # the pipeline runs ONCE per video and the resulting styled_df is
    # rendered twice, the second time with the styling columns flattened.
    #
    # That is not only faster, it is the only way the pair is a valid
    # A/B. WhisperX with batching is not bit-deterministic; two separate
    # runs of the same file can differ in segmentation and word
    # boundaries. If that happened, the two conditions would differ in
    # transcript AND styling, and any effect you measured could be either.
    # Sharing one styled_df guarantees identical text, identical timings,
    # identical segmentation, identical emotion labels -- styling is the
    # only thing that varies.
    #
    # WHAT "PLAIN" MEANS IS A STUDY-DESIGN CHOICE, NOT A TECHNICAL ONE.
    # PLAIN_SPEC below is written out explicitly for that reason. The
    # default strips every channel this pipeline adds: colour, per-word
    # size, weight, italic, animation, motion, letter-spacing, pause
    # gaps, the reveal wipe, and the emotion-driven capitals/exclamations.
    # What remains is white text, one font, one size, appearing per line.
    # If your control should keep some of those (e.g. you want to isolate
    # colour alone), change PLAIN_SPEC rather than editing the flattener.
    #
    # ONE CONFOUND WORTH KNOWING: absolute layout measures text to place
    # it, so uniform font size can wrap lines differently than varied
    # font size does. Line breaks may therefore differ between the pair.
    # That is inherent to the manipulation, not a bug -- but say so in the
    # write-up rather than claiming the two differ only in styling.
    # =====================================================================
    SOURCE_VIDEO_DIR = "OriginalVideo"
    EVAL_OUT_DIR = "evaluationoutput"
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

    # Output naming: "{stem}{VERSION_TAG}_withUniqueSubtitles". VERSION_TAG
    # comes from CELL 1, so bumping it to "v19" renames every output with
    # no edit here.
    UNIQUE_SUFFIX = f"{VERSION_TAG}_withUniqueSubtitles"
    PLAIN_SUFFIX = f"{VERSION_TAG}_withoutUniqueSubtitles"

    # ---------------- FORMAT NORMALISATION ----------------
    # One stimulus (TheMentalist) carries baked-in black bars offset to a
    # corner, which means the encoded frame is not the picture. Two
    # separate causes produce that and both are handled below:
    #   * letterbox/pillarbox burned into the pixels -> detected with
    #     ffmpeg's cropdetect and cropped away.
    #   * non-square pixels (SAR != 1) -> the frame is stored at one
    #     shape and meant to be displayed at another. setsar=1 after
    #     scaling to the display size fixes it.
    # After cropping, everything is scaled to fit NORM_TARGET and padded
    # to exactly that size. Uniform canvas matters for a study: stimulus
    # size should not vary between conditions or between clips, or it is a
    # confound sitting alongside the one you are testing. Padding adds
    # bars back for non-16:9 sources, but symmetric and known, rather
    # than baked in and offset.
    NORMALISE_FORMAT = True
    NORM_TARGET = (1920, 1080)
    NORM_DIR = "outputs/normalised"
    CROPDETECT_PROBES = 3        # sample points, so one dark scene can't
    CROPDETECT_SECONDS = 4       # decide the crop for the whole film

    def ffprobe_stream(path):
        """-> dict of the first video stream's geometry, or {} on failure."""
        try:
            _r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries",
                 "stream=width,height,sample_aspect_ratio,"
                 "display_aspect_ratio,duration,codec_name",
                 "-show_entries", "format=duration",
                 "-of", "json", path],
                capture_output=True, text=True, check=True)
            _j = json.loads(_r.stdout)
            _s = (_j.get("streams") or [{}])[0]
            _dur = _s.get("duration") or (_j.get("format") or {}).get("duration")
            _s["_duration"] = float(_dur) if _dur else None
            return _s
        except Exception as _e:
            print(f"    ffprobe failed on {path}: {_e}")
            return {}

    _CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")

    def detect_crop(path, duration=None, probes=CROPDETECT_PROBES,
                    seconds=CROPDETECT_SECONDS):
        """-> (w, h, x, y) of the largest content box found, or None.

        Samples several points rather than the opening frames, because
        fades and dark establishing shots make cropdetect over-crop. The
        UNION (largest box) is taken across probes: cropping to the
        intersection would clip picture that is genuinely there in a
        brighter scene.
        """
        if duration is None or duration <= 0:
            _offsets = [0.0]
        else:
            # skip the first and last 10% -- titles and credits are often
            # letterboxed differently from the body of the clip
            _lo, _hi = duration * 0.1, duration * 0.9
            if _hi <= _lo:
                _offsets = [max(duration * 0.5, 0.0)]
            else:
                _offsets = [_lo + (_hi - _lo) * _i / max(probes - 1, 1)
                            for _i in range(probes)]
        _boxes = []
        for _off in _offsets:
            try:
                _r = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-nostats",
                     "-ss", f"{_off:.2f}", "-i", path,
                     "-t", str(seconds), "-vf", "cropdetect=24:2:0",
                     "-f", "null", "-"],
                    capture_output=True, text=True)
                _found = _CROP_RE.findall(_r.stderr or "")
                if _found:
                    _w, _h, _x, _y = (int(v) for v in _found[-1])
                    if _w > 0 and _h > 0:
                        _boxes.append((_w, _h, _x, _y))
            except Exception:
                continue
        if not _boxes:
            return None
        # union: smallest x/y, largest right/bottom edge
        _x0 = min(b[2] for b in _boxes)
        _y0 = min(b[3] for b in _boxes)
        _x1 = max(b[2] + b[0] for b in _boxes)
        _y1 = max(b[3] + b[1] for b in _boxes)
        # even dimensions -- yuv420p needs them
        _w = (_x1 - _x0) // 2 * 2
        _h = (_y1 - _y0) // 2 * 2
        return (_w, _h, _x0, _y0)

    def normalise_video(path, out_dir=NORM_DIR, target=NORM_TARGET,
                        force=False):
        """Crop baked-in bars, square the pixels, fit to `target`.

        Returns the path to use downstream: the normalised file, or the
        original when nothing needed changing (so conformant clips are
        not needlessly re-encoded).
        """
        os.makedirs(out_dir, exist_ok=True)
        _stem = Path(path).stem
        _out = f"{out_dir}/{_stem}_norm.mp4"
        _tw, _th = target

        _info = ffprobe_stream(path)
        if not _info:
            print(f"    could not probe -- using original")
            return path
        _w, _h = int(_info.get("width") or 0), int(_info.get("height") or 0)
        _sar = str(_info.get("sample_aspect_ratio") or "1:1")
        _dur = _info.get("_duration")
        if _w <= 0 or _h <= 0:
            print(f"    no video stream geometry -- using original")
            return path

        _crop = detect_crop(path, duration=_dur)
        _needs_crop = bool(_crop and (_crop[0] < _w or _crop[1] < _h))
        _needs_sar = _sar not in ("1:1", "0:1", "N/A", "")
        _needs_scale = (_w, _h) != (_tw, _th)

        print(f"    {_w}x{_h} sar={_sar}"
              + (f" | content box {_crop[0]}x{_crop[1]}+{_crop[2]}+{_crop[3]}"
                 if _crop else " | cropdetect found nothing"))

        if not (_needs_crop or _needs_sar or _needs_scale) and not force:
            print("    already conformant -- no re-encode")
            return path
        if os.path.exists(_out) and not force:
            print(f"    reusing {_out}")
            return _out

        _chain = []
        if _needs_crop:
            _chain.append(f"crop={_crop[0]}:{_crop[1]}:{_crop[2]}:{_crop[3]}")
        # scale to FIT inside the target, preserving aspect, then pad out
        _chain.append(f"scale={_tw}:{_th}:force_original_aspect_ratio="
                      f"decrease:flags=lanczos")
        _chain.append(f"pad={_tw}:{_th}:(ow-iw)/2:(oh-ih)/2:color=black")
        _chain.append("setsar=1")
        _vf = ",".join(_chain)

        _reasons = [r for r, f in (("cropped bars", _needs_crop),
                                    ("squared pixels", _needs_sar),
                                    ("resized", _needs_scale)) if f]
        print(f"    normalising ({', '.join(_reasons)}) -> {_out}")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-nostats", "-i", path,
                 "-vf", _vf, "-c:v", "libx264", "-preset", "medium",
                 "-crf", "18", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", _out],
                capture_output=True, text=True, check=True)
            return _out
        except subprocess.CalledProcessError as _e:
            print(f"    NORMALISE FAILED, falling back to original:\n"
                  f"      {(_e.stderr or '')[-400:]}")
            return path

    # ---------------- THE PLAIN (CONTROL) VARIANT ----------------
    # Every column the ASS builder reads, and what the control sets it to.
    # Written as data rather than code so the control condition is
    # inspectable and quotable -- "the baseline was PLAIN_SPEC" is a
    # statement you can put in a methods section.
    PLAIN_SPEC = {
        "color_ass": "&H00FFFFFF",   # opaque white, ASS is &HAABBGGRR
        "bold": 0,
        "italic": 0,
        "anim": "flat",
        "gesture": "none",           # gates ALL motion tags off
        "motion_strength": 0.0,
        "tracking": 0.0,             # no calm letter-spacing
        "intensity": 0.0,
        "is_yell": False,
        "is_emotion_shout": False,
        "blended": False,
    }
    # render-time params for the control. dim_alpha=0 neuters the reveal:
    # the tag becomes a transition from opaque to opaque, so words simply
    # appear with their line instead of wiping in.
    PLAIN_RENDER = dict(reveal_mode="snap", dim_alpha=0, motion_style="none",
                        pause_hold=False, squash_enabled=False)

    def flatten_to_plain(styled_df, base_font, spec=PLAIN_SPEC):
        """-> a copy of styled_df with every kinetic channel neutralised.

        word_display is reset from `word`, which is what undoes the
        emotion-driven ALL CAPS and the trailing '!' marks -- those were
        written INTO word_display by apply_emotion_shout/apply_yell_case,
        so clearing the boolean flags alone would leave the text shouting.
        """
        out = styled_df.copy()
        for _col, _val in spec.items():
            if _col in out.columns:
                out[_col] = _val
        out["font"] = EMOTION_STYLES["neutral"]["font"]
        out["font_size"] = int(base_font)
        # the text itself, stripped back to what was transcribed
        if "word" in out.columns:
            out["word_display"] = out["word"].astype(str)
        # keep pause_after present (the builder reads it) but inert:
        # PLAIN_RENDER passes pause_hold=False, so it is never consulted
        return out

    # ---------------- THE BATCH ----------------
    def find_source_videos(src_dir=SOURCE_VIDEO_DIR, exts=VIDEO_EXTS):
        """Every video in src_dir, sorted. Not name-dependent: drop a new
        file in the folder and it joins the batch."""
        _p = Path(src_dir)
        if not _p.is_dir():
            return []
        return sorted((f for f in _p.iterdir()
                       if f.is_file() and f.suffix.lower() in exts),
                      key=lambda f: f.name.lower())

    def safe_stem(path):
        """Filesystem-safe stem. The stimulus names carry parentheses
        ('BreakingBad(Happy)'), which survive ffmpeg fine via argv lists
        but are awkward in shells and in some analysis tools. Parens
        become underscores; everything else unusual is dropped."""
        _s = Path(path).stem
        _s = _s.replace("(", "_").replace(")", "")
        _s = re.sub(r"[^A-Za-z0-9_.-]+", "_", _s)
        return re.sub(r"_+", "_", _s).strip("_")

    def run_evaluation_batch(src_dir=SOURCE_VIDEO_DIR,
                             out_dir=EVAL_OUT_DIR,
                             normalise=NORMALISE_FORMAT,
                             only=None, skip_existing=True):
        """Render every video in src_dir as a matched unique/plain pair.

        only          list of stems to restrict to (for re-running one)
        skip_existing skip a video whose BOTH outputs already exist
        """
        _vids = find_source_videos(src_dir)
        if not _vids:
            print(f"No videos found in '{src_dir}/'.\n"
                  f"  cwd is {os.getcwd()}\n"
                  f"  expected extensions: {sorted(VIDEO_EXTS)}")
            return pd.DataFrame()

        os.makedirs(out_dir, exist_ok=True)
        print(f"found {len(_vids)} video(s) in {src_dir}/:")
        for _v in _vids:
            print(f"  {_v.name}")
        print(f"\noutputs -> {out_dir}/video/  (tag prefix {VERSION_TAG})\n")

        _rows = []
        for _i, _vid in enumerate(_vids, 1):
            _stem = safe_stem(_vid)
            if only and _stem not in only and _vid.stem not in only:
                continue
            _uniq_tag = f"{_stem}{UNIQUE_SUFFIX}"
            _plain_tag = f"{_stem}{PLAIN_SUFFIX}"
            _uniq_mp4 = f"{out_dir}/video/{_uniq_tag}.mp4"
            _plain_mp4 = f"{out_dir}/video/{_plain_tag}.mp4"

            print("=" * 70)
            print(f"[{_i}/{len(_vids)}] {_vid.name}")
            print("=" * 70)

            if (skip_existing and os.path.exists(_uniq_mp4)
                    and os.path.exists(_plain_mp4)):
                print("  both outputs exist -- skipping "
                      "(skip_existing=False to force)")
                _rows.append({"source": _vid.name, "stem": _stem,
                              "status": "skipped",
                              "unique": _uniq_mp4, "plain": _plain_mp4})
                continue

            _src = str(_vid)
            if normalise:
                print("  [format] checking geometry")
                _src = normalise_video(_src)

            try:
                print("  [unique] full kinetic styling")
                (_u_path, _u_ass, _seg_df, _styled,
                 _audio) = process_any_video(
                    _src, out_tag=_uniq_tag, use_bg_video=True,
                    out_dir=out_dir)

                print("  [plain] flattened control render")
                _flat = flatten_to_plain(_styled, BASE_FONT_SIZE)
                _p_path, _p_ass = render_long_video(
                    _audio, _flat, EMOTION_STYLES, out_dir=out_dir,
                    caption_mode=CAPTION_MODE,
                    hold_max_tail=HOLD_MAX_TAIL,
                    min_line_dur=MIN_LINE_DURATION,
                    layout_mode=LAYOUT_MODE,
                    margin_x=LAYOUT_MARGIN_X, margin_v=LAYOUT_MARGIN_V,
                    line_gap=LAYOUT_LINE_GAP, space_scale=LAYOUT_SPACE_SCALE,
                    rate_enforce=READING_RATE_ENFORCE,
                    rate_max_cps=READING_RATE_MAX_CPS,
                    bg_video_path=_src, tag=_plain_tag,
                    **PLAIN_RENDER)

                styling_coverage(_styled, label=f"{_stem} (unique)")
                _rows.append({
                    "source": _vid.name, "stem": _stem, "status": "ok",
                    "normalised_src": _src if _src != str(_vid) else "",
                    "unique": _u_path, "plain": _p_path,
                    "n_segments": int(len(_seg_df)),
                    "n_words": int(len(_styled)),
                    "emotions": ", ".join(
                        f"{_k}:{_v}" for _k, _v in
                        _seg_df["pred_emotion"].value_counts().items()),
                })
                print(f"  DONE  unique -> {_u_path}")
                print(f"        plain  -> {_p_path}")
            except Exception as _e:
                # one bad stimulus must not kill a long batch
                print(f"  FAILED on {_vid.name}: {type(_e).__name__}: {_e}")
                _rows.append({"source": _vid.name, "stem": _stem,
                              "status": f"failed: {type(_e).__name__}",
                              "unique": "", "plain": ""})
                continue

        _manifest = pd.DataFrame(_rows)
        if len(_manifest):
            _mpath = f"{out_dir}/manifest_{VERSION_TAG}.csv"
            _manifest.to_csv(_mpath, index=False)
            print("\n" + "=" * 70)
            print(f"batch complete -- manifest -> {_mpath}")
            print(_manifest[["source", "status"]].to_string(index=False))
        return _manifest

    print("batch runner ready.")
    print(f"  source folder : {SOURCE_VIDEO_DIR}/")
    print(f"  output folder : {EVAL_OUT_DIR}/video/")
    print(f"  naming        : <stem>{UNIQUE_SUFFIX}.mp4")
    print(f"                  <stem>{PLAIN_SUFFIX}.mp4")
    print(f"  found now     : {[f.name for f in find_source_videos()]}")
    print("\ncall run_evaluation_batch() to render everything.")
    return (run_evaluation_batch,)


@app.cell
def _(mo):
    mo.md("""
    ### Running the batch

    ```python
    run_evaluation_batch()                      # everything in OriginalVideo/
    run_evaluation_batch(only=["TheMentalist_Threathen"])   # one stimulus
    run_evaluation_batch(skip_existing=False)   # force re-render
    run_evaluation_batch(normalise=False)       # skip the format pass
    ```

    Outputs land in `evaluationoutput/video/` as
    `<stem>v18_withUniqueSubtitles.mp4` and
    `<stem>v18_withoutUniqueSubtitles.mp4`, with the `.ass` files in
    `evaluationoutput/ass/` and a `manifest_v18.csv` recording what was
    rendered, from which source, with which emotions found. Bump
    `VERSION_TAG` in CELL 1 to `"v19"` and every name follows.

    Adding a stimulus is dropping a file into `OriginalVideo/` — the runner
    globs the folder, so nothing here is keyed to a filename.
    """)
    return


@app.cell
def _(run_evaluation_batch):
    run_evaluation_batch(only=["TheMentalist_Threathen"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Decision log — V18

    **What changed, and what it costs.**

    V18 is a model swap, not a feature release. `clf_v2` was trained on
    RAVDESS and quoted at 60.6% by comments scattered through this notebook.
    `clf_v3` (`train_emotionsV3`) is trained IEMOCAP-primary with RAVDESS as
    a down-weighted support set, and measured leave-one-IEMOCAP-session-out
    at roughly **36% accuracy / 0.23 macro-F1 on 8 classes** (chance 0.125).

    **That number is lower than 60.6% and it is the better number.** The old
    figure was measured on acted, studio-recorded, scripted speech — the
    domain RAVDESS occupies and this pipeline does not. Measured against
    conversational speech, an acted-speech-only model scores **0.14 accuracy
    on the same 8-class space, barely above chance**; even restricted to the
    6 classes it knows, it reaches only 0.23. Conversational emotion
    recognition from acoustics alone is simply a harder problem than the
    RAVDESS number implied. V18 renders a weaker-looking classifier that is
    measured honestly against the thing it actually runs on.

    Do not quote 60.6% anywhere. Where it still appears in comments below
    (cells 7, 13), it is describing `clf_v2` and is left as historical
    context, not as a claim about the loaded model. CELL 6 now prints the
    loaded bundle's own provenance, so a run states its own numbers.

    ### 1. CELL 6 reads provenance, and validates rather than trusts

    The loader takes a candidate list (`clf_v3` -> `clf_v2` -> in-notebook
    14-feature retrain) and says which it got, because a silent fallback to
    the old model is the failure mode most likely to waste a render batch.
    It prints corpus, protocol, CV numbers, normalisation scope, and RAVDESS
    weight, and warns if the extractor label disagrees with the feature
    count (an `egemaps` bundle that is not 88-wide is silent garbage
    downstream, not an error). `CLF_N_CLASSES` and `CLF_CLASSES` are now
    exported so nothing downstream has to hardcode the class count again.

    ### 2. CELL 6b is new: the label-space contract check

    The style lookup is `styles.get(emotion, styles["neutral"])` in five
    places. That default keeps a batch from crashing, but it means **a class
    the model emits and the palette lacks renders as near-white flat neutral,
    silently.** Against `clf_v3`, V17's palette would have done exactly that
    to `frustrated` and `excited` — and `frustrated` is one of IEMOCAP's
    largest classes, so this was not an edge case. CELL 6b now refuses to
    proceed (`STRICT_LABEL_SPACE = True`) when the model can emit a class the
    palette cannot draw, and separately warns about dead styles and dead
    `EXCLAIM_EMOTIONS` entries, which fail inertly rather than loudly.

    ### 3. EMOTION_STYLES gained two entries — and the audit is now stale

    `frustrated` at 20deg rust (angry-adjacent by semantics, separated on
    value and saturation) and `excited` at 315deg magenta (placed in the
    largest remaining hue gap, by discriminability rather than semantics).

    **These two placements are proposed, not audited.** The V13 hue values
    came out of a documented CVD-simulated dE2000 search over *seven*
    classes. Nine styles on the same wheel necessarily lowers worst-case
    pairwise separation, and the figures quoted in CELL 12 (normal 24.0 /
    protan 13.7 / deutan 16.7 / achromatopsia 4.8) **describe the old
    7-class palette and will drop.** Re-run CELL 12d before this feeds a
    study. The tightest pair is `angry`/`frustrated` at 20deg apart, which
    will be worst under protanopia; if that fails your floor, push
    `frustrated` toward 30-35deg and re-audit.

    The cleaner alternative is upstream: `MERGE_EXCITED_INTO_HAPPY=True` in
    `train_emotionsV3` CELL 1 folds excited into happy, returning the
    palette to a size the existing audit covers. Most published IEMOCAP work
    merges those two anyway, on the grounds that annotators confuse them.

    `disgust` is kept in the palette although `clf_v3` will usually not emit
    it — IEMOCAP has too few disgust utterances to clear `MIN_CLASS_COUNT`.
    An unused entry is inert; a missing one is a silent neutral render.

    ### 4. The shout trigger: chance-relative, and deliberately loose again

    Two changes, and the second one matters more than the first.

    **The floors are chance-relative.** `apply_emotion_shout` hardcoded
    `p_top >= 0.40` and `p_second >= 0.30`, reasoned explicitly against
    "7 classes, ~60.6% accuracy". Both halves of that moved: chance went
    0.143 -> 0.125, and **a less accurate model produces flatter
    posteriors**, so a fixed 0.40 floor is a considerably higher bar than it
    was. On clf_v3 that clause fired rarely to never — and nothing printed a
    fire rate, so the only symptom would have been a render that quietly
    stopped shouting. Floors are now multiples of chance, so a label-space
    change cannot silently re-tune the trigger.

    **The emotion set was widened, which is the bigger lever.**
    `EXCLAIM_EMOTIONS` was `{"angry", "happy"}`, written when the model could
    only emit RAVDESS classes. It is now
    `{"angry", "frustrated", "excited", "happy"}`. On IEMOCAP's label space
    "mad" has two words in it — `angry` is flaring, `frustrated` is
    sustained — and **frustrated is one of the largest classes in
    conversational speech**, so the old set was blind to most of the footage
    that is, in plain terms, someone being mad. `excited` is likewise a real
    class now rather than a synonym for happy. This changes how many
    segments are *eligible*, which moves the fire rate far more than any
    threshold does.

    **Sensitivity is a named preset, not a bare number.** `SHOUT_SENSITIVITY`
    in CELL 12 takes `conservative` / `balanced` / `trigger_happy`, shipping
    on **`trigger_happy`**. The previous state of this dial was two constants
    in the function body, and the V17 comment above them described a 0.50
    floor while the code held 0.40 — the comment and the constant had already
    drifted apart, which is exactly how "we tuned this once and lost which
    way" happens. Naming the state makes it recoverable and quotable.

    At 8 classes the presets resolve to `p_top >=` 0.350 / 0.250 / 0.169.
    Note what `trigger_happy` means against this model: 0.169 is ~1.35x
    chance, and clf_v3 sits at ~36% accuracy, so the capitals will land on
    plenty of segments the classifier is close to guessing on. **That is the
    requested trade — expressive over accurate — and it is legitimate for a
    stylistic render, but it changes what the capitals MEAN.** At this
    setting they mark "the model leaned mad/excited here", not "this was
    shouted". Worth one line in any write-up rather than left implicit,
    especially since the acoustic yell detector (CELL 17/18) is still ORed in
    underneath and *does* measure vocal effort.

    `apply_emotion_shout` now prints its own fire rate (`SHOUT: n/m segments
    capitalised`), warns when nothing fires, and warns above 80% — at that
    rate the capitals no longer contrast with anything, which defeats the
    emphasis. Tune against that number rather than by watching renders.

    ### 5. Not changed, deliberately

    - `VERSION_TAG` is now `"v18"`, so filenames move; nothing else about
      naming changed.
    - The confidence curve (`confidence_scale`, CELL 12b) already took
      `n_classes` as an argument and needed no edit — it was written for
      exactly this.
    - No re-tuning of `EMOTION_SELF_BIAS` / `MIN_DWELL_S`. Flatter posteriors
      mean the Viterbi smoother is doing *more* of the work than it was under
      `clf_v2`, and its parameters were swept against a peakier emission
      distribution. That sweep should be redone; it is not done here, and
      pretending otherwise would be the same mistake as the 0.40 threshold.

    ## Decision log — V12

    25. **Motion is isolated by absolute layout, not by attenuation.** V11
        animated `\fscy` on a word inside a single sentence-level `Dialogue`
        event. libass derives a line box from the tallest run in it, so scaling
        one word grew the box and shifted every other word's baseline: the whole
        caption jumped. Attenuating the swell would only have made the problem
        smaller, not fixed it. V12 measures each word with the font file
        fontconfig reports (the same one libass resolves) and emits one `\pos`'d
        event per word, wrapped and stacked by hand. Positioned events are laid
        out independently, so a word's swell is now provably local. Words are
        anchored `\an1` on a shared baseline so growth is upward out of the
        baseline; `\org` is set to the word's own centre so rotations pivot
        correctly rather than swinging from the corner. `LAYOUT_MODE="flow"`
        restores the V11 renderer for A/B. Cost: caption wrapping is now our
        responsibility, and a font substitution would desync measurement from
        rendering — the CELL 14 check exists to catch that.

    26. **Emphasis is directional.** V11 scored `w * |z|`, which treated a word
        that was unusually quiet, low and short exactly like one that was
        unusually loud, high and long. Since unstressed function words are
        reliably *negative* outliers on every one of those features, the metric
        was actively selecting for them — this is why "OR" kept rendering huge.
        V12 clips the emphasis features at zero so only above-average loudness,
        pitch and lengthening earn points. HNR stays two-sided, because creak
        and breathiness are both marked voice qualities.

    27. **Duration is a residual, not a raw value.** Raw duration conflates
        "this word has four syllables" with "the speaker leaned on this word".
        V12 fits duration against syllable count across the clip and scores the
        log residual, so length earns points only when it exceeds what the
        word's own shape predicts. This also makes the feature immune to
        speaking-rate drift.

    28. **Pitch is tracked once, with a speaker-adapted range.** V11 re-ran
        `to_pitch()` inside each ~150ms word, where Praat has almost no context
        and octave errors are common; `f0_range` as `max - min` then turned a
        single halved frame into an enormous apparent range. V12 tracks the
        whole clip with a two-pass floor/ceiling, slices per word, drops frames
        more than 1.6x from the word's median, and reports the 10th-90th
        percentile spread. A word needs four clean voiced frames before its
        pitch features count at all — otherwise they are marked missing and
        excluded from the segment statistics rather than silently read as zero.

    29. **A word-class prior, with a prosodic override.** Determiners,
        prepositions, conjunctions and auxiliaries are damped (conj lowest at
        0.30); nouns, verbs, adjectives and adverbs carry full weight; negation
        is boosted to 1.15 because "not" is very often the stressed word in its
        clause. The damping is deliberately *conditional*: a function word whose
        best positive z clears `CLASS_OVERRIDE_Z` recovers weight linearly and
        is back at parity by `CLASS_OVERRIDE_FULL`. Contrastive stress on a
        conjunction is real ("with milk OR without") and must survive; what is
        filtered is the flat, throwaway case. Implemented as a closed-class
        lexicon rather than a tagger because English function words are a finite
        set, so the lookup is exact and dependency-free; `POS_SOURCE="spacy"`
        swaps in a real tagger where one is installed.

    30. **An emphasis quota.** Even with correct scoring, a line where a third
        of the words are large reads as a line with no emphasis at all.
        `EMPHASIS_QUOTA_FRAC` caps the proportion of a segment allowed to run
        hot; the rest are damped rather than dropped, so the ranking is
        preserved. Font size also gained a gamma so mid-range intensities stay
        near the base size and only genuine peaks read as large.

    31. **ASR quality is a model-size problem first.** V11 ran whisper "base"
        with no language hint. Size dominates WER, so V12 defaults to large-v3
        on CUDA and medium on CPU, pins the language (mis-detection on the first
        30s is a common cause of wholesale garbage), and turns
        `condition_on_previous_text` off, which is the main driver of Whisper's
        runaway repetition loops. Temperature fallback with log-probability and
        compression-ratio gates lets a bad chunk be re-decoded rather than kept,
        and a post-hoc n-gram run detector drops any loop that still gets
        through. Better transcription is not cosmetic here: every prosodic
        feature is measured inside word boundaries the aligner produced, so
        alignment error propagates straight into the emphasis scores.

    ## Decision log — V15 (dynamics, sequence, and compaction)

    42. **The salience budget cannot represent a uniformly loud line.**
        Intensity is derived from a word's SHARE of a fixed 100 points, so a
        segment in which every word is shouted gives every word fair share,
        every share_ratio near 1.0, and every intensity near zero. The angriest
        delivery in a clip rendered at the smallest type in the system. This is
        a boundary of the project's own best idea rather than a coding error:
        making emphasis relative and scarce is exactly what makes it robust on
        ordinary speech, and exactly what blinds it to sustained high arousal.
        A gain would not fix it, because any multiplier on zero is zero. Each
        segment's absolute arousal, z-scored across the clip and squashed
        through a tanh, now sets a floor via
        `intensity = floor + (1 - floor) * intensity_raw`, which is monotone in
        intensity_raw (word ordering inside a line is untouched) and bounded by
        1. Deliberately emotion-agnostic: it fires on loud speech whatever the
        classifier called it, so it cannot bias one class in the evaluation.
        `SEGMENT_AROUSAL_FLOOR = 0.0` reproduces the old behaviour exactly, so
        this is an A/B condition rather than a one-way change.
        `EMOTION_FLOOR_BONUS` exists for per-class hand-weighting and ships at
        zero for all seven; using it must be declared in the write-up, because
        a mute test cannot otherwise separate "the mapping works" from "anger
        was given more ink than the other six".

    43. **Capitals are a standard, not a styling idea.** DCMP's Captioning Key
        already prescribes mixed case for readability, reserves capitals for
        screaming or shouting, and forbids them for general emphasis. Human
        captioners apply this by listening; the contribution here is applying
        it from a measurement. That forces the gate to be vocal effort and NOT
        salience, NOT loudness alone, and NOT the predicted emotion, since
        people shout when happy and can be quietly furious. Loudness cannot
        distinguish a shout from a close microphone or a pushed gain, so the
        detector uses the alpha ratio (energy above 1 kHz over energy below, in
        dB), which is a ratio and therefore gain-invariant. Two test failures
        shaped the final form. A weighted sum still fired on merely-loud
        speech, because a 20 dB jump dominates the sum, so the tilt term must
        now clear its own threshold independently: a conjunction, not a sum. And
        on an evenly delivered clip the robust spread collapses and a 1 dB
        difference becomes a z of 2 — the same degenerate-MAD failure
        `SCALE_REL_FLOOR` exists to catch in the budget — so an absolute dB
        margin over the clip median was added. It matters more here than in the
        budget: a spurious flag does not over-emphasise a word, it tells a deaf
        viewer that someone shouted when nobody did. Capitals also cost
        readability (they remove the ascender/descender cues word-shape
        recognition uses), which is why coverage is capped and reported, and
        why the reading-rate ceiling tightens on recased lines. Whisper's "!"
        is a language-model guess, not an acoustic measurement, so punctuation
        only lowers the threshold and can never cross it alone; letting it
        trigger capitals would quietly turn part of the pipeline into a
        text-to-visual system, which is the thing this project differentiates
        itself from.

    44. **Emotion is a sequence, not seven independent decisions.** Classifying
        each segment alone assumed a speaker's emotion is redrawn from scratch
        every sentence. Five angry sentences are five samples of one state. With
        the classifier at 60.6 percent, independent argmax flips the caption's
        colour on posterior differences (0.34 / 0.29 / 0.36) far smaller than
        the model's own error, so the palette strobes while the delivery has not
        changed. The fix is a first-order HMM over the segment sequence
        (Rabiner 1989): emissions are the posteriors already computed, the
        transition matrix favours staying put, and Viterbi decodes the most
        likely path so a switch must be paid for in likelihood. Two properties
        make this the right tool rather than a blur. It is confidence-weighted
        for free — a sharp posterior overrules its neighbours, a flat one is
        carried by them — and it can only choose labels the classifier actually
        proposed, so no colour can appear that no segment supported. The
        self-transition bias was SWEPT, not guessed: 0.40-0.60 collapses the
        wobble while preserving a confident (p=0.81) one-segment outburst,
        whereas >=0.65 erases that outburst because the path pays two
        transitions to visit it; a genuine sustained change survives at every
        value. Default 0.55, mid-window. The full posterior is now retained per
        segment (a label plus two scalars had already discarded the evidence
        smoothing needs), failed segments emit a FLAT posterior rather than a
        confident "neutral" so they inherit their neighbours, and p_top and
        conf_scale are rewritten from the smoothed distribution so the strength
        on screen refers to the colour on screen. `pred_emotion_raw` and
        `switched_by_smoother` are kept for audit. Known limit: there is no
        diarisation, so a fast two-person exchange will be blended across the
        speaker change — state it, do not hide it.

    45. **Compaction is only safe if it is proved.** 52 cells became 32 by
        merging function-definition and self-check cells while keeping every
        expensive boundary (transcription, alignment, each render) able to fail
        and re-run on its own. The hazard is specific to marimo: `_`-prefixed
        names are cell-local, so two cells may each own a `_cond`, and
        concatenating them puts both in one scope where a nested function
        reading `_cond` at call time sees whichever assignment ran last — a
        silent wrong answer, not a crash. Eight such collisions were detected
        and the colliding locals renamed apart before merging. Equivalence was
        then verified rather than assumed: identical set of 537 public
        definitions, and zero code lines lost or added once the renames are
        undone.

    46. **Environment failures should degrade, not halt.** `FONT_STRICT`
        shipped True, so a font missing from a lab machine raised and blocked
        every downstream cell. A missing font is an environment problem, not a
        defect in the work, so the default is now loud-and-continue, with strict
        reserved for the renders that go into the dissertation and the study.
        Relatedly, `torch.cuda.is_available()` only reports that a driver and a
        device exist; creating a context costs VRAM, so on a full shared card it
        by allocating, requires 2 GiB free, and falls back to CPU with the
        reason printed. The diagnostic cell wraps every CUDA call, because a
        diagnostic that raises is worse than no diagnostic.

    47. **One version string.** Output names were built from three separate
        literals, which is why corpus renders were still labelled v12 two
        versions on and any-video runs were tagged "anyvideo". All output
        naming now derives from `VERSION_TAG`. A stale filename on a stimulus
        is not cosmetic: it is how the wrong render ends up in the study.

    ## Decision log — V13 (the audited palette)

    32. **Palette hues are chosen by measurement, not by eye.** A CIEDE2000
        maximin search under Machado et al. (2009) protanopia/deuteranopia
        simulation, with anger/happy/sad/fear anchored and disgust constrained
        to the bile range (88–112°), moved surprised 27° → 190° and disgust
        40° → 94°. The audit's headline pair (happy/surprised) actually
        measured an acceptable 31.6 ΔE; the genuinely broken pairs were
        surprised/disgust (7.5 ΔE under protanopia) and sad/fearful, i.e. the
        hue-crowding problem and the CVD problem were the same problem. V13's
        worst pair: normal 17.5 → 24.0, protanopia 7.5 → 13.7, deuteranopia
        10.2 → 16.7. The unconstrained optimum (magenta surprise, cyan
        disgust) beats this by only 2.3 ΔE, which is the measured cost of
        keeping disgust semantically bile. The search plateau across 88–112°
        is flat to ~0.7 ΔE; 94° was picked because it favours the deuteranope
        case (the more common deficiency) and sits mid-bile. Cell 12d re-runs
        the audit on every start, so a future palette edit re-derives its own
        numbers.

    33. **Fearful's arousal inversion is fixed in both channels it appeared
        in.** V12 encoded fear at saturation 0.28 / value 0.88, which the
        Valdez & Mehrabian arousal equation (−0.31·V + 0.60·S) scores at
        −0.10 — *below sad* (+0.08), making fear the lowest-arousal chromatic
        class; and its anim group was "soft" at 1.50×, the slowest tempo,
        grouped with sad. Both contradict the source literature (fear is
        +arousal in the PAD octants; Lee et al. put trembling with the
        high-arousal states). V13: (0.70, 0.78) → +0.18, above sad, and a new
        "tremor" group at 0.70× whose words also get a lowered wobble gate —
        the V7 wobble was built for nervous, unsteady words, and fear is the
        class it was built for. The fear-above-sad ordering is now an
        assertion, so the bug cannot silently return. Caveat recorded: the
        equations map colour → felt emotion and are used here in reverse.

    34. **The colour science self-checks or refuses to run.** ΔE2000 follows
        Sharma, Wu & Dalal (2005) and asserts against six of their published
        reference pairs on every start (off-line it matches the
        colour-science reference library to zero error on all 34 pairs and
        400 random ones). CVD uses Machado's precomputed severity-1.0
        matrices — digit-exact against the reference implementation, rows
        summing to 1 so neutrals are preserved *exactly* (asserted; neutral
        drift is the failure that sank the earlier hand-derived tritanopia
        attempt). Collapse checks are ratio-based (protan red/green must fall
        below 0.65× its normal-vision distance), not the absolute cutoff that
        was left unverified before. Tritanopia proper is still not simulated:
        Machado's model treats tritan as a shift-paradigm approximation and
        explicitly refrains from claiming true tritanopia, so the key exists
        with that caveat and stays out of the headline audit. It is also
        ~1000× rarer than red-green deficiency.

    35. **The channel rule is resolved as an explicit, evidenced switch.**
        `CHANNEL_MODE="hue_only"` enforces one-variable-per-channel (Position
        A, matching AffType's single-channels-win result and giving clean
        attribution in an A/B); `"redundant"` keeps font/italic/tempo as
        designed redundancy (Position B). The default is Position B on
        measurement, not taste: under achromatopsia the best minimum pairwise
        ΔE any hue assignment can achieve is ~4.8, with 8/21 pairs under 15 —
        hue alone cannot serve viewers without colour vision, so the extra
        channels are the accessibility path, and the "hue_only" condition
        exists precisely so the evaluation can price what they cost.

    36. **The confidence curve stops overstating the classifier.** The legacy
        formula divided by (0.5 − chance), saturating at p = 0.5, so a 0.43
        posterior rendered at ~90% strength. The calibrated curve spans
        [chance, 1] with a gamma of 1.5: 0.43 → ~0.60. Both prediction paths
        now share one function (cell 12b), removing the duplicated inline
        formula that had to be kept in sync by hand. The write-up reframe:
        styling strength reflects the classifier's *relative preference*, and
        the honest benchmark for its certainty is human agreement on emotion
        labelling (EmoLex micro-averaged κ = 0.29), not the near-ceiling
        agreement of transcription-like tasks.

    37. **The swell is volume-conserving squash-and-stretch.** Every `\fscy`
        keyframe now pairs an inverse `\fscx = 100·(100/fscy)^c` (c = 0.45
        reproduces the audit's example of `\fscy110` with `\fscx≈96`; 1.0 is
        exact area conservation). Lee et al. define squash-and-stretch by the
        volume-conserving property — that is what makes it read as
        deformation rather than scale change. The gesture amplitude fractions
        moved into shared constants so the tagger and the layout cannot
        disagree, and the layout reserves the horizontal room a drop or
        wobble's widened phase occupies (wrapping and advancing on the
        reserved width). Squash applies only in absolute layout, where the
        per-word `\pos` events make the squeeze provably local; the flow path
        is untouched, preserving the V11 A/B condition verbatim.

    38. **Reading rate is enforced, bounded, and reported.** Broadcast caption
        practice works to a characters-per-second ceiling; V13 extends a
        line's end time when its natural duration implies a faster read, but
        only up to the next segment's start — a caption cannot borrow time
        the next one owns — and lines that still exceed the ceiling are
        printed rather than silently sped past. Held pauses add display time
        and the per-word layout changes line geometry, which is exactly why
        this is a check and not an assumption. The 17 cps value is the usual
        band of BBC-style guidance; verify the current BBC/Ofcom documents
        before citing them (the same verification the audit asks for on the
        colour-means-speaker convention).

    39. **Legibility parameters are visible, and substitution is a failure.**
        Outline and shadow were hardcoded in the ASS header; they are
        load-bearing for legibility over arbitrary video (AffType's original
        per-emotion colour scheme died on readability), so they are documented
        dials now. Font checking asks `fc-match` — the same resolver libass
        uses — what each family actually resolves to; with `FONT_STRICT` a
        substitution raises before measurement, and the renderers re-check at
        render time, because in absolute layout a silent substitution desyncs
        measurement from rendering and is only observed after the burn.

    40. **The adjective/adverb tier ships off.** EmoLex's POS breakdown
        (adjectives 68%, adverbs 67% emotion-associated vs nouns 46%)
        motivates a slight prior tier for morphologically marked adj/adv. The
        first heuristic failed its own examples ("table" as t+able, "family"
        as fami+ly); the shipped version uses stem-length and vowel guards
        plus a trap-word exclusion list, and a self-test cell measures it on
        a 95-word labelled sample rather than trusting it (98% recall,
        57/58, with zero false positives across 37 trap words). Suffix-less
        adjectives are out of scope by design and keep the plain content
        prior. Default OFF until an A/B says otherwise, per the audit — and
        per the EmoLex caveat that its annotations are of words out of
        context.

    41. **The saturation floor is a named trade-off.** At intensity 0 every
        style renders at `s × SAT_FLOOR_FRAC`, and the palette audit shows the
        worst-case pair under deuteranopia dropping to ~8 ΔE there: the floor
        trades the intensity channel's dynamic range against baseline
        discriminability. Recorded so the dial is tuned knowingly; raising the
        floor helps CVD viewers at rest and costs intensity contrast.
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

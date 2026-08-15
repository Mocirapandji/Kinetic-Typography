import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # V1.2
    # ---------------------------------------------------------------------
    # plus the three fixes that came out of the BreakingBad(Happy)
    # failure. All three are marked NEW in the code so they can be
    # found, reviewed or reverted individually.
    #
    #   1. LOOP SCREEN, TWICE WRONG (CELL 1b + CELL 8a)
    #      BreakingBad(Happy) failed the batch with
    #      "KeyError: Column not found: word" -- an error two stages
    #      downstream of its cause. The transcript was fine:
    #
    #        "$672,000. All of it? No. Each. Each. Each. $672,000 each.
    #         Each. Yes! Hell yeah!..."
    #
    #      LOOP_WORD_RE strips digits, so "$672,000" tokenises to NOTHING
    #      and the five "each"es collapsed into one unbroken run of 5,
    #      clearing the old ASR_LOOP_MIN_REPEATS of 4. The only segment was
    #      dropped, word_df came back empty, and compute_salience raised on
    #      a missing column. Fixed in two places: the threshold is 6 (real
    #      Whisper loops run 8-20+), and looks_like_loop now also requires
    #      the repeated gram to COVER most of the segment -- a hallucination
    #      loop IS the segment, genuine repetition sits inside other speech.
    #
    #   2. RAPID SEGMENTS RENDERED UNREADABLY (merge_rapid_segments)
    #      With the transcript recovered, twelve of seventeen caption lines
    #      rendered for ~0.16 s each. build_ass_events clamps every line to
    #      the NEXT line's start, which silently overrides MIN_LINE_DURATION
    #      -- so on rapid exclamatory speech ("Each." "Yes!" "Come on!")
    #      each line gets only the gap to its successor. merge_rapid_
    #      segments is the inverse of split_on_long_pauses: it joins
    #      consecutive segments whose gap is below a threshold, bounded by a
    #      character cap so a merged line cannot become its own reading-rate
    #      problem. It runs BEFORE the splitter, which can still cut the
    #      result at a genuinely long pause.
    #
    #   3. VERSION_TAG IS "v19" (CELL 1)
    #      The code has been at this level since the strict/tiered shout work; the tag
    #      still said v18, so every render was labelled with the version
    #      before the one that produced it. Bumping it re-renders the batch
    #      (skip_existing will not find v19 files), which is correct: the
    #      v18 outputs predate the merge fix and should not go in a study.
    #      Set it back to "v18" if you want the old names.
    #
    # ---- INHERITED ----
    #
    #   SHOUT TRIGGER, STRICT AND TIERED — SHOUT_PRIMARY_EMOTIONS (angry)
    #     clears a reachable posterior floor plus a raised-voice check;
    #     SHOUT_EXTREME_EMOTIONS (excited) clears a much higher bar. The
    #     torn/runner-up clause is off, a hard cap limits how much of a clip
    #     may shout, and the acoustic yell detector is vetoable by emotion.
    #
    #   SILENCE AS PUNCTUATION AND AS A LINE BREAK — a measured gap becomes
    #     an ellipsis, and past a longer threshold a new subtitle event, so
    #     the caption reader waits for a withheld word the way the hearing
    #     audience does.
    #
    # ---- INHERITED ----
    #
    #   clf_v3 (IEMOCAP-primary) with provenance printing, and a
    #   label-space contract check that fails loudly when the model can
    #   emit a class the palette cannot draw.
    #
    # ---- INHERITED FROM V1.0/V1.1 ----------------------------------------
    #
    #   PALETTE — hues chosen by CIEDE2000 search under CVD simulation
    #     rather than by eye, self-checked against Sharma et al. (2005).
    #   SEGMENT AROUSAL FLOOR — a segment's absolute arousal sets the floor
    #     the whole line renders at, so a uniformly shouted line no longer
    #     renders like a flat one.
    #   EMOTION SMOOTHING — a first-order HMM over the segment sequence
    #     (currently off; see EMOTION_SMOOTH).
    #   CONFIDENCE — calibrated curve spanning [chance, 1] with a gamma.
    #   CHANNEL RULE — explicit switch, defaulted to designed redundancy on
    #     the achromatopsia measurement.
    #   MOTION — volume-conserving squash-and-stretch.
    #
    # V1.0's three fixes (motion isolation, directional salience, ASR model
    # size) are all carried forward unchanged.

    # CELL 0 — IMPORTS
    import marimo as mo
    import os
    import re
    import random
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
        random,
        re,
        subprocess,
        torch,
        whisperx,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # V1.2: provenance, contracts, and the evaluation batch

    Four strands, and the first two are about the pipeline being able to state
    what it is doing rather than about anything on screen.

    **The model can finally say where it came from.** The bundle path was
    hardcoded and only four keys were read, which meant the file could not
    record the corpus, the fold protocol or the class count it was measured
    under — and that is exactly how the accuracy figure quoted throughout this
    notebook became untraceable. Loading now searches a list of candidates,
    reads twelve provenance keys defensively, and prints the model's own
    pedigree on every run. An older bundle still loads and says so loudly.

    **The palette and the label space are now checked against each other before
    anything renders.** They are set in two different cells by two different
    decisions and nothing verified they agreed. The style lookup falls back to
    neutral by design, so rendering never crashes mid-batch — but that means a
    class the model can emit and the palette lacks renders as flat near-white,
    silently, forever. With the new bundle that would have hit `frustrated`,
    one of the corpus's largest classes. The check is cheap, it runs before any
    render, and it is the difference between "the palette is wrong" and "the
    study used the wrong palette for three weeks".

    **Capitals went back to strict.** The emotion-driven shout trigger had been
    widened to four classes at the loosest sensitivity, and the result was a
    clip that shouted constantly. It is now tiered — anger on the primary bar,
    excitement admitted only on a much higher one — with arousal corroboration,
    a hard cap on what fraction of a clip may be recased, and named
    sensitivity presets so the tuning cannot be lost again.

    **Silence became punctuation, and rapid speech became readable.** Widened
    letter spacing has a ceiling: a wide gap and a very wide gap look nearly
    alike, and neither conveys what a long pause actually does, which is
    withhold the next word. The same measurement now drives an ellipsis on the
    same line at one threshold and a line break at a higher one, so a caption
    reader is not handed the payload during the exact silence the speaker is
    using to withhold it. Separately, `merge_rapid_segments` joins segments too
    close together to read, because the subtitle builder clamps each line to
    the next one's start and silently overrides the minimum duration.

    Three of these came out of a single stimulus, `BreakingBad(Happy)`, failing
    in two different ways.

    **The loop screen deleted a real line.** `LOOP_WORD_RE` strips digits, so
    `$672,000` tokenised to nothing and five "each"es became one unbroken run.
    The only segment was dropped, `word_df` came back empty, and
    `compute_salience` raised `KeyError: 'Column not found: word'` two stages
    later. The threshold is now 6, and the detector additionally requires the
    repeated gram to cover most of the segment — a hallucination loop *is* the
    segment; genuine repetition sits inside other speech.

    **Rapid speech rendered unreadably.** With the transcript recovered, twelve
    of seventeen lines were on screen for about 0.16 seconds each — burned in
    correctly and impossible to read. `build_ass_events` clamps every line to
    the next line's start, which overrides `MIN_LINE_DURATION`, so on
    exclamatory dialogue each caption gets only the gap to its successor.
    `merge_rapid_segments` is the inverse of `split_on_long_pauses`: it joins
    segments that are too close together, bounded by a character cap, and runs
    before the splitter so a genuinely long pause can still cut the result.

    **The version tag was a version behind.** Outputs were labelled `v18` by
    code that had run ahead of it for two revisions.
    """)
    return


@app.cell
def _(os, torch):
    # CELL 1 — CONFIG + DATASET SWITCH
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
    #
    # NEW: one tag drives every output name. The code has been ahead of the tag since the strict/tiered
    # shout work, so every render so far has been labelled with the version
    # BEFORE the one that made it -- the exact failure the paragraph above
    # describes, committed by the file that describes it. Bumping this means
    # run_evaluation_batch's skip_existing will not find the old pairs and
    # will re-render everything, which is correct: those files predate
    # merge_rapid_segments and are not the renders you want in a study.
    VERSION_TAG = "v1_2"

    features_csv = "outputs/features.csv"     # 14-feature cache (fallback model)

    iemocap_csv = "outputs/features_iemocap.csv"
    if not os.path.exists(iemocap_csv):
        iemocap_csv = "/run/media/s5812886/T7 Shield/kinetic_outputs/features_iemocap.csv"

    # ---------- device is PROBED, not just queried --------
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
    return (
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
def _():
    # CELL 1b — ASR QUALITY DIALS  (FIX 3)
    # ---------------------------------------------------------------------
    # V0.7 used "base", which is the second-smallest Whisper checkpoint and is
    # simply not good enough for conversational audio, crosstalk or film
    # dialogue. Model size is by far the biggest lever on WER; everything
    # else here is a second-order guard against known failure modes.
    #
    #   "auto"  -> large-v3 on CUDA, medium on CPU
    #   Force a size by replacing "auto" with "small" | "medium" | "large-v3".
    #   large-v3 is ~3GB and slow on CPU; medium is the sane CPU default.
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
    # NEW: 4 -> 6. BreakingBad(Happy) transcribed as
    #   "$672,000. All of it? No. Each. Each. Each. $672,000 each. Each..."
    # and LOOP_WORD_RE strips digits, so "$672,000" tokenised to NOTHING and
    # the five "each"es became one unbroken run of 5. At a threshold of 4
    # that read as a hallucination, the only segment was dropped, and the
    # clip reached compute_salience with an empty word_df -- surfacing as
    # "KeyError: Column not found: word", two stages downstream of the
    # cause. Genuine Whisper loops run 8-20+ repeats, so 6 still catches
    # them. See looks_like_loop in CELL 8a for the structural half of the
    # fix, which is the part that does not depend on this number.
    ASR_LOOP_MIN_REPEATS = 6
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
def _(call, np, parselmouth):
    # CELL 4 — CLIP-LEVEL 14-FEATURE EXTRACTOR (fallback model + 14-feat path)
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
    # CELL 5 — LOAD 14-FEATURE CACHE (only used by the fallback model)
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
    # CELL 6 — LOAD THE MODEL BUNDLE  (clf_v3 preferred)
    # ---------------------------------------------------------------------
    # NEW. The previous version hardcoded clf_v2.joblib and read exactly 4 keys, which
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
    MODEL_BUNDLE_CANDIDATES = [
        "outputs/clf_v3.joblib",      # NEW: IEMOCAP-primary, RAVDESS support
        "outputs/clf_v2.joblib",      # V1.1 legacy, kept so old runs reproduce
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
            print("  WARNING: this bundle records no provenance (pre-provenance "
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

    print(list(clf_full.classes_))
    return (
        CLF_CLASSES,
        CLF_EXTRACTOR,
        CLF_META,
        CLF_NORMALISED,
        CLF_N_CLASSES,
        clf_feature_cols,
        clf_full,
    )


@app.cell
def _(CONF_CURVE, CONF_GAMMA, np):
    # CELL 12b — CONFIDENCE CURVE  (audit 5.4)
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
    return (confidence_scale,)


@app.cell
def _(
    CLF_EXTRACTOR,
    CLF_NORMALISED,
    call,
    confidence_scale,
    extract_clip_features,
    extract_clip_features_from_sound,
    np,
    opensmile,
    os,
    parselmouth,
    pd,
):
    # CELL 7 — eGeMAPS extraction + per-segment prediction (top-2 for blend)
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
        """V1.1 -- post-Viterbi hygiene pass.

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

        V1.1: `speaker_ids`, if given (a per-segment array aligned with
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

        V1.1: after the path is decoded, enforce_min_dwell runs as a second,
        duration-based pass -- self_bias controls the PER-STEP cost of
        switching, `min_dwell_s` controls the MINIMUM ON-SCREEN duration of
        whatever the path decided. They catch different failure modes:
        self_bias alone can still let a single noisy segment win a close
        call; the dwell pass mops that up afterward without touching
        genuinely sustained runs.

        V1.1: `speaker_ids`, a dict {segment_id: speaker_label} from
        diarization (or None), removes the single biggest source of wrong
        persistence: self_bias and min_dwell_s both assume one continuous
        voice, which is false on fast multi-speaker dialogue -- there, a
        real, isolated angry outburst looks identical to classifier noise,
        and both features fixed above will happily erase it. When speaker
        change is known, the Viterbi transition (and the EMA blend) resets
        to unbiased at every speaker boundary instead of applying self_bias
        uniformly, and enforce_min_dwell refuses to merge a short run
        across a speaker change at all. speaker_ids=None reproduces plain
        V1.1 behaviour exactly (uniform persistence, no boundaries).

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
                  "V1.1. Re-run prediction; returning unsmoothed.")
            out["pred_emotion_raw"] = out["pred_emotion"]
            out["switched_by_smoother"] = False
            return out

        P = out[cols].to_numpy(dtype=float)
        P = np.clip(P, eps, None)
        P = P / P.sum(axis=1, keepdims=True)
        n = len(P)

        # resolve speaker_ids (a {segment_id: label} dict) against
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
            # the reset matrix used at a speaker boundary instead --
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

        # duration-based cleanup pass, independent of self_bias.
        # speaker-aware -- a short run is never merged across a
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
            # shared calibrated curve (cell 12b) — the inline formula
            # here and the one in cell 15 had to be kept in sync by hand
            conf   = confidence_scale(p_top, len(clf.classes_))
            _row = {"segment_id": m["segment_id"], "start": m["start"],
                    "end": m["end"], "pred_emotion": pred, "pred_emotion2": pred2,
                    "p_top": round(p_top, 3), "p_second": round(p_sec, 3),
                    "conf_scale": round(conf, 3), "normalised": bool(do_norm)}
            # keep the WHOLE posterior, not just the top two. Smoothing
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
def _(gc, torch):
    # CELL 7c — VRAM HYGIENE  
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
    return (free_vram,)


@app.cell
def _(
    ASR_BEAM_SIZE,
    ASR_COMPRESSION_RATIO_THRESHOLD,
    ASR_CONDITION_ON_PREV,
    ASR_INITIAL_PROMPT,
    ASR_LOGPROB_THRESHOLD,
    ASR_NO_SPEECH_THRESHOLD,
    ASR_TEMPERATURE_FALLBACK,
    re,
    whisperx,
):
    # CELL 8a — ASR LOADER + LOOP SCREEN  (FIX 3)
    # ---------------------------------------------------------------------
    # whisperx's load_model signature and the set of keys it accepts inside
    # asr_options both drift between releases, so this walks a ladder from
    # the richest call down to the barest one and takes the first that binds.
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

    # NEW: minimum share of a segment the repeated gram must occupy
    # before the segment is called a hallucination loop. See
    # looks_like_loop for why this exists; it is the half of the fix that
    # does NOT depend on tuning a repeat count against one clip.
    ASR_LOOP_MIN_COVERAGE = 0.6

    def looks_like_loop(text, min_repeats=6, min_coverage=ASR_LOOP_MIN_COVERAGE):
        """Whisper's classic failure: one phrase emitted N times in a row.

        NEW -- two changes, both forced by BreakingBad(Happy), which this
        function killed by reading a real line as a hallucination.

        The transcript was:

            "$672,000. All of it? No. Each. Each. Each. $672,000 each.
             Each. Yes! Hell yeah! Hey, come on, baby! Come on! Yes!..."

        LOOP_WORD_RE strips digits, so "$672,000" tokenises to NOTHING and
        the five "each"es collapsed into one unbroken run of 5 -- clearing
        the old threshold of 4. The only segment was dropped, word_df came
        back empty, and compute_salience raised
        "KeyError: Column not found: word" two stages downstream of the
        cause. A dropped stimulus with a misleading error is the expensive
        version of this mistake.

        1. min_repeats 4 -> 6. Genuine Whisper loops run 8-20+ repeats, so
           the screen still catches them with room to spare.

        2. min_coverage, which is the structural fix. A real hallucination
           loop IS the segment -- Whisper emits "Thank you." twelve times
           and nothing else. Genuine repetition is EMBEDDED in other
           speech: here the repeated gram was 5 of 23 tokens, 22% of the
           line. Requiring the run to cover most of the segment separates
           the two cases on a property they actually differ in, rather
           than on a count tuned to whichever clip broke last.

        Set min_coverage=0.0 to get the pure repeat-count behaviour back.
        """
        toks = LOOP_WORD_RE.findall(str(text).lower())
        if len(toks) < min_repeats:
            return False
        # The unit length is derived, not hardcoded. A fixed (1, 2, 3) was
        # sized to the phrases seen so far ("Thank you." at 2, "you" at 1)
        # and cannot catch a 4-word cycle: chop a 4-word unit into 3-word
        # grams and the phase shifts on every repetition, so no run of
        # identical grams ever forms. max_size is the longest unit that
        # could still repeat min_repeats times inside this segment.
        max_size = len(toks) // min_repeats
        for size in range(1, max_size + 1):
            grams = [tuple(toks[i:i + size]) for i in range(0, len(toks) - size + 1, size)]
            run, best = 1, 1
            for a, b in zip(grams, grams[1:]):
                run = run + 1 if a == b else 1
                best = max(best, run)
            # NEW: the run must be long AND must be most of the segment
            if best >= min_repeats and (best * size) / len(toks) >= min_coverage:
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

    return (
        ASR_LOOP_MIN_COVERAGE,
        load_asr_model,
        looks_like_loop,
        screen_segments,
    )


@app.cell
def _(
    ASR_BATCH_SIZE,
    ASR_CHUNK_SIZE,
    ASR_DROP_LOOPS,
    ASR_LANGUAGE,
    ASR_LOOP_MIN_REPEATS,
    ASR_MODEL_SIZE,
    audio_file,
    compute_type,
    device,
    free_vram,
    load_asr_model,
    screen_segments,
    whisperx,
):
    # CELL 8b — WHISPERX TRANSCRIPTION
    # hand torch's cache back to the driver BEFORE CTranslate2 asks
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
    return asr_model, audio, result


@app.cell
def _(
    DIARIZE_ENABLE,
    DIARIZE_MAX_SPEAKERS,
    DIARIZE_MIN_SPEAKERS,
    HF_TOKEN,
    audio,
    audio_file,
    device,
    result,
    whisperx,
):
    # CELL 9 — WORD-LEVEL ALIGNMENT
    align_model, align_metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    aligned = whisperx.align(
        result["segments"], align_model, align_metadata, audio, device,
        return_char_alignments=False,
    )
    print(f"aligned {len(aligned['word_segments'])} words")
    aligned["word_segments"]

    # ---------------------------------------------------------------------
    # 9d — SPEAKER DIARIZATION 
    # ---------------------------------------------------------------------
    # Stamps a "speaker" field onto every segment and word in `aligned`,
    # in place. Downstream, smooth_segment_emotions reads that field to
    # decide where a scene's persistence assumption should NOT apply.
    # Failure here is loud, not fatal -- a missing/invalid token or an
    # unaccepted model licence should not block the rest of the notebook,
    # it should just mean no speaker-aware smoothing this run.
    #
    # marimo: this is deliberately NOT its own cell. It REASSIGNS
    # `aligned`, and marimo forbids two cells defining the same public
    # name -- and giving it a new name would be worse, because
    # assign_word_speakers mutates the dict in place, so CELL 9's object
    # would silently change underneath any cell holding the old name.
    # Alignment and diarization belong to one reactive unit.
    # ---------------------------------------------------------------------
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
    return (aligned,)


@app.cell
def _(ADJ_ADV_TIER, CONTENT_MOD_PRIOR, np, re):
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
                   # (audit 5.2): suffix-marked adjectives/adverbs. The
                   # EmoLex POS breakdown (adjectives 68% / adverbs 67%
                   # emotion-associated vs nouns 46%) supports a SLIGHT tier
                   # above plain content. Applied only when ADJ_ADV_TIER is
                   # on; the actual value comes from CONTENT_MOD_PRIOR.
                   "content_mod": 1.00}

    # ------------------------------------------------------------------
    # suffix heuristic for adjective/adverb detection (audit 5.2).
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
        # promote suffix-marked adjectives/adverbs to their own tier.
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

    return classify_words, count_syllables, looks_adj_adv, normalise_token


@app.cell
def _(looks_adj_adv):
    # CELL 9c — ADJ/ADV HEURISTIC SELF-TEST  (audit 5.2)
    # ---------------------------------------------------------------------
    # The first draft of this heuristic failed its own examples: "table"
    # parsed as t+able and "family" as fami+ly. Measure, don't assume.
    # Scope note: the positive set is DERIVATIONALLY SUFFIXED adj/adv only —
    # suffix-less adjectives (big, sad, good) are out of the heuristic's
    # scope by design and keep the plain content prior, so they are not
    # counted as misses here.
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
    return


@app.cell
def _(ASR_LOOP_MIN_COVERAGE, ASR_LOOP_MIN_REPEATS, looks_like_loop):
    # CELL 9e — LOOP SCREEN SELF-TEST  (new)
    # ---------------------------------------------------------------------
    # looks_like_loop deleted a real stimulus and surfaced it as a KeyError
    # in a different function two stages later. A screen that can silently
    # remove an entire clip from a study needs its own test, on both the
    # case it exists to catch and the case that broke it.
    _LOOP_SHOULD_CATCH = [
        "Thank you. Thank you. Thank you. Thank you. Thank you. Thank you. "
        "Thank you. Thank you.",
        "you you you you you you you you you you",
        "Subscribe to my channel. Subscribe to my channel. Subscribe to my "
        "channel. Subscribe to my channel. Subscribe to my channel. "
        "Subscribe to my channel.",
    ]
    _LOOP_SHOULD_KEEP = [
        # the line this whole fix exists for
        "$672,000. All of it? No. Each. Each. Each. $672,000 each. Each. "
        "Yes! Hell yeah! Hey, come on, baby! Come on! Yes! Come on! Yes! Yes!",
        # ordinary emphatic repetition
        "No, no, no, no, I told you already, that is not what happened.",
        "Run! Run! Run! Get to the car before they see us!",
        "",
    ]
    _loop_missed = [t for t in _LOOP_SHOULD_CATCH
                    if not looks_like_loop(t, ASR_LOOP_MIN_REPEATS)]
    _loop_killed = [t for t in _LOOP_SHOULD_KEEP
                    if looks_like_loop(t, ASR_LOOP_MIN_REPEATS)]
    print(f"loop screen (min_repeats={ASR_LOOP_MIN_REPEATS}, "
          f"min_coverage={ASR_LOOP_MIN_COVERAGE}): "
          f"caught {len(_LOOP_SHOULD_CATCH) - len(_loop_missed)}/"
          f"{len(_LOOP_SHOULD_CATCH)} hallucinations, "
          f"kept {len(_LOOP_SHOULD_KEEP) - len(_loop_killed)}/"
          f"{len(_LOOP_SHOULD_KEEP)} real lines")
    assert not _loop_killed, (
        "loop screen would delete a REAL transcript -- this is the "
        f"BreakingBad(Happy) failure returning: {_loop_killed}")
    assert not _loop_missed, (
        f"loop screen no longer catches a known hallucination: {_loop_missed}")
    return


@app.cell
def _(call, np, parselmouth, pd):
    # CELL 10 — PER-WORD PROSODY EXTRACTOR — ROBUST  (FIX 2, part 2)
    # ---------------------------------------------------------------------
    # Four changes from V0.7, all aimed at the false-positive problem:
    #
    #  1. Pitch is tracked ONCE over the whole clip with a speaker-adapted
    #     floor/ceiling, then sliced per word. V0.7 re-ran to_pitch() on each
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
    MIN_VOICED_FRAMES = 4
    OCTAVE_TOL = 1.6          # drop frames >1.6x or <1/1.6x the word median
    PITCH_TIME_STEP = 0.005

    def estimate_pitch_range(snd):
        """Two-pass floor/ceiling (De Looze & Hirst). A fixed 75-600Hz window
        badly over-searches for one speaker and truncates another."""
        try:
            p0 = snd.to_pitch(time_step=0.01, pitch_floor=60.0, pitch_ceiling=600.0)
            v = p0.selected_array["frequency"]
            v = v[v > 0]
            if len(v) < 10:
                return 75.0, 500.0
            q15, q65 = np.percentile(v, 15), np.percentile(v, 65)
            floor = max(50.0, 0.72 * q15)
            ceil  = min(700.0, 1.9 * q65)
            if ceil - floor < 60.0:
                floor, ceil = max(50.0, floor - 30.0), floor + 90.0
            return float(floor), float(ceil)
        except Exception:
            return 75.0, 500.0

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

            # ---- spectral tilt (alpha ratio) ----------------------
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
            # NEW: an empty frame here is what turned a dropped segment
            # into "KeyError: Column not found: word" inside
            # compute_salience, two stages away from anything that could
            # explain it. The columns are declared so the failure surfaces
            # where it belongs -- as zero words, not as a missing column.
            return pd.DataFrame(columns=[
                "word", "start", "end", "duration", "pause_after", "syllables",
                "n_voiced", "f0_mean", "f0_range", "f0_slope", "rms",
                "intensity_db", "hnr", "alpha_ratio", "dur_expected",
                "dur_resid"])

        # DURATION RESIDUAL  (the other half of the "OR" fix)
        # -----------------------------------------------------------------
        # V0.7 fed raw duration into salience, which conflates two different
        # things: a word can be long because it has four syllables, or long
        # because the speaker leaned on it. Only the second is emphasis.
        # Fit duration ~ a + b*syllables across the clip and score the
        # RESIDUAL. "Extraordinarily" no longer earns points for being long,
        # and a drawn-out "or" still can.
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

    return PITCH_TIME_STEP, estimate_pitch_range, extract_word_features


@app.cell
def _(aligned, audio_file, count_syllables, extract_word_features):
    # CELL 11 — RUN PER-WORD EXTRACTION
    word_df = extract_word_features(audio_file, aligned["word_segments"],
                                    count_syllables)
    word_df
    return (word_df,)


@app.cell
def _(
    PITCH_TIME_STEP,
    audio_file,
    estimate_pitch_range,
    np,
    os,
    parselmouth,
    pd,
    word_df,
):
    # CELL 11b — EXPORT FIGURE DATA (dissertation prosody figure)
    # ---------------------------------------------------------------------
    # Writes to outputs/figure_data/, then zips it into one file:
    #   clip.wav            copy of the source audio for this run
    #   word_timestamps.csv word_df exactly as CELL 11 produced it
    #   pitch_contour.csv   the CONTINUOUS pitch track (time, f0) this
    #                       pipeline tracks ONCE across the whole clip
    #                       (CELL 10) before slicing it per word -- the
    #                       real series, not per-word aggregates, so the
    #                       figure can draw one unbroken contour.
    import shutil

    FIGDATA_DIR = "outputs/figure_data"
    os.makedirs(FIGDATA_DIR, exist_ok=True)

    # 1. the audio itself
    _fig_wav = f"{FIGDATA_DIR}/clip.wav"
    shutil.copy(audio_file, _fig_wav)
    print(f"wrote {_fig_wav}  (source: {audio_file})")

    # 2. word-level timestamps, exactly as extracted
    _fig_words_csv = f"{FIGDATA_DIR}/word_timestamps.csv"
    word_df.to_csv(_fig_words_csv, index=False)
    print(f"wrote {_fig_words_csv}  ({len(word_df)} words)")

    # 3. the continuous pitch contour -- same two-pass floor/ceiling method
    #    CELL 10 uses, so this is the real contour, not a re-derivation.
    _snd_fig = parselmouth.Sound(audio_file)
    _floor_fig, _ceil_fig = estimate_pitch_range(_snd_fig)
    _pitch_fig = _snd_fig.to_pitch(time_step=PITCH_TIME_STEP,
                                   pitch_floor=_floor_fig, pitch_ceiling=_ceil_fig)
    _t_fig = np.asarray(_pitch_fig.xs(), dtype=float)
    _f_fig = np.asarray(_pitch_fig.selected_array["frequency"], dtype=float)
    # f0=0.0 marks an unvoiced frame -- kept, not dropped, so the figure can
    # show real gaps in the contour rather than a falsely continuous line
    pitch_contour_df = pd.DataFrame({"time": np.round(_t_fig, 4),
                                     "f0": np.round(_f_fig, 2)})
    _fig_pitch_csv = f"{FIGDATA_DIR}/pitch_contour.csv"
    pitch_contour_df.to_csv(_fig_pitch_csv, index=False)
    print(f"wrote {_fig_pitch_csv}  ({len(pitch_contour_df)} frames, "
          f"floor={_floor_fig:.0f}Hz ceiling={_ceil_fig:.0f}Hz)")

    # 4. bundle into one file so there's just one thing to upload
    _zip_path = shutil.make_archive(f"{FIGDATA_DIR}_bundle", "zip", FIGDATA_DIR)
    print(f"\nzipped -> {_zip_path}\nupload this one file back to Claude.")
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part C: The expressive budget + all styling dials
    """)
    return


@app.cell
def _(os):
    # CELL 12 — TUNABLE PARAMETERS
    # ----- budget -----
    # intensity_db replaces rms (log domain), dur_resid replaces duration
    # (syllable-corrected). Weights favour loudness and lengthening, which
    # are the two most reliable acoustic correlates of English stress.
    SALIENCE_WEIGHTS = {"f0_mean": 0.9, "f0_range": 0.7,
                        "intensity_db": 1.2, "dur_resid": 1.0, "hnr": 0.3}

    # THE HEADLINE FIX. Features here score only when the word is ABOVE the
    # segment norm. V0.7 used |z| on everything, so being unusually quiet and
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

    # ----- SEGMENT AROUSAL FLOOR --------------------------------
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
    SEGMENT_AROUSAL_FLOOR = 0.45     # 0.0 = no floor
    AROUSAL_FEATURES = {"intensity_db": 1.0, "f0_mean": 0.6}
    AROUSAL_SPREAD = 1.5             # z at which the floor is ~76% of max

    # ----- PER-EMOTION FLOOR BONUS (declare this if you use it) --
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

    # ----- SHOUTING DETECTION AND CASE --------------------------
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
    # ----- TEMPORAL SMOOTHING OF EMOTION -------------------------
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
    # switched off. On fast multi-speaker dialogue (12 Angry Men)
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
    # raised 0.55 -> 0.68. The swept "usable window" above (0.40-0.65)
    # was chosen to protect a confident one-segment outburst from being
    # erased -- but 0.55 sits close to a coin flip on the stay/switch
    # decision, and with a 60%-accurate classifier that is not enough
    # persistence to stop ordinary noise from flipping colour. Pushing
    # self_bias alone past ~0.65 starts erasing genuine short outbursts
    # (per the sweep above), so it is paired with here with a
    # separate, duration-based guard (MIN_DWELL_S) that catches short
    # noisy runs self_bias would otherwise let through.
    EMOTION_SELF_BIAS = 0.68     # P(stay in the same emotion) per segment
    # minimum on-screen duration for a smoothed colour run.
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

    # ----- SPEAKER DIARIZATION -----------------------------------
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
    # behaviour (speaker_ids=None everywhere -> no resets, same as before).
    DIARIZE_ENABLE = True
    HF_TOKEN = os.environ.get("HF_TOKEN")
    DIARIZE_MIN_SPEAKERS = None   # int, or None to let pyannote decide
    DIARIZE_MAX_SPEAKERS = None

    YELL_DETECT = True
    YELL_CASE = "upper"        # "upper" | "off" (detect but do not recase)
    # ----- normalisation reference  ------------------------------
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

    # ----- EMOTION-DRIVEN SHOUT MARKERS ------------------------
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
    # acoustic yelling (the pre-existing acoustic-only behaviour).
    EXCLAIM_ENABLE = True

    # NEW — BACK TO STRICT, AND TIERED (reverts the earlier widening)
    # ---------------------------------------------------------------------
    # The previous pass shipped EXCLAIM_EMOTIONS = {angry, frustrated, excited, happy} at
    # SHOUT_SENSITIVITY="trigger_happy" (p_top >= 1.35x chance, ~0.169 on 8
    # classes). That is the loosest state this dial has ever been in: four
    # of eight classes eligible, on a bar barely above chance. It was asked
    # for, it did exactly what its comment said it would, and it is now
    # being reverted -- deliberately, not by accident. The note below
    # ("if you want the capitals back to their old rarity, shrink this set
    # first and touch SHOUT_SENSITIVITY second") is the instruction being
    # followed here; both have been done.
    #
    # The new ask is narrower than the earlier state, so the emotion set is
    # no longer FLAT. One bar for every eligible class cannot express "caps
    # for anger, but for excitement only when it is overwhelming", so the
    # set is split into two tiers that clear different bars:
    #
    #   SHOUT_PRIMARY_EMOTIONS   what capitals are actually for. Strict
    #                            posterior floor + a raised-voice check.
    #   SHOUT_EXTREME_EMOTIONS   admitted only on the "super super" bar: a
    #                            much higher posterior AND clearly raised
    #                            vocal arousal. This is what stops
    #                            `excited` becoming the new `happy`.
    #
    # `happy` is out entirely: a happy line is not a shouted line, and it
    # was the single biggest source of stray capitals in the previous pass.
    #
    # `frustrated` is out too, and that one is worth a sentence because it
    # is not obvious. It is IEMOCAP's SUSTAINED-mad class -- the quiet,
    # grinding kind -- and one of the largest classes in the corpus, so
    # including it is most of what made the previous pass shout constantly. Someone
    # quietly seething is the exact case DCMP's rule says must NOT be in
    # capitals. If you disagree, move the string into
    # SHOUT_PRIMARY_EMOTIONS; it is a one-line change and the fire rate
    # printed by apply_emotion_shout will tell you what it cost.
    SHOUT_PRIMARY_EMOTIONS = {"angry"}
    SHOUT_EXTREME_EMOTIONS = {"excited"}
    # Kept as the union so the CELL 6b label-space contract check and every
    # existing call site keep working unchanged -- they only ever needed to
    # know WHICH classes are eligible, not which tier they sit in.
    EXCLAIM_EMOTIONS = SHOUT_PRIMARY_EMOTIONS | SHOUT_EXTREME_EMOTIONS
    EXCLAIM_MIN_MARKS = 1       # marks on the mildest qualifying segment
    EXCLAIM_MAX_MARKS = 3       # marks on the most intense one

    # NEW — SHOUT SENSITIVITY, AS A NAMED PRESET
    # ---------------------------------------------------------------------
    # This dial was previously two bare numbers (0.40 / 0.30) sitting in the
    # body of apply_emotion_shout, which is how "we tuned this once and then
    # lost which way we tuned it" happens. The comment above them even
    # described a 0.50 floor while the code held 0.40 -- the comment and the
    # constant had already drifted apart. Naming the state makes it
    # recoverable: you can say "we shipped trigger_happy" in a write-up and
    # the notebook can prove what that meant.
    #
    # Floors are multiples of CHANCE (1/n_classes), not absolute posteriors,
    # so a label-space change does not silently re-tune the trigger. See
    # apply_emotion_shout for why absolute floors broke on the model swap.
    #
    #   conservative  reproduces the earlier 0.40/0.30 at 7 classes.
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
    # NEW: "strict" is the new default and the new entry in the table.
    # It sits between balanced and conservative on the posterior floor,
    # because on its own a posterior floor is the WRONG place to buy
    # strictness on this model: conservative (0.350 on 8 classes) is above
    # what clf_v3 usually produces even when it is right, so it does not
    # make the capitals rarer, it makes them impossible. That is the "off
    # by accident" state the table already warns about, and shipping it
    # would look like a working feature that never fires.
    #
    # So strictness is bought in three places instead of one, and each is
    # a different kind of evidence:
    #   1. a reachable-but-real posterior floor          (here)
    #   2. a raised-voice check on the AUDIO             (SHOUT_*_AROUSAL_Z)
    #   3. a hard cap on how much of a clip may shout    (SHOUT_MAX_FRAC)
    # A segment has to pass all three. Any one of them alone is gameable
    # by an unlucky clip; together they are not.
    SHOUT_SENSITIVITY = "strict"  # "strict"|"conservative"|"balanced"|"trigger_happy"

    SHOUT_PRESETS = {
        "strict":        {"p_floor_mult": 2.40, "p_second_mult": 2.40},
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

    # ----- NEW: the "super super" bar for SHOUT_EXTREME_EMOTIONS -------
    # A multiple of chance, like the others. At 8 classes this is 0.400 --
    # above clf_v3's usual ceiling for a merely-correct call, which is the
    # point: `excited` should only reach capitals when the classifier is
    # unusually sure AND (below) the voice backs it up. If excited never
    # fires on your footage, that is the intended failure direction.
    SHOUT_EXTREME_P_MULT = 3.20

    # ----- NEW: raised-voice corroboration -----------------------------
    # The posterior says WHICH emotion. It says nothing about whether the
    # line was delivered with any force, and that is half of what capitals
    # claim. segment_arousal_floor already computes exactly the missing
    # quantity -- each segment's loudness and pitch, robust-z-scored ACROSS
    # the clip -- so this reuses it rather than measuring anything new.
    #
    # Stated in z (robust sigma above the clip's own median segment) rather
    # than in arousal_floor units, because arousal_floor is a tanh-squashed
    # 0..floor_max number whose midpoint is an AVERAGE segment, not a quiet
    # one -- a threshold written in those units reads as if it means
    # something absolute when it does not. apply_emotion_shout inverts the
    # squash to get back to z. Same convention as YELL_TILT_MIN_Z.
    #
    #   0.35  a bit louder/higher than this speaker's own median segment
    #   0.90  clearly raised -- near the top of an ordinary clip's spread
    #
    # Degrades honestly: on a clip with fewer than 3 segments the arousal
    # floor is all-zero by design (nothing to compare against), and rather
    # than block every segment forever the gate switches itself off and
    # says so.
    SHOUT_AROUSAL_GATE = True
    SHOUT_MIN_AROUSAL_Z = 0.35          # primary tier (angry)
    SHOUT_EXTREME_MIN_AROUSAL_Z = 0.90  # extreme tier (excited)

    # ----- NEW: the torn / "close to mad" clause, now OFF --------------
    # The earlier comment: "At trigger_happy the floor is barely above
    # chance, so this clause is doing most of the widening." It let a
    # segment the model called something else entirely be capitalised on
    # the strength of a runner-up. That is a reasonable thing to want for
    # COLOUR (a blend degrades gracefully); it is not reasonable for
    # capitals, which are binary and which a viewer reads as a claim that
    # someone raised their voice. Off by default. When re-enabled it now
    # runs on the extreme bar, not the primary one.
    SHOUT_ALLOW_TORN = False

    # ----- NEW: hard cap on how much of a clip may shout ---------------
    # Structural, not tuning. YELL_MAX_FRAC already does this for the
    # acoustic detector and the reasoning is identical: capitals only carry
    # meaning while they stay rare, so if the trigger wants more than this
    # share of the clip it is miscalibrated for that footage and only the
    # strongest segments are kept. This is the gate that holds when the
    # other two are defeated by an unusual clip -- an argument scene where
    # every segment is genuinely angry and genuinely loud would otherwise
    # render entirely in capitals, which reads as no emphasis at all.
    # Ranking is by posterior, so what survives is what the model was
    # surest about.
    SHOUT_MAX_FRAC = 0.20

    # ----- NEW: does the ACOUSTIC yell detector get an emotion veto? ---
    # Read the note at the top of this section first, because this
    # setting contradicts it on purpose. detect_yelling is deliberately
    # emotion-blind -- "people shout when happy and are quietly furious" --
    # and as a MEASUREMENT that is the correct design. But its output is
    # OR'd into the same capitals, so leaving it unrestricted means a
    # delighted yell still renders as caps no matter how narrow
    # EXCLAIM_EMOTIONS gets, and "caps only when angry" would be false.
    #
    # True  = a segment must ALSO land in EXCLAIM_EMOTIONS for an acoustic
    #         yell to reach capitals. is_yell stays set either way, so the
    #         audit numbers and yell_report still describe what the
    #         detector actually measured -- only the recasing is vetoed.
    # False = the earlier behaviour: measured vocal effort is sufficient.
    #
    # Worth declaring in a write-up: at True the capitals are keyed on
    # "angry AND loud", not on "loud".
    YELL_EMOTION_GATE = True

    print(f"shout sensitivity: {SHOUT_SENSITIVITY} "
          f"(p_floor={SHOUT_P_FLOOR_MULT}x chance, "
          f"p_second={SHOUT_P_SECOND_MULT}x chance) | "
          f"primary={sorted(SHOUT_PRIMARY_EMOTIONS)} "
          f"extreme={sorted(SHOUT_EXTREME_EMOTIONS)} "
          f"@{SHOUT_EXTREME_P_MULT}x | arousal gate "
          f"{'on' if SHOUT_AROUSAL_GATE else 'off'} "
          f"(z>={SHOUT_MIN_AROUSAL_Z}/{SHOUT_EXTREME_MIN_AROUSAL_Z}) | "
          f"cap {SHOUT_MAX_FRAC:.0%} of segments")

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
    # "tremor": a fast group for fear. Lee et al. (2002) put trembling
    # with the HIGH-arousal states; it had fearful in "soft" at 1.50x, the
    # slowest setting, grouped with sad — the same arousal inversion as the
    # colour bug fixed in EMOTION_STYLES below (audit 2.4).
    MOTION_TEMPO = {"pop": 0.80, "soft": 1.50, "flat": 1.00, "tremor": 0.70}
    # words in a "tremor" segment wobble more readily: their wobble gate is
    # WOBBLE_RANGE_HZ * this factor (the V7 wobble was built for exactly
    # these nervous, unsteady words — fear is the class it was made for)
    TREMOR_WOBBLE_FACTOR = 0.6

    # ============ FIX 1: LAYOUT MODE ================================
    # "absolute" — measure the text, emit one \pos'd Dialogue per word.
    #              A word's \fscy swell is then purely local: nothing else
    #              on the line moves. This is the fix for "the whole text
    #              jumps".
    # "flow"     — the V0.7 behaviour: one Dialogue per sentence, libass
    #              flows it. Kept for A/B comparison.
    LAYOUT_MODE = "absolute"
    LAYOUT_MARGIN_X = 60        # px from each edge before wrapping
    LAYOUT_MARGIN_V = 60        # px from the bottom to the last baseline
    LAYOUT_LINE_GAP = 0.22      # extra leading, as a fraction of row height
    LAYOUT_SPACE_SCALE = 1.0    # multiplier on the inter-word space
    MOTION_ANCHOR = "baseline"  # "baseline" (grows upward) | "center"

    # ----- CHANNEL 1: SATURATION = INTENSITY -----
    SATURATION_INTENSITY = True
    SAT_FLOOR_FRAC = 0.5

    # ----- CHANNEL 2: UNCERTAINTY HUE BLEND -----
    BLEND_MODE = "blend"        # "off" | "blend" | "gradient"
    # tightened 0.20 -> 0.08. Blending between the top-2 emotion
    # colours now only fires when the classifier is genuinely close to a
    # coin flip, not merely "somewhat unsure" -- most segments now render
    # as one clean, decisive colour.
    BLEND_MARGIN = 0.08
    # lowered 0.30 -> 0.0. Every word inside an ambiguous segment now
    # gets the SAME blend ratio (p2 / (p1 + p2)), instead of each word's
    # own emphasis intensity nudging it independently -- that per-word
    # nudging was what produced a visible colour gradient across a single
    # sentence even when the segment's classification never changed.
    BLEND_PERWORD_SWING = 0.0

    # ----- CHANNEL 3: LETTER SPACING = CALM -----
    TRACKING_CALM = True
    CALM_SPACING_MAX = 6.0

    # ----- CHANNEL 4: HELD SPACE = SILENCE -----
    PAUSE_HOLD = True
    PAUSE_HOLD_THRESH = 0.40
    PAUSE_HOLD_FULL = 1.20
    PAUSE_HOLD_MAX_FSP = 40.0

    # ----- NEW CHANNEL 4a: MERGE SEGMENTS TOO SHORT TO READ ----------
    # The inverse of the split below, and it exists because of a failure
    # that only shows up on rapid exclamatory speech.
    #
    # build_ass_events clamps every caption to the NEXT caption's start:
    #
    #     if si < len(seg_ids) - 1:
    #         e0 = min(e0, seg_starts[seg_ids[si + 1]])
    #
    # That clamp is load-bearing -- two absolute-layout lines at the same
    # screen position would overlap -- but it OVERRIDES MIN_LINE_DURATION.
    # So when Whisper segments faster than one line per second, every
    # MIN_LINE_DURATION in the file is dead and each caption gets only the
    # gap to its successor.
    #
    # On BreakingBad(Happy) ("Each." "Yes!" "Come on!" "Yes!") that meant
    # 16 segments across ~35 s, twelve of which rendered for about 0.16
    # seconds each -- burned in correctly, laid out correctly, and
    # impossible to read. The reading-rate report was the only sign, and
    # it reads as a formatting complaint rather than as "nobody can see
    # these".
    #
    # merge_rapid_segments joins consecutive segments whose gap is under
    # MERGE_MAX_GAP_S, bounded by MERGE_MAX_CHARS so a merged line cannot
    # become its own reading-rate problem. It runs BEFORE
    # split_on_long_pauses, which can still cut the merged result at a
    # genuinely long silence -- merge first, split second, never the
    # reverse, or the splitter's line breaks get merged straight back.
    #
    # WHY 0.45 s -- it sits above the ~0.2 s inter-word gaps of fluent
    # connected speech (so it does not glue together lines a speaker
    # genuinely separated) and below PAUSE_DOTS_MIN_S = 0.28... which it
    # does NOT, and that overlap is deliberate: a gap between 0.28 and
    # 0.45 gets an ellipsis AND is merged, so the beat is marked in the
    # text while the two halves share one readable caption. Above 0.45 the
    # segments stay separate; above 0.90 the splitter cuts them anyway.
    #
    # MERGE_MAX_CHARS = 42 is one comfortable subtitle line at this font
    # size. Raise it and merged lines wrap to two rows; lower it and rapid
    # speech stays fragmented.
    MERGE_RAPID_ENABLE = True
    MERGE_MAX_GAP_S = 0.45      # join across gaps below this
    MERGE_MAX_CHARS = 42        # but never past this many characters
    MERGE_MIN_LINE_S = 1.0      # reported only; the real floor is the
                                # clamp in build_ass_events

    # ----- NEW CHANNEL 4b: SILENCE AS PUNCTUATION AND AS A LINE BREAK ---
    # Channel 4 above renders a pause as extra letter-spacing. That works
    # for a beat inside a phrase, but it has a ceiling: a wide gap and a
    # very wide gap look nearly the same on screen, and neither reads as
    # the specific thing a long menacing pause is doing, which is WITHHOLD-
    # ING the next word. "Who should I kill ... You" arriving as one line
    # hands the viewer the answer before the character gives it. The
    # hearing audience waits; the caption reader does not.
    #
    # So the same measurement (the gap between two words) now drives two
    # different renderings, on two thresholds:
    #
    #   gap >= PAUSE_DOTS_MIN_S  -> ellipsis, SAME line.  "Good... job."
    #   gap >= PAUSE_SPLIT_MIN_S -> ellipsis, and the rest of the sentence
    #                               becomes its own subtitle event.
    #                               "Who should I kill...?" / "You!"
    #
    # The split is the honest option for a long gap: the first line is on
    # screen for the whole silence with its dots showing, and the payload
    # word appears when it is actually spoken. It preserves the timing the
    # audio has instead of flattening it into one simultaneous line.
    #
    # WHY 0.28 / 0.90 -- these are perceptual, not arbitrary. Inter-word
    # gaps inside fluent connected speech are mostly under ~0.2 s, so 0.28
    # is comfortably outside normal articulation and marks a deliberate
    # beat. Around 0.5-1.0 s a silence stops reading as phrasing and starts
    # reading as a held pause, which is also roughly where a caption line
    # can be cut without the two halves feeling severed. 0.90 keeps the
    # split for gaps that are unambiguously dramatic. Both are dials: lower
    # PAUSE_SPLIT_MIN_S for a more staccato, comic-timing feel, raise it if
    # captions start flickering line to line on ordinary speech.
    PAUSE_SPLIT_ENABLE = True
    PAUSE_DOTS_MIN_S = 0.28     # gap at/above this -> "..." on the same line
    PAUSE_SPLIT_MIN_S = 0.90    # gap at/above this -> new subtitle event too
    PAUSE_DOTS = "..."          # three periods, not U+2026: libass renders
                                # the single glyph inconsistently across the
                                # font families in EMOTION_STYLES, and the
                                # layout measures what it draws
    # Hold the first half on screen through the silence, so the dots are
    # visible during the pause rather than vanishing into a blank frame
    # HOLD_MAX_TAIL after the last word. The line still ends exactly when
    # the next one begins -- build_ass_events clamps it -- so this can
    # never overlap the payload.
    PAUSE_SPLIT_HOLD = True
    # A word that got dots does not also need the wide \fsp gap after it:
    # that would be the same pause billed twice, and the gap pushes the
    # dots away from the word they belong to. Channel 4 keeps the pauses
    # BELOW PAUSE_DOTS_MIN_S, where it is the only signal available.
    PAUSE_DOTS_REPLACE_GAP = True
    # Ellipsis is skipped where the line already ends in "!" -- the shout
    # marks from apply_emotion_shout own that ending, and "OUT!!!..." is
    # not a thing anyone writes. The SPLIT still happens; only the dots
    # are suppressed.
    PAUSE_DOTS_SKIP_ON_BANG = True

    # ----- CHANNEL RULE, made explicit (audit Part 1) --------------
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

    # ----- CONFIDENCE CURVE (audit 5.4) ----------------------------
    # "legacy" divided by (0.5 - chance): saturated at p=0.5, so a 0.43
    # posterior rendered at ~90% strength — more certainty than either the
    # classifier or human annotators possess (EmoLex Table 8: micro-avg
    # kappa 0.29 on this task). "calibrated" spans the full [chance, 1]
    # range with a gamma; 0.43 now lands at ~60%. Monotone either way.
    CONF_CURVE = "calibrated"        # "calibrated" | "legacy"
    CONF_GAMMA = 1.5

    # ----- SQUASH AND STRETCH (audit 4.3) --------------------------
    # Pair every \fscy with an inverse \fscx so the swell conserves
    # apparent area and reads as deformation, not scale change (Lee et al.
    # 2002 after Lasseter/Thomas & Johnston). 1.0 = exact area
    # conservation; 0.45 reproduces the audit's example pairing
    # (\fscy110 with \fscx~96). Only active in absolute layout, where the
    # per-word \pos events make the horizontal squeeze provably local; the
    # layout reserves the widening a drop/wobble needs (see 13c).
    SQUASH_STRETCH = True
    SQUASH_CONSERVATION = 0.45

    # ----- READING RATE (audit Part 8 item 3) ----------------------
    # Broadcast caption practice works to a characters-per-second ceiling
    # (BBC guidance is commonly summarised around 160-180 wpm; ~15-20 cps
    # is the usual band — verify the current BBC/Ofcom documents before
    # citing, per audit 2.9). When a line's natural duration implies a
    # faster rate, its end time is extended up to the next segment's start;
    # lines that still exceed the ceiling are reported loudly.
    READING_RATE_ENFORCE = True
    READING_RATE_MAX_CPS = 17.0

    # ----- CONTRAST OVER ARBITRARY VIDEO (audit Part 8 item 2) -----
    # These were hardcoded in the ASS header; they are load-bearing for
    # legibility (AffType's original per-emotion colour scheme died on
    # readability) so they are dials now. Outline/shadow in px at PlayRes.
    CAPTION_OUTLINE_PX = 3
    CAPTION_SHADOW_PX = 1
    CAPTION_OUTLINE_COLOUR = "&H00000000"   # black outline (ASS &HAABBGGRR)
    CAPTION_BACK_COLOUR = "&H64000000"      # translucent shadow

    # ----- FONT SUBSTITUTION (audit Part 8 item 8) -----------------
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

    # ----- ADJ/ADV PRIOR TIER (audit 5.2) — default OFF ------------
    # EmoLex Table 4: adjectives 68% / adverbs 67% associated with at
    # least one emotion vs nouns 46%, so suffix-marked adj/adv get a
    # slight boost over other content words. Heuristic (suffix + guards,
    # measured in the self-test cell); OFF until A/B'd, per the audit.
    ADJ_ADV_TIER = False
    CONTENT_MOD_PRIOR = 1.08

    # ----- EmoLex lexicon for stimulus screening (audit 7.1) -------
    # Download "NRC Word-Emotion Association Lexicon" from Saif Mohammad's
    # NRC page (free for research) and point this at the word-level file.
    EMOLEX_PATH = "resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

    # ----- COLOUR channel base per emotion (audited palette) ----------------
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
    #     fearful=violet 252deg kept BUT s/v corrected: the earlier (0.28, 0.88)
    #     gave Valdez & Mehrabian arousal -0.10, BELOW sad (+0.08) — fear
    #     was the lowest-arousal chromatic class. (0.70, 0.78) gives +0.18.
    #     Note purple is Jonauskaite's least stable term cross-nationally
    #     (.659): record first language + upbringing country at intake.
    #   Neutral: near-white, kept deliberately — a false positive of
    #     "relief" (white's anchor) harms less than mid-gray's "sadness".
    # Worst-case pairwise dE2000, before -> after the search (from cell 12d):
    #   normal 17.5 -> 24.0 | protan 7.5 -> 13.7 | deutan 10.2 -> 16.7
    #   achromatopsia 1.8 -> 4.8 (and <=4.8 for ANY hues: see CHANNEL_MODE)
    # NEW — TWO NEW CLASSES: frustrated, excited
    # ---------------------------------------------------------------------
    # clf_v3's label space comes from IEMOCAP, not RAVDESS, and IEMOCAP has
    # two categories RAVDESS never had. the previous palette had no entry for either, and the
    # lookup is styles.get(emotion, styles["neutral"]) in five places -- so
    # a frustrated segment rendered as near-white flat neutral
    # with no warning at all. On IEMOCAP, frustrated is one of the LARGEST
    # classes, so that silent fallback was not an edge case: it was a large
    # fraction of conversational speech being drawn as "no emotion".
    #
    # These two placements are PROPOSED, not audited. The audited hue values
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
    #     size the audit actually covers, and is what most published
    #     IEMOCAP work does anyway.
    #
    # "disgust" is kept below even though clf_v3 will usually NOT emit it:
    # IEMOCAP has only a handful of disgust utterances, so MIN_CLASS_COUNT
    # =40 drops the class at training time. An unused style entry is inert;
    # a missing one is a silent neutral render. Keep it.
    EMOTION_STYLES = {
        #             hue     sat   val  italic  anim      font
        "angry":      {"h": 0.0000, "s": 0.85, "v": 0.80, "i": 0, "anim": "pop",    "font": "DejaVu Sans Condensed"},
        "frustrated": {"h": 0.0556, "s": 0.70, "v": 0.62, "i": 0, "anim": "flat",   "font": "DejaVu Sans Condensed"},  # NEW: 20deg rust
        "happy":      {"h": 0.1400, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},
        "excited":    {"h": 0.8750, "s": 0.85, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},             # NEW: 315deg magenta
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
        MERGE_MAX_CHARS,
        MERGE_MAX_GAP_S,
        MERGE_MIN_LINE_S,
        MERGE_RAPID_ENABLE,
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
        PAUSE_DOTS,
        PAUSE_DOTS_MIN_S,
        PAUSE_DOTS_REPLACE_GAP,
        PAUSE_DOTS_SKIP_ON_BANG,
        PAUSE_HOLD,
        PAUSE_HOLD_FULL,
        PAUSE_HOLD_MAX_FSP,
        PAUSE_HOLD_THRESH,
        PAUSE_SPLIT_ENABLE,
        PAUSE_SPLIT_HOLD,
        PAUSE_SPLIT_MIN_S,
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
        SHOUT_ALLOW_TORN,
        SHOUT_AROUSAL_GATE,
        SHOUT_EXTREME_EMOTIONS,
        SHOUT_EXTREME_MIN_AROUSAL_Z,
        SHOUT_EXTREME_P_MULT,
        SHOUT_MAX_FRAC,
        SHOUT_MIN_AROUSAL_Z,
        SHOUT_PRIMARY_EMOTIONS,
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
        YELL_EMOTION_GATE,
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
    # CELL 6b — LABEL-SPACE CONTRACT CHECK  (new)
    # ---------------------------------------------------------------------
    # The single most dangerous coupling in this notebook: the classifier's
    # label space and EMOTION_STYLES are set in two different cells, by two
    # different people's decisions, and NOTHING checked that they agree.
    #
    # The style lookup is styles.get(emotion, styles["neutral"]) in five
    # places. That default is deliberate -- rendering must not crash
    # mid-batch -- but it means a class the model emits and the palette
    # lacks renders as near-white flat neutral, silently, forever. Under
    # clf_v3 that would have hit `frustrated`, one of IEMOCAP's
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
def _(colorsys, np):
    # CELL 12c — COLOUR SCIENCE  (audit Part 8 item 6 + 2.10)
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
    return delta_e_2000, emotion_rgb01, simulate_cvd, srgb_to_lab


@app.cell
def _(
    EMOTION_STYLES,
    SAT_FLOOR_FRAC,
    delta_e_2000,
    emotion_rgb01,
    pd,
    simulate_cvd,
    srgb_to_lab,
):
    # CELL 12d — PALETTE AUDIT  (audit 2.2/2.4/2.6/2.10, Part 8 item 6)
    # ---------------------------------------------------------------------
    # Re-derives, on every run, the numbers the palette was chosen by,
    # so a future palette edit is audited automatically. Reports:
    #   * pairwise dE2000 between all 7 styles under normal vision,
    #     protanopia, deuteranopia and achromatopsia
    #   * the same at the saturation floor (intensity=0 words render at
    #     s * SAT_FLOOR_FRAC, the worst case for discriminability)
    #   * the Valdez & Mehrabian implied arousal ordering — the check that
    #     caught the fear bug (arousal = -0.31*V + 0.60*S applied to the
    #     palette's own S/V; fear must land ABOVE sad)
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

    _n_pairs = len(EMOTION_STYLES) * (len(EMOTION_STYLES) - 1) // 2
    print("pairwise dE2000 by condition (full saturation):")
    for _cond_m1 in AUDIT_CONDITIONS:
        _sub = palette_audit_df[palette_audit_df["condition"] == _cond_m1]
        _worst = _sub.nsmallest(3, "delta_e")
        _n_low = int((_sub["delta_e"] < 15.0).sum())
        print(f"  {_cond_m1:13s} min={_sub['delta_e'].min():5.1f}  "
              f"pairs<15: {_n_low:2d}/{_n_pairs}  worst: "
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
        "V1.0 bug (fear at -0.10 vs sad +0.08) made permanent")
    # value stays fixed per emotion while only saturation carries intensity,
    # so the ramp does not fight the arousal equation (audit 2.1's check)
    palette_audit_df
    return (palette_audit_df,)


@app.cell
def _(
    EMOTION_STYLES,
    emotion_rgb01,
    matplotlib,
    np,
    os,
    palette_audit_df,
    plt,
    simulate_cvd,
):
    # CELL 12e — CVD FIGURE  (audit 2.10 — "highest value-per-hour")
    # ---------------------------------------------------------------------
    # Renders every emotion label in its palette colour, outlined the way
    # libass outlines it, on a video-grey ground, under each simulated
    # condition. Saved to outputs/figures/cvd_palette.png for the
    # dissertation. Tritanopia is deliberately absent: see cell 12c.
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
    cvd_fig.suptitle("V1.0 palette under colour-vision-deficiency simulation "
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
    # CELL 13 — STYLE HELPERS (colour, calm, pause gap, motion, styling)
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
            # (audit 2.4): fear's segments sit in the "tremor" anim
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
            # no gesture without a trustworthy pitch contour. V0.7 let a
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

        # the segment's absolute arousal sets a FLOOR, then the
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
            # Position A (audit Part 1): one variable per channel,
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

    def _arousal_to_z(frac_arr, spread, eps=1e-6):
        """Undo segment_arousal_floor's tanh squash, back to robust z.

        That function stores gain = 0.5 * (1 + tanh(z / spread)), so an
        AVERAGE segment sits at 0.5, not at 0. A threshold written in
        arousal_floor units therefore looks absolute while meaning
        something entirely relative, which is exactly the confusion
        YELL_MIN_DB_OVER_MEDIAN exists to avoid elsewhere. Inverting is
        exact and costs nothing, so the dial can be stated in sigma.
        """
        u = np.clip(2.0 * np.asarray(frac_arr, dtype=float) - 1.0,
                    -1.0 + eps, 1.0 - eps)
        return float(spread) * np.arctanh(u)

    def apply_emotion_shout(styled_df, emotions, blend_margin,
                            max_marks=3, min_marks=2,
                            n_classes=7, p_floor_mult=2.8, p_second_mult=2.1,
                            primary_emotions=None, extreme_emotions=None,
                            extreme_p_mult=3.2,
                            arousal_gate=True, min_arousal_z=0.35,
                            extreme_min_arousal_z=0.90,
                            arousal_floor_max=0.45, arousal_spread=1.5,
                            allow_torn=False, max_frac=0.20,
                            yell_emotion_gate=True):
        """V1.1 — a SECOND, emotion-driven shout trigger, independent of
        the acoustic yell detector (CELL 17/18). Strict and tiered.

        Capitalises every word in a segment that clears ALL of the gates
        for its tier, and appends 1-N '!' to that segment's LAST word,
        scaled by the segment's mean intensity.

        The changes, in order of how much they reduce the fire rate:

        1. TWO TIERS. `primary_emotions` (anger) clear a strict posterior
           floor; `extreme_emotions` (excitement) clear a much higher one.
           A flat set could not express "caps for anger, and for excitement
           only when it is overwhelming".
        2. RAISED-VOICE GATE. The posterior says which emotion, not whether
           the line had any force behind it. The segment's arousal --
           already computed by segment_arousal_floor, robust-z across the
           clip -- must clear a sigma threshold too. Conjunction, not a
           sum, for the same reason the tilt gate in detect_yelling is:
           either signal alone has a known way of being wrong.
        3. TORN CLAUSE OFF. Capitalising on the strength of a RUNNER-UP
           was doing most of the widening. Colour degrades gracefully
           when the classifier is torn; capitals do not -- they are binary
           and a viewer reads them as a claim about the audio.
        4. HARD CAP. No more than `max_frac` of segments may shout,
           whatever the thresholds decided. Ranked by the posterior that
           qualified them, so what survives is what the model was surest
           of.
        5. ACOUSTIC YELLS ARE VETOABLE. detect_yelling is emotion-blind by
           design and its result is OR'd into the same capitals, so with
           `yell_emotion_gate` a measured yell on a non-eligible emotion no
           longer reaches capitals. `is_yell` itself is left untouched --
           the audit trail still records what was measured, only the
           recasing is withheld.

        Must run AFTER assign_styles (needs emotion/emotion2/p_top/
        p_second/intensity/arousal_floor) and BEFORE any width measurement
        or render step, for the same reason apply_yell_case must:
        word_display has to be final before layout measures it, or the
        burned-in text and the measured geometry desync.

        Ordering note: this runs BEFORE merge_rapid_segments, so a
        merged caption can carry the '!' marks of more than one qualifying
        segment ("YES!! COME ON!!"). That is the intended reading -- each
        fired segment earned its marks -- but it is worth knowing before
        wondering where the doubled marks came from.
        """
        out = styled_df.copy()
        out["shout_tier"] = ""
        out["shout_block"] = ""
        if "is_yell" not in out.columns:
            out["is_yell"] = False
        if not emotions or not len(out):
            out["is_emotion_shout"] = False
            return out

        _primary = set(primary_emotions) if primary_emotions is not None \
            else (set(emotions) - set(extreme_emotions or ()))
        _extreme = set(extreme_emotions or ())
        _eligible = _primary | _extreme

        # NEW: both floors are CHANCE-RELATIVE, driven by the
        # SHOUT_SENSITIVITY preset in CELL 12 rather than by constants
        # buried here.
        #
        # The earlier code hardcoded 0.40 and 0.30, reasoned against "7 classes, ~60.6%
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
        _p_extreme = _chance * float(extreme_p_mult)

        # ---- one row per SEGMENT: the decision is segment-level ---------
        _seg = (out.groupby("segment_id", sort=False)
                   .agg(emotion=("emotion", "first"),
                        emotion2=("emotion2", "first"),
                        p_top=("p_top", "first"),
                        p_second=("p_second", "first"),
                        is_yell=("is_yell", "any"))
                   .reset_index())

        # ---- raised-voice evidence, in sigma ----------------------------
        _ar_col = ("arousal_floor" if "arousal_floor" in out.columns
                   else ("arousal_floor_used" if "arousal_floor_used" in out.columns
                         else None))
        _use_arousal = bool(arousal_gate) and _ar_col is not None \
            and float(arousal_floor_max) > 0.0
        if _use_arousal:
            _ar = (out.groupby("segment_id", sort=False)[_ar_col]
                      .mean().reindex(_seg["segment_id"]).to_numpy(dtype=float))
            if not np.any(_ar > 0.0):
                # segment_arousal_floor returns all-zero when there are
                # fewer than 3 segments, or when the level features never
                # computed. Blocking every segment forever on a signal that
                # was never measured is the wrong failure -- say so and
                # fall back to the posterior gates alone.
                _use_arousal = False
                print("SHOUT: arousal floor is all-zero (fewer than 3 "
                      "segments, or level features missing) — the "
                      "raised-voice gate is OFF for this clip and the "
                      "capitals rest on the classifier alone. Treat them "
                      "as weaker evidence than usual.")
                _ar_z = np.zeros(len(_seg))
            else:
                _ar_z = _arousal_to_z(_ar / float(arousal_floor_max),
                                      arousal_spread)
        else:
            _ar_z = np.zeros(len(_seg))

        # ---- gate each segment, and remember what stopped it ------------
        _fired = np.zeros(len(_seg), dtype=bool)
        _tier = np.array([""] * len(_seg), dtype=object)
        _why = np.array([""] * len(_seg), dtype=object)
        _evidence = np.zeros(len(_seg))

        for _i, _r in _seg.iterrows():
            _e1 = _r["emotion"]
            _e2 = _r["emotion2"]
            _p1 = float(_r["p_top"] or 0.0)
            _p2 = float(_r["p_second"] or 0.0)
            _z = float(_ar_z[_i])

            if _e1 in _extreme:
                _tier[_i] = "extreme"
                _need_p, _need_z = _p_extreme, float(extreme_min_arousal_z)
                _ev = _p1
            elif _e1 in _primary:
                _tier[_i] = "primary"
                _need_p, _need_z = _p_floor, float(min_arousal_z)
                _ev = _p1
            elif (allow_torn and isinstance(_e2, str) and _e2 in _eligible
                  and (_p1 - _p2) <= blend_margin):
                # the runner-up path, on the EXTREME bar when enabled at all
                _tier[_i] = "torn"
                _need_p, _need_z = max(_p2_floor, _p_extreme), \
                    float(extreme_min_arousal_z)
                _ev = _p2
            else:
                _why[_i] = "emotion"
                continue

            _evidence[_i] = _ev
            if _ev < _need_p:
                _why[_i] = f"p({_ev:.3f}<{_need_p:.3f})"
                continue
            if _use_arousal and _z < _need_z:
                _why[_i] = f"arousal(z {_z:+.2f}<{_need_z:+.2f})"
                continue
            _fired[_i] = True

        # ---- structural cap: capitals only mean anything while rare -----
        _n_seg = len(_seg)
        _allowed = max(1, int(np.ceil(_n_seg * float(max_frac))))
        if _fired.sum() > _allowed:
            _order = np.argsort(-_evidence)
            _keep = np.zeros(_n_seg, dtype=bool)
            _keep[[i for i in _order if _fired[i]][:_allowed]] = True
            _dropped = int(_fired.sum() - _keep.sum())
            for _i in np.where(_fired & ~_keep)[0]:
                _why[_i] = f"max_frac({max_frac:.0%} cap)"
            _fired = _keep
            print(f"SHOUT: {_dropped} segment(s) cleared every gate but were "
                  f"dropped by the {max_frac:.0%} cap — kept the {_allowed} "
                  f"with the strongest posterior. If this recurs the "
                  f"footage may genuinely be wall-to-wall shouting, in "
                  f"which case capitals cannot mark anything within it.")

        _seg["_fired"] = _fired
        _seg["_tier"] = _tier
        _seg["_why"] = _why
        _fmap = dict(zip(_seg["segment_id"], _seg["_fired"]))
        out["is_emotion_shout"] = out["segment_id"].map(_fmap).fillna(False).astype(bool)
        out["shout_tier"] = out["segment_id"].map(
            dict(zip(_seg["segment_id"], _seg["_tier"]))).fillna("")
        out["shout_block"] = out["segment_id"].map(
            dict(zip(_seg["segment_id"], _seg["_why"]))).fillna("")

        # ---- the acoustic detector's own capitals, optionally vetoed ----
        # is_yell is NOT modified: it is a measurement and the audit cells
        # read it as one. Only the recasing is withheld, and because
        # apply_yell_case is the sole writer of word_display before this
        # point, restoring the original is exactly `word`.
        _yell = out["is_yell"].astype(bool)
        _yell_caps = _yell.copy()
        if yell_emotion_gate:
            _ok_emotion = out["emotion"].isin(_eligible)
            _yell_caps = _yell & _ok_emotion
            _vetoed = _yell & ~_ok_emotion
            if _vetoed.any():
                out.loc[_vetoed, "word_display"] = \
                    out.loc[_vetoed, "word"].astype(str)
                _v_segs = int(out.loc[_vetoed, "segment_id"].nunique())
                print(f"SHOUT: emotion veto un-capitalised {_v_segs} "
                      f"acoustically-yelled segment(s) whose emotion is not "
                      f"in {sorted(_eligible)} (is_yell left set for the "
                      f"audit). YELL_EMOTION_GATE=False to allow them.")

        # capitalise: OR together with the acoustic yell decision, never
        # overwrite it -- a word already capitalised for being genuinely
        # loud stays capitalised even if this trigger doesn't also fire
        _shout = out["is_emotion_shout"] | _yell_caps
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

        # NEW: report the FIRE RATE. "Too trigger happy" and "never fires"
        # are both statements about a number nobody was printing -- earlier versions ran
        # this trigger silently, so the only way to find out how often it
        # fired was to watch the render. Segment-level, not word-level,
        # because the decision is per segment; word counts just track
        # segment length.
        _n_fired = int(_fired.sum())
        _frac = _n_fired / max(_n_seg, 1)
        _by_tier = _seg.loc[_seg["_fired"], "_tier"].value_counts().to_dict()
        print(f"SHOUT: {_n_fired}/{_n_seg} segments capitalised "
              f"({_frac:.0%}) {_by_tier or ''} | floors p_top>={_p_floor:.3f} "
              f"extreme>={_p_extreme:.3f} (chance {_chance:.3f}, "
              f"{n_classes} classes) | arousal z>="
              f"{min_arousal_z:g}/{extreme_min_arousal_z:g}"
              f"{'' if _use_arousal else ' [OFF]'} | "
              f"primary={sorted(_primary)} extreme={sorted(_extreme)}")

        # NEW: when nothing fires, say WHICH gate stopped what -- the same
        # reasoning as yell_report. "No capitals appeared" has four causes
        # that look identical from the outside, and nudging thresholds to
        # find out which is how a strict detector gets loosened into a
        # broken one.
        if _n_fired == 0 and _n_seg:
            _cand = _seg[_seg["_why"] != "emotion"]
            # group by the GATE, not by the detail string it carries --
            # "p(0.20<0.30)" and "p(0.18<0.40)" are the same gate twice
            _counts = (_seg["_why"].astype(str).str.split("(").str[0]
                       .value_counts())
            _summary = "; ".join(f"{_k}: {_v}" for _k, _v in _counts.items()
                                 if _k)
            print(f"  nothing fired. Blocked by -> {_summary}")
            if len(_cand):
                _near = _cand.sort_values("p_top", ascending=False).head(3)
                for _, _r in _near.iterrows():
                    print(f"    seg {_r['segment_id']}: {_r['emotion']} "
                          f"p={float(_r['p_top']):.3f} tier={_r['_tier']} "
                          f"-> {_r['_why']}")
            else:
                print(f"    no segment was even classified into "
                      f"{sorted(_eligible)} — this is an emotion-set "
                      f"question, not a threshold one.")
        elif _frac >= 0.50 and _n_seg >= 5:
            print(f"  {_frac:.0%} of segments are in capitals. Above roughly "
                  f"half, capitals stop contrasting with anything and the "
                  f"emphasis reads as none -- lower SHOUT_MAX_FRAC, or "
                  f"raise SHOUT_MIN_AROUSAL_Z.")

        return out

    # ---------------------------------------------------------------------
    # NEW — MERGE SEGMENTS TOO SHORT TO READ
    # ---------------------------------------------------------------------
    # The inverse of split_on_long_pauses below, and the two are a matched
    # pair: one joins captions that are too close together, the other cuts
    # captions that hold a silence. Merge runs FIRST -- reversed, the
    # splitter's line breaks would be merged straight back.
    # ---------------------------------------------------------------------
    def merge_rapid_segments(df, enable=True, max_gap_s=0.45, max_chars=42,
                             min_line_s=1.0, verbose=True):
        """Join consecutive segments too close together to be readable.

        THE BUG THIS EXISTS FOR. build_ass_events ends every caption at the
        NEXT caption's start:

            if si < len(seg_ids) - 1:
                e0 = min(e0, seg_starts[seg_ids[si + 1]])

        That clamp is load-bearing -- in absolute layout two lines share the
        same screen position, so an overlap would draw them on top of each
        other -- but it silently OVERRIDES MIN_LINE_DURATION. When Whisper
        segments faster than one line per second, MIN_LINE_DURATION is dead
        and each caption gets only the gap to its successor.

        On BreakingBad(Happy) -- "Each." "Yes!" "Come on!" "Yes!" -- that
        produced 16 segments over ~35 s, twelve of which rendered for about
        0.16 seconds each: laid out correctly, burned in correctly, and
        impossible to read. Nothing crashed. The only symptom was a
        READING RATE line that looks like a formatting complaint rather
        than "no one can see these captions".

        WHAT IT DOES. Walks the segments in time order and joins the next
        one into the current caption while BOTH hold:

          * the gap between them is under `max_gap_s`, so speech the
            speaker genuinely separated stays separated; and
          * the combined text stays under `max_chars`, so a merged line
            cannot become its own reading-rate problem.

        Only ever merges ACROSS an existing boundary; it never splits, and
        it never reorders. `segment_id` is renumbered contiguously in time
        order and the pre-merge value is kept in `segment_id_premerge` so
        the change is auditable.

        WHY THIS IS SAFE HERE AND NOWHERE EARLIER. Every per-segment
        quantity (emotion, colour, salience, arousal floor, yell flag,
        shout marks) has already been computed and written PER WORD by the
        time this runs, so nothing downstream re-joins on the old id -- the
        ASS builder is the only remaining consumer and it wants exactly
        these coarser units. Same argument split_on_long_pauses makes for
        itself.

        ONE VISIBLE CONSEQUENCE. A merged caption inherits the words of
        two or more classified segments, so if they disagreed the line can
        carry two colours. That is correct -- the colour is per word, not
        per line -- but it is a change in how the render reads, and worth
        a sentence in a write-up.
        """
        out = df.sort_values("start", kind="mergesort").reset_index(drop=True)
        if not enable or len(out) < 2:
            out["segment_id_premerge"] = out.get("segment_id", 0)
            if verbose and not enable:
                print("MERGE: disabled (MERGE_RAPID_ENABLE=False)")
            return out

        _seg = out["segment_id"].to_numpy()
        _start = out["start"].astype(float).to_numpy()
        _end = out["end"].astype(float).to_numpy()
        _txt_len = (out["word_display"].astype(str).str.strip().str.len()
                    .to_numpy() if "word_display" in out.columns
                    else out["word"].astype(str).str.strip().str.len().to_numpy())

        _new = np.zeros(len(out), dtype=int)
        _cur = 0
        _chars = 0
        _joins = 0
        for _i in range(len(out)):
            _new[_i] = _cur
            _chars += int(_txt_len[_i]) + 1          # +1 for the space
            if _i == len(out) - 1:
                break
            if _seg[_i + 1] == _seg[_i]:
                continue                              # same segment, carry on
            # a real boundary: decide whether to keep it
            _gap = float(_start[_i + 1]) - float(_end[_i])
            if _gap >= float(max_gap_s) or _chars >= int(max_chars):
                _cur += 1
                _chars = 0
            else:
                _joins += 1

        out["segment_id_premerge"] = _seg
        out["segment_id"] = _new

        if verbose:
            _before = int(len(np.unique(_seg)))
            _after = int(len(np.unique(_new)))
            # report how many lines are still under the readable floor, so
            # a clip that needs a wider max_gap_s says so rather than
            # quietly rendering flashes again
            _dur = (out.groupby("segment_id")["end"].max()
                    - out.groupby("segment_id")["start"].min())
            _short = int((_dur < float(min_line_s)).sum())
            print(f"MERGE: {_before} segment(s) -> {_after} caption line(s) "
                  f"({_joins} join(s), gap<{max_gap_s:g}s, "
                  f"max {max_chars} chars)")
            if _short:
                print(f"  {_short}/{_after} merged line(s) still span under "
                      f"{min_line_s:g}s of speech. build_ass_events extends a "
                      f"line to the next line's start, so those are only a "
                      f"problem if the NEXT line starts immediately -- if "
                      f"captions still flash, raise MERGE_MAX_GAP_S.")
        return out

    # ---------------------------------------------------------------------
    # NEW — SILENCE AS PUNCTUATION, AND AS A LINE BREAK
    # ---------------------------------------------------------------------
    # Channel 4 (PAUSE_HOLD) renders a pause as letter-spacing. These two
    # functions add the other two renderings a pause can have: an ellipsis,
    # and — past a longer threshold — a hard cut to a new subtitle event so
    # the word after the pause is not visible before it is spoken.
    #
    # Both must run BEFORE any measurement or layout, for the same reason
    # apply_yell_case and apply_emotion_shout must: they change
    # word_display, and the absolute layout sizes each word's box from it.
    # ---------------------------------------------------------------------
    _TRAIL_PUNCT_RE = re.compile(r"([\.\!\?,;:\u2026]+)\s*$")

    def add_pause_ellipsis(text, dots="...", skip_on_bang=True):
        """Append trailing dots, respecting punctuation already there.

        The placement rule follows how the pause is actually being read:

          "kill?"  -> "kill...?"   the dots are the WAIT; the question mark
                                   still terminates the sentence, so it
                                   stays last.
          "job."   -> "job..."     a full stop and an ellipsis are both
                                   terminal, so the ellipsis replaces it
                                   rather than stacking into four dots.
          "Good,"  -> "Good..."    same, the comma was the weaker mark.
          "OUT!!!" -> "OUT!!!"     untouched. The shout marks own that
                                   ending and "OUT!!!..." is not something
                                   anyone writes. The line may still SPLIT;
                                   only the dots are suppressed.
          "wait…"  -> "wait…"      already waiting; never doubled.
        """
        s = str(text).rstrip()
        if not s:
            return s
        if s.endswith("\u2026") or s.endswith(".."):
            return s
        m = _TRAIL_PUNCT_RE.search(s)
        if not m:
            return s + dots
        run, head = m.group(1), s[:m.start()]
        if "!" in run and skip_on_bang:
            return s
        if "?" in run or "!" in run:
            return head + dots + run
        return head + dots

    def split_on_long_pauses(df, enable=True, dots_min_s=0.28,
                             split_min_s=0.90, dots="...",
                             hold_through_gap=True, replace_gap=True,
                             skip_on_bang=True, verbose=True):
        """Turn measured silence into punctuation and, past a threshold,
        into a subtitle break.

        Two thresholds on one measurement (the gap between a word's end and
        the next word's start, computed WITHIN a segment):

          gap >= dots_min_s   ellipsis, same line.   "Good... job."
          gap >= split_min_s  ellipsis, and everything after the gap
                              becomes its own subtitle event.
                              "Who should I kill...?" then "You!"

        Why the split matters and spacing alone does not: a caption that
        shows the whole sentence at once hands the viewer the payload word
        during the silence the speaker is using to withhold it. The joke,
        the threat or the reveal lands early, and only for the caption
        reader. Cutting the line preserves the timing the audio has.

        The gap is recomputed here rather than read from `pause_after`,
        because pause_after was measured across the whole word stream and
        so reports the gap to the next SEGMENT on a segment's last word.
        Splitting there would be a no-op at best (it is already a line
        break) and a duplicated line at worst.

        Renumbers `segment_id` contiguously in time order and keeps the
        original in `segment_src`. Safe to do here and nowhere earlier:
        every per-segment quantity (emotion, salience, arousal, yell) has
        already been computed and written per word, so nothing downstream
        re-joins on the old id — the ASS builder is the only remaining
        consumer, and it wants exactly these finer units.

        NEW: runs AFTER merge_rapid_segments, so the ids it reads are
        already the merged ones. That order matters -- reversed, merge
        would join back together exactly the lines this function cut.
        """
        out = df.copy()
        if not enable or not len(out):
            out["segment_src"] = out.get("segment_id", 0)
            out["pause_dots"] = False
            out["line_hold_until"] = 0.0
            return out

        out = (out.sort_values(["segment_id", "start"], kind="mergesort")
                  .reset_index(drop=True))
        _seg = out["segment_id"].to_numpy()
        _start = out["start"].astype(float).to_numpy()
        _end = out["end"].astype(float).to_numpy()

        # gap to the next word, but only within the same original segment
        _gap = np.zeros(len(out))
        if len(out) > 1:
            _same = _seg[1:] == _seg[:-1]
            _gap[:-1] = np.where(_same, np.maximum(_start[1:] - _end[:-1], 0.0), 0.0)

        _dots_at = _gap >= float(dots_min_s)
        _split_at = _gap >= float(split_min_s)

        # ---- renumber: a split point ends the current subtitle line ----
        _new_id = np.zeros(len(out), dtype=int)
        _n = 0
        for _i in range(len(out)):
            _new_id[_i] = _n
            _ends_line = _split_at[_i] or (_i == len(out) - 1) or _seg[_i + 1] != _seg[_i]
            if _ends_line:
                _n += 1

        out["segment_src"] = _seg
        out["segment_id"] = _new_id
        out["pause_dots"] = _dots_at

        # ---- the text itself -------------------------------------------
        if _dots_at.any():
            _idx = out.index[_dots_at]
            out.loc[_idx, "word_display"] = [
                add_pause_ellipsis(_t, dots, skip_on_bang)
                for _t in out.loc[_idx, "word_display"].astype(str)]

        # ---- do not bill the same pause twice --------------------------
        # A word that now carries dots does not also need the wide \fsp
        # gap: it reads as the pause being counted twice, and the gap
        # pushes the dots away from the word they belong to. Channel 4
        # keeps everything below dots_min_s, where it is the only signal.
        if replace_gap and "pause_after" in out.columns and _dots_at.any():
            out.loc[out.index[_dots_at], "pause_after"] = 0.0

        # ---- hold the first half on screen through the silence ---------
        # Without this the line vanishes HOLD_MAX_TAIL after its last word
        # and the dots — the whole point — are gone before the pause is.
        # build_ass_events clamps every line to the next line's start, so
        # this can never overlap the payload it is waiting for.
        out["line_hold_until"] = 0.0
        if hold_through_gap:
            _starts = out.groupby("segment_id")["start"].min()
            _ids = list(_starts.index)
            _next_start = {_ids[_k]: float(_starts.iloc[_k + 1])
                           for _k in range(len(_ids) - 1)}
            _split_ids = set(out.loc[out.index[_split_at], "segment_id"].tolist())
            out["line_hold_until"] = [
                _next_start.get(_s, 0.0) if _s in _split_ids else 0.0
                for _s in out["segment_id"]]

        if verbose:
            _n_split = int(_split_at.sum())
            _n_dots = int(_dots_at.sum()) - _n_split
            _n_lines = int(out["segment_id"].nunique())
            _src_lines = int(len(np.unique(_seg)))
            print(f"PAUSE: {_n_dots} inline ellipsis (>={dots_min_s:g}s), "
                  f"{_n_split} line split(s) (>={split_min_s:g}s) — "
                  f"{_src_lines} caption line(s) -> {_n_lines}")
            if _n_split:
                for _pos in np.flatnonzero(_split_at)[:3]:
                    _rw = out.iloc[int(_pos)]
                    print(f"    split after "
                          f"'{str(_rw['word_display']).strip()}' @ "
                          f"{float(_rw['end']):.2f}s (gap {_gap[_pos]:.2f}s)")
        return out

    return (
        add_pause_ellipsis,
        apply_emotion_shout,
        assign_styles,
        attach_motion,
        merge_rapid_segments,
        pause_gap,
        split_on_long_pauses,
    )


@app.cell
def _(merge_rapid_segments, pd):
    # CELL 13d — MERGE SELF-TEST  (new)
    # ---------------------------------------------------------------------
    # merge_rapid_segments changes how many caption events a clip renders
    # as, which is the most visible thing in the pipeline. It gets a test
    # on the case it was written for and on the case it must not touch.
    def _merge_fixture(times, seg_ids, words=None):
        """Build the minimum frame merge_rapid_segments reads."""
        _w = words or [f"w{_i}" for _i in range(len(times))]
        return pd.DataFrame({
            "word": _w,
            "word_display": _w,
            "start": [t[0] for t in times],
            "end": [t[1] for t in times],
            "segment_id": seg_ids,
        })

    # rapid exclamations, each its own segment, ~0.15 s apart -- the
    # BreakingBad(Happy) shape
    _mt_rapid = _merge_fixture(
        [(0.0, 0.30), (0.45, 0.75), (0.90, 1.20), (1.35, 1.65)],
        [0, 1, 2, 3], ["Each.", "Each.", "Yes!", "Yes!"])
    _mr = merge_rapid_segments(_mt_rapid, max_gap_s=0.45, max_chars=42,
                               verbose=False)
    assert _mr["segment_id"].nunique() == 1, (
        "rapid segments 0.15s apart should merge into one readable line, "
        f"got {_mr['segment_id'].nunique()}")

    # genuinely separated speech -- 1.5 s apart, must stay separate
    _mt_slow = _merge_fixture(
        [(0.0, 0.50), (2.0, 2.50), (4.0, 4.50)], [0, 1, 2])
    _ms = merge_rapid_segments(_mt_slow, max_gap_s=0.45, verbose=False)
    assert _ms["segment_id"].nunique() == 3, (
        "segments 1.5s apart must not be merged, got "
        f"{_ms['segment_id'].nunique()}")

    # the character cap must break a long rapid run rather than build one
    # unreadable line out of it
    _mt_long = _merge_fixture(
        [(_i * 0.4, _i * 0.4 + 0.25) for _i in range(12)],
        list(range(12)),
        ["antidisestablish"] * 12)
    _ml = merge_rapid_segments(_mt_long, max_gap_s=0.45, max_chars=42,
                               verbose=False)
    assert _ml["segment_id"].nunique() > 1, (
        "max_chars must stop a rapid run from merging into one giant line")

    # order and word count are never altered
    assert list(_mr["word"]) == list(_mt_rapid["word"]), \
        "merge must not reorder or drop words"
    print(f"merge self-test passed (rapid -> 1 line, spaced -> 3 lines, "
          f"char cap -> {_ml['segment_id'].nunique()} lines)")
    return


@app.cell
def _(subprocess):
    # CELL 13b — TEXT MEASUREMENT  (FIX 1, part 1)
    # ---------------------------------------------------------------------
    # To position words individually we have to know how wide they are. We
    # ask fontconfig for the exact file libass will pick, then measure with
    # FreeType through Pillow. Because PlayResX is set to the video width,
    # one ASS unit is one pixel, so \fs32 and a 32px PIL font agree.
    try:
        from PIL import ImageFont
        HAVE_PIL = True
    except Exception:
        ImageFont = None
        HAVE_PIL = False
        print("Pillow not installed -> LAYOUT_MODE='absolute' will fall back "
              "to 'flow'. pip install Pillow to enable isolated motion.")

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
        """(audit Part 8 item 8): substitution is a FAILURE, not a
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

    return HAVE_PIL, font_vmetrics, space_width, text_width, verify_fonts


@app.cell
def _(font_vmetrics, pause_gap, space_width, text_width):
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
    # ---- V1.0 squash-and-stretch geometry (audit 4.3) --------------------
    # The gesture amplitude fractions live HERE, in one place, because two
    # cells must agree on them: the tagger (20a) uses them to write the \t
    # transforms, and this layout uses them to reserve the horizontal room
    # a squashed word will occupy. V1.0 had 0.6/0.5/0.4 as magic numbers in
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

            # V1.0: horizontal headroom for the squash (see swell_headroom_x)
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

    return (
        GESTURE_DROP_FRAC,
        GESTURE_WOBBLE_HI,
        GESTURE_WOBBLE_LO,
        layout_segment,
        squash_x,
    )


@app.cell
def _(EMOTION_STYLES, FONT_STRICT, verify_fonts):
    # CELL 14 — FONT AVAILABILITY CHECK
    # (audit Part 8 item 8): substitution is a hard failure, not a
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
    confidence_scale,
    np,
    pd,
):
    # CELL 15 — PREDICT THE CLIP'S EMOTION (top-2 for the blend)
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
        # V1.0: calibrated curve from cell 12b (0.43 -> ~0.60, not ~0.90)
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
def _(classify_words, np, pd):
    # CELL 17 — BUDGET MACHINERY  (FIX 2, part 4)
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

        The V0.7 bug was `salience += w * abs(z)`. Emphasis is directional:
        a word is emphasised by being LOUDER, HIGHER and LONGER than its
        neighbours, not merely different from them. Features listed in
        `positive_only` are clipped at zero so that being quiet, low and
        rushed — the signature of an unstressed function word — earns
        nothing instead of earning full marks.

        NEW: the empty-frame guard. BreakingBad(Happy) reached here with
        zero words (its only segment had been deleted by the loop screen),
        and `df.groupby("segment_id")["word"]` raised
        "KeyError: Column not found: word" -- an error that names a column
        rather than the actual problem, in a function three stages from
        where the problem happened. extract_word_features now returns a
        correctly-columned empty frame, and this says plainly what is
        wrong if one ever arrives anyway."""
        df = words_df.copy()
        if not len(df) or "word" not in df.columns:
            raise ValueError(
                "compute_salience got no words. This is not a salience "
                "problem: it means transcription or alignment produced "
                "nothing for this clip. Check the [2/6] transcribing line "
                "for 'dropped N repetition-loop segment(s)' -- if the whole "
                "clip was dropped, see looks_like_loop in CELL 8a and "
                "ASR_LOOP_MIN_REPEATS in CELL 1b.")
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
                  "extracted before V1.1. Re-run the feature extraction.")
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
    # CELL 18 — RUN THE BUDGET
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
    # V1.1: recover the between-segment dynamics the per-segment z-scoring
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
    return budget_df, seg_list


@app.cell
def _(budget_df):
    # CELL 18b — EMPHASIS AUDIT
    # ---------------------------------------------------------------------
    # Read this when a word looks wrong on screen. It shows what the word
    # was scored on and whether the class prior held it back or an override
    # let it through, so a false positive can be traced to a dial.
    _aud = budget_df.copy()
    _aud["word"] = _aud["word"].astype(str).str.strip()
    _aud = _aud.sort_values("intensity_raw", ascending=False)
    print("TOP 15 BY FINAL INTENSITY")
    print(_aud[["word", "word_class", "zmax_pos", "prominence", "class_weight",
                "salience_raw", "salience", "intensity_raw"]]
          .head(15).round(2).to_string(index=False))

    # (audit 5.3): the EMPIRICAL version of the z-bound argument. The
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
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part D: Style mapping
    """)
    return


@app.cell
def _(
    AROUSAL_SPREAD,
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
    MERGE_MAX_CHARS,
    MERGE_MAX_GAP_S,
    MERGE_MIN_LINE_S,
    MERGE_RAPID_ENABLE,
    MOTION_MIN_INTENSITY,
    MOTION_SOURCE,
    PAUSE_DOTS,
    PAUSE_DOTS_MIN_S,
    PAUSE_DOTS_REPLACE_GAP,
    PAUSE_DOTS_SKIP_ON_BANG,
    PAUSE_HOLD,
    PAUSE_HOLD_THRESH,
    PAUSE_SPLIT_ENABLE,
    PAUSE_SPLIT_HOLD,
    PAUSE_SPLIT_MIN_S,
    SATURATION_INTENSITY,
    SAT_FLOOR_FRAC,
    SEGMENT_AROUSAL_FLOOR,
    SHOUT_ALLOW_TORN,
    SHOUT_AROUSAL_GATE,
    SHOUT_EXTREME_EMOTIONS,
    SHOUT_EXTREME_MIN_AROUSAL_Z,
    SHOUT_EXTREME_P_MULT,
    SHOUT_MAX_FRAC,
    SHOUT_MIN_AROUSAL_Z,
    SHOUT_PRIMARY_EMOTIONS,
    SHOUT_P_FLOOR_MULT,
    SHOUT_P_SECOND_MULT,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    TRACKING_CALM,
    TREMOR_WOBBLE_FACTOR,
    WOBBLE_RANGE_HZ,
    YELL_EMOTION_GATE,
    apply_emotion_shout,
    assign_styles,
    attach_motion,
    budget_df,
    conf_scale,
    merge_rapid_segments,
    p_second,
    p_top,
    pd,
    pred_emotion,
    pred_emotion2,
    split_on_long_pauses,
):
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
    # V1.1: emotion-driven caps + exclamation marks, layered on top of
    # the acoustic yell decision assign_styles just carried through.
    # Must run before attach_motion/measurement -- word_display has to
    # be final before anything measures its width.
    if EXCLAIM_ENABLE:
        styled_word_df = apply_emotion_shout(
            styled_word_df, EXCLAIM_EMOTIONS, BLEND_MARGIN,
            max_marks=EXCLAIM_MAX_MARKS, min_marks=EXCLAIM_MIN_MARKS,
            n_classes=CLF_N_CLASSES,          # NEW: chance-relative floors
            p_floor_mult=SHOUT_P_FLOOR_MULT,
            p_second_mult=SHOUT_P_SECOND_MULT,
            primary_emotions=SHOUT_PRIMARY_EMOTIONS,   # NEW: strict tiers
            extreme_emotions=SHOUT_EXTREME_EMOTIONS,
            extreme_p_mult=SHOUT_EXTREME_P_MULT,
            arousal_gate=SHOUT_AROUSAL_GATE,
            min_arousal_z=SHOUT_MIN_AROUSAL_Z,
            extreme_min_arousal_z=SHOUT_EXTREME_MIN_AROUSAL_Z,
            arousal_floor_max=SEGMENT_AROUSAL_FLOOR,
            arousal_spread=AROUSAL_SPREAD,
            allow_torn=SHOUT_ALLOW_TORN,
            max_frac=SHOUT_MAX_FRAC,
            yell_emotion_gate=YELL_EMOTION_GATE)
    # NEW: MERGE FIRST, SPLIT SECOND. merge_rapid_segments joins captions
    # too close together to read (the BreakingBad(Happy) failure: twelve of
    # seventeen lines on screen for ~0.16s each, because build_ass_events
    # clamps every line to the next line's start and that overrides
    # MIN_LINE_DURATION). split_on_long_pauses then cuts the merged result
    # at genuinely long silences. Reversed, merge would join back together
    # exactly the lines the splitter had just separated.
    styled_word_df = merge_rapid_segments(
        styled_word_df, enable=MERGE_RAPID_ENABLE,
        max_gap_s=MERGE_MAX_GAP_S, max_chars=MERGE_MAX_CHARS,
        min_line_s=MERGE_MIN_LINE_S)
    # NEW: silence -> ellipsis, and past a longer threshold -> a new
    # subtitle line. Runs LAST of everything that touches word_display and
    # BEFORE attach_motion/measurement, so the text the layout measures is
    # the text that gets burned in (same invariant as apply_yell_case).
    styled_word_df = split_on_long_pauses(
        styled_word_df, enable=PAUSE_SPLIT_ENABLE,
        dots_min_s=PAUSE_DOTS_MIN_S, split_min_s=PAUSE_SPLIT_MIN_S,
        dots=PAUSE_DOTS, hold_through_gap=PAUSE_SPLIT_HOLD,
        replace_gap=PAUSE_DOTS_REPLACE_GAP,
        skip_on_bang=PAUSE_DOTS_SKIP_ON_BANG)
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
    return (styled_word_df,)


@app.cell
def _(PAUSE_HOLD, PAUSE_HOLD_THRESH, pd, styled_word_df):
    # CELL 19b — STYLING COVERAGE  (audit 2.7 / Part 8 item 4)
    # ---------------------------------------------------------------------
    # "Emotion-responsive" needs a number attached to it. If ~8% of words
    # style non-neutrally, the phrase is doing a lot of work for a small
    # effect; if ~60% do, that is the over-styling AffType's all-features
    # condition predicts will hurt. Either way the number belongs in the
    # Results, and it is nearly free to produce — so it prints on every
    # run, here and inside process_any_video.
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
            # NEW: caption LINES, not just words. The BreakingBad failure
            # was invisible in every number above -- 25 words all styled
            # correctly, across 17 lines nobody could read.
            "lines": int(styled_df["segment_id"].nunique())
            if "segment_id" in styled_df else 0,
        }
        print(f"styling coverage [{label}]: "
              f"{100 * _stats['non_neutral_frac']:.0f}% of {_n} words carry a "
              f"non-neutral emotion style | intensity>0.3: "
              f"{100 * _stats['intensity>0.3']:.0f}% | >0.6: "
              f"{100 * _stats['intensity>0.6']:.0f}% | bold: "
              f"{100 * _stats['bold_frac']:.0f}% | moving: "
              f"{100 * _stats['gesture_frac']:.0f}% | blended: "
              f"{100 * _stats['blended_frac']:.0f}% | lines: "
              f"{_stats['lines']}")
        return _stats

    styling_coverage(styled_word_df,
                     label="current clip" + ("" if PAUSE_HOLD else ""))
    _ = PAUSE_HOLD_THRESH  # coverage reads no thresholds; keep cell honest
    return (styling_coverage,)


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

    Set `LAYOUT_MODE="flow"` to get the V0.7 renderer back for comparison.

    **NEW — read the clamp in `build_ass_events` before touching timing.**
    Every line ends at the next line's start, which silently overrides
    `MIN_LINE_DURATION`. That is why `merge_rapid_segments` exists and why
    raising `MIN_LINE_DURATION` would not have helped.
    """)
    return


@app.cell
def _(
    GESTURE_DROP_FRAC,
    GESTURE_WOBBLE_HI,
    GESTURE_WOBBLE_LO,
    HAVE_PIL,
    MERGE_MAX_GAP_S,
    layout_segment,
    pause_gap,
    squash_x,
):
    # CELL 20a — SHARED ASS BUILDER (both layout modes)
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
        # (audit Part 8 item 2): outline and shadow are parameters now —
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
        safe: it changes nothing but this word. `isolated=False` is the V0.7
        flow path, kept verbatim for A/B fidelity — squash is only applied
        when isolated, because only the per-word \\pos architecture
        guarantees the horizontal squeeze is local (and only there does the
        layout reserve the room it needs).

        (audit 4.3): with squash on, every \\fscy carries an inverse
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

    # ---------- FLOW path (V0.7 behaviour, kept for A/B) -------------------
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
                         rate_enforce=False, rate_max_cps=17.0,
                         flash_warn_s=0.5):
        lines = []
        rate_violations = []
        flash_lines = []

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
            # V1.0 reading rate (audit Part 8 item 3): when the natural
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
            # NEW: a line that was cut at a long pause holds until the line
            # after it starts, so its trailing "..." is on screen FOR the
            # silence rather than disappearing hold_max_tail after the last
            # word and leaving a blank frame. Requested end only — the
            # clamp immediately below still owns the hard limit, so this
            # can never overlap the line it is waiting for.
            if "line_hold_until" in seg.columns:
                _hold = float(seg["line_hold_until"].max() or 0.0)
                if _hold > 0.0:
                    e0 = max(e0, _hold)
            # ---- THE CLAMP (NEW: read the note before changing timing) --
            # This is the line that overrides MIN_LINE_DURATION, and it has
            # to: in absolute layout every caption occupies the same screen
            # position, so an overlap draws two lines on top of each other.
            # But it means MIN_LINE_DURATION is only advisory -- when
            # segments arrive closer together than min_line_dur, each line
            # gets exactly the gap to its successor and nothing above can
            # give it more. On BreakingBad(Happy) that produced twelve
            # captions of ~0.16s. merge_rapid_segments exists to stop the
            # segments arriving that close in the first place; raising
            # MIN_LINE_DURATION here would change nothing at all.
            if si < len(seg_ids) - 1:
                e0 = min(e0, seg_starts[seg_ids[si + 1]])
            e0 = max(e0, s0 + 0.10)
            if rate_enforce:
                cps = n_chars / max(e0 - s0, 1e-9)
                if cps > rate_max_cps + 0.05:
                    rate_violations.append((sid, round(cps, 1), n_chars))
            # NEW: a caption too brief to read is a DIFFERENT failure
            # from one that is too dense, and the reading-rate report
            # cannot distinguish them -- both show up as a high cps. This
            # reports on-screen DURATION directly, which is the thing a
            # viewer actually experiences.
            if (e0 - s0) < float(flash_warn_s):
                flash_lines.append((sid, round(e0 - s0, 2), n_chars))

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
        # NEW: the report that would have caught BreakingBad(Happy) on
        # sight. "Too dense to read" and "gone before you can read it" are
        # different problems with different fixes, and only one of them is
        # about characters per second.
        if flash_lines:
            _n_lines = len(seg_ids)
            print(f"FLASH: {len(flash_lines)}/{_n_lines} caption line(s) are on "
                  f"screen under {flash_warn_s:g}s (segment, seconds, chars): "
                  f"{flash_lines[:8]}" + (" ..." if len(flash_lines) > 8 else ""))
            print(f"  These are drawn correctly and cannot be read. The line's "
                  f"end is clamped to the NEXT line's start, so "
                  f"MIN_LINE_DURATION cannot help -- raise MERGE_MAX_GAP_S "
                  f"(now {MERGE_MAX_GAP_S:g}s) so the segments join into one "
                  f"readable caption, or check MERGE_RAPID_ENABLE is True.")
        return lines

    def write_ass(path, width, height, lines, wrap_style=0, outline_px=3,
                  shadow_px=1, outline_colour="&H00000000",
                  back_colour="&H64000000"):
        with open(path, "w") as f:
            f.write(ass_header(width, height, wrap_style, outline_px,
                               shadow_px, outline_colour, back_colour)
                    + "\n".join(lines) + "\n")
        return path

    return build_ass_events, make_word_tagger, write_ass


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
    build_ass_events,
    make_word_tagger,
    os,
    out_tag,
    pred_emotion,
    styled_word_df,
    subprocess,
    verify_fonts,
    write_ass,
):
    # CELL 20b — LEGACY RENDER (black screen)
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

        # V1.0: refuse to measure in one face and render in another
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
    return


@app.cell
def _(
    EMOTION_SELF_BIAS,
    EMOTION_SMOOTH,
    MIN_DWELL_S,
    NORM_MIN_SEGMENTS,
    SEGMENT_NORM,
    audio_file,
    clf_feature_cols,
    clf_full,
    confidence_scale,
    predict_segment_emotions_v9,
    seg_list,
    smooth_segment_emotions,
):
    # CELL 21 — PER-SEGMENT PREDICTION on the test clip
    seg_emotion_df = predict_segment_emotions_v9(
        audio_file, seg_list, clf_full, clf_feature_cols,
        normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
    # V1.1: {segment_id: speaker_label} from diarization, or all-None if
    # diarization was skipped/failed -- either way smooth_segment_emotions
    # handles it correctly (None-vs-None is never treated as a boundary).
    _speaker_ids_21 = {_i: _s.get("speaker") for _i, _s in enumerate(seg_list)}
    # V1.1: min_dwell_s wired through -- the second guard alongside the
    # raised self_bias (see CELL 12).
    seg_emotion_df = smooth_segment_emotions(
        seg_emotion_df, list(clf_full.classes_), mode=EMOTION_SMOOTH,
        self_bias=EMOTION_SELF_BIAS, min_dwell_s=MIN_DWELL_S,
        speaker_ids=_speaker_ids_21,
        confidence_scale_fn=confidence_scale)
    seg_emotion_df
    return


@app.cell
def _(
    AROUSAL_FEATURES,
    AROUSAL_SPREAD,
    ASR_BATCH_SIZE,
    ASR_CHUNK_SIZE,
    ASR_DROP_LOOPS,
    ASR_LOOP_MIN_COVERAGE,
    ASR_LOOP_MIN_REPEATS,
    ASR_NO_SPEECH_THRESHOLD,
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
    HAVE_PIL,
    HOLD_MAX_TAIL,
    LAYOUT_LINE_GAP,
    LAYOUT_MARGIN_V,
    LAYOUT_MARGIN_X,
    LAYOUT_MODE,
    LAYOUT_SPACE_SCALE,
    MERGE_MAX_CHARS,
    MERGE_MAX_GAP_S,
    MERGE_MIN_LINE_S,
    MERGE_RAPID_ENABLE,
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
    PAUSE_DOTS,
    PAUSE_DOTS_MIN_S,
    PAUSE_DOTS_REPLACE_GAP,
    PAUSE_DOTS_SKIP_ON_BANG,
    PAUSE_HOLD,
    PAUSE_HOLD_FULL,
    PAUSE_HOLD_MAX_FSP,
    PAUSE_HOLD_THRESH,
    PAUSE_SPLIT_ENABLE,
    PAUSE_SPLIT_HOLD,
    PAUSE_SPLIT_MIN_S,
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
    SHOUT_ALLOW_TORN,
    SHOUT_AROUSAL_GATE,
    SHOUT_EXTREME_EMOTIONS,
    SHOUT_EXTREME_MIN_AROUSAL_Z,
    SHOUT_EXTREME_P_MULT,
    SHOUT_MAX_FRAC,
    SHOUT_MIN_AROUSAL_Z,
    SHOUT_PRIMARY_EMOTIONS,
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
    YELL_EMOTION_GATE,
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
    build_ass_events,
    clf_feature_cols,
    clf_full,
    compute_salience,
    confidence_scale,
    count_syllables,
    detect_yelling,
    device,
    extract_word_features,
    free_vram,
    json,
    make_word_tagger,
    merge_rapid_segments,
    np,
    os,
    predict_segment_emotions_v9,
    screen_segments,
    segment_arousal_floor,
    smooth_segment_emotions,
    split_on_long_pauses,
    styling_coverage,
    subprocess,
    verify_fonts,
    whisperx,
    write_ass,
    yell_report,
):
    # CELL 22 — RENDER onto a REAL video (per-segment colour)
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

        # V1.0: substitution desyncs measurement from rendering — fail here,
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

    def process_any_video(video_path, out_tag=None, use_bg_video=True,
                          out_dir="outputs"):
        os.makedirs("outputs/audio", exist_ok=True)
        stem = Path(video_path).stem
        extracted_audio = f"outputs/audio/{stem}.wav"

        print(f"[1/6] extracting audio <- {video_path}")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
                        extracted_audio], capture_output=True, text=True, check=True)

        print("[2/6] transcribing (whisperx)")
        free_vram()          # V1.1 — see CELL 7c
        audio = whisperx.load_audio(extracted_audio)
        try:
            result = asr_model.transcribe(audio, batch_size=ASR_BATCH_SIZE,
                                          chunk_size=ASR_CHUNK_SIZE)
        except TypeError:
            result = asr_model.transcribe(audio, batch_size=ASR_BATCH_SIZE)

        _n_before = len(result["segments"])
        result["segments"], _dropped = screen_segments(
            result["segments"], ASR_DROP_LOOPS, ASR_LOOP_MIN_REPEATS)
        if _dropped:
            print(f"        dropped {_dropped} repetition-loop segment(s)")

        align_model, align_meta = whisperx.load_align_model(
            language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], align_model, align_meta, audio, device,
                                 return_char_alignments=False)
        # V1.1: this function loads a SECOND alignment model on top of the
        # one cell 9 already holds. Dropping the local name is not enough,
        # because torch keeps the freed blocks in its own cache; free_vram
        # returns them to the driver so the next call can transcribe.
        align_model = None
        free_vram()
        seg_list = aligned.get("segments") or result["segments"]
        print(f"        {len(seg_list)} segment(s), {len(aligned['word_segments'])} word(s)")

        # NEW: fail HERE, where the cause is visible, rather than three
        # stages later inside compute_salience with a message about a
        # missing column. BreakingBad(Happy) reached [4/6] and raised
        # "KeyError: Column not found: word", which named neither the clip
        # problem nor the loop screen that caused it.
        if not len(aligned["word_segments"]):
            raise RuntimeError(
                f"no words survived transcription for '{stem}'. "
                f"{_n_before} raw segment(s), {_dropped} dropped as "
                f"repetition loops -> {len(seg_list)} kept. If the drop count "
                f"accounts for all of them, the loop screen ate a real "
                f"transcript: check looks_like_loop (CELL 8a), "
                f"ASR_LOOP_MIN_REPEATS={ASR_LOOP_MIN_REPEATS} and "
                f"ASR_LOOP_MIN_COVERAGE={ASR_LOOP_MIN_COVERAGE}. If nothing "
                f"was dropped, the clip has no intelligible speech -- try "
                f"lowering ASR_NO_SPEECH_THRESHOLD "
                f"(now {ASR_NO_SPEECH_THRESHOLD}), or accept that it is not "
                f"a usable stimulus.")

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
        # V1.1: same speaker-aware wiring as the single-clip test path.
        _speaker_ids_pav = {_i: _s.get("speaker") for _i, _s in enumerate(seg_list)}
        # V1.1: min_dwell_s wired through here too, so the demo/production
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
        # V1.1: same emotion-driven caps + exclamation marks as the
        # single-clip test path -- must run before attach_motion.
        if EXCLAIM_ENABLE:
            styled_df = apply_emotion_shout(
                styled_df, EXCLAIM_EMOTIONS, BLEND_MARGIN,
                max_marks=EXCLAIM_MAX_MARKS, min_marks=EXCLAIM_MIN_MARKS,
                n_classes=CLF_N_CLASSES,      # NEW: chance-relative floors
                p_floor_mult=SHOUT_P_FLOOR_MULT,
                p_second_mult=SHOUT_P_SECOND_MULT,
                primary_emotions=SHOUT_PRIMARY_EMOTIONS,  # NEW: strict tiers
                extreme_emotions=SHOUT_EXTREME_EMOTIONS,
                extreme_p_mult=SHOUT_EXTREME_P_MULT,
                arousal_gate=SHOUT_AROUSAL_GATE,
                min_arousal_z=SHOUT_MIN_AROUSAL_Z,
                extreme_min_arousal_z=SHOUT_EXTREME_MIN_AROUSAL_Z,
                arousal_floor_max=SEGMENT_AROUSAL_FLOOR,
                arousal_spread=AROUSAL_SPREAD,
                allow_torn=SHOUT_ALLOW_TORN,
                max_frac=SHOUT_MAX_FRAC,
                yell_emotion_gate=YELL_EMOTION_GATE)
        # NEW: merge first (join captions too brief to read), split
        # second (cut captions that hold a silence). Same order as the
        # single-clip path above -- reversed, merge undoes the split.
        styled_df = merge_rapid_segments(
            styled_df, enable=MERGE_RAPID_ENABLE,
            max_gap_s=MERGE_MAX_GAP_S, max_chars=MERGE_MAX_CHARS,
            min_line_s=MERGE_MIN_LINE_S)
        # NEW: silence -> ellipsis / new subtitle line. Last writer of
        # word_display, before anything measures it.
        styled_df = split_on_long_pauses(
            styled_df, enable=PAUSE_SPLIT_ENABLE,
            dots_min_s=PAUSE_DOTS_MIN_S, split_min_s=PAUSE_SPLIT_MIN_S,
            dots=PAUSE_DOTS, hold_through_gap=PAUSE_SPLIT_HOLD,
            replace_gap=PAUSE_DOTS_REPLACE_GAP,
            skip_on_bang=PAUSE_DOTS_SKIP_ON_BANG)
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
        # (audit 2.7): the styled-proportion number the write-up needs
        styling_coverage(styled_df, label=stem)
        print("wrote:", out_path)
        return out_path, ass_path, seg_emotion_df, styled_df, extracted_audio

    return process_any_video, render_long_video


@app.cell
def _(Path, iemocap_dir, os, process_any_video, subprocess):
    # CELL 24 — DEMO: pull a whole conversation off IEMOCAP
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

    return


@app.cell
def _(
    Path,
    SOURCE_VIDEO_DIR,
    VERSION_TAG,
    find_source_videos,
    normalise_video,
    os,
    process_any_video,
    random,
    safe_stem,
):
    # CELL 25 — RUN ON ONE OF THE REAL SOURCE VIDEOS
    # ---------------------------------------------------------------------
    # Was hardcoded to "12AngryMenTest.mp4". Two things wrong with that: the
    # file is not in the repo, so on a fresh checkout this cell printed a
    # can't-find message and silently did nothing at all; and judging a
    # styling change on one fixed clip is how you end up tuned to that
    # clip's quirks rather than to the effect you meant to build. The
    # stimuli are already discoverable -- find_source_videos() returns the
    # same roster the batch runner renders -- so this picks from it.
    #
    # RANDOM PICKS AND REACTIVITY FIGHT EACH OTHER. marimo re-runs this cell
    # whenever anything upstream of it changes, so with a fresh pick every
    # time, editing a threshold in CELL 12 would re-transcribe a DIFFERENT
    # video and leave you with nothing to compare against -- two variables
    # moved, one conclusion available. The seed actually used is therefore
    # always printed, and pinning it is a one-line edit. Pin it before
    # A/B-ing a dial; leave it None when you want fresh footage to check
    # that a setting generalises past the clip you tuned it on.
    #
    # NEW: TEST_CLIP_ENABLE. This cell is the single most expensive thing
    # in the notebook that nobody asked for -- editing any dial in CELL 12
    # re-transcribes a whole video here before you can look at the batch.
    # Set it False while iterating and True when you actually want a
    # one-clip look.
    TEST_CLIP_ENABLE = True
    TEST_CLIP_MODE = "random"        # "random" | "fixed"
    TEST_CLIP_SEED = None            # None = fresh pick; int = reproducible
    TEST_CLIP_NAME = ""              # mode="fixed": stem or filename
    TEST_CLIP_NORMALISE = True       # the same geometry fix the batch
                                     # applies, so a test render is
                                     # comparable to its stimulus
    TEST_CLIP_FALLBACK = "12AngryMenTest.mp4"   # only if the folder is
                                                # missing or empty

    _roster = find_source_videos()
    _pick = None

    if not TEST_CLIP_ENABLE:
        print("CELL 25 disabled (TEST_CLIP_ENABLE=False) — no single-clip "
              "render. Set it True for a one-clip look; leave it False while "
              "iterating on dials so an edit does not re-transcribe a video "
              "you did not ask for.")
    elif _roster:
        print(f"source roster ({len(_roster)} in {SOURCE_VIDEO_DIR}/): "
              + ", ".join(f.name for f in _roster))
        if TEST_CLIP_MODE == "fixed":
            _want = str(TEST_CLIP_NAME).strip().lower()
            _hits = [f for f in _roster
                     if _want and _want in (f.name.lower(), f.stem.lower())]
            if _hits:
                _pick = _hits[0]
                print(f"fixed pick: {_pick.name}")
            else:
                print(f"TEST_CLIP_MODE='fixed' but TEST_CLIP_NAME="
                      f"{TEST_CLIP_NAME!r} matches nothing above — "
                      f"falling back to a random pick.")
        if _pick is None:
            # A fresh seed is DRAWN and printed rather than left implicit,
            # so an interesting render is always reproducible after the
            # fact. "It did something odd on some clip" is not a bug report.
            _seed = (random.randrange(2 ** 31) if TEST_CLIP_SEED is None
                     else int(TEST_CLIP_SEED))
            _pick = random.Random(_seed).choice(_roster)
            print(f"random pick: {_pick.name}   (seed {_seed} — set "
                  f"TEST_CLIP_SEED = {_seed} to re-run this exact clip)")
    else:
        print(f"No videos found in {SOURCE_VIDEO_DIR}/ (looked from "
              f"{os.getcwd()}).")
        if os.path.exists(TEST_CLIP_FALLBACK):
            _pick = Path(TEST_CLIP_FALLBACK)
            print(f"  falling back to {TEST_CLIP_FALLBACK}")
        else:
            print(f"  and no {TEST_CLIP_FALLBACK} either, so there is "
                  f"nothing to run. Drop a video into {SOURCE_VIDEO_DIR}/ "
                  f"and re-run this cell.")

    if _pick is not None:
        _src_clip = str(_pick)
        if TEST_CLIP_NORMALISE:
            _src_clip = normalise_video(_src_clip)
        # out_tag via safe_stem, matching the batch runner: stimulus names
        # carry parentheses ("BreakingBad(Happy)") and process_any_video's
        # own default would put them straight into the filename.
        print(f"\n--- rendering {_pick.name} ---")
        (test_clip_out, test_clip_ass, test_clip_seg_emotions,
         test_clip_styled_df, test_clip_audio) = process_any_video(
            _src_clip, out_tag=f"{VERSION_TAG}_{safe_stem(_pick)}",
            use_bg_video=True)
        print(f"\nvideo : {test_clip_out}\nsubs  : {test_clip_ass}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part F: Evaluation infrastructure — stimulus screening (V1.0)

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
    # CELL 26 — EMOLEX STIMULUS SCREENING  (audit 7.1)
    # ---------------------------------------------------------------------
    # The NRC Word-Emotion Association Lexicon is free for research; place
    # the word-level file at EMOLEX_PATH. Screening is a rough gate, not a
    # verdict: it reports WHICH words carry core-emotion associations so
    # they can be judged in context (EmoLex annotates words in isolation —
    # its own authors flag that as a limit, and so should the write-up).
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
def _(Path, VERSION_TAG, json, os, re, subprocess):
    # CELL 26b — EVALUATION CONFIG + FORMAT NORMALISATION
    # ---------------------------------------------------------------------
    # Split out of the EmoLex cell. These are constants and pure
    # functions; EmoLex screening opens a 14 MB lexicon off disk. Sharing a
    # cell meant editing PLAIN_SPEC re-read the lexicon, which is the sort
    # of tax that makes people stop editing dials.
    SOURCE_VIDEO_DIR = str(Path(__file__).resolve().parent.parent / "source_clips_raw")
    EVAL_OUT_DIR = str(Path(__file__).resolve().parent.parent / "evaluationoutput")
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

    # Output naming: "{stem}{VERSION_TAG}_withUniqueSubtitles". VERSION_TAG
    # comes from CELL 1, so bumping it renames every output with no edit
    # here. Bumping it means skip_existing will not find the
    # v18 pairs and the batch re-renders -- which is correct, since those
    # predate merge_rapid_segments.
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

    return (
        EVAL_OUT_DIR,
        NORMALISE_FORMAT,
        PLAIN_SUFFIX,
        SOURCE_VIDEO_DIR,
        UNIQUE_SUFFIX,
        VIDEO_EXTS,
        normalise_video,
    )


@app.cell
def _(EMOTION_STYLES, PAUSE_DOTS, PAUSE_DOTS_SKIP_ON_BANG, add_pause_ellipsis):
    # CELL 26c — THE PLAIN (CONTROL) VARIANT
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

        Note: segment_id has already been through
        merge_rapid_segments and split_on_long_pauses by the time this
        runs, so the control inherits exactly the same caption BOUNDARIES
        as the styled render. That is required, not incidental -- if the
        two conditions were cut into lines differently they would differ in
        reading load as well as in styling, and the mute test could not
        attribute an effect to either.
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
        # NEW: ...except the pause ellipsis, which is put back. The dots
        # are a TIMING mark, not a kinetic one -- they encode a silence
        # that is in the audio and is equally available to both conditions.
        # The line breaks survive this reset anyway (segment_id was already
        # renumbered), so dropping the dots would leave the control with an
        # unexplained mid-sentence cut and quietly make the two renders
        # differ in their text as well as their styling, which is the one
        # thing a mute-test control must not do.
        if "pause_dots" in out.columns and "word" in out.columns:
            _pd_m = out["pause_dots"].astype(bool)
            if _pd_m.any():
                out.loc[_pd_m, "word_display"] = [
                    add_pause_ellipsis(_t, PAUSE_DOTS, PAUSE_DOTS_SKIP_ON_BANG)
                    for _t in out.loc[_pd_m, "word"].astype(str)]
        # keep pause_after present (the builder reads it) but inert:
        # PLAIN_RENDER passes pause_hold=False, so it is never consulted
        return out

    return PLAIN_RENDER, PLAIN_SPEC, flatten_to_plain


@app.cell
def _(
    BASE_FONT_SIZE,
    CAPTION_MODE,
    EMOTION_STYLES,
    EVAL_OUT_DIR,
    HOLD_MAX_TAIL,
    LAYOUT_LINE_GAP,
    LAYOUT_MARGIN_V,
    LAYOUT_MARGIN_X,
    LAYOUT_MODE,
    LAYOUT_SPACE_SCALE,
    MIN_LINE_DURATION,
    NORMALISE_FORMAT,
    PLAIN_RENDER,
    PLAIN_SUFFIX,
    Path,
    READING_RATE_ENFORCE,
    READING_RATE_MAX_CPS,
    SOURCE_VIDEO_DIR,
    UNIQUE_SUFFIX,
    VERSION_TAG,
    VIDEO_EXTS,
    flatten_to_plain,
    normalise_video,
    os,
    pd,
    process_any_video,
    re,
    render_long_video,
    styling_coverage,
):
    # CELL 26d — THE BATCH RUNNER
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
                    # NEW: caption LINES after merge+split, which is what a
                    # viewer counts. n_segments is what the classifier saw.
                    "n_lines": int(_styled["segment_id"].nunique()),
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
    return find_source_videos, run_evaluation_batch, safe_stem


@app.cell
def _(
    BASE_FONT_SIZE,
    CAPTION_MODE,
    EMOTION_STYLES,
    EVAL_OUT_DIR,
    HOLD_MAX_TAIL,
    LAYOUT_LINE_GAP,
    LAYOUT_MARGIN_V,
    LAYOUT_MARGIN_X,
    LAYOUT_MODE,
    LAYOUT_SPACE_SCALE,
    MIN_LINE_DURATION,
    NORMALISE_FORMAT,
    PLAIN_RENDER,
    PLAIN_SPEC,
    PLAIN_SUFFIX,
    READING_RATE_ENFORCE,
    READING_RATE_MAX_CPS,
    SOURCE_VIDEO_DIR,
    VERSION_TAG,
    find_source_videos,
    normalise_video,
    os,
    pd,
    process_any_video,
    render_long_video,
    safe_stem,
):
    # CELL 28 — PLAIN-ONLY BATCH, PURE SUBTITLES  (V20)
    # ---------------------------------------------------------------------
    # Paste as a NEW cell after CELL 26d (the batch runner). Nothing above
    # is modified, so the existing withUniqueSubtitles renders and
    # run_evaluation_batch() still behave exactly as before.
    #
    # Two changes relative to flatten_to_plain / run_evaluation_batch:
    #
    #  1. No pause marks. The unique condition puts "..." into the control on the
    #     argument that a silence is a timing fact rather than a kinetic
    #     one. For a pure-subtitle baseline that is still an editorial
    #     insertion the transcript did not contain, so it goes. With
    #     PURE_SPLIT_ON_PAUSES = False the whole pause stage is switched
    #     off for this run, which also removes the pause-driven LINE BREAK
    #     and the line_hold_until extension -- build_ass_events reads that
    #     column regardless of pause_hold, so clearing the dots alone would
    #     have left the control holding lines through silences.
    #
    #  2. No unique render. process_any_video always ends by burning the
    #     kinetic version, which is the expensive half of the batch and is
    #     not wanted here. render_long_video is stubbed out for the
    #     duration of that call only, so the pipeline still produces the
    #     styled dataframe the flattener needs, and the real renderer is
    #     restored before the plain pass.
    #
    # NOTE ON COMPARABILITY: with PURE_SPLIT_ON_PAUSES = False the control
    # is cut into caption lines differently from the unique condition
    # (merge_rapid_segments still runs; split_on_long_pauses does not).
    # That is the intended meaning of "pure", but it does mean the two
    # conditions now differ in line boundaries as well as in styling. Set
    # it True to keep the unique condition's boundaries and drop only the
    # dots and the hold.
    import contextlib

    PURE_SPLIT_ON_PAUSES = False   # False = no pause dots, no pause splits
    PURE_SUFFIX = PLAIN_SUFFIX     # overwrites the old withoutUniqueSubtitles
    # PURE_SUFFIX = f"{VERSION_TAG}_withoutUniqueSubtitles_pure"  # keep both


    def flatten_to_plain_pure(styled_df, base_font, spec=PLAIN_SPEC):
        """flatten_to_plain, minus every pause-derived edit.

        word_display is rebuilt from `word`, so the ALL CAPS, the added
        '!' marks and any ellipsis are all gone in one move. pause_dots,
        pause_after and line_hold_until are then zeroed so nothing
        downstream can reintroduce a pause through a column the render
        params do not gate.
        """
        out = styled_df.copy()
        for _col, _val in spec.items():
            if _col in out.columns:
                out[_col] = _val
        out["font"] = EMOTION_STYLES["neutral"]["font"]
        out["font_size"] = int(base_font)
        if "word" in out.columns:
            out["word_display"] = out["word"].astype(str)
        if "pause_dots" in out.columns:
            out["pause_dots"] = False
        if "pause_after" in out.columns:
            out["pause_after"] = 0.0
        if "line_hold_until" in out.columns:
            out["line_hold_until"] = 0.0
        return out


    @contextlib.contextmanager
    def _skip_unique_render():
        """Neutralise render_long_video for the analysis pass only."""
        _g = globals()
        _real = _g["render_long_video"]

        def _stub(*_a, **_k):
            print("  [unique] render skipped (plain-only run)")
            return ("", "")

        _g["render_long_video"] = _stub
        try:
            yield
        finally:
            _g["render_long_video"] = _real


    @contextlib.contextmanager
    def _pause_stage(enabled):
        """Temporarily override PAUSE_SPLIT_ENABLE, which process_any_video
        reads from globals at call time."""
        _g = globals()
        _old = _g["PAUSE_SPLIT_ENABLE"]
        _g["PAUSE_SPLIT_ENABLE"] = bool(enabled)
        try:
            yield
        finally:
            _g["PAUSE_SPLIT_ENABLE"] = _old


    def run_plain_only_batch(src_dir=SOURCE_VIDEO_DIR,
                             out_dir=EVAL_OUT_DIR,
                             normalise=NORMALISE_FORMAT,
                             only=None, skip_existing=False,
                             split_on_pauses=None,
                             suffix=None):
        """Re-render only the plain condition, with pure subtitle text.

        only            list of stems to restrict to
        skip_existing   default False -- the point of this run is to
                        replace files that already exist
        split_on_pauses overrides PURE_SPLIT_ON_PAUSES for one call
        suffix          overrides PURE_SUFFIX for one call
        """
        _split = (PURE_SPLIT_ON_PAUSES if split_on_pauses is None
                  else bool(split_on_pauses))
        _suffix = suffix or PURE_SUFFIX

        _vids = find_source_videos(src_dir)
        if not _vids:
            print(f"No videos found in '{src_dir}/'. cwd is {os.getcwd()}")
            return pd.DataFrame()

        os.makedirs(out_dir, exist_ok=True)
        print(f"plain-only run over {len(_vids)} video(s) in {src_dir}/")
        print(f"  pause stage   : {'ON (dots + splits)' if _split else 'OFF (pure)'}")
        print(f"  writing       : {out_dir}/video/<stem>{_suffix}.mp4\n")

        _rows = []
        for _i, _vid in enumerate(_vids, 1):
            _stem = safe_stem(_vid)
            if only and _stem not in only and _vid.stem not in only:
                continue
            _tag = f"{_stem}{_suffix}"
            _mp4 = f"{out_dir}/video/{_tag}.mp4"

            print("=" * 70)
            print(f"[{_i}/{len(_vids)}] {_vid.name}")
            print("=" * 70)

            if skip_existing and os.path.exists(_mp4):
                print("  output exists -- skipping")
                _rows.append({"source": _vid.name, "stem": _stem,
                              "status": "skipped", "plain": _mp4})
                continue

            _src = str(_vid)
            if normalise:
                print("  [format] checking geometry")
                _src = normalise_video(_src)

            try:
                with _pause_stage(_split), _skip_unique_render():
                    (_u_path, _u_ass, _seg_df, _styled,
                     _audio) = process_any_video(
                        _src, out_tag=_tag, use_bg_video=True,
                        out_dir=out_dir)

                print("  [plain] flattened control render, pure text")
                _flat = flatten_to_plain_pure(_styled, BASE_FONT_SIZE)
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
                    bg_video_path=_src, tag=_tag,
                    **PLAIN_RENDER)

                # cheap proof the text really is pure
                _dots = int(_flat["word_display"].astype(str)
                            .str.contains(r"\.\.\.|\u2026", regex=True).sum())
                print(f"  ellipsis-bearing words in output: {_dots} (want 0)")

                _rows.append({
                    "source": _vid.name, "stem": _stem, "status": "ok",
                    "plain": _p_path, "ass": _p_ass,
                    "n_words": int(len(_flat)),
                    "n_lines": int(_flat["segment_id"].nunique()),
                    "ellipses": _dots,
                })
                print(f"  DONE  plain -> {_p_path}")
            except Exception as _e:
                print(f"  FAILED on {_vid.name}: {type(_e).__name__}: {_e}")
                _rows.append({"source": _vid.name, "stem": _stem,
                              "status": f"failed: {type(_e).__name__}",
                              "plain": ""})
                continue

        _manifest = pd.DataFrame(_rows)
        if len(_manifest):
            _mpath = f"{out_dir}/manifest_{VERSION_TAG}_plainonly.csv"
            _manifest.to_csv(_mpath, index=False)
            print("\n" + "=" * 70)
            print(f"plain-only batch complete -- manifest -> {_mpath}")
            print(_manifest[["source", "status"]].to_string(index=False))
        return _manifest


    print("plain-only runner ready.")
    print(f"  pause stage default : "
          f"{'ON' if PURE_SPLIT_ON_PAUSES else 'OFF (pure subtitles)'}")
    print(f"  naming              : <stem>{PURE_SUFFIX}.mp4")
    print("\ncall run_plain_only_batch() to re-render every control video.")
    return (run_plain_only_batch,)


@app.cell
def _(run_plain_only_batch):
    run_plain_only_batch()
    return


@app.cell
def _(mo):
    mo.md("""
    ### Running the batch

    ```python
    run_evaluation_batch()                      # everything in source_clips_raw/
    run_evaluation_batch(only=["BreakingBad_Happy"])        # one stimulus
    run_evaluation_batch(skip_existing=False)   # force re-render
    run_evaluation_batch(normalise=False)       # skip the format pass
    ```

    Outputs land in `evaluationoutput/video/` as
    `<stem>v19_withUniqueSubtitles.mp4` and
    `<stem>v19_withoutUniqueSubtitles.mp4`, with the `.ass` files in
    `evaluationoutput/ass/` and a `manifest_v19.csv` recording what was
    rendered, from which source, with which emotions found and how many
    caption lines resulted.

    **Bumping `VERSION_TAG`**, so `skip_existing` will not
    find the old `v18` pairs and the batch re-renders everything. That is
    intended: the v18 files predate `merge_rapid_segments` and contain the
    0.16-second captions.

    Adding a stimulus is dropping a file into `source_clips_raw/` — the runner
    globs the folder, so nothing here is keyed to a filename.
    """)
    return


@app.cell
def _(ASR_LOOP_MIN_REPEATS, run_evaluation_batch):
    # CELL 27 — RE-RUN ONE STIMULUS  
    # ---------------------------------------------------------------------
    # BreakingBad(Happy) failed the v18 batch with "KeyError: Column not
    # found: word" and then, once recovered, rendered twelve of seventeen
    # captions for about 0.16 seconds each. Both causes are fixed upstream
    # (CELL 1b, CELL 8a, merge_rapid_segments), so this cell exists mainly
    # to re-render that one stimulus without waiting for the whole batch.
    #
    # REDO_ENABLE ships False. An auto-running render cell costs minutes on
    # every notebook start and on every upstream edit, which is exactly the
    # tax CELL 25 already charges -- one is enough. Flip it True when you
    # want the render, and read the MERGE / FLASH / PAUSE lines in the
    # output rather than watching the video to decide whether it worked.
    #
    # skip_existing=False matters: with VERSION_TAG at "v19" no v19 outputs
    # exist yet, but leaving it True would silently skip the clip as soon
    # as one render succeeds, which makes a second attempt look like a
    # no-op.
    REDO_ENABLE = False
    REDO_STEM = "BreakingBad_Happy"

    if not REDO_ENABLE:
        print(f"CELL 27 idle (REDO_ENABLE=False). Set it True to re-render "
              f"{REDO_STEM} alone, or call run_evaluation_batch() for the "
              f"whole set.")
        redo_manifest = None
    elif ASR_LOOP_MIN_REPEATS < 6:
        print(f"ASR_LOOP_MIN_REPEATS is {ASR_LOOP_MIN_REPEATS}, which is what "
              f"deleted this clip's only segment in the first place. CELL 1b "
              f"should be 6.")
        redo_manifest = None
    else:
        redo_manifest = run_evaluation_batch(
            only=[REDO_STEM], skip_existing=False)

    redo_manifest
    return


@app.cell
def _(run_evaluation_batch):
    manifest = run_evaluation_batch()
    manifest
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Decision log

    The logs for the earlier milestones live in their own files; this one covers
    what changed here.

    1. **A model bundle has to carry its own provenance.** The loader hardcoded
        one path and read four keys, which meant the file could not say what it
        was trained on — and that is precisely how the accuracy figure quoted
        throughout this notebook became untraceable: no bundle recorded the
        corpus, the fold protocol or the class count it was measured under, so
        the number outlived the model it described. The search is now a list, not
        a path, and twelve provenance keys are read defensively with `.get()`
        because an older bundle has none of them and must still load. Every run
        prints its own pedigree; a bundle without one says so. The fallback is
        loud on purpose, since a silent fall back to the previous model is the
        failure mode most likely to waste a render batch.

    2. **The palette and the label space are checked against each other before
        anything renders.** They are set in two different cells by two different
        decisions and nothing verified they agreed. The style lookup defaults to
        neutral in five places, which is deliberate — rendering must not crash
        mid-batch — but it means a class the model emits and the palette lacks
        renders as flat near-white, silently, forever. Three directions are
        checked because they fail differently: a class with no style is fatal, a
        style no class emits is inert but usually means the palette is stale, and
        a shout trigger naming an unemittable class silently never fires.

    3. **Two classes were added to the palette, and one unusable one was kept.**
        The new label space carries `frustrated` and `excited`, so both needed
        hues. `disgust` stays even though the model can rarely emit it: an unused
        style entry is inert, a missing one is a silent neutral render. The
        placements are marked as proposed rather than audited, and the recorded
        consequence is stated — the published colour-distance figures were
        computed for seven classes, so nine hues on the same wheel necessarily
        lower the worst pair, and the audit cell needs re-running before this
        feeds a study.

    4. **Capitals went back to strict, and tiered.** The emotion-driven trigger
        had been widened to four classes at the loosest sensitivity, which is
        most of what made a clip shout constantly. Anger sits on the primary bar
        and excitement is admitted only on a much higher one; happiness is
        excluded because a happy line is not a shouted line, and frustration
        because it is the sustained, quietly grinding kind of anger — exactly the
        case the standard says must not be capitalised. The acoustic detector now
        also gets an emotion veto, which reverses its original design principle
        and is flagged as such next to the switch that restores it.

    5. **The sensitivity presets exist so the tuning cannot be lost.** The
        thresholds had been hardcoded numbers reasoned against a class count and
        an accuracy that no longer applied, which is how "nobody remembers which
        way we tuned this" happens. They are named presets now, an unknown value
        raises rather than falling through, and the per-segment diagnostic
        records which tier fired and which gate blocked.

    6. **Silence became punctuation as well as space.** Widened letter spacing
        has a ceiling: a wide gap and a very wide gap look nearly alike, and
        neither conveys what a long pause does, which is *withhold* the next
        word. A line like "Who should I kill … You" arriving whole hands the
        caption reader the payload during the exact silence the speaker is using
        to withhold it — the hearing audience waits and the caption reader does
        not, which is an accessibility defect rather than a styling preference.
        One measurement now drives two renderings on two thresholds: dots on the
        same line at the lower one, dots plus a line break at the higher one.
        Three periods rather than the ellipsis glyph, because libass renders the
        single character inconsistently across the font families in use and the
        layout measures what it draws.

    7. **Both pause thresholds are reasoned, not measured, and say so.** They
        come from typical fluent-speech inter-word intervals and from the region
        where a silence stops reading as phrasing and starts reading as a held
        pause. They need calibrating against the actual stimulus set before they
        are anything more than design choices, and the renderer prints the split
        count and the first few split points with their gaps for exactly that
        purpose.

    8. **`merge_rapid_segments` is the inverse of the pause splitter.** The
        subtitle builder clamps every line to the next line's start, which
        silently overrides the minimum line duration, so on rapid exclamatory
        dialogue twelve of seventeen lines rendered for about a sixth of a second
        each — burned in correctly and impossible to read. Merging is bounded by
        a character cap so a merged line does not become its own reading-rate
        problem, and it runs *before* the splitter so a genuinely long pause can
        still cut the result. The two operate at different thresholds and must
        not fight.

    9. **The loop screen deleted a real stimulus, so it now has its own test.**
        Digits are stripped before tokenising, so `$672,000` became nothing and
        five consecutive "each"es read as a hallucination; the only segment was
        dropped and the failure surfaced two stages later as a missing column.
        The threshold rose and a coverage requirement was added — a hallucination
        loop *is* the segment, genuine repetition sits inside other speech — and
        the repeating-unit length is now derived from the token count rather than
        hardcoded, because a fixed set of gram sizes was sized to the phrases
        seen so far and cannot detect a longer cycle. A screen that can silently
        remove a clip from a study gets a self-test on both the case it exists to
        catch and the case that broke it.

    10. **One pass, two renders, and that is a methodological point rather than
        an optimisation.** The expensive stages are transcription, alignment,
        prosody and per-segment classification; the render is comparatively free.
        Running the pipeline once and rendering the styled result twice is not
        only faster, it is the only way the pair is a valid comparison — the ASR
        is not bit-deterministic, so two separate runs of the same file can
        differ in segmentation, and the two conditions would then differ in
        transcript *and* styling with no way to attribute an effect to either.

    11. **What "plain" means is a study-design choice, so it is written down.**
        The control's specification is explicit rather than implied by whatever
        the flattening code happens to do, and a stricter variant exists that
        also removes the pause marks, on the grounds that a silence is a timing
        fact but an inserted ellipsis is still an editorial addition the
        transcript did not contain.

    12. **One confound is disclosed rather than hidden.** Absolute layout
        measures text in order to place it, so a uniform font size can wrap lines
        differently from a varied one. Line breaks may therefore differ between
        the two conditions. That is inherent to the manipulation rather than a
        bug, but it belongs in the write-up rather than a claim that the pair
        differs only in styling.

    13. **Bumping the version tag re-renders everything, which is correct.** The
        tag drives every output filename and the skip check in the batch runner,
        so a bump means the existing outputs are not found and the batch runs
        again. The earlier outputs predate the merge fix and should not go in a
        study.
    """)
    return


if __name__ == "__main__":
    app.run()

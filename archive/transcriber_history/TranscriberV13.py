import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():

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

    return (
        Path,
        RandomForestClassifier,
        call,
        colorsys,
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
        whisperx,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # TranscriberV13: the audited palette

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
def _(os):
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

    features_csv = "outputs/features.csv"     # 14-feature cache (fallback model)

    iemocap_csv = "outputs/features_iemocap.csv"
    if not os.path.exists(iemocap_csv):
        iemocap_csv = "/run/media/s5812886/T7 Shield/kinetic_outputs/features_iemocap.csv"

    # ---------- FIX 3: device / precision are now detected, not hardcoded --
    try:
        import torch
        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"
    except Exception:
        device, compute_type = "cpu", "int8"

    if DATASET == "ravdess":
        audio_file = f"{ravdess_dir}/Actor_01/03-01-06-01-02-01-01.wav"
    else:
        _utt = "Ses01F_impro01_F012"
        _utt = _utt.replace(".wav", "")
        _sess = f"Session{int(_utt[3:5])}"
        _dialog = _utt.rsplit("_", 1)[0]
        audio_file = f"{iemocap_dir}/{_sess}/sentences/wav/{_dialog}/{_utt}.wav"

    out_tag = "v12_" + DATASET + "_" + audio_file.split("/")[-1].replace(".wav", "")
    print(f"dataset={DATASET}\nclip={audio_file}\ndevice={device} ({compute_type})")
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
def _():
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
    ## Part A: Clip-level classifier (load the V2 model)
    """)
    return


@app.cell
def _(call, np, parselmouth):
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
    return (clip_df,)


@app.cell
def _(RandomForestClassifier, clip_df, joblib, os):
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
    return CLF_EXTRACTOR, CLF_NORMALISED, clf_feature_cols, clf_full


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
            # V13: shared calibrated curve (cell 12b) — the inline formula
            # here and the one in cell 15 had to be kept in sync by hand
            conf   = confidence_scale(p_top, len(clf.classes_))
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

    return load_asr_model, screen_segments


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
    load_asr_model,
    screen_segments,
    whisperx,
):
    # =====================================================================
    # CELL 8b — WHISPERX TRANSCRIPTION
    # =====================================================================
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
def _(audio, device, result, whisperx):
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
    return (aligned,)


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

    return classify_words, count_syllables, looks_adj_adv, normalise_token


@app.cell
def _(looks_adj_adv):
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

    _hits = [_w for _w in _SHOULD_MATCH if looks_adj_adv(_w)]
    _false = [_w for _w in _SHOULD_NOT if looks_adj_adv(_w)]
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
def _(call, np, parselmouth, pd):
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

    return (extract_word_features,)


@app.cell
def _(aligned, audio_file, count_syllables, extract_word_features):
    # =====================================================================
    # CELL 11 — RUN PER-WORD EXTRACTION
    # =====================================================================
    word_df = extract_word_features(audio_file, aligned["word_segments"],
                                    count_syllables)
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
    BLEND_MARGIN = 0.20
    BLEND_PERWORD_SWING = 0.30

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

    # ----- V13: FONT SUBSTITUTION IS A FAILURE (audit Part 8 item 8) ----
    # In absolute layout a silent substitution desyncs measurement from
    # rendering (words measured in one face, drawn in another). True =
    # refuse to render when fontconfig resolves any family to a different
    # one; False = warn loudly and continue.
    FONT_STRICT = True

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
    EMOTION_STYLES = {
        #             hue     sat   val  italic  anim      font
        "angry":     {"h": 0.0000, "s": 0.85, "v": 0.80, "i": 0, "anim": "pop",    "font": "DejaVu Sans Condensed"},
        "happy":     {"h": 0.1400, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},
        "surprised": {"h": 0.5278, "s": 0.90, "v": 1.00, "i": 0, "anim": "pop",    "font": "DejaVu Sans"},   # 190deg cyan (was 27deg)
        "sad":       {"h": 0.6100, "s": 0.55, "v": 0.82, "i": 0, "anim": "soft",   "font": "Liberation Serif"},
        "fearful":   {"h": 0.7000, "s": 0.70, "v": 0.78, "i": 0, "anim": "tremor", "font": "DejaVu Serif"},  # arousal fix (was s.28 v.88, anim soft)
        "disgust":   {"h": 0.2611, "s": 0.55, "v": 0.72, "i": 1, "anim": "flat",   "font": "Liberation Mono"},  # 94deg bile (was 40deg)
        "neutral":   {"h": 0.0000, "s": 0.00, "v": 0.95, "i": 0, "anim": "flat",   "font": "Liberation Sans"},
    }
    return (
        ADJ_ADV_TIER,
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
        DIM_ALPHA,
        EMOLEX_PATH,
        EMOTION_STYLES,
        EMPHASIS_QUOTA_FRAC,
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
        ZERO_MEANS_MISSING,
    )


@app.cell
def _(CONF_CURVE, CONF_GAMMA, np):
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
    return (confidence_scale,)


@app.cell
def _(colorsys, np):
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
        for _cond in conditions:
            _labs = {_n: srgb_to_lab(simulate_cvd(
                emotion_rgb01(styles, _n, sat_frac), _cond)) for _n in _names}
            for _i, _a in enumerate(_names):
                for _b in _names[_i + 1:]:
                    _rows.append({"condition": _cond, "emotion_a": _a,
                                  "emotion_b": _b,
                                  "delta_e": round(delta_e_2000(_labs[_a],
                                                                _labs[_b]), 1)})
        return pd.DataFrame(_rows)

    palette_audit_df = audit_palette(EMOTION_STYLES)

    print("pairwise dE2000 by condition (full saturation):")
    for _cond in AUDIT_CONDITIONS:
        _sub = palette_audit_df[palette_audit_df["condition"] == _cond]
        _worst = _sub.nsmallest(3, "delta_e")
        _n_low = int((_sub["delta_e"] < 15.0).sum())
        print(f"  {_cond:13s} min={_sub['delta_e'].min():5.1f}  "
              f"pairs<15: {_n_low:2d}/21  worst: "
              + ", ".join(f"{_r.emotion_a}/{_r.emotion_b}={_r.delta_e}"
                          for _r in _worst.itertuples()))

    _floor_df = audit_palette(EMOTION_STYLES, sat_frac=SAT_FLOOR_FRAC)
    for _cond in ("normal", "deuteranopia"):
        _sub = _floor_df[_floor_df["condition"] == _cond]
        _w = _sub.nsmallest(1, "delta_e").iloc[0]
        print(f"  at the saturation floor (s x{SAT_FLOOR_FRAC}), {_cond}: "
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

    _arousal = {_n: vm_pad(_f["s"], _f["v"])["arousal"]
                for _n, _f in EMOTION_STYLES.items()}
    print("implied arousal ordering:",
          "  ".join(f"{_n}{_arousal[_n]:+.2f}" for _n in
                    sorted(_arousal, key=_arousal.get, reverse=True)))
    assert _arousal["fearful"] > _arousal["sad"], (
        "fear must sit above sad on implied arousal — this assert is the "
        "V12 bug (fear at -0.10 vs sad +0.08) made permanent")
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
    for _ax, _cond in zip(np.atleast_1d(_axes), _FIG_CONDS):
        _ax.set_facecolor("#3f3f3f")
        _ax.set_xlim(0, len(_emos))
        _ax.set_ylim(0, 1)
        _ax.set_xticks([]); _ax.set_yticks([])
        _sub = palette_audit_df[palette_audit_df["condition"] == _cond]
        _ax.set_ylabel(f"{_cond}\nmin dE {_sub['delta_e'].min():.1f}",
                       rotation=0, ha="right", va="center", fontsize=9)
        for _j, _e in enumerate(_emos):
            _rgb = simulate_cvd(emotion_rgb01(EMOTION_STYLES, _e), _cond)
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
def _(colorsys, np):
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
                      channel_mode="redundant"):
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

    return assign_styles, attach_motion, pause_gap


@app.cell
def _():
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
    return HAVE_PIL, ImageFont


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
def _(LIBASS_REF_SIZE, libass_scale, ref_font_metrics, subprocess):
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

    return font_vmetrics, space_width, text_width, verify_fonts


@app.cell
def _(font_vmetrics, pause_gap, space_width, text_width):
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
            txt  = str(rw["word"]).strip()
            if not txt:
                continue
            w = text_width(txt, fam, size, bold, ital, trk)
            asc, desc = font_vmetrics(fam, size, bold, ital)

            # Reserve the room a swell will need. Leading is our
            # responsibility now, and \fscy grows a word upward out of its
            # baseline — without this a lifting word on the lower row can
            # climb into the row above it.
            _gesture = str(rw.get("gesture", "none"))
            if _gesture != "none":
                k = float(rw.get("motion_strength", 0.0))
                asc_reserved = asc * (1.0 + (swell_peak / 100.0) * k)
            else:
                k = 0.0
                asc_reserved = asc

            # V13: horizontal headroom for the squash (see swell_headroom_x)
            w_adv = w * swell_headroom_x(_gesture, k, swell_peak,
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
        for _emo, _fam in EMOTION_STYLES.items():
            print(f"{_emo:10s} -> {_fam['font']}")
        _subs = verify_fonts(EMOTION_STYLES, strict=FONT_STRICT)
        if not _subs:
            print("all families resolve to themselves — measurement and "
                  "rendering agree")
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
    return conf_scale, p_second, p_top, pred_emotion, pred_emotion2


@app.cell
def _(conf_scale, pred_emotion, true_emotion):
    # =====================================================================
    # CELL 16 — PREDICTION vs LABEL VERDICT
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
def _(classify_words, np):
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
        assign_words_to_segments,
        compute_salience,
    )


@app.cell
def _(
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
    SOFTMAX_TEMPERATURE,
    USE_WORD_CLASS_PRIOR,
    ZERO_MEANS_MISSING,
    aligned,
    allocate_points,
    apply_emphasis_quota,
    apply_word_class_prior,
    assign_words_to_segments,
    compute_salience,
    np,
    result,
    word_df,
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
    return


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
    EMOTION_STYLES,
    FONT_GAMMA,
    FONT_SWING,
    MOTION_MIN_INTENSITY,
    MOTION_SOURCE,
    PAUSE_HOLD,
    PAUSE_HOLD_THRESH,
    SATURATION_INTENSITY,
    SAT_FLOOR_FRAC,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    TRACKING_CALM,
    TREMOR_WOBBLE_FACTOR,
    WOBBLE_RANGE_HZ,
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
        channel_mode=CHANNEL_MODE)
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

    Set `LAYOUT_MODE="flow"` to get the V11 renderer back for comparison.
    """)
    return


@app.cell
def _(
    GESTURE_DROP_FRAC,
    GESTURE_WOBBLE_HI,
    GESTURE_WOBBLE_LO,
    HAVE_PIL,
    layout_segment,
    pause_gap,
    squash_x,
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
            chunks.append("{" + tags_fn(rw, line_start) + "}" + str(rw["word"]).strip())
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
                txt = "{\\an5" + tags_fn(rw, s_t) + "}" + str(rw["word"]).strip()
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
            n_chars = int(seg["word"].astype(str).str.strip().str.len().sum()
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
                            tag="v13_demo"):
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

    v13_video, v13_ass = render_budget_video(
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
    print("Wrote:", v13_video, "and", v13_ass)
    return


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
    # =====================================================================
    # CELL 21 — PER-SEGMENT PREDICTION on the test clip
    # =====================================================================
    seg_emotion_df = predict_segment_emotions_v9(
        audio_file, seg_list, clf_full, clf_feature_cols,
        normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
    seg_emotion_df
    return


@app.cell
def _(
    HAVE_PIL,
    build_ass_events,
    json,
    make_word_tagger,
    os,
    subprocess,
    verify_fonts,
    write_ass,
):
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
                          bg_video_path=None, tag="v13_demo"):
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
    DIM_ALPHA,
    EMOTION_STYLES,
    EMPHASIS_QUOTA_FRAC,
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
    SEGMENT_NORM,
    SLOPE_DEADZONE,
    SLOPE_FULL,
    SOFTMAX_TEMPERATURE,
    SQUASH_CONSERVATION,
    SQUASH_STRETCH,
    TRACKING_CALM,
    TREMOR_WOBBLE_FACTOR,
    USE_WORD_CLASS_PRIOR,
    WOBBLE_RANGE_HZ,
    ZERO_MEANS_MISSING,
    allocate_points,
    apply_emphasis_quota,
    apply_word_class_prior,
    asr_model,
    assign_styles,
    assign_words_to_segments,
    attach_motion,
    clf_feature_cols,
    clf_full,
    compute_salience,
    count_syllables,
    device,
    extract_word_features,
    np,
    os,
    predict_segment_emotions_v9,
    render_long_video,
    screen_segments,
    styling_coverage,
    subprocess,
    whisperx,
):
    # =====================================================================
    # CELL 23 — "INSERT ANY VIDEO": full pipeline, real video out
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

        print("[5/6] emotion per segment + styling")
        seg_emotion_df = predict_segment_emotions_v9(
            extracted_audio, seg_list, clf_full, clf_feature_cols,
            normalise_mode=SEGMENT_NORM, norm_min_segments=NORM_MIN_SEGMENTS)
        styled_df = assign_styles(
            budget_df, seg_emotion_df, EMOTION_STYLES,
            base_font=BASE_FONT_SIZE, font_swing=FONT_SWING, font_gamma=FONT_GAMMA,
            bold_thresh=BOLD_THRESHOLD,
            saturation_intensity=SATURATION_INTENSITY, sat_floor_frac=SAT_FLOOR_FRAC,
            blend_mode=BLEND_MODE, blend_margin=BLEND_MARGIN,
            blend_perword_swing=BLEND_PERWORD_SWING,
            tracking_calm=TRACKING_CALM, calm_spacing_max=CALM_SPACING_MAX,
            channel_mode=CHANNEL_MODE)
        styled_df = attach_motion(styled_df, motion_source=MOTION_SOURCE,
                                  motion_min_intensity=MOTION_MIN_INTENSITY,
                                  slope_deadzone=SLOPE_DEADZONE, slope_full=SLOPE_FULL,
                                  wobble_range_hz=WOBBLE_RANGE_HZ,
                                  tremor_wobble_factor=TREMOR_WOBBLE_FACTOR)

        print("[6/6] rendering + burning onto the original video")
        tag = out_tag or ("anyvideo_" + stem)
        out_path, ass_path = render_long_video(
            extracted_audio, styled_df, EMOTION_STYLES,
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
        return out_path, ass_path, seg_emotion_df, styled_df

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
        long_out, long_ass, long_seg_emotions, long_styled_df = process_any_video(_demo_video)
    else:
        print("No dialog video at the expected path -- pass any video path straight "
              "into process_any_video('/path/to/your/clip.mp4').")
    return


@app.cell
def _(os, process_any_video):
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
def _(mo):
    mo.md(r"""
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


if __name__ == "__main__":
    app.run()

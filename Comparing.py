import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    # =====================================================================
    # CELL 0 — IMPORTS  (standalone notebook, nothing from V15 is reused)
    # =====================================================================
    import marimo as mo
    import os
    import re
    import json
    import time
    import gc
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, confusion_matrix,
                                 classification_report)

    # openSMILE is only needed for the legacy arm. torch and transformers
    # are only needed for the two new arms. All three are imported
    # defensively so that a missing install disables ONE arm rather than
    # stopping the whole notebook. You can get legacy numbers on a machine
    # with no GPU and no transformers.
    try:
        import opensmile
    except Exception as _e_smile:
        opensmile = None
        print(f"opensmile unavailable ({_e_smile}) — legacy arm will be off")

    try:
        import torch
    except Exception as _e_torch:
        torch = None
        print(f"torch unavailable ({_e_torch}) — wav2vec2 arm will be off")

    try:
        import soundfile as sf
    except Exception:
        sf = None

    import joblib

    return (
        LogisticRegression,
        Path,
        RandomForestClassifier,
        StandardScaler,
        StratifiedGroupKFold,
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        gc,
        json,
        make_pipeline,
        mo,
        np,
        opensmile,
        os,
        pd,
        plt,
        re,
        sf,
        time,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # SER architecture benchmark

    One question: of the three ways of deciding what emotion a segment carries,
    which one is actually most accurate on labelled data?

    Three arms, one set of folds, one label set.

    1. **legacy** — eGeMAPS functionals into a Random Forest. This is the V15
       classifier, reproduced here so the comparison is against the real
       baseline rather than a remembered number.
    2. **wav2vec2** — a frozen self-supervised audio transformer replaces
       openSMILE. Mean-pooled embeddings into a linear probe.
    3. **multimodal** — the legacy acoustic posterior fused with a text emotion
       classifier reading the transcript.

    Every arm is scored on the same speaker-independent folds, with the same
    seven classes, on the same clips. That is the only way the three numbers
    can be put in a table next to each other.
    """)
    return


@app.cell
def _(os):
    # =====================================================================
    # CELL 1 — CONFIG
    # =====================================================================
    # Which corpus to benchmark on. This choice is not cosmetic, see the
    # warning printed at the bottom of this cell.
    BENCH_DATASET = "iemocap"          # "ravdess" | "iemocap" | "both"

    RAVDESS_DIR = "/run/media/s5812886/T7 Shield/RAVDESS"
    IEMOCAP_DIR = "/run/media/s5812886/T7 Shield/IEMOCAP_full_release"
    CACHE_DIR   = "/run/media/s5812886/T7 Shield/kinetic_outputs/bench_cache"

    if not os.path.isdir(os.path.dirname(CACHE_DIR)):
        CACHE_DIR = "outputs/bench_cache"          # fall back to local disk
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Which arms to run. Turn one off if its library is missing or if you
    # only want to re-score one thing.
    RUN_LEGACY     = True
    RUN_WAV2VEC2   = True
    RUN_MULTIMODAL = True

    # Cap per class. Embedding a whole corpus through a transformer on CPU
    # is slow, so start capped, confirm the pipeline runs end to end, then
    # set None for the real run.
    MAX_PER_CLASS = 300               # None = use everything

    N_FOLDS = 5
    RANDOM_STATE = 42

    # Seven classes. RAVDESS calm is dropped; IEMOCAP excited is folded into
    # happy and its junk labels are dropped, so both corpora land on the
    # same label set and can be scored with one report.
    CANON_CLASSES = ["angry", "disgust", "fearful", "happy",
                     "neutral", "sad", "surprised"]

    RAVDESS_EMOTION_MAP = {"01": "neutral", "02": "calm", "03": "happy",
                           "04": "sad", "05": "angry", "06": "fearful",
                           "07": "disgust", "08": "surprised"}
    DROP_EMOTIONS = {"calm", "unknown", "xxx", "oth", "frustrated", "fru"}
    IEMOCAP_LABEL_FOLD = {"exc": "happy", "excited": "happy", "ang": "angry",
                          "hap": "happy", "neu": "neutral", "sad": "sad",
                          "fea": "fearful", "dis": "disgust",
                          "sur": "surprised", "surprise": "surprised"}

    # Per-speaker z-normalisation of the acoustic features. V15's clf_v2
    # uses this and it is worth a few points, because a lot of what eGeMAPS
    # measures is the speaker rather than the emotion. Statistics are
    # computed within a speaker and never use labels, so it does not leak.
    SPEAKER_NORMALISE = True

    # ---- model ids -------------------------------------------------------
    W2V2_MODEL = "facebook/wav2vec2-base"
    TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

    # Fusion. The sweep cell finds the best weight empirically; this is just
    # the value the headline table uses.
    FUSION_W_ACOUSTIC = 0.6
    FUSION_TEXT_TEMP  = 1.5

    W2V2_BATCH   = 8                  # embedding batch size
    W2V2_MAX_S   = 10.0               # truncate long clips, centre kept
    W2V2_MIN_S   = 0.5                # pad very short clips

    # Where the transcript for the multimodal arm comes from.
    #   "corpus"  IEMOCAP's own transcription files. Free, instant, and an
    #             optimistic upper bound because they are human transcripts.
    #   "asr"     run WhisperX per clip. Slower, but it is what the deployed
    #             pipeline actually sees, so it is the honest number.
    TEXT_SOURCE = "corpus"            # "corpus" | "asr"

    print(f"benchmark corpus : {BENCH_DATASET}")
    print(f"cache            : {CACHE_DIR}")
    print(f"arms             : legacy={RUN_LEGACY} wav2vec2={RUN_WAV2VEC2} "
          f"multimodal={RUN_MULTIMODAL}")
    if BENCH_DATASET == "ravdess":
        print("\n*** WARNING about the multimodal arm on RAVDESS ***")
        print("RAVDESS has exactly two sentences, both deliberately")
        print("semantically neutral, spoken in every emotion. The words")
        print("therefore carry ZERO emotion information by design. A text")
        print("classifier cannot do better than chance on them, and fusing")
        print("it in can only drag the acoustic model down. If you want a")
        print("meaningful multimodal number you need IEMOCAP, whose")
        print("dialogue is spontaneous and whose words vary with emotion.")
    return (
        BENCH_DATASET,
        CACHE_DIR,
        CANON_CLASSES,
        DROP_EMOTIONS,
        FUSION_TEXT_TEMP,
        FUSION_W_ACOUSTIC,
        IEMOCAP_DIR,
        IEMOCAP_LABEL_FOLD,
        MAX_PER_CLASS,
        N_FOLDS,
        RANDOM_STATE,
        RAVDESS_DIR,
        RAVDESS_EMOTION_MAP,
        RUN_LEGACY,
        RUN_MULTIMODAL,
        RUN_WAV2VEC2,
        SPEAKER_NORMALISE,
        TEXT_MODEL,
        TEXT_SOURCE,
        W2V2_BATCH,
        W2V2_MAX_S,
        W2V2_MIN_S,
        W2V2_MODEL,
    )


@app.cell
def _(
    BENCH_DATASET,
    CANON_CLASSES,
    DROP_EMOTIONS,
    IEMOCAP_DIR,
    IEMOCAP_LABEL_FOLD,
    MAX_PER_CLASS,
    Path,
    RANDOM_STATE,
    RAVDESS_DIR,
    RAVDESS_EMOTION_MAP,
    os,
    pd,
    re,
):
    # =====================================================================
    # CELL 2 — MANIFEST: one row per clip, with path, label, speaker, text
    #          (pandas 3 fix, see the note on the sampling step below)
    # ---------------------------------------------------------------------
    # Everything downstream indexes off this dataframe, so a clip that is
    # missing here is missing from all three arms and the comparison stays
    # aligned. Speaker ids are prefixed by corpus so RAVDESS actor 3 and
    # IEMOCAP speaker 3 cannot be treated as the same person by the group
    # split, which would leak a speaker across the fold boundary.
    # =====================================================================

    # RAVDESS ships two fixed sentences. Statement code is field 5.
    RAVDESS_STATEMENTS = {"01": "Kids are talking by the door.",
                          "02": "Dogs are sitting by the door."}

    IEM_LINE_RE = re.compile(r"^(\S+)\s+\[[\d.\s\-]+\]:\s*(.*)$")

    def load_iemocap_transcripts(root):
        """utt_id -> transcript, read from the corpus's own files."""
        out = {}
        for sess in sorted(Path(root).glob("Session*")):
            tdir = sess / "dialog" / "transcriptions"
            if not tdir.is_dir():
                continue
            for tf in sorted(tdir.glob("*.txt")):
                try:
                    for line in tf.read_text(errors="ignore").splitlines():
                        m = IEM_LINE_RE.match(line.strip())
                        if m:
                            out[m.group(1)] = m.group(2).strip()
                except Exception:
                    continue
        return out

    IEM_EMO_RE = re.compile(
        r"^\[[\d.\s\-]+\]\s+(\S+)\s+(\S+)\s+\[", re.MULTILINE)

    def load_iemocap_labels(root):
        """utt_id -> raw categorical label, from EmoEvaluation summary lines."""
        out = {}
        for sess in sorted(Path(root).glob("Session*")):
            edir = sess / "dialog" / "EmoEvaluation"
            if not edir.is_dir():
                continue
            for ef in sorted(edir.glob("*.txt")):
                try:
                    txt = ef.read_text(errors="ignore")
                except Exception:
                    continue
                for utt, lab in IEM_EMO_RE.findall(txt):
                    out[utt] = lab.strip().lower()
        return out

    def fold_label(raw):
        lab = str(raw).strip().lower()
        lab = IEMOCAP_LABEL_FOLD.get(lab, lab)
        if lab in DROP_EMOTIONS or lab not in CANON_CLASSES:
            return None
        return lab

    def build_ravdess_rows(root):
        rows = []
        for p in sorted(Path(root).glob("Actor_*/*.wav")):
            parts = p.stem.split("-")
            if len(parts) < 7:
                continue
            lab = fold_label(RAVDESS_EMOTION_MAP.get(parts[2], "unknown"))
            if lab is None:
                continue
            rows.append({"uid": p.stem, "path": str(p), "emotion": lab,
                         "speaker": f"rav_{int(parts[6]):02d}",
                         "corpus": "ravdess",
                         "text": RAVDESS_STATEMENTS.get(parts[4], "")})
        return rows

    def build_iemocap_rows(root):
        transcripts = load_iemocap_transcripts(root)
        labels = load_iemocap_labels(root)
        rows = []
        for utt, raw in labels.items():
            lab = fold_label(raw)
            if lab is None:
                continue
            try:
                sess = f"Session{int(utt[3:5])}"
            except Exception:
                continue
            dialog = utt.rsplit("_", 1)[0]
            wav = f"{root}/{sess}/sentences/wav/{dialog}/{utt}.wav"
            if not os.path.exists(wav):
                continue
            # Speaker identity is the session number plus the F/M marker in
            # the utterance id, which is the actual person, not the actor
            # whose scene it is.
            spk = f"iem_{utt[3:5]}{utt.rsplit('_', 1)[1][0]}"
            rows.append({"uid": utt, "path": wav, "emotion": lab,
                         "speaker": spk, "corpus": "iemocap",
                         "text": transcripts.get(utt, "")})
        return rows

    _rows = []
    if BENCH_DATASET in ("ravdess", "both"):
        _r = build_ravdess_rows(RAVDESS_DIR)
        print(f"RAVDESS : {len(_r)} usable clips")
        _rows += _r
    if BENCH_DATASET in ("iemocap", "both"):
        _r = build_iemocap_rows(IEMOCAP_DIR)
        print(f"IEMOCAP : {len(_r)} usable clips")
        _rows += _r

    manifest_full = pd.DataFrame(_rows)

    # ---- THE FIX -------------------------------------------------------
    # This was groupby("emotion").apply(lambda g: g.sample(...)). In pandas
    # 2.x the group being worked on still carried its own grouping column,
    # so "emotion" survived into the result. pandas 3 changed that: the
    # grouping column is now excluded from the frame handed to apply, so
    # the concatenated result came back WITHOUT an emotion column and every
    # later cell that asked for manifest["emotion"] raised KeyError.
    #
    # Shuffle once, then take the first N of each group. head() slices rows
    # out of the original frame rather than rebuilding it, so every column
    # survives and there is no apply involved to change behaviour again.
    if MAX_PER_CLASS:
        manifest = (manifest_full
                    .sample(frac=1.0, random_state=RANDOM_STATE)
                    .groupby("emotion", group_keys=False)
                    .head(MAX_PER_CLASS))
    else:
        manifest = manifest_full

    manifest = manifest.sort_values("uid").reset_index(drop=True)

    # A schema loss like the one above should stop the notebook here, where
    # it is one line to diagnose, not six cells later after a transformer
    # has spent twenty minutes embedding audio.
    _need_cols = {"uid", "path", "emotion", "speaker", "text"}
    _missing_cols = _need_cols - set(manifest.columns)
    assert not _missing_cols, f"manifest lost columns: {_missing_cols}"
    assert len(manifest), "manifest is empty: check the corpus paths"

    print(f"\nbenchmark set: {len(manifest)} clips, "
          f"{manifest['speaker'].nunique()} speakers")
    print(manifest["emotion"].value_counts().to_string())
    _no_text = int((manifest["text"].astype(str).str.strip() == "").sum())
    print(f"clips with no transcript: {_no_text} "
          f"({100*_no_text/max(len(manifest),1):.1f}%)")
    manifest.head(10)
    return (manifest,)


@app.cell
def _(
    CACHE_DIR,
    RUN_LEGACY,
    RUN_MULTIMODAL,
    manifest,
    np,
    opensmile,
    os,
    pd,
    time,
):
    # =====================================================================
    # CELL 3 — ARM 1 FEATURES: eGeMAPS functionals
    # ---------------------------------------------------------------------
    # 88 functionals per clip from openSMILE's eGeMAPSv02 set. Cached to
    # disk because extraction takes minutes and you will re-run the scoring
    # cell far more often than the extraction cell.
    # =====================================================================
    EGEMAPS_CACHE = os.path.join(CACHE_DIR, "egemaps.parquet")

    def extract_egemaps(paths):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals)
        out, bad = [], 0
        t0 = time.time()
        for i, p in enumerate(paths, 1):
            try:
                out.append(smile.process_file(str(p)).iloc[0].to_dict())
            except Exception:
                out.append(None)
                bad += 1
            if i % 250 == 0:
                print(f"  eGeMAPS {i}/{len(paths)} "
                      f"({time.time()-t0:.0f}s elapsed)")
        print(f"  eGeMAPS done: {len(paths)-bad} ok, {bad} failed")
        return out

    if (RUN_LEGACY or RUN_MULTIMODAL) and opensmile is not None:
        if os.path.exists(EGEMAPS_CACHE):
            _cached = pd.read_parquet(EGEMAPS_CACHE)
            _have = set(_cached["uid"])
            _need = manifest[~manifest["uid"].isin(_have)]
            if len(_need):
                print(f"cache hit for {len(_have)} clips, extracting "
                      f"{len(_need)} new")
                _new = extract_egemaps(_need["path"].tolist())
                _ndf = pd.DataFrame([r if r else {} for r in _new])
                _ndf.insert(0, "uid", _need["uid"].to_numpy())
                _cached = pd.concat([_cached, _ndf], ignore_index=True)
                _cached.to_parquet(EGEMAPS_CACHE, index=False)
            ege_df = _cached
        else:
            print(f"extracting eGeMAPS for {len(manifest)} clips...")
            _raw = extract_egemaps(manifest["path"].tolist())
            ege_df = pd.DataFrame([r if r else {} for r in _raw])
            ege_df.insert(0, "uid", manifest["uid"].to_numpy())
            ege_df.to_parquet(EGEMAPS_CACHE, index=False)
            print(f"cached -> {EGEMAPS_CACHE}")

        ege_df = (manifest[["uid"]].merge(ege_df, on="uid", how="left")
                  .set_index("uid").loc[manifest["uid"]].reset_index())
        EGE_COLS = [c for c in ege_df.columns if c != "uid"]
        X_ege = ege_df[EGE_COLS].to_numpy(dtype=float)
        X_ege = np.nan_to_num(X_ege, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"X_ege {X_ege.shape}")
    else:
        X_ege, EGE_COLS = None, []
        print("legacy arm off (RUN_LEGACY False or openSMILE missing)")
    return (X_ege,)


@app.cell
def _(
    CACHE_DIR,
    RUN_WAV2VEC2,
    W2V2_BATCH,
    W2V2_MAX_S,
    W2V2_MIN_S,
    W2V2_MODEL,
    gc,
    manifest,
    np,
    os,
    sf,
    time,
    torch,
):
    # =====================================================================
    # CELL 4 — ARM 2 FEATURES: wav2vec2 embeddings
    # ---------------------------------------------------------------------
    # facebook/wav2vec2-base is used as a FROZEN encoder: no fine-tuning,
    # no gradient, just the mean-pooled last hidden state as a 768-dim
    # description of the clip. That keeps the comparison honest, because
    # fine-tuning a transformer against a Random Forest is not a comparison
    # of representations, it is a comparison of budgets.
    #
    # Batch size is forced to 1 when the feature extractor returns no
    # attention mask. wav2vec2-base was pretrained without one, so padding
    # a batch feeds real zeros into the convolutional front end and quietly
    # corrupts every pooled vector in the batch.
    # =====================================================================
    W2V2_CACHE = os.path.join(CACHE_DIR, "w2v2_base_emb.npz")

    def read_wav_16k(path, min_s=W2V2_MIN_S, max_s=W2V2_MAX_S, sr=16000):
        if sf is None:
            raise RuntimeError("soundfile not installed: pip install soundfile")
        wav, in_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if in_sr != sr:                      # linear resample, no extra dep
            n_out = int(round(len(wav) * sr / float(in_sr)))
            wav = np.interp(np.linspace(0, len(wav) - 1, n_out),
                            np.arange(len(wav)), wav).astype(np.float32)
        need = int(min_s * sr)
        if len(wav) < need:
            wav = np.pad(wav, (0, need - len(wav)))
        cap = int(max_s * sr)
        if len(wav) > cap:
            off = (len(wav) - cap) // 2
            wav = wav[off:off + cap]
        return wav.astype(np.float32)

    def embed_wav2vec2(paths, model_id=W2V2_MODEL, batch=W2V2_BATCH):
        from transformers import AutoFeatureExtractor, AutoModel
        dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        fe = AutoFeatureExtractor.from_pretrained(model_id)
        mdl = AutoModel.from_pretrained(model_id).to(dev).eval()
        if batch > 1 and not getattr(fe, "return_attention_mask", False):
            print("  batch forced to 1: no attention mask from this extractor")
            batch = 1
        out, t0 = [], time.time()
        with torch.no_grad():
            for i in range(0, len(paths), batch):
                chunk = [read_wav_16k(p) for p in paths[i:i + batch]]
                inp = fe(chunk, sampling_rate=16000, return_tensors="pt",
                         padding=True)
                inp = {k: v.to(dev) for k, v in inp.items()}
                hid = mdl(**inp).last_hidden_state
                mask = inp.get("attention_mask")
                if mask is not None:
                    try:
                        lens = mdl._get_feat_extract_output_lengths(
                            mask.sum(-1)).clamp(max=hid.shape[1])
                        fm = (torch.arange(hid.shape[1], device=dev)[None, :]
                              < lens[:, None]).unsqueeze(-1).to(hid.dtype)
                        pooled = (hid * fm).sum(1) / fm.sum(1).clamp(min=1.0)
                    except Exception:
                        pooled = hid.mean(1)
                else:
                    pooled = hid.mean(1)
                out.append(pooled.float().cpu().numpy())
                if (i // max(batch, 1)) % 25 == 0:
                    print(f"  w2v2 {min(i+batch, len(paths))}/{len(paths)} "
                          f"({time.time()-t0:.0f}s, device={dev})")
        # give the card back before anything else asks for it
        mdl.to("cpu")
        del mdl, fe
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return np.vstack(out)

    if RUN_WAV2VEC2 and torch is not None:
        _uids = manifest["uid"].to_numpy()
        if os.path.exists(W2V2_CACHE):
            _z = np.load(W2V2_CACHE, allow_pickle=True)
            _lut = {u: i for i, u in enumerate(_z["uids"])}
            _missing = [u for u in _uids if u not in _lut]
            if _missing:
                print(f"cache hit {len(_uids)-len(_missing)}, embedding "
                      f"{len(_missing)} new")
                _mp = manifest.set_index("uid").loc[_missing, "path"].tolist()
                _newE = embed_wav2vec2(_mp)
                _allU = np.concatenate([_z["uids"], np.array(_missing)])
                _allE = np.vstack([_z["E"], _newE])
                np.savez_compressed(W2V2_CACHE, uids=_allU, E=_allE)
                _lut = {u: i for i, u in enumerate(_allU)}
                _E = _allE
            else:
                _E = _z["E"]
                print(f"loaded cached embeddings {_E.shape}")
        else:
            print(f"embedding {len(_uids)} clips with {W2V2_MODEL}...")
            _E = embed_wav2vec2(manifest["path"].tolist())
            np.savez_compressed(W2V2_CACHE, uids=_uids, E=_E)
            _lut = {u: i for i, u in enumerate(_uids)}
            print(f"cached -> {W2V2_CACHE}")
        X_w2v = np.vstack([_E[_lut[u]] for u in _uids])
        print(f"X_w2v {X_w2v.shape}")
    else:
        X_w2v = None
        print("wav2vec2 arm off (RUN_WAV2VEC2 False or torch missing)")
    return (X_w2v,)


@app.cell
def _(
    CACHE_DIR,
    CANON_CLASSES,
    RUN_MULTIMODAL,
    TEXT_MODEL,
    gc,
    manifest,
    np,
    os,
    torch,
):
    # =====================================================================
    # CELL 5 — ARM 3 FEATURES: text emotion posteriors
    #          (now caches per clip instead of all-or-nothing)
    # ---------------------------------------------------------------------
    # j-hartmann/emotion-english-distilroberta-base emits anger, disgust,
    # fear, joy, neutral, sadness, surprise, which is a one-to-one match
    # onto the seven classes left after calm is dropped. It is used
    # zero-shot, with no training on this corpus, so there is no fold to
    # leak across and the same posterior can be reused in every fold.
    #
    # The old cache logic was all-or-nothing: one new clip in the manifest
    # and the whole corpus got re-scored. It now scores only the uids it
    # has never seen and merges them into the existing file, so changing
    # MAX_PER_CLASS costs seconds rather than a full pass.
    # =====================================================================
    TEXT_CACHE = os.path.join(CACHE_DIR, "text_probs.npz")

    LABEL_KEYS = [
        ("angry",     ("ang", "anger", "angry", "mad")),
        ("disgust",   ("dis", "disgust", "disgusted")),
        ("fearful",   ("fea", "fear", "fearful", "scared", "afraid")),
        ("happy",     ("hap", "happy", "happiness", "joy", "joyful",
                       "exc", "excited")),
        ("neutral",   ("neu", "neutral", "calm")),
        ("sad",       ("sad", "sadness", "sorrow")),
        ("surprised", ("sur", "surprise", "surprised", "amazed")),
    ]

    def canon_label(raw):
        s = str(raw).strip().lower().replace("label_", "")
        for canon, keys in LABEL_KEYS:
            if s in keys:
                return canon
        for canon, keys in LABEL_KEYS:
            for k in keys:
                if s.startswith(k) or k.startswith(s):
                    return canon
        return None

    def align_probs(raw_probs, raw_labels, classes, floor=0.005):
        """Foreign posterior -> canonical class order. Classes the foreign
        model has no head for get `floor`, then the row is renormalised."""
        P = np.atleast_2d(np.asarray(raw_probs, dtype=float))
        classes = list(classes)
        out = np.full((P.shape[0], len(classes)), float(floor))
        hit = set()
        for j, lab in enumerate(raw_labels):
            c = canon_label(lab)
            if c is None or c not in classes:
                continue
            ci = classes.index(c)
            if ci not in hit:
                out[:, ci] = 0.0
                hit.add(ci)
            out[:, ci] += P[:, j]
        out = np.clip(out, 1e-9, None)
        return out / out.sum(axis=1, keepdims=True), sorted(
            classes[i] for i in hit)

    def text_posteriors(texts, classes, model_id=TEXT_MODEL, batch=32):
        from transformers import pipeline
        dev = 0 if (torch is not None and torch.cuda.is_available()) else -1
        pipe = pipeline("text-classification", model=model_id,
                        top_k=None, device=dev, truncation=True)
        texts = [str(t or "").strip() for t in texts]
        has = np.array([len(t) > 0 for t in texts], dtype=bool)
        P = np.full((len(texts), len(classes)), 1.0 / len(classes))
        idx = [i for i, h in enumerate(has) if h]
        for b in range(0, len(idx), batch):
            sl = idx[b:b + batch]
            res = pipe([texts[i] for i in sl])
            if res and isinstance(res[0], dict):
                res = [res]
            for i, r in zip(sl, res):
                row, _ = align_probs([[float(d["score"]) for d in r]],
                                     [d["label"] for d in r], classes)
                P[i] = row[0]
            if (b // batch) % 20 == 0:
                print(f"  text {min(b+batch, len(idx))}/{len(idx)}")
        try:
            pipe.model.to("cpu")
        except Exception:
            pass
        del pipe
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return P, has

    if RUN_MULTIMODAL:
        _uids = manifest["uid"].to_numpy()
        _texts = manifest["text"].tolist()

        _cU = np.array([], dtype=object)
        _cP = np.zeros((0, len(CANON_CLASSES)))
        _cH = np.array([], dtype=bool)
        if os.path.exists(TEXT_CACHE):
            _z = np.load(TEXT_CACHE, allow_pickle=True)
            _cU, _cP, _cH = _z["uids"], _z["P"], _z["has"]

        _lut = {u: i for i, u in enumerate(_cU)}
        _todo = [i for i, u in enumerate(_uids) if u not in _lut]
        if _todo:
            print(f"cache has {len(_lut)}; scoring {len(_todo)} new "
                  f"transcripts with {TEXT_MODEL}...")
            _nP, _nH = text_posteriors([_texts[i] for i in _todo],
                                       CANON_CLASSES)
            _cU = np.concatenate([_cU, _uids[_todo]])
            _cP = np.vstack([_cP, _nP]) if len(_cP) else _nP
            _cH = np.concatenate([_cH, _nH])
            np.savez_compressed(TEXT_CACHE, uids=_cU, P=_cP, has=_cH)
            _lut = {u: i for i, u in enumerate(_cU)}
            print(f"cached -> {TEXT_CACHE}")
        else:
            print(f"all {len(_uids)} text posteriors served from cache")

        P_text = np.vstack([_cP[_lut[u]] for u in _uids])
        has_text = np.array([bool(_cH[_lut[u]]) for u in _uids])

        # A sanity number worth printing: how good is TEXT ALONE? On IEMOCAP
        # this is usually surprisingly high. On RAVDESS it will sit at
        # chance, which is the design of the corpus showing through.
        _txt_pred = np.array(CANON_CLASSES)[P_text.argmax(axis=1)]
        _acc = float((_txt_pred == manifest["emotion"].to_numpy()).mean())
        print(f"text-only accuracy (zero-shot, no training): {_acc:.3f}  "
              f"| chance {1/len(CANON_CLASSES):.3f}")
        print(f"clips with usable text: {int(has_text.sum())}/{len(has_text)}")
    else:
        P_text, has_text = None, None
        print("multimodal arm off")
    return P_text, has_text


@app.cell
def _(
    CANON_CLASSES,
    N_FOLDS,
    RANDOM_STATE,
    StratifiedGroupKFold,
    manifest,
    np,
):
    # =====================================================================
    # CELL 6 — THE FOLDS
    # ---------------------------------------------------------------------
    # Computed ONCE and shared by all three arms. This is the single most
    # important cell in the notebook: if each arm generated its own split,
    # the differences between the three numbers would partly be differences
    # between three random splits, and the comparison would be worthless.
    #
    # Grouping by speaker means no speaker appears in both train and test.
    # Without that a classifier can score well by recognising the person,
    # and speaker-dependent accuracy on RAVDESS is roughly twenty points
    # higher than speaker-independent, which is why the literature reports
    # the latter.
    # =====================================================================
    y_all = manifest["emotion"].to_numpy()
    groups_all = manifest["speaker"].to_numpy()
    classes_used = [c for c in CANON_CLASSES if c in set(y_all)]

    _sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=RANDOM_STATE)
    folds = list(_sgkf.split(np.zeros(len(y_all)), y_all, groups=groups_all))

    print(f"{len(folds)} folds over {len(y_all)} clips, "
          f"{len(set(groups_all))} speakers, {len(classes_used)} classes")
    for _i, (_tr, _te) in enumerate(folds):
        _spk_tr = set(groups_all[_tr]); _spk_te = set(groups_all[_te])
        assert not (_spk_tr & _spk_te), "speaker leaked across a fold"
        print(f"  fold {_i}: train {len(_tr):5d} / test {len(_te):5d} "
              f"| test speakers {sorted(_spk_te)}")
    print("no speaker appears on both sides of any fold")
    return classes_used, folds, groups_all, y_all


@app.cell
def _(np):
    # =====================================================================
    # CELL 7 — HELPERS: speaker normalisation, fusion, prob reindexing
    # =====================================================================
    def speaker_normalise(X, groups, train_idx=None):
        """Z-score each feature within each speaker.

        No labels are involved, so this is not leakage in the usual sense;
        it is the same operation V15 applies at inference time, where a
        clip's own segments provide the statistics. Speakers unseen in
        training are normalised with their own statistics, which is exactly
        the deployed condition.
        """
        Xn = np.array(X, dtype=float, copy=True)
        for g in np.unique(groups):
            m = (groups == g)
            mu = Xn[m].mean(axis=0)
            sd = Xn[m].std(axis=0)
            sd[sd == 0] = 1.0
            Xn[m] = (Xn[m] - mu) / sd
        return np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    def reindex_proba(proba, model_classes, target_classes):
        """sklearn orders columns by its own classes_. Put them in the
        canonical order so two models can be fused or compared."""
        P = np.zeros((proba.shape[0], len(target_classes)))
        lut = {c: i for i, c in enumerate(target_classes)}
        for j, c in enumerate(model_classes):
            if c in lut:
                P[:, lut[c]] = proba[:, j]
        s = P.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return P / s

    def temper(P, t=1.0):
        P = np.clip(np.asarray(P, dtype=float), 1e-12, None)
        if abs(float(t) - 1.0) < 1e-9:
            return P / P.sum(axis=1, keepdims=True)
        Q = P ** (1.0 / float(t))
        return Q / Q.sum(axis=1, keepdims=True)

    def late_fuse(P_acoustic, P_text_rows, has_text_rows, w_acoustic=0.6,
                  text_temp=1.0):
        """Weighted average of two posteriors, per row.

        A row with no transcript gets weight 1.0 on acoustic, which is the
        fallback: no text means no vote, not a uniform vote averaged in.
        """
        A = np.clip(np.asarray(P_acoustic, dtype=float), 1e-12, None)
        A = A / A.sum(axis=1, keepdims=True)
        T = temper(P_text_rows, text_temp)
        w = np.where(np.asarray(has_text_rows, dtype=bool),
                     float(w_acoustic), 1.0).reshape(-1, 1)
        F = w * A + (1.0 - w) * T
        return F / F.sum(axis=1, keepdims=True)

    return late_fuse, reindex_proba, speaker_normalise


@app.cell
def _(
    FUSION_TEXT_TEMP,
    FUSION_W_ACOUSTIC,
    LogisticRegression,
    P_text,
    RUN_LEGACY,
    RUN_MULTIMODAL,
    RUN_WAV2VEC2,
    RandomForestClassifier,
    SPEAKER_NORMALISE,
    StandardScaler,
    X_ege,
    X_w2v,
    classes_used,
    folds,
    groups_all,
    has_text,
    late_fuse,
    make_pipeline,
    np,
    reindex_proba,
    speaker_normalise,
    time,
    y_all,
):
    # =====================================================================
    # CELL 8 — RUN THE BENCHMARK
    # ---------------------------------------------------------------------
    # Out-of-fold prediction: every clip is predicted exactly once, by a
    # model that never saw its speaker. The stored posteriors let the later
    # cells sweep fusion weights and run significance tests without
    # retraining anything.
    # =====================================================================
    K = len(classes_used)
    N = len(y_all)

    oof = {}          # arm -> (N, K) posterior
    fold_acc = {}     # arm -> list of per-fold accuracies
    timings = {}

    if RUN_LEGACY and X_ege is not None:
        _Xl = speaker_normalise(X_ege, groups_all) if SPEAKER_NORMALISE else X_ege
        _P = np.zeros((N, K)); _fa = []
        _t0 = time.time()
        for _fi, (_tr, _te) in enumerate(folds):
            _clf = RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=-1).fit(_Xl[_tr], y_all[_tr])
            _P[_te] = reindex_proba(_clf.predict_proba(_Xl[_te]),
                                    list(_clf.classes_), classes_used)
            _a = float((np.array(classes_used)[_P[_te].argmax(1)] ==
                        y_all[_te]).mean())
            _fa.append(_a)
            print(f"  legacy fold {_fi}: {_a:.3f}")
        oof["legacy"] = _P; fold_acc["legacy"] = _fa
        timings["legacy"] = time.time() - _t0

    if RUN_WAV2VEC2 and X_w2v is not None:
        _P = np.zeros((N, K)); _fa = []
        _t0 = time.time()
        for _fi, (_tr, _te) in enumerate(folds):
            _clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, C=1.0,
                                   class_weight="balanced")
            ).fit(X_w2v[_tr], y_all[_tr])
            _P[_te] = reindex_proba(_clf.predict_proba(X_w2v[_te]),
                                    list(_clf.classes_), classes_used)
            _a = float((np.array(classes_used)[_P[_te].argmax(1)] ==
                        y_all[_te]).mean())
            _fa.append(_a)
            print(f"  wav2vec2 fold {_fi}: {_a:.3f}")
        oof["wav2vec2"] = _P; fold_acc["wav2vec2"] = _fa
        timings["wav2vec2"] = time.time() - _t0

    if RUN_MULTIMODAL and P_text is not None and "legacy" in oof:
        _P = late_fuse(oof["legacy"], P_text, has_text,
                       w_acoustic=FUSION_W_ACOUSTIC,
                       text_temp=FUSION_TEXT_TEMP)
        oof["multimodal"] = _P
        fold_acc["multimodal"] = [
            float((np.array(classes_used)[_P[_te].argmax(1)] ==
                   y_all[_te]).mean()) for _tr, _te in folds]
        for _fi, _a in enumerate(fold_acc["multimodal"]):
            print(f"  multimodal fold {_fi}: {_a:.3f}")

    # bonus arm, free: the transformer embeddings fused with text
    if "wav2vec2" in oof and P_text is not None and RUN_MULTIMODAL:
        _P = late_fuse(oof["wav2vec2"], P_text, has_text,
                       w_acoustic=FUSION_W_ACOUSTIC,
                       text_temp=FUSION_TEXT_TEMP)
        oof["w2v2+text"] = _P
        fold_acc["w2v2+text"] = [
            float((np.array(classes_used)[_P[_te].argmax(1)] ==
                   y_all[_te]).mean()) for _tr, _te in folds]

    # text alone, for reference
    if P_text is not None and RUN_MULTIMODAL:
        oof["text_only"] = P_text
        fold_acc["text_only"] = [
            float((np.array(classes_used)[P_text[_te].argmax(1)] ==
                   y_all[_te]).mean()) for _tr, _te in folds]

    print(f"\narms scored: {list(oof.keys())}")
    return fold_acc, oof


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    classes_used,
    f1_score,
    fold_acc,
    np,
    oof,
    pd,
    y_all,
):
    # =====================================================================
    # CELL 9 — THE NUMBERS
    # ---------------------------------------------------------------------
    # Accuracy is the headline, but on a class-imbalanced corpus it can be
    # bought by predicting the majority class, so balanced accuracy (which
    # is unweighted average recall, the standard SER metric) and macro F1
    # are reported next to it. If accuracy and balanced accuracy disagree
    # sharply, trust balanced accuracy.
    # =====================================================================
    _rows = []
    for _arm, _P in oof.items():
        _pred = np.array(classes_used)[_P.argmax(axis=1)]
        _rows.append({
            "arm": _arm,
            "accuracy": accuracy_score(y_all, _pred),
            "balanced_acc": balanced_accuracy_score(y_all, _pred),
            "macro_f1": f1_score(y_all, _pred, average="macro",
                                 zero_division=0),
            "fold_mean": float(np.mean(fold_acc[_arm])),
            "fold_std": float(np.std(fold_acc[_arm])),
        })

    results = (pd.DataFrame(_rows)
               .sort_values("balanced_acc", ascending=False)
               .reset_index(drop=True))
    _chance = 1.0 / len(classes_used)

    print(f"{len(y_all)} clips, {len(classes_used)} classes, "
          f"chance = {_chance:.3f}, speaker-independent\n")
    print(results.to_string(
        index=False,
        formatters={c: "{:.3f}".format for c in
                    ("accuracy", "balanced_acc", "macro_f1",
                     "fold_mean", "fold_std")}))

    _best = results.iloc[0]
    print(f"\nWINNER: {_best['arm']} at {_best['balanced_acc']:.3f} "
          f"balanced accuracy ({_best['accuracy']:.3f} raw), "
          f"{_best['balanced_acc']/_chance:.1f}x chance")
    results
    return (results,)


@app.cell
def _(
    classes_used,
    classification_report,
    confusion_matrix,
    np,
    oof,
    plt,
    y_all,
):
    # =====================================================================
    # CELL 10 — PER-CLASS BREAKDOWN + CONFUSION MATRICES
    # ---------------------------------------------------------------------
    # The aggregate number hides the thing you actually need for the
    # dissertation: WHICH emotions each architecture gets right. If the
    # transformer wins overall but loses on disgust and surprised, and
    # those are the two classes your palette leans on hardest, the overall
    # win is not the whole story.
    # =====================================================================
    for _arm, _P in oof.items():
        _pred = np.array(classes_used)[_P.argmax(axis=1)]
        print(f"\n===== {_arm} =====")
        print(classification_report(y_all, _pred, labels=classes_used,
                                    zero_division=0, digits=3))

    _n = len(oof)
    _fig, _axes = plt.subplots(1, _n, figsize=(4.2 * _n, 4.0))
    if _n == 1:
        _axes = [_axes]
    for _ax, (_arm, _P) in zip(_axes, oof.items()):
        _pred = np.array(classes_used)[_P.argmax(axis=1)]
        _cm = confusion_matrix(y_all, _pred, labels=classes_used,
                               normalize="true")
        _ax.imshow(_cm, vmin=0, vmax=1, cmap="magma")
        _ax.set_title(_arm, fontsize=10)
        _ax.set_xticks(range(len(classes_used)))
        _ax.set_xticklabels(classes_used, rotation=90, fontsize=7)
        _ax.set_yticks(range(len(classes_used)))
        _ax.set_yticklabels(classes_used, fontsize=7)
        for _i in range(len(classes_used)):
            for _j in range(len(classes_used)):
                _ax.text(_j, _i, f"{_cm[_i, _j]:.2f}", ha="center",
                         va="center", fontsize=6,
                         color=("white" if _cm[_i, _j] < 0.5 else "black"))
    _axes[0].set_ylabel("true")
    _fig.suptitle("Row-normalised confusion, speaker-independent out-of-fold")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(
    P_text,
    RUN_MULTIMODAL,
    classes_used,
    has_text,
    late_fuse,
    np,
    oof,
    pd,
    plt,
    y_all,
):
    # =====================================================================
    # CELL 11 — FUSION WEIGHT SWEEP
    # ---------------------------------------------------------------------
    # FUSION_W_ACOUSTIC = 0.6 was a design choice, not a measurement. This
    # sweeps it so the dissertation can report the value that actually
    # maximises accuracy alongside the value the renderer ships with, and
    # can say honestly whether the two differ. w=1.0 IS the legacy arm, so
    # the left-hand end of the curve is the baseline by construction.
    # =====================================================================
    if RUN_MULTIMODAL and P_text is not None:
        _grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
        _rows = []
        for _base in [k for k in ("legacy", "wav2vec2") if k in oof]:
            for _w in _grid:
                _F = late_fuse(oof[_base], P_text, has_text,
                               w_acoustic=_w, text_temp=1.5)
                _pred = np.array(classes_used)[_F.argmax(axis=1)]
                _rows.append({"base": _base, "w_acoustic": float(_w),
                              "accuracy": float((_pred == y_all).mean())})
        sweep = pd.DataFrame(_rows)

        _fig2, _ax2 = plt.subplots(figsize=(7, 4))
        for _base, _g in sweep.groupby("base"):
            _ax2.plot(_g["w_acoustic"], _g["accuracy"], marker="o",
                      markersize=3, label=_base)
            _b = _g.loc[_g["accuracy"].idxmax()]
            _ax2.scatter([_b["w_acoustic"]], [_b["accuracy"]], s=70,
                         zorder=5, facecolors="none", edgecolors="black")
            print(f"{_base}: best w_acoustic = {_b['w_acoustic']:.2f} "
                  f"at {_b['accuracy']:.3f} "
                  f"(w=1.0, acoustic only = "
                  f"{_g[_g['w_acoustic']==1.0]['accuracy'].iloc[0]:.3f})")
        _ax2.set_xlabel("weight on the acoustic branch "
                        "(1.0 = acoustic only, 0.0 = text only)")
        _ax2.set_ylabel("out-of-fold accuracy")
        _ax2.legend(); _ax2.grid(alpha=0.3)
        _ax2.set_title("Late fusion weight sweep")
        _fig2.tight_layout()
        _fig2
    else:
        sweep = None
        print("sweep skipped: multimodal arm is off")
    return (sweep,)


@app.cell
def _(classes_used, fold_acc, np, oof, pd, y_all):
    # =====================================================================
    # CELL 12 — IS THE DIFFERENCE REAL?
    # ---------------------------------------------------------------------
    # Two tests, because they answer different questions.
    #
    # McNemar works on the paired per-clip outcomes and asks whether one
    # system is right where the other is wrong more often than chance would
    # allow. It is the right test for comparing two classifiers on the same
    # test set, and it does not care about fold structure.
    #
    # The per-fold difference is coarser (five paired numbers) but it is
    # what most SER papers report, so it is here for comparability.
    # =====================================================================
    def mcnemar_exact(correct_a, correct_b):
        """Exact binomial McNemar. -> (b, c, p) where b is 'a right, b
        wrong' and c is the reverse."""
        from scipy.stats import binomtest
        b = int(np.sum(correct_a & ~correct_b))
        c = int(np.sum(~correct_a & correct_b))
        if b + c == 0:
            return b, c, 1.0
        return b, c, float(binomtest(b, b + c, 0.5).pvalue)

    _correct = {}
    for _arm, _P in oof.items():
        _pred = np.array(classes_used)[_P.argmax(axis=1)]
        _correct[_arm] = (_pred == y_all)

    _arms = list(_correct.keys())
    _rows = []
    for _i in range(len(_arms)):
        for _j in range(_i + 1, len(_arms)):
            _a, _b = _arms[_i], _arms[_j]
            try:
                _nb, _nc, _p = mcnemar_exact(_correct[_a], _correct[_b])
            except Exception as _e:
                _nb = _nc = -1; _p = float("nan")
                print(f"  (scipy missing? {_e})")
            _da = float(np.mean(fold_acc[_a]) - np.mean(fold_acc[_b]))
            _rows.append({"A": _a, "B": _b,
                          "A_right_B_wrong": _nb, "B_right_A_wrong": _nc,
                          "mcnemar_p": _p, "fold_mean_diff": _da})
    sig = pd.DataFrame(_rows)
    print(sig.to_string(index=False,
                        formatters={"mcnemar_p": "{:.4g}".format,
                                    "fold_mean_diff": "{:+.3f}".format}))
    print("\np < 0.05 means the two systems disagree in a way that is "
          "unlikely to be sampling noise. A big accuracy gap with a large "
          "p usually means too few test clips, not a real tie.")
    sig
    return (sig,)


@app.cell
def _(
    BENCH_DATASET,
    CACHE_DIR,
    MAX_PER_CLASS,
    N_FOLDS,
    SPEAKER_NORMALISE,
    TEXT_SOURCE,
    classes_used,
    json,
    np,
    oof,
    os,
    pd,
    results,
    sig,
    sweep,
    y_all,
):
    # =====================================================================
    # CELL 13 — SAVE EVERYTHING FOR THE WRITE-UP
    # ---------------------------------------------------------------------
    # Out-of-fold posteriors are saved, not just the summary table, so the
    # methodology chapter can be re-analysed months later without re-running
    # a single transformer.
    # =====================================================================
    _stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    _base = os.path.join(CACHE_DIR, f"bench_{BENCH_DATASET}_{_stamp}")

    results.to_csv(_base + "_summary.csv", index=False)
    if sig is not None:
        sig.to_csv(_base + "_significance.csv", index=False)
    if sweep is not None:
        sweep.to_csv(_base + "_fusion_sweep.csv", index=False)
    np.savez_compressed(_base + "_oof.npz",
                        y=y_all, classes=np.array(classes_used),
                        **{f"P_{k}": v for k, v in oof.items()})
    with open(_base + "_config.json", "w") as _f:
        json.dump({"dataset": BENCH_DATASET, "n_clips": int(len(y_all)),
                   "n_folds": N_FOLDS, "classes": classes_used,
                   "max_per_class": MAX_PER_CLASS,
                   "speaker_normalise": SPEAKER_NORMALISE,
                   "text_source": TEXT_SOURCE}, _f, indent=2)
    print(f"saved:\n  {_base}_summary.csv\n  {_base}_oof.npz\n"
          f"  {_base}_config.json")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

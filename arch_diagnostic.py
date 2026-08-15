#!/usr/bin/env python3
"""
arch_diagnostic.py — settle the architecture-benchmark question.

Runs eGeMAPS+RandomForest and frozen wav2vec2 on the SAME five IEMOCAP
leave-one-session-out folds used by Train_EmotionV3 CELL 9, and reports
accuracy, macro-F1 and balanced accuracy for both.

Why this exists
---------------
The thesis currently claims a neural encoder does not beat hand-crafted
features "on this data", but the original benchmark used a different fold
scheme and its corpus is unverified. This re-runs the comparison under the
shipped protocol so the claim is either supported or corrected.

Usage
-----
    python arch_diagnostic.py --part a      # provenance only, seconds, no GPU
    python arch_diagnostic.py --part b      # full comparison (needs torch)
    python arch_diagnostic.py               # both

Outputs a JSON blob at the end. Paste that back and the thesis text can be
updated from measured numbers.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIG — must match Train_EmotionV3 CELL 1
# --------------------------------------------------------------------------
OUT_DIR = "outputs"
EXTRACTOR = "egemaps"
COMBINED_CSV = f"{OUT_DIR}/features_combined_{EXTRACTOR}.csv"
BUNDLE_PATH = f"{OUT_DIR}/clf_v3.joblib"
W2V_CACHE = f"{OUT_DIR}/features_wav2vec2.npz"

RAVDESS_WEIGHT = 0.20
RF_KWARGS = dict(n_estimators=600, random_state=42, n_jobs=-1,
                 class_weight="balanced_subsample", min_samples_leaf=2)

NORM_SCOPE = "dialog"       # CELL 8 default
NORM_MIN_ROWS = 4
NORM_STD_FLOOR = 1e-6

W2V_MODEL = "facebook/wav2vec2-base"
W2V_SR = 16000
W2V_MAX_SECONDS = 12.0      # clip long utterances so memory stays bounded


# ==========================================================================
# PART A — provenance. Answers "what corpus / label space is in play".
# ==========================================================================
def part_a():
    print("=" * 70)
    print("PART A — PROVENANCE")
    print("=" * 70)
    report = {}

    # --- the shipped bundle -------------------------------------------------
    if os.path.exists(BUNDLE_PATH):
        import joblib
        b = joblib.load(BUNDLE_PATH)
        clf = b["clf"]
        classes = [str(c) for c in clf.classes_]
        report["bundle"] = {
            "path": BUNDLE_PATH,
            "n_features": len(b["feature_cols"]),
            "extractor": b.get("extractor"),
            "classes": classes,
            "n_classes": len(classes),
            "chance": round(1.0 / len(classes), 4),
            "cv_protocol": b.get("cv_protocol", "<unrecorded>"),
            "cv_accuracy": b.get("cv_pooled_accuracy"),
            "cv_macro_f1": b.get("cv_pooled_macro_f1"),
            "ravdess_weight": b.get("ravdess_weight"),
            "norm_scope": b.get("norm_scope", "<unrecorded>"),
        }
        print(f"bundle      : {BUNDLE_PATH}")
        print(f"  extractor : {b.get('extractor')}  "
              f"({len(b['feature_cols'])} features)")
        print(f"  classes   : {len(classes)} -> {classes}")
        print(f"  chance    : {1.0/len(classes):.4f}")
        print(f"  cv        : {b.get('cv_protocol', '<unrecorded>')}")
        print(f"              acc={b.get('cv_pooled_accuracy')}  "
              f"macroF1={b.get('cv_pooled_macro_f1')}")

        # The corpus tell: these two classes exist only in IEMOCAP.
        iem_only = {"frustrated", "excited"} & set(classes)
        rav_only = {"calm"} & set(classes)
        if iem_only:
            verdict = (f"IEMOCAP-era label space (contains {sorted(iem_only)}, "
                       f"which RAVDESS does not have)")
        elif rav_only:
            verdict = (f"RAVDESS-era label space (contains {sorted(rav_only)}, "
                       f"which IEMOCAP does not have)")
        else:
            verdict = "ambiguous — no corpus-exclusive class present"
        report["bundle"]["corpus_verdict"] = verdict
        print(f"  VERDICT   : {verdict}")
    else:
        print(f"bundle      : NOT FOUND at {BUNDLE_PATH}")
        report["bundle"] = None

    # --- the feature table --------------------------------------------------
    if os.path.exists(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV)
        by_source = df.groupby("source")["label"].value_counts().unstack(
            fill_value=0)
        report["combined_csv"] = {
            "path": COMBINED_CSV,
            "n_rows": len(df),
            "by_source": df["source"].value_counts().to_dict(),
            "sessions": sorted(df[df["source"] == "iemocap"]
                               ["session"].unique().tolist()),
            "class_counts_by_source": by_source.to_dict(),
        }
        print(f"\nfeature csv : {COMBINED_CSV}  ({len(df)} rows)")
        print(f"  by source : {df['source'].value_counts().to_dict()}")
        print(f"  sessions  : "
              f"{sorted(df[df['source']=='iemocap']['session'].unique())}")
        print("\n  class counts by source:")
        print(by_source.to_string())
    else:
        print(f"\nfeature csv : NOT FOUND at {COMBINED_CSV}")
        print("  -> run Train_EmotionV3 through CELL 7 first.")
        report["combined_csv"] = None

    return report


# ==========================================================================
# shared helpers — copied to match Train_EmotionV3 CELL 8 / CELL 9 exactly
# ==========================================================================
def normalise(df, cols, scope=NORM_SCOPE, min_rows=NORM_MIN_ROWS,
              std_floor=NORM_STD_FLOOR):
    """Per-dialogue z-score. Mirrors CELL 8."""
    out = df.copy()
    if scope == "off":
        return out
    if scope == "speaker":
        key = out["speaker"]
    elif scope == "dialog":
        key = out["dialog"]
    elif scope == "dialog_speaker":
        key = out["dialog"].astype(str) + "|" + out["speaker"].astype(str)
    else:
        raise ValueError(f"unknown scope {scope}")

    grp = out.groupby(key)
    sizes = grp[cols[0]].transform("size")
    mu = grp[cols].transform("mean")
    sd = grp[cols].transform("std").fillna(0.0)
    sd = sd.where(sd > std_floor, 1.0)
    z = (out[cols] - mu) / sd
    big = sizes >= min_rows
    out.loc[big, cols] = z.loc[big].fillna(0.0).values
    return out


def run_loso(df, cols, ravdess_weight, fit_predict, label="model"):
    """Leave-one-IEMOCAP-session-out. Mirrors CELL 9.

    fit_predict(Xtr, ytr, w, Xte) -> yhat, so the same fold loop can drive
    any architecture.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score)

    iem = df[df["source"] == "iemocap"]
    rav = df[df["source"] == "ravdess"]
    sessions = sorted(iem["session"].unique())

    per_fold, preds, truths = [], [], []
    for s in sessions:
        te = iem[iem["session"] == s]
        tr_iem = iem[iem["session"] != s]
        tr = (pd.concat([tr_iem, rav], ignore_index=True, sort=False)
              if ravdess_weight > 0 and len(rav) else tr_iem)
        if te.empty or tr.empty:
            continue

        # speaker-leak assertion: no speaker on both sides of the fold
        leak = set(tr["speaker"]) & set(te["speaker"])
        assert not leak, f"SPEAKER LEAK in fold {s}: {sorted(leak)[:5]}"

        w = np.where(tr["source"].to_numpy() == "ravdess",
                     float(ravdess_weight), 1.0)
        yp = fit_predict(tr[cols].to_numpy(dtype=float),
                         tr["label"].to_numpy(), w,
                         te[cols].to_numpy(dtype=float))
        yt = te["label"].to_numpy()
        preds.extend(list(yp))
        truths.extend(list(yt))
        per_fold.append({
            "session": int(s),
            "n_test": len(te),
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "macro_f1": round(float(f1_score(yt, yp, average="macro",
                                             zero_division=0)), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(yt, yp)), 4),
        })
        print(f"    session {s}: acc={per_fold[-1]['accuracy']:.3f}  "
              f"macroF1={per_fold[-1]['macro_f1']:.3f}  "
              f"balAcc={per_fold[-1]['balanced_accuracy']:.3f}  "
              f"(n={len(te)})")

    pooled = {
        "accuracy": round(float(accuracy_score(truths, preds)), 4),
        "macro_f1": round(float(f1_score(truths, preds, average="macro",
                                         zero_division=0)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(truths, preds)), 4),
        "n": len(truths),
    }
    print(f"  POOLED {label}: acc={pooled['accuracy']:.4f}  "
          f"macroF1={pooled['macro_f1']:.4f}  "
          f"balAcc={pooled['balanced_accuracy']:.4f}  (n={pooled['n']})")
    return per_fold, pooled


# ==========================================================================
# wav2vec2 embeddings
# ==========================================================================
def extract_w2v_embeddings(paths):
    """Mean-pooled last-hidden-state from a frozen wav2vec2 base."""
    import torch
    import soundfile as sf
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(W2V_MODEL)
    model = Wav2Vec2Model.from_pretrained(W2V_MODEL).to(device).eval()

    embs, ok = [], []
    for i, p in enumerate(paths, 1):
        try:
            wav, sr = sf.read(p, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != W2V_SR:
                import scipy.signal as ss
                wav = ss.resample_poly(wav, W2V_SR, sr).astype(np.float32)
            wav = wav[: int(W2V_MAX_SECONDS * W2V_SR)]
            if len(wav) < W2V_SR // 10:      # under 100ms, unusable
                embs.append(None); ok.append(False); continue
            inp = fe(wav, sampling_rate=W2V_SR, return_tensors="pt")
            with torch.no_grad():
                h = model(inp.input_values.to(device)).last_hidden_state
            embs.append(h.mean(dim=1).squeeze(0).cpu().numpy())
            ok.append(True)
        except Exception as e:
            if i < 5:
                print(f"    failed {p}: {type(e).__name__}: {e}")
            embs.append(None); ok.append(False)
        if i % 250 == 0:
            print(f"    {i}/{len(paths)}  ({sum(ok)} ok)")
    return embs, ok


# ==========================================================================
# PART B — the comparison
# ==========================================================================
def part_b():
    print("\n" + "=" * 70)
    print("PART B — ARCHITECTURE COMPARISON ON IEMOCAP LOSO FOLDS")
    print("=" * 70)

    if not os.path.exists(COMBINED_CSV):
        print(f"missing {COMBINED_CSV} — run Train_EmotionV3 to CELL 7 first.")
        return None

    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(COMBINED_CSV)
    meta = {"utt_id", "path", "label", "raw_code", "speaker", "dialog",
            "session", "source", "intensity", "statement"}
    ege_cols = [c for c in df.columns if c not in meta]
    print(f"loaded {len(df)} rows, {len(ege_cols)} eGeMAPS columns")
    print(f"classes: {sorted(df['label'].unique())}")

    results = {"n_classes": int(df["label"].nunique()),
               "chance": round(1.0 / df["label"].nunique(), 4)}

    def rf_fit_predict(Xtr, ytr, w, Xte):
        clf = RandomForestClassifier(**RF_KWARGS)
        clf.fit(Xtr, ytr, sample_weight=w)
        return clf.predict(Xte)

    # ---- ARM 1: eGeMAPS + Random Forest -----------------------------------
    print("\n[ARM 1] eGeMAPS + Random Forest")
    ege_df = normalise(df, ege_cols)
    folds1, pooled1 = run_loso(ege_df, ege_cols, RAVDESS_WEIGHT,
                               rf_fit_predict, "egemaps_rf")
    results["egemaps_rf"] = {"folds": folds1, "pooled": pooled1}

    # ---- ARM 2: frozen wav2vec2 + Random Forest ---------------------------
    print("\n[ARM 2] frozen wav2vec2 (mean-pooled) + Random Forest")
    if os.path.exists(W2V_CACHE):
        z = np.load(W2V_CACHE, allow_pickle=True)
        X, keep = z["X"], z["keep"]
        print(f"  loaded cache: {X.shape} from {W2V_CACHE}")
    else:
        print(f"  extracting embeddings for {len(df)} clips "
              f"(slow, one time only)")
        embs, ok = extract_w2v_embeddings(df["path"].tolist())
        keep = np.array(ok)
        X = np.vstack([e for e in embs if e is not None])
        os.makedirs(OUT_DIR, exist_ok=True)
        np.savez_compressed(W2V_CACHE, X=X, keep=keep)
        print(f"  cached -> {W2V_CACHE}  ({X.shape}, "
              f"{(~keep).sum()} failed)")

    w2v_cols = [f"w2v_{i}" for i in range(X.shape[1])]
    w2v_df = df[keep].reset_index(drop=True).copy()
    for i, c in enumerate(w2v_cols):
        w2v_df[c] = X[:, i]
    print(f"  {len(w2v_df)} rows survive extraction "
          f"({(~keep).sum()} dropped)")
    w2v_df = normalise(w2v_df, w2v_cols)
    folds2, pooled2 = run_loso(w2v_df, w2v_cols, RAVDESS_WEIGHT,
                               rf_fit_predict, "wav2vec2_rf")
    results["wav2vec2_rf"] = {"folds": folds2, "pooled": pooled2}

    # ---- verdict ----------------------------------------------------------
    print("\n" + "-" * 70)
    d_acc = pooled2["accuracy"] - pooled1["accuracy"]
    d_bal = pooled2["balanced_accuracy"] - pooled1["balanced_accuracy"]
    spread = float(np.std([f["accuracy"] for f in folds1]))
    print(f"wav2vec2 - eGeMAPS:  accuracy {d_acc:+.4f}   "
          f"balanced accuracy {d_bal:+.4f}")
    print(f"eGeMAPS cross-fold accuracy SD: ±{spread:.4f}")
    if abs(d_acc) < spread:
        verdict = ("difference smaller than cross-fold spread — the original "
                   "claim holds on conversational speech")
    elif d_acc > 0:
        verdict = ("wav2vec2 beats eGeMAPS by more than the fold spread — "
                   "the thesis claim needs correcting")
    else:
        verdict = ("eGeMAPS beats wav2vec2 by more than the fold spread — "
                   "claim holds and is stronger than stated")
    print(f"VERDICT: {verdict}")
    results["delta_accuracy"] = round(d_acc, 4)
    results["delta_balanced_accuracy"] = round(d_bal, 4)
    results["egemaps_fold_sd"] = round(spread, 4)
    results["verdict"] = verdict
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    out = {}
    if args.part in ("a", "both"):
        out["part_a"] = part_a()
    if args.part in ("b", "both"):
        out["part_b"] = part_b()

    print("\n" + "=" * 70)
    print("JSON RESULT — paste this back")
    print("=" * 70)
    print(json.dumps(out, indent=2, default=str))
    with open(f"{OUT_DIR}/arch_diagnostic_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nalso written to {OUT_DIR}/arch_diagnostic_result.json")


if __name__ == "__main__":
    main()

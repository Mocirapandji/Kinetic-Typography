import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Word Budget Audit

    A standalone, self-contained reproduction of the salience &rarr; softmax
    &rarr; threshold &rarr; arousal-floor pipeline from Section&nbsp;4 (Word
    budgeting), using synthetic per-word prosodic values instead of a real
    WhisperX/Praat run, so it needs no audio and no models.

    Every constant below (weights, temperature, thresholds, class priors) is
    copied directly from the shipped transcriber, not re-guessed for this demo.
    """)
    return


@app.cell
def _():
    # ---- copied verbatim from the shipped transcriber's CELL 12 -----------
    SALIENCE_WEIGHTS = {
        "f0_mean": 0.9, "f0_range": 0.7,
        "intensity_db": 1.2, "dur_resid": 1.0, "hnr": 0.3,
    }
    POSITIVE_ONLY_FEATURES = {"f0_mean", "f0_range", "intensity_db", "dur_resid"}
    SOFTMAX_TEMPERATURE = 1.5
    MIN_POINTS = 2.0
    FULL_DRAMA_RATIO = 2.5          # R: share ratio at which intensity maxes out

    BASE_FONT_SIZE = 40
    FONT_SWING = 24
    FONT_GAMMA = 1.6
    BOLD_THRESHOLD = 0.62

    CLASS_PRIOR = {
        "neg": 1.15, "content": 1.00, "modal": 0.85, "degree": 0.80,
        "pron": 0.55, "aux": 0.45, "prep": 0.40, "det": 0.35,
        "conj": 0.30, "filler": 0.25,
    }
    CLASS_OVERRIDE_Z = 0.90
    CLASS_OVERRIDE_REL = 0.75
    CLASS_OVERRIDE_FULL = 1.00

    SEGMENT_AROUSAL_FLOOR = 0.45
    AROUSAL_FEATURES = {"intensity_db": 1.0, "f0_mean": 0.6}
    AROUSAL_SPREAD = 1.5
    return (
        AROUSAL_FEATURES,
        AROUSAL_SPREAD,
        BASE_FONT_SIZE,
        BOLD_THRESHOLD,
        CLASS_OVERRIDE_FULL,
        CLASS_OVERRIDE_REL,
        CLASS_OVERRIDE_Z,
        CLASS_PRIOR,
        FONT_GAMMA,
        FONT_SWING,
        FULL_DRAMA_RATIO,
        MIN_POINTS,
        POSITIVE_ONLY_FEATURES,
        SALIENCE_WEIGHTS,
        SEGMENT_AROUSAL_FLOOR,
        SOFTMAX_TEMPERATURE,
    )


@app.cell
def _():
    # ---- three example segments, standing in for one short "clip" --------
    # SEGMENT A: a word genuinely stands out prosodically -> the case the
    #            word budget is designed for.
    # SEGMENT B: an unremarkable, flat delivery -> the threshold should
    #            keep this segment completely unstyled.
    # SEGMENT C: every word is shouted almost equally -> the case that
    #            needs the segment arousal floor, not the word-level ratio.
    SEGMENTS = {
        "A — \u201cI really HATE this movie\u201d (genuine emphasis)": {
            "word":         ["I", "really", "HATE", "this", "movie"],
            "class":        ["pron", "degree", "content", "det", "content"],
            "f0_mean":      [110, 115, 210, 112, 108],
            "f0_range":     [10, 12, 60, 9, 11],
            "intensity_db": [55, 58, 78, 54, 56],
            "dur_resid":    [0.00, 0.02, 0.35, 0.00, 0.05],
            "hnr":          [12, 12, 8, 13, 12],
        },
        "B — \u201cThe cat sat on the mat\u201d (flat, neutral)": {
            "word":         ["The", "cat", "sat", "on", "the", "mat"],
            "class":        ["det", "content", "content", "prep", "det", "content"],
            "f0_mean":      [105, 105, 105, 105, 105, 105],
            "f0_range":     [8, 8, 8, 8, 8, 8],
            "intensity_db": [50, 50, 50, 50, 50, 50],
            "dur_resid":    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            "hnr":          [13, 13, 13, 13, 13, 13],
        },
        "C — \u201cI HATE YOU\u201d (uniformly shouted)": {
            "word":         ["I", "HATE", "YOU"],
            "class":        ["pron", "content", "pron"],
            "f0_mean":      [206, 206, 206],
            "f0_range":     [56, 56, 56],
            "intensity_db": [80, 80, 80],
            "dur_resid":    [0.10, 0.10, 0.10],
            "hnr":          [8, 8, 8],
        },
    }
    FEATURE_NAMES = ["f0_mean", "f0_range", "intensity_db", "dur_resid", "hnr"]
    return FEATURE_NAMES, SEGMENTS


@app.cell
def _(SEGMENTS, mo):
    segment_picker = mo.ui.dropdown(
        options=list(SEGMENTS.keys()),
        value=list(SEGMENTS.keys())[0],
        label="Segment to render",
    )
    floor_toggle = mo.ui.switch(value=True, label="Apply segment arousal floor")
    mo.hstack([segment_picker, floor_toggle], justify="start", gap=2)
    return floor_toggle, segment_picker


@app.cell
def _(np):
    # ---- helper functions, all copied from the shipped transcriber --------
    def center_scale(vals, rel_floor=0.02, abs_floor=1e-3):
        vals = np.asarray(vals, dtype=float)
        c = float(np.median(vals))
        mad = float(np.median(np.abs(vals - c)))
        s = 1.4826 * mad
        if s < 1e-9:
            s = float(np.std(vals))
        if s <= max(abs_floor, rel_floor * abs(c)):
            return c, 0.0
        return c, s

    def compute_salience(feat_matrix, feat_names, weights, positive_only):
        n = feat_matrix.shape[0]
        salience = np.zeros(n)
        zmax_pos = np.zeros(n)
        for j, feat in enumerate(feat_names):
            if feat not in weights:
                continue
            w = weights[feat]
            vals = feat_matrix[:, j]
            c, s = center_scale(vals)
            z = (vals - c) / s if s > 1e-9 else np.zeros(n)
            if feat in positive_only:
                zc = np.maximum(z, 0.0)
                zmax_pos = np.maximum(zmax_pos, zc)
            else:
                zc = np.abs(z)
            salience += w * zc
        return salience, zmax_pos

    def apply_word_class_prior(salience, zmax_pos, classes, class_prior,
                                override_z, override_rel, override_full):
        priors = np.array([class_prior.get(c, 1.0) for c in classes])
        peak = zmax_pos.max() if zmax_pos.max() > 1e-9 else 1.0
        rel = zmax_pos / peak
        gate = np.clip((zmax_pos - override_z) / 0.5, 0.0, 1.0)
        t = np.clip((rel - override_rel) / max(override_full - override_rel, 1e-9),
                    0.0, 1.0) * gate
        eff = np.where(priors >= 1.0, priors, priors + (1.0 - priors) * t)
        return salience * eff, eff

    def allocate_points(salience, temperature, min_points):
        n = len(salience)
        floor = min(min_points, 100.0 / n)
        pool = 100.0 - floor * n
        logits = salience / temperature
        logits = logits - logits.max()
        w = np.exp(logits)
        w = w / w.sum()
        points = floor + pool * w
        fair = 100.0 / n
        share_ratio = points / fair
        return points, share_ratio

    def intensity_from_ratio(share_ratio, R):
        return np.clip((share_ratio - 1.0) / (R - 1.0), 0.0, 1.0)

    def segment_arousal_floor(seg_means, floor_max, spread):
        # seg_means: dict {segment_key: weighted absolute score}, already
        # combined across AROUSAL_FEATURES for that segment.
        keys = list(seg_means.keys())
        vals = np.array([seg_means[k] for k in keys], dtype=float)
        c, s = center_scale(vals)
        z = (vals - c) / s if s > 1e-9 else np.zeros(len(vals))
        gain = 0.5 * (1.0 + np.tanh(z / max(spread, 1e-9)))
        return dict(zip(keys, floor_max * gain))

    def font_size_fn(intensity, base, swing, gamma):
        return base + swing * (intensity ** gamma)

    return (
        allocate_points,
        apply_word_class_prior,
        compute_salience,
        font_size_fn,
        intensity_from_ratio,
        segment_arousal_floor,
    )


@app.cell
def _(
    AROUSAL_FEATURES,
    CLASS_OVERRIDE_FULL,
    CLASS_OVERRIDE_REL,
    CLASS_OVERRIDE_Z,
    CLASS_PRIOR,
    FEATURE_NAMES,
    POSITIVE_ONLY_FEATURES,
    SALIENCE_WEIGHTS,
    SEGMENTS,
    apply_word_class_prior,
    compute_salience,
    np,
):
    # word-level salience for every segment, plus each segment's own
    # absolute arousal score (needed later to compare segments to each
    # other, exactly as segment_arousal_floor does across a whole clip).
    per_segment = {}
    segment_arousal_score = {}
    for _key, _seg in SEGMENTS.items():
        _feat_matrix = np.array([_seg[f] for f in FEATURE_NAMES], dtype=float).T
        _salience_raw, _zmax_pos = compute_salience(
            _feat_matrix, FEATURE_NAMES, SALIENCE_WEIGHTS, POSITIVE_ONLY_FEATURES)
        _salience_final, _eff = apply_word_class_prior(
            _salience_raw, _zmax_pos, _seg["class"], CLASS_PRIOR,
            CLASS_OVERRIDE_Z, CLASS_OVERRIDE_REL, CLASS_OVERRIDE_FULL)
        per_segment[_key] = {
            "salience_raw": _salience_raw,
            "class_weight": _eff,
            "salience": _salience_final,
        }
        _score = 0.0
        for _feat, _w in AROUSAL_FEATURES.items():
            _score += _w * float(np.mean(_seg[_feat]))
        segment_arousal_score[_key] = _score
    return per_segment, segment_arousal_score


@app.cell
def _(
    AROUSAL_SPREAD,
    SEGMENT_AROUSAL_FLOOR,
    segment_arousal_floor,
    segment_arousal_score,
):
    segment_floor = segment_arousal_floor(
        segment_arousal_score, SEGMENT_AROUSAL_FLOOR, AROUSAL_SPREAD)
    return (segment_floor,)


@app.cell
def _(
    BASE_FONT_SIZE,
    BOLD_THRESHOLD,
    FONT_GAMMA,
    FONT_SWING,
    FULL_DRAMA_RATIO,
    MIN_POINTS,
    SEGMENTS,
    SOFTMAX_TEMPERATURE,
    allocate_points,
    floor_toggle,
    font_size_fn,
    intensity_from_ratio,
    pd,
    per_segment,
    segment_floor,
    segment_picker,
):
    _key = segment_picker.value
    _seg = SEGMENTS[_key]
    _salience = per_segment[_key]["salience"]

    _points, _share_ratio = allocate_points(
        _salience, SOFTMAX_TEMPERATURE, MIN_POINTS)
    _intensity_raw = intensity_from_ratio(_share_ratio, FULL_DRAMA_RATIO)

    _floor = segment_floor[_key] if floor_toggle.value else 0.0
    _intensity = _floor + (1.0 - _floor) * _intensity_raw

    _font_size = font_size_fn(_intensity, BASE_FONT_SIZE, FONT_SWING, FONT_GAMMA)
    _bold = _intensity >= BOLD_THRESHOLD

    audit = pd.DataFrame({
        "word": _seg["word"],
        "class": _seg["class"],
        "salience_raw": per_segment[_key]["salience_raw"].round(2),
        "class_weight": per_segment[_key]["class_weight"].round(2),
        "salience": _salience.round(2),
        "points /100": _points.round(1),
        "share_ratio": _share_ratio.round(2),
        "intensity_raw": _intensity_raw.round(2),
        "arousal_floor": round(_floor, 2),
        "intensity_final": _intensity.round(2),
        "font_size": _font_size.round(1),
        "bold": _bold,
    })
    return (audit,)


@app.cell
def _(audit, mo, segment_picker):
    mo.vstack([
        mo.md(f"### Audit table &mdash; {segment_picker.value}"),
        mo.ui.table(audit, selection=None),
    ])
    return


@app.cell
def _(audit, plt, segment_picker):
    # a plain-text "rendered caption" preview: font size and bold follow
    # the audit table directly, colour intensity fades in as a proxy for
    # the confidence/emotion-hue channel (not modelled in this demo).
    _fig, _ax = plt.subplots(figsize=(9.5, 2.4))
    _ax.set_xlim(0, 1)
    _ax.set_ylim(0, 1)
    _ax.axis("off")
    _ax.set_facecolor("#1c1c1c")
    _fig.patch.set_facecolor("#1c1c1c")

    _n = len(audit)
    _x = 0.03
    for _i in range(_n):
        _row = audit.iloc[_i]
        _size = float(_row["font_size"])
        _inten = float(_row["intensity_final"])
        _weight = "bold" if bool(_row["bold"]) else "normal"
        _color = (1.0, 1.0 - 0.55 * _inten, 1.0 - 0.75 * _inten)
        _txt = _ax.text(
            _x, 0.5, str(_row["word"]), fontsize=_size, fontweight=_weight,
            color=_color, va="center", ha="left", family="DejaVu Sans")
        _fig.canvas.draw()
        _bbox = _txt.get_window_extent(renderer=_fig.canvas.get_renderer())
        _bbox_data = _ax.transAxes.inverted().transform(_bbox)
        _x = _bbox_data[1][0] + 0.02

    _ax.set_title(
        f"Rendered preview \u2014 {segment_picker.value}",
        color="white", fontsize=10, pad=10)
    plt.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

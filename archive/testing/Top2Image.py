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
    import colorsys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return colorsys, mo, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Top-two emotion signal

    A standalone reproduction of `resolve_word_color` and `confidence_scale`
    from the shipped transcriber: given a segment's full probability
    distribution over emotion classes, this shows the top-two bars, whether
    the hue-blend gate fires, and the resulting rendered colour.

    No audio, no model, no WhisperX &mdash; every constant (hues, blend
    margin, confidence gamma) is copied directly from the transcriber.
    """)
    return


@app.cell
def _():
    # ---- copied verbatim from EMOTION_STYLES in the shipped transcriber ---
    # (h, s, v only -- font/italic/anim are not needed for this figure)
    EMOTION_STYLES = {
        "angry":      {"h": 0.0000, "s": 0.85, "v": 0.80},
        "frustrated": {"h": 0.0556, "s": 0.70, "v": 0.62},
        "happy":      {"h": 0.1400, "s": 0.90, "v": 1.00},
        "excited":    {"h": 0.8750, "s": 0.85, "v": 1.00},
        "surprised":  {"h": 0.5278, "s": 0.90, "v": 1.00},
        "sad":        {"h": 0.6100, "s": 0.55, "v": 0.82},
        "fearful":    {"h": 0.7000, "s": 0.70, "v": 0.78},
        "neutral":    {"h": 0.0000, "s": 0.00, "v": 0.95},
    }
    # the 8 live classes the shipped classifier actually emits (Section 4.1.1);
    # "disgust" is an inert style entry and is excluded here for that reason.
    N_CLASSES = 8
    BLEND_MARGIN = 0.08     # margin at/below which the top-two hues blend
    SAT_FLOOR_FRAC = 0.5    # saturation never drops below this fraction
    CONF_GAMMA = 1.5
    return BLEND_MARGIN, CONF_GAMMA, EMOTION_STYLES, N_CLASSES, SAT_FLOOR_FRAC


@app.cell
def _():
    # ---- two example segments, chosen to sit on either side of the gate ---
    EXAMPLES = {
        "Decisive \u2014 large margin (single colour)": {
            "text": "GET OUT OF MY HOUSE",
            "dist": {
                "angry": 0.62, "frustrated": 0.15, "fearful": 0.06,
                "excited": 0.04, "happy": 0.03, "sad": 0.04,
                "surprised": 0.03, "neutral": 0.03,
            },
        },
        "Ambiguous \u2014 small margin (blended colour)": {
            "text": "I can't believe you did that",
            "dist": {
                "surprised": 0.29, "excited": 0.24, "happy": 0.14,
                "frustrated": 0.10, "angry": 0.08, "fearful": 0.07,
                "sad": 0.05, "neutral": 0.03,
            },
        },
    }
    return (EXAMPLES,)


@app.cell
def _(EXAMPLES, mo):
    segment_picker = mo.ui.dropdown(
        options=list(EXAMPLES.keys()),
        value=list(EXAMPLES.keys())[0],
        label="Segment to render",
    )
    segment_picker
    return (segment_picker,)


@app.cell
def _(colorsys):
    # ---- helper functions, copied from the shipped transcriber ------------
    def confidence_scale(p_top, n_classes, gamma):
        chance = 1.0 / max(int(n_classes), 2)
        p = min(1.0, max(0.0, p_top))
        t = min(1.0, max(0.0, (p - chance) / (1.0 - chance))) ** gamma
        return 0.5 + 0.5 * t

    def emotion_hsv(emotion, styles):
        fam = styles.get(emotion, styles["neutral"])
        return fam["h"], fam["s"], fam["v"]

    def resolve_word_color(emo1, emo2, p1, p2, blend_margin, styles,
                            sat_floor_frac, strength):
        h1, s1, v1 = emotion_hsv(emo1, styles)
        r1, g1, b1 = colorsys.hsv_to_rgb(h1, s1, v1)
        do_blend = (emo2 is not None) and (emo1 != emo2) and \
                   ((p1 - p2) <= blend_margin)
        if do_blend:
            h2, s2, v2 = emotion_hsv(emo2, styles)
            r2, g2, b2 = colorsys.hsv_to_rgb(h2, s2, v2)
            denom = (p1 + p2) if (p1 + p2) > 1e-9 else 1.0
            t = p2 / denom
            r = r1 + (r2 - r1) * t
            g = g1 + (g2 - g1) * t
            b = b1 + (b2 - b1) * t
        else:
            r, g, b = r1, g1, b1
            t = 0.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = s * (sat_floor_frac + (1.0 - sat_floor_frac) * strength)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return do_blend, t, (r, g, b)

    return confidence_scale, resolve_word_color


@app.cell
def _(
    BLEND_MARGIN,
    CONF_GAMMA,
    EMOTION_STYLES,
    EXAMPLES,
    N_CLASSES,
    SAT_FLOOR_FRAC,
    confidence_scale,
    pd,
    resolve_word_color,
    segment_picker,
):
    _example = EXAMPLES[segment_picker.value]
    _dist = _example["dist"]
    _ranked = sorted(_dist.items(), key=lambda kv: -kv[1])
    _emo1, _p1 = _ranked[0]
    _emo2, _p2 = _ranked[1]
    _margin = _p1 - _p2

    _strength = confidence_scale(_p1, N_CLASSES, CONF_GAMMA)
    _do_blend, _blend_t, _rgb = resolve_word_color(
        _emo1, _emo2, _p1, _p2, BLEND_MARGIN, EMOTION_STYLES,
        SAT_FLOOR_FRAC, _strength)
    _hex = "#{:02X}{:02X}{:02X}".format(
        round(_rgb[0] * 255), round(_rgb[1] * 255), round(_rgb[2] * 255))

    dist_df = pd.DataFrame(
        [{"emotion": k, "probability": v} for k, v in _ranked])

    audit = pd.DataFrame([{
        "segment text": _example["text"],
        "top emotion": _emo1, "p_top": _p1,
        "second emotion": _emo2, "p_second": _p2,
        "margin (p_top - p_second)": round(_margin, 3),
        "blend gate fires (margin \u2264 0.08)": _do_blend,
        "blend weight t = p2/(p1+p2)": round(_blend_t, 3) if _do_blend else "\u2014",
        "confidence strength": round(_strength, 3),
        "rendered colour": _hex,
    }])
    rendered_hex = _hex
    rendered_text = _example["text"]
    top_emo, second_emo = _emo1, _emo2
    return audit, dist_df, rendered_hex, rendered_text, second_emo, top_emo


@app.cell
def _(audit, mo):
    mo.vstack([
        mo.md("### Audit"),
        mo.ui.table(audit, selection=None),
    ])
    return


@app.cell
def _(dist_df, plt, rendered_hex, rendered_text, second_emo, top_emo):
    # one combined figure: probability distribution on top, the resulting
    # rendered caption underneath -- this is the single image to screenshot,
    # since the figure caption promises both the predictions AND the colour
    # they produce, not one or the other.
    _fig, (_ax1, _ax2) = plt.subplots(
        2, 1, figsize=(7.5, 4.6), gridspec_kw={"height_ratios": [3, 1.2]})

    _colors = []
    for _e in dist_df["emotion"]:
        if _e == top_emo:
            _colors.append("#3b7dd8")
        elif _e == second_emo:
            _colors.append("#e07b1a")
        else:
            _colors.append("#b0b0b0")
    _bars = _ax1.bar(dist_df["emotion"], dist_df["probability"], color=_colors)
    for _b, _p in zip(_bars, dist_df["probability"]):
        _ax1.text(_b.get_x() + _b.get_width() / 2, _p + 0.01, f"{_p:.2f}",
                   ha="center", va="bottom", fontsize=8)
    _ax1.set_ylim(0, max(dist_df["probability"]) * 1.25)
    _ax1.set_ylabel("probability")
    _ax1.set_title("Per-class distribution \u2014 top two highlighted")
    _ax1.tick_params(axis="x", rotation=30)

    _ax2.set_xlim(0, 1)
    _ax2.set_ylim(0, 1)
    _ax2.axis("off")
    _ax2.set_facecolor("#1c1c1c")
    _ax2.text(0.5, 0.5, rendered_text, fontsize=24, fontweight="bold",
               color=rendered_hex, ha="center", va="center",
               family="DejaVu Sans", transform=_ax2.transAxes)
    _ax2.set_title("Rendered caption preview", fontsize=10, pad=6)

    plt.tight_layout()
    _fig
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

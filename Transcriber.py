import marimo

__generated_with = "0.23.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import whisperx
    import parselmouth
    from parselmouth.praat import call
    import pandas as pd

    return call, mo, parselmouth, pd, whisperx


@app.cell
def _():
    device = "cpu"
    compute_type = "int8"
    audio_file = "/run/media/s5812886/T7 Shield/RAVDESS/Actor_01/03-01-06-01-02-01-01.wav"
    return audio_file, compute_type, device


@app.cell
def _(mo):
    mo.md("""
    ## Stage 1: WhisperX transcription → word-level timestamps
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
def _(mo):
    mo.md("""
    ## Stage 2: Per-word prosody extraction
    """)
    return


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

            # pitch (F0) — drop unvoiced frames (0 Hz)
            pitch = word_snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0v = f0[f0 > 0]
            f0_mean = float(f0v.mean()) if len(f0v) else 0.0
            f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0

            # intensity / loudness (RMS energy)
            rms = call(word_snd, "Get root-mean-square", 0, 0)

            # voice quality (HNR) — drop undefined frames (-200 dB sentinel)
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
    features_df = extract_word_features(audio_file, aligned["word_segments"])
    features_df
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

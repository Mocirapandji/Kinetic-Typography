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

            # pause length = gap to the NEXT word's start; 0 for the last word
            if i < len(word_segments) - 1:
                pause_after = float(word_segments[i + 1]["start"]) - end
            else:
                pause_after = 0.0

            word_snd = snd.extract_part(from_time=start, to_time=end, preserve_times=True)

            # drop unvoiced frames (0 Hz)
            pitch = word_snd.to_pitch()
            f0 = pitch.selected_array["frequency"]
            f0v = f0[f0 > 0]
            f0_mean = float(f0v.mean()) if len(f0v) else 0.0
            f0_range = float(f0v.max() - f0v.min()) if len(f0v) else 0.0

            rms = call(word_snd, "Get root-mean-square", 0, 0)

            # drop undefined HNR frames (-200 dB sentinel)
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
    return (features_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Stage 3: Style mapping (THROWAWAY crude rules)
    """)
    return


@app.cell
def _(features_df):
    def assign_styles(_df):
        df = _df.copy()
        rms_min, rms_max = df["rms"].min(), df["rms"].max()
        rms_span = (rms_max - rms_min) or 1.0  # avoid divide-by-zero

        def rgb_to_ass(r, g, b):
            # ASS colour is &HBBGGRR&, NOT RGB
            return f"&H{b:02X}{g:02X}{r:02X}&"

        sizes, colors, bolds = [], [], []
        for _, row in df.iterrows():
            # size from loudness, normalised within this clip -> 44..96 px
            norm = (row["rms"] - rms_min) / rms_span
            sizes.append(int(44 + norm * 52))

            # colour from pitch: high = red, mid = amber, low/unvoiced = blue
            if row["f0_mean"] >= 160:
                colors.append(rgb_to_ass(255, 60, 60))
            elif row["f0_mean"] >= 120:
                colors.append(rgb_to_ass(255, 200, 60))
            else:
                colors.append(rgb_to_ass(120, 180, 255))

            bolds.append(1 if row["f0_range"] >= 60 else 0)

        df["font_size"] = sizes
        df["color_ass"] = colors
        df["bold"] = bolds
        return df

    styled_df = assign_styles(features_df)
    styled_df
    return (styled_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Stage 4: Render .ass + FFmpeg burn-in
    """)
    return


@app.cell
def _(audio_file, styled_df):
    def render_styled_video(audio_path, df, out_dir="outputs"):
        import os, subprocess
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

        # one Dialogue event per word, so each word can carry its own override
        lines = []
        for _, row in df.iterrows():
            start = sec_to_ass(float(row["start"]))
            end = sec_to_ass(float(row["end"]))
            override = f"{{\\fs{int(row['font_size'])}\\c{row['color_ass']}\\b{int(row['bold'])}}}"
            text = override + str(row["word"]).strip()
            lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

        ass_content = header + "\n".join(lines) + "\n"
        ass_path = f"{out_dir}/ass/demo.ass"
        with open(ass_path, "w") as f:
            f.write(ass_content)

        duration = float(df["end"].max()) + 0.5
        out_path = f"{out_dir}/video/demo.mp4"
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

    output_video, output_ass = render_styled_video(audio_file, styled_df)
    print("Wrote:", output_video, "and", output_ass)
    return


if __name__ == "__main__":
    app.run()

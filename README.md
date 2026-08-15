cat > README.md << 'EOF'
# Emotion-Responsive Kinetic Typography
*A Prosody-Driven Captioning Pipeline for Accessibility*

## Overview
A captioning pipeline that reads the emotional tone of spoken audio and lets that
tone drive the visual style of the subtitles — colour, size, font, animation and
pacing — instead of rendering flat, static text. Built to give deaf and
hard-of-hearing (DHH) viewers, and anyone watching on mute, access to the
emotional layer of speech that plain captions strip out.

## Pipeline
Six stages, plus an integration layer that wires them into a single callable tool.
The original proposal described four stages (transcription -> prosody -> style ->
rendering); the build ended up with six once word budgeting and style mapping
were split out into their own stages (see dissertation Section 4.2.1).

1. **Transcription** — WhisperX (large-v3) converts audio to word-level timed text.
2. **Prosody extraction** — Parselmouth/Praat measures per-word pitch, loudness,
   duration and related acoustic features.
3. **Emotion classification** — a Random Forest classifier (trained on eGeMAPS
   features) predicts an emotion per spoken segment; an HMM sequence smoother
   keeps predictions consistent across a speaker's dialogue.
4. **Word budgeting** — a salience-scoring system allocates a limited "emphasis
   budget" per segment, so only words that genuinely stand out prosodically get
   styled.
5. **Style mapping** — emotion and salience are converted into typography
   (colour, font, size, weight, motion), one meaning per visual channel.
6. **Rendering** — styled captions are written as an Advanced SubStation Alpha
   (.ass) file and burned into the source video with FFmpeg + libass.

## Repo structure
- `Final_Code/` — the pipeline as submitted.
  - `TranscriberV1.2.py` — the transcriber/rendering pipeline described in the
    dissertation methodology.
  - `Train_EmotionV3(Final).py` — trains the emotion classifier (`clf_v3.joblib`)
    on RAVDESS + IEMOCAP.
  - `OlderVersions/` — earlier consolidated iterations (v0.1–v1.1), a documented
    failed attempt (v1.3-FAILED), and the post-evaluation build used to generate
    the final polished renders (v1.3(Post_Evaluation)).
- `archive/` — full incremental development history, kept as process evidence:
  - `transcriber_history/` — every raw iteration of the transcriber (V1–V21)
    before consolidation.
  - `training_archive/` — earlier emotion-classifier training scripts (V1, V2,
    V4, windowed variant).
  - `testing/` — rejected experiments (sliding-window classification, MLP vs.
    Random Forest comparison, architecture diagnostics).
- `outputs/` — generated `.ass` subtitle files, extracted feature CSVs, and
  figures. Rendered video/audio and cached intermediates are excluded (see
  Supplementary Materials).
- `evaluationoutput/` — `.ass` files and manifests from the user study;
  rendered video excluded.
- `.gitignore` — excludes large/generated files, raw datasets, and copyrighted
  test clips.

## Setup
```bash
git clone git@github.com:Mocirapandji/Kinetic-Typography.git
cd Kinetic-Typography
uv sync
```
## Overview
    Random Forest comparison, architecture diagnostics).
- `outputs/` — generated `.ass` subtitle files, extracted feature CSVs, and
  figures. Rendered video/audio and cached intermediates are excluded (see
  Supplementary Materials).
- `evaluationoutput/` — `.ass` files and manifests from the user study;
  rendered video excluded.
- `.gitignore` — excludes large/generated files, raw datasets, and copyrighted
  test clips.

## Setup
```bash
git clone git@github.com:Mocirapandji/Kinetic-Typography.git
cd Kinetic-Typography
uv sync
```

Requires a HuggingFace access token for pyannote's gated speaker-diarisation
models. Request access at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

then set:
```bash
export HF_TOKEN=your_token_here
```

## Usage
Run from the repo root, not from inside `Final_Code/` — paths resolve relative
to the repo root:
```bash
marimo run Final_Code/TranscriberV1.2.py
```
Drop new video files into `source_clips_raw/` (excluded from this repo — see
Supplementary Materials) before running.

To retrain the emotion classifier:
```bash
marimo run "Final_Code/Train_EmotionV3(Final).py"
```

## Datasets
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) —
  used to train the emotion classifier. Publicly available under CC BY-NC-SA
  4.0 (Livingstone & Russo, 2018). https://zenodo.org/record/1188976
- **IEMOCAP** (Interactive Emotional Dyadic Motion Capture) — used to test
  conversational generalisation beyond acted speech. Accessed under Bournemouth
  University's institutional licence, arranged via project supervisor. Raw
  audio is not redistributed here per USC's data use agreement — request
  access via https://sail.usc.edu/iemocap/

Raw audio for both datasets is excluded from this repository. Extracted
eGeMAPS feature CSVs are shared in Supplementary Materials where the licence
permits derived data — see below.

## Classifier
`clf_v3.joblib` (Random Forest, 600 trees, eGeMAPS 88-feature set, 260MB)
exceeds GitHub's 100MB file limit and is not included in this repo. Available
via Supplementary Materials.

## Supplementary Materials
Source clips, rendered outputs, the trained classifier, and derived feature
datasets are available via Google Drive: [DRIVE LINK] (shared with module
supervisor).

## Thesis / Context
MSc AI for Media, Master Project (NCCA7035), National Centre for Computer
Animation, Bournemouth University.

Dissertation: *Emotion-Responsive Kinetic Typography: A Prosody-Driven
Captioning Pipeline for Accessibility*
Author: Moses Sirapandji, August 2026
Requires a HuggingFace access token for pyannote's gated speaker-diarisation
models. Request access at:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

then set:
```bash
export HF_TOKEN=your_token_here
```

## Usage
Run from the repo root, not from inside `Final_Code/` — paths resolve relative
to the repo root:
```bash
marimo run Final_Code/TranscriberV1.2.py
```
Drop new video files into `source_clips_raw/` (excluded from this repo — see
Supplementary Materials) before running.

To retrain the emotion classifier:
```bash
marimo run "Final_Code/Train_EmotionV3(Final).py"
```

## Datasets
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) —
  used to train the emotion classifier. Publicly available under CC BY-NC-SA
  4.0 (Livingstone & Russo, 2018). https://zenodo.org/record/1188976
- **IEMOCAP** (Interactive Emotional Dyadic Motion Capture) — used to test
  conversational generalisation beyond acted speech. Accessed under Bournemouth
  University's institutional licence, arranged via project supervisor. Raw
  audio is not redistributed here per USC's data use agreement — request
  access via https://sail.usc.edu/iemocap/

Raw audio for both datasets is excluded from this repository. Extracted
eGeMAPS feature CSVs are shared in Supplementary Materials where the licence
permits derived data.

## Classifier
`clf_v3.joblib` (Random Forest, 600 trees, eGeMAPS 88-feature set, 260MB)
exceeds GitHub's 100MB file limit and is not included in this repo. Available
via Supplementary Materials.

## Supplementary Materials
Large files that don't fit on GitHub live in a shared Google Drive folder:
[https://drive.google.com/drive/folders/1AetMzZlmX-mcCO5W5b0InimW9omcbT8V?usp=sharing](https://drive.google.com/drive/folders/1AetMzZlmX-mcCO5W5b0InimW9omcbT8V?usp=drive_link)

- `Classifier_MODELWEIGHTS/` — `clf_v3.joblib` (260MB) and `clf_v2.joblib`,
  the saved emotion classifier bundles referenced in Section 4.1 of the
  dissertation.
- `source_clips_raw/` — the raw commercial video clips (Breaking Bad, The
  Walking Dead, Interstellar, etc.) used as pipeline test/evaluation inputs.
  Excluded from GitHub for copyright reasons.
- `rendered_output/` — curated before/after renders from the evaluation,
  split into `Final_Rendered_Output/` (kinetic captions) and
  `Normal_Caption_Control/` (plain captions), matching the two conditions
  compared in the user study (Section 6).

## Thesis / Context
MSc AI for Media, Master Project (NCCA7035), National Centre for Computer
Animation, Bournemouth University.

Dissertation: *Emotion-Responsive Kinetic Typography: A Prosody-Driven
Captioning Pipeline for Accessibility*
Author: Moses Sirapandji, August 2026
EOF

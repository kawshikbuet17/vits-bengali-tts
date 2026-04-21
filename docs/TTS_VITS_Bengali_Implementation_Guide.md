# TTS VITS Bengali Implementation Guide

This project turns a raw Bangla/Bengali speech dataset into runnable VITS training pipelines. It supports a stable single-speaker path and a separate multi-speaker path; both use Bengali transcript text stored in `.json` files with matching `.flac` audio, and both produce trained models that generate `.wav` speech from Bengali text.

This guide documents the Bengali single-speaker and multi-speaker VITS support added to this fork and gives beginner-friendly, sequential runbooks for preparing data, training, and inference.

For a concise summary of the major changes from the original VITS repository, see [Major_Changes_From_Original_VITS.md](Major_Changes_From_Original_VITS.md).

The real dataset is on the remote Linux server. The local repository does not assume access to that dataset.

## 1. Purpose

This fork adapts the original `jaywalnut310/vits` style codebase into runnable Bengali TTS baselines.

Main design choices:

- Keep the original VITS model and training flow as intact as possible.
- Keep single-speaker and multi-speaker workflows separate.
- Use Bengali Unicode characters directly, not phonemes.
- Convert raw `.flac` audio to VITS-compatible `.wav` during dataset preparation.
- Generate original VITS-style single-speaker filelists: `wav_path|text`.
- Generate original VITS-style multi-speaker filelists: `wav_path|speaker_index|text`.

## 2. Why Single-Speaker Remains The First Baseline

This Bengali adaptation keeps the single-speaker VITS baseline as the first recommended path.

The purpose is to first prove that the complete Bengali pipeline works end to end:

```text
raw .flac + .json dataset
  -> Bengali transcript extraction
  -> audio conversion to WAV
  -> VITS filelists
  -> Bengali text preprocessing
  -> training
  -> inference
```

Single-speaker training is the safest first target because it keeps the model setup close to the original LJ Speech-style VITS path:

```text
wav_path|text
```

The current config uses the single-speaker model path:

```json
"n_speakers": 0
```

and trains with:

```bash
python train.py -c configs/bengali_base.json -m bengali_base
```

The `--speaker-id` argument in dataset preparation selects exactly one voice from the larger raw dataset. For example:

```bash
--speaker-id 01332512906
```

means:

```text
scan the full raw dataset
keep only JSON records where speaker_id == 01332512906
skip other speakers
write single-speaker filelists
```

This explains why a preparation run may find many files but keep only a small subset. For example, a previous run found `148026` JSON files and kept `224` records for one selected speaker. That is expected for this baseline.

For another single-speaker model, run preparation again with a different `--speaker-id`, then preprocess and train a separate model name.

## 3. Multi-Speaker Path Summary

Multi-speaker Bengali VITS is now added as a separate path. It does not replace the single-speaker baseline.

The multi-speaker path trains one model that can synthesize multiple voices. The model learns a speaker embedding for every speaker index.

Multi-speaker filelists use:

```text
wav_path|speaker_index|text
```

The original raw dataset speaker IDs are mapped to numeric indices in:

```text
speaker_map.json
```

Example:

```json
{
  "01332512906": 0,
  "01800000001": 1,
  "01700000001": 2
}
```

Multi-speaker training uses:

```bash
python train_ms.py -c configs/bengali_ms.generated.json -m bengali_ms
```

Single-speaker and multi-speaker commands are different:

```text
single-speaker: wav_path|text                 -> train.py
multi-speaker:  wav_path|speaker_index|text   -> train_ms.py
```

## 4. Files Added

- `scripts/prepare_bengali_dataset.py`
  Single-speaker only. Recursively scans the raw Bengali dataset, filters one speaker, pairs `.json` and `.flac`, extracts transcript text, converts audio, and writes `wav_path|text` train/val/test filelists.

- `scripts/prepare_bengali_ms_dataset.py`
  Multi-speaker only. Recursively scans the raw Bengali dataset, keeps multiple speakers, creates `speaker_map.json`, converts audio, and writes `wav_path|speaker_index|text` train/val/test filelists.

- `scripts/infer_bengali.py`
  Single-speaker command-line inference script for Bengali checkpoints.

- `scripts/infer_bengali_ms.py`
  Multi-speaker command-line inference script. It accepts either `--speaker-index` or `--speaker-id` with `--speaker-map`.

- `configs/bengali_base.json`
  Single-speaker Bengali training config for `train.py`.

- `configs/bengali_ms.json`
  Multi-speaker Bengali config template for `train_ms.py`. The multi-speaker prep script can generate `configs/bengali_ms.generated.json` with the correct `n_speakers`.

- `docs/TTS_VITS_Bengali_Implementation_Guide.md`
  This complete implementation and running guide.

- `docs/sample_bengali_raw_dataset/`
  Small non-runnable dummy folder showing the expected raw dataset layout and JSON schema.

- `environment_snapshots/`
  Exported package, Python, PyTorch/CUDA, and GPU/driver details from the working remote server environment.

## 5. Files Modified

- `text/symbols.py`
  Adds Bengali symbols and the Bengali Unicode block to the original symbol table.

- `text/cleaners.py`
  Adds `bengali_cleaners`, a simple Unicode character-based cleaner.

- `requirements.txt`
  Updates runtime dependency pins for the modern PyTorch environment and adds `imageio-ffmpeg`.

- `mel_processing.py`
  Updates `librosa.filters.mel` calls for modern `librosa` and sets `torch.stft(..., return_complex=False)` for newer PyTorch compatibility.

- `train.py` and `train_ms.py`
  Use valid distributed port `29500`.

- `monotonic_align/__init__.py`
  Adds a Python fallback if the Cython extension is not built.

- `monotonic_align/setup.py`
  Creates the expected nested output folder before building the extension.

- `README.md`
  Points to this Bengali implementation guide.

## 6. Dataset Contract

The remote raw dataset root is expected to look approximately like this:

```text
/home/kawshik/TTS_Dataset/
  Male/
    01332512906/
      2024-05-07/
        00d4da35-bebc-471b-aa92-0d9398388e98.flac
        00d4da35-bebc-471b-aa92-0d9398388e98.json
  Female/
    ...
```

Extra nested folders are allowed. The script scans recursively, so it does not require one fixed depth.

Each utterance should have:

- one `.flac` audio file
- one matching `.json` metadata file
- transcript text under `annotation[*]["sentence"]`

Example JSON:

```json
{
  "duration": 6.66,
  "speaker_id": "01332512906",
  "gender": "পুরুষ",
  "script_source": "manually curated",
  "path": "Male/01332512906/2024-05-07/00d4da35-bebc-471b-aa92-0d9398388e98.flac",
  "speech_id": "00d4da35-bebc-471b-aa92-0d9398388e98",
  "annotation": [
    {
      "tagList": [],
      "start": 1.07355922,
      "end": 6.08486152,
      "id": "rPboUgsw",
      "sentence": "তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।",
      "words": [
        {"start": 1.17355922, "end": 1.43355922, "id": "rPboUgsw-1", "word": "তার"}
      ]
    }
  ]
}
```

### Original VITS Dataset Pattern Vs Bengali Dataset Pattern

The original VITS repository does not train directly from raw annotation JSON files. It expects the dataset to already be represented as VITS filelists that point to readable WAV audio.

Original VITS single-speaker pattern, such as LJ Speech:

```text
dataset already has WAV files
filelist line format:
wav_path|text
```

Example:

```text
DUMMY1/LJ001-0001.wav|Printing, in the only sense with which we are at present concerned...
```

Original VITS multi-speaker pattern, such as VCTK:

```text
dataset already has WAV files
filelist line format:
wav_path|speaker_id|text
```

Example:

```text
DUMMY2/p225/p225_001.wav|0|Please call Stella.
```

The Bengali raw dataset starts in a different pattern:

```text
/home/kawshik/TTS_Dataset/
  Male/
    <speaker_id>/
      optional_date_folder/
        utterance.flac
        utterance.json
  Female/
    <speaker_id>/
      optional_date_folder/
        utterance.flac
        utterance.json
```

The important differences are:

- Original VITS expects filelists first; the Bengali dataset starts as raw folders and JSON metadata.
- Original VITS expects WAV audio; the Bengali dataset audio is `.flac`.
- Original VITS reads text directly from the filelist; the Bengali transcript is inside `annotation[*]["sentence"]`.
- Original VITS examples use fixed known datasets like LJ Speech or VCTK; the Bengali dataset can have variable folder depth.
- The Bengali single-speaker baseline must filter one `speaker_id` before training.

Therefore `scripts/prepare_bengali_dataset.py` converts the Bengali raw pattern into the original VITS training pattern:

```text
raw Bengali .flac + .json
  -> select one speaker_id
  -> extract annotation[*]["sentence"]
  -> convert .flac to 22050 Hz mono WAV
  -> write wav_path|text filelists
```

After preparation, Bengali training uses the same single-speaker filelist style expected by original VITS:

```text
/home/kawshik/TTS_Dataset_vits_bengali_22050/wavs/.../utterance.wav|তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।
```

## 7. Dummy Dataset Example

A small structural example is included here:

```text
docs/sample_bengali_raw_dataset/
```

It shows the folder shape and a sample JSON file. It is intentionally not a real training dataset because it does not include valid audio. Use it only to understand structure and schema.

Sample folder:

```text
docs/sample_bengali_raw_dataset/
  README.md
  Male/
    01332512906/
      2024-05-07/
        00d4da35-bebc-471b-aa92-0d9398388e98.json
        PUT_MATCHING_FLAC_HERE.txt
    01700000001/
      male-no-date-0001.json
      PUT_MATCHING_FLAC_HERE.txt
  Female/
    01800000001/
      2024-05-08/
        female-date-0001.json
        PUT_MATCHING_FLAC_HERE.txt
    01900000001/
      female-no-date-0001.json
      PUT_MATCHING_FLAC_HERE.txt
```

Your real remote dataset must contain a valid matching `.flac` file beside, or discoverable from, each JSON file.

## 8. Transcript Extraction

`scripts/prepare_bengali_dataset.py` extracts text like this:

- Reads each JSON using UTF-8.
- Uses all non-empty `annotation[*]["sentence"]` values.
- Sorts annotation entries by `start` when present.
- Joins multiple sentence fragments with spaces.
- Replaces `|` with spaces because VITS filelists use `|` as the separator.
- Falls back to top-level `sentence`, `transcript`, or `text` only if annotation sentences are missing.
- Skips malformed JSON, missing text, missing audio, or failed audio conversion.

Skipped examples are recorded in:

```text
<PREPARED_ROOT>/bengali_prep_skipped.tsv
```

## 9. Audio Conversion

The raw dataset audio is `.flac`, but original VITS reads WAV using `scipy.io.wavfile.read`.

So preparation converts:

```text
.flac -> mono 16-bit PCM .wav at 22050 Hz
```

The script first uses `ffmpeg` if it is available on `PATH`. If not, it uses the pip package `imageio-ffmpeg`, which is installed through `requirements.txt`.

Prepared audio is written under:

```text
<PREPARED_ROOT>/wavs/
```

Example:

```text
/home/kawshik/TTS_Dataset_vits_bengali_22050/wavs/...
```

## 10. Bengali Text Processing

This baseline uses Bengali characters directly.

`bengali_cleaners`:

- Applies Unicode NFC normalization.
- Removes BOM, zero-width joiner, and zero-width non-joiner.
- Replaces `|` with spaces.
- Collapses whitespace.
- Preserves Bengali Unicode block characters.
- Preserves ASCII letters/digits for occasional code-mixed text.
- Preserves common Bengali punctuation including `।`.
- Does not use English phonemization, espeak, or transliteration.

Model representation:

```text
Bengali text -> cleaned Bengali Unicode characters -> symbol IDs
```

Bengali conjuncts are represented as normal Unicode character sequences using `্`.

## 11. Filelist Generation

Single-speaker preparation writes original VITS single-speaker filelists:

```text
wav_path|text
```

Generated files:

```text
filelists/bengali_audio_text_train_filelist.txt
filelists/bengali_audio_text_val_filelist.txt
filelists/bengali_audio_text_test_filelist.txt
```

After text preprocessing, VITS trains from:

```text
filelists/bengali_audio_text_train_filelist.txt.cleaned
filelists/bengali_audio_text_val_filelist.txt.cleaned
```

These are already configured in `configs/bengali_base.json`.

Multi-speaker preparation writes original VITS multi-speaker filelists:

```text
wav_path|speaker_index|text
```

Generated files:

```text
filelists/bengali_ms_audio_sid_text_train_filelist.txt
filelists/bengali_ms_audio_sid_text_val_filelist.txt
filelists/bengali_ms_audio_sid_text_test_filelist.txt
```

After text preprocessing, multi-speaker VITS trains from:

```text
filelists/bengali_ms_audio_sid_text_train_filelist.txt.cleaned
filelists/bengali_ms_audio_sid_text_val_filelist.txt.cleaned
```

The multi-speaker prep script also writes:

```text
<PREPARED_ROOT>/speaker_map.json
<PREPARED_ROOT>/speaker_stats.tsv
configs/bengali_ms.generated.json
```

`configs/bengali_ms.generated.json` is created from `configs/bengali_ms.json` and contains the correct `n_speakers` value for that prepared speaker set.

## 12. Step-By-Step Running Guide

Run these steps on the remote Linux server from the repository root.

### Step 1: Enter The Repository

```bash
cd ~/vits-bengali-tts
```

Use the actual path if your repo is somewhere else.

### Step 2: Create And Activate Environment

For RTX 30/40/50-series GPUs, use a modern PyTorch CUDA build. Python 3.10 is recommended.

```bash
conda create -n vits-bn310 python=3.10 -y
conda activate vits-bn310
python --version
```

### Step 3: Install PyTorch And Requirements

For RTX 5090 / recent GPU support:

```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

This also installs `imageio-ffmpeg`, so no separate system `ffmpeg` install is required for Bengali dataset preparation.

The working server environment has also been recorded under:

```text
environment_snapshots/
```

Those files are snapshots for reproducibility and debugging. They are not used directly by training.

### Step 4: Verify CUDA

For GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda'); print((x@x).shape)"
```

Expected:

```text
True
NVIDIA GeForce RTX ...
torch.Size([64, 64])
```

Inside the process, physical GPU 1 appears as CUDA device `0`. That is normal.

### Step 5: Optional Faster Alignment Build

The code has a Python fallback, so this build is optional. If it succeeds, training is faster.

```bash
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

If you see an output-folder error:

```bash
cd monotonic_align
mkdir -p monotonic_align
touch monotonic_align/__init__.py
python setup.py build_ext --inplace
cd ..
```

### Step 6A: Prepare Dataset For Single-Speaker Training

Set these values:

```text
DATASET_ROOT=/home/kawshik/TTS_Dataset
PREPARED_ROOT=/home/kawshik/TTS_Dataset_vits_bengali_22050
SPEAKER_ID=01332512906
```

Run:

```bash
python scripts/prepare_bengali_dataset.py \
  --dataset-root /home/kawshik/TTS_Dataset \
  --output-root /home/kawshik/TTS_Dataset_vits_bengali_22050 \
  --speaker-id 01332512906 \
  --sample-rate 22050 \
  --val-ratio 0.02 \
  --test-ratio 0.02 \
  --seed 1234 \
  --min-duration 0.5 \
  --max-duration 11.5 \
  --max-text-chars 250
```

Expected successful summary looks like:

```text
JSON files found: ...
FLAC files found: ...
Records kept: ...
Train/val/test: .../.../...
Skipped: ...
Wrote filelists:
  train: ...
  val: ...
  test: ...
Report: ...
```

Observed sample result for speaker `01332512906` on the remote dataset:

```text
JSON files found: 148026
FLAC files found: 148026
Records kept: 224
Train/val/test: 216/4/4
Skipped: 147802
Wrote filelists:
  train: /home/kawshik/vits-bengali-tts/filelists/bengali_audio_text_train_filelist.txt
  val:   /home/kawshik/vits-bengali-tts/filelists/bengali_audio_text_val_filelist.txt
  test:  /home/kawshik/vits-bengali-tts/filelists/bengali_audio_text_test_filelist.txt
Report: /home/kawshik/TTS_Dataset_vits_bengali_22050/bengali_prep_report.json
```

The high skipped count is expected here because the script scanned all speakers but kept only one selected `speaker_id`.

### Step 7A: Review Single-Speaker Prep Output

```bash
cat /home/kawshik/TTS_Dataset_vits_bengali_22050/bengali_prep_report.json
head /home/kawshik/TTS_Dataset_vits_bengali_22050/bengali_prep_skipped.tsv
head filelists/bengali_audio_text_train_filelist.txt
```

Check:

- `records_kept` is greater than `0`.
- `train_count` is greater than `0`.
- filelist paths point to prepared `.wav` files.

A large skipped count is normal when filtering one speaker from a multi-speaker dataset.

### Step 8A: Preprocess Single-Speaker Text

```bash
python preprocess.py \
  --text_index 1 \
  --text_cleaners bengali_cleaners \
  --filelists \
  filelists/bengali_audio_text_train_filelist.txt \
  filelists/bengali_audio_text_val_filelist.txt \
  filelists/bengali_audio_text_test_filelist.txt
```

Confirm:

```bash
ls -lh filelists/bengali_audio_text_*cleaned
head filelists/bengali_audio_text_train_filelist.txt.cleaned
```

### Step 9A: Start Single-Speaker Training

For physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c configs/bengali_base.json -m bengali_base
```

Successful early output includes:

```text
[INFO] Train Epoch: 1 [0%]
[INFO] Saving model and optimizer state ...
[INFO] ====> Epoch: 1
```

Observed early training sample:

```text
[INFO] ====> Epoch: 1
[INFO] ====> Epoch: 2
[INFO] ====> Epoch: 3
[INFO] ====> Epoch: 4
[INFO] ====> Epoch: 5
[INFO] Train Epoch: 6 [56%]
[INFO] [2.1879518032073975, 2.4627513885498047, 3.9181230068206787, 40.8165283203125, 2.0494723320007324, 1.859900712966919, 100, 0.00019987503124609398]
[INFO] ====> Epoch: 6
[INFO] Train Epoch: 12 [11%]
[INFO] [2.5117287635803223, 2.326852560043335, 2.2720282077789307, 33.08064270019531, 2.1187636852264404, 1.3832086324691772, 200, 0.00019972517181056292]
```

The loss list is:

```text
[discriminator_loss, generator_loss, feature_matching_loss, mel_loss, duration_loss, kl_loss, global_step, learning_rate]
```

This sample indicates that training started successfully, completed multiple epochs, saved checkpoints, and the mel loss dropped from the first-step value.

Logs and checkpoints:

```text
logs/bengali_base/
```

Watch training:

```bash
tail -f logs/bengali_base/train.log
watch -n 2 nvidia-smi
```

### Step 10A: Find A Single-Speaker Checkpoint

```bash
ls logs/bengali_base/G_*.pth
```

Example:

```text
logs/bengali_base/G_1000.pth
```

### Step 11A: Run Single-Speaker Inference

Replace `<STEP>` with a real checkpoint step:

```bash
python scripts/infer_bengali.py \
  --config configs/bengali_base.json \
  --checkpoint logs/bengali_base/G_<STEP>.pth \
  --text "তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।" \
  --output outputs/bengali_sample.wav \
  --noise-scale 0.667 \
  --noise-scale-w 0.8 \
  --length-scale 1.0
```

Output:

```text
outputs/bengali_sample.wav
```

### Step 6B: Prepare Dataset For Multi-Speaker Training

Multi-speaker preparation keeps multiple speakers and writes `wav_path|speaker_index|text` filelists.

For the first multi-speaker experiment, start with a small clean subset such as 5-10 speakers. You can select speakers explicitly:

```bash
python scripts/prepare_bengali_ms_dataset.py \
  --dataset-root /home/kawshik/TTS_Dataset \
  --output-root /home/kawshik/TTS_Dataset_vits_bengali_ms_22050 \
  --speaker-ids 01332512906,01800000001,01700000001 \
  --sample-rate 22050 \
  --val-ratio 0.02 \
  --test-ratio 0.02 \
  --seed 1234 \
  --min-duration 0.5 \
  --max-duration 11.5 \
  --max-text-chars 250 \
  --min-utterances-per-speaker 50 \
  --max-utterances-per-speaker 300
```

Or let the script choose the top speakers by kept utterance count:

```bash
python scripts/prepare_bengali_ms_dataset.py \
  --dataset-root /home/kawshik/TTS_Dataset \
  --output-root /home/kawshik/TTS_Dataset_vits_bengali_ms_22050 \
  --sample-rate 22050 \
  --val-ratio 0.02 \
  --test-ratio 0.02 \
  --seed 1234 \
  --min-duration 0.5 \
  --max-duration 11.5 \
  --max-text-chars 250 \
  --min-utterances-per-speaker 200 \
  --max-speakers 10 \
  --max-utterances-per-speaker 500
```

Expected multi-speaker outputs:

```text
filelists/bengali_ms_audio_sid_text_train_filelist.txt
filelists/bengali_ms_audio_sid_text_val_filelist.txt
filelists/bengali_ms_audio_sid_text_test_filelist.txt
/home/kawshik/TTS_Dataset_vits_bengali_ms_22050/speaker_map.json
/home/kawshik/TTS_Dataset_vits_bengali_ms_22050/speaker_stats.tsv
configs/bengali_ms.generated.json
```

### Step 7B: Review Multi-Speaker Prep Output

```bash
cat /home/kawshik/TTS_Dataset_vits_bengali_ms_22050/bengali_ms_prep_report.json
cat /home/kawshik/TTS_Dataset_vits_bengali_ms_22050/speaker_map.json
head /home/kawshik/TTS_Dataset_vits_bengali_ms_22050/speaker_stats.tsv
head filelists/bengali_ms_audio_sid_text_train_filelist.txt
```

Check:

- `speakers_kept` is greater than `1` for real multi-speaker training.
- `speakers_kept` equal to `1` is only useful as a smoke test for the multi-speaker code path.
- `records_kept` is large enough for training.
- `configs/bengali_ms.generated.json` has `n_speakers` equal to the speaker count.
- filelist lines look like `wav_path|speaker_index|text`.

### Step 8B: Preprocess Multi-Speaker Text

Multi-speaker text is column index `2`:

```text
wav_path|speaker_index|text
```

Run:

```bash
python preprocess.py \
  --text_index 2 \
  --text_cleaners bengali_cleaners \
  --filelists \
  filelists/bengali_ms_audio_sid_text_train_filelist.txt \
  filelists/bengali_ms_audio_sid_text_val_filelist.txt \
  filelists/bengali_ms_audio_sid_text_test_filelist.txt
```

Confirm:

```bash
ls -lh filelists/bengali_ms_audio_sid_text_*cleaned
head filelists/bengali_ms_audio_sid_text_train_filelist.txt.cleaned
```

### Step 9B: Start Multi-Speaker Training

For physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ms.py -c configs/bengali_ms.generated.json -m bengali_ms
```

Use a different model name for different speaker sets:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ms.py -c configs/bengali_ms.generated.json -m bengali_ms_top10
```

### Step 10B: Find A Multi-Speaker Checkpoint

```bash
ls logs/bengali_ms/G_*.pth
```

Example:

```text
logs/bengali_ms/G_1000.pth
```

### Step 11B: Run Multi-Speaker Inference

Using the original raw dataset speaker ID:

```bash
python scripts/infer_bengali_ms.py \
  --config configs/bengali_ms.generated.json \
  --checkpoint logs/bengali_ms/G_<STEP>.pth \
  --speaker-map /home/kawshik/TTS_Dataset_vits_bengali_ms_22050/speaker_map.json \
  --speaker-id 01332512906 \
  --text "আমি বাংলা ভাষায় কথা বলি।" \
  --output outputs/bengali_ms_01332512906.wav \
  --noise-scale 0.667 \
  --noise-scale-w 0.8 \
  --length-scale 1.0
```

Using a direct numeric speaker index:

```bash
python scripts/infer_bengali_ms.py \
  --config configs/bengali_ms.generated.json \
  --checkpoint logs/bengali_ms/G_<STEP>.pth \
  --speaker-index 0 \
  --text "আমি বাংলা ভাষায় কথা বলি।" \
  --output outputs/bengali_ms_speaker0.wav
```

## 13. Common Problems And Fixes

### Old PyTorch GPU Error

Error:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Cause:

Old PyTorch was installed and does not support the GPU architecture.

Fix:

Use Python 3.10 and install a modern CUDA PyTorch wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Missing TensorBoard

Error:

```text
ModuleNotFoundError: No module named 'tensorboard'
```

Fix:

```bash
pip install -r requirements.txt
```

### Librosa Mel API Error

Error:

```text
TypeError: mel() takes 0 positional arguments but 5 were given
```

Fix:

Use the updated `mel_processing.py` from this repo.

### Missing Cleaned Filelists

Error:

```text
FileNotFoundError: filelists/bengali_audio_text_train_filelist.txt.cleaned
```

Fix for single-speaker:

Run Step 8A.

Fix for multi-speaker:

Run Step 8B and make sure `--text_index 2` was used.

### CUDA Out Of Memory

Reduce batch size in `configs/bengali_base.json`:

```json
"batch_size": 8
```

or:

```json
"batch_size": 4
```

### Very Small Dataset

If only a few hundred utterances are kept for one speaker, the pipeline can run, but quality may be weak from scratch. More clean data for one speaker usually improves results.

## 14. Important Assumptions

- The single-speaker path trains one model for one selected `speaker_id`.
- The multi-speaker path trains one model for multiple speaker indices.
- For single-speaker, the selected `speaker_id` should represent one consistent voice.
- For multi-speaker, `speaker_map.json` maps raw speaker IDs to numeric speaker indices.
- Audio is converted to 22050 Hz because the config uses `sampling_rate: 22050`.
- Text is character-based Bengali, not phoneme-based Bengali.
- The prep scripts rely on JSON metadata duration for duration filtering.
- The real dataset must contain valid `.flac` audio; the docs sample dataset is only structural.

## 15. Multi-Speaker Notes

Multi-speaker Bengali VITS is now available as a separate runnable path.

The goal of multi-speaker training would be to train one model that can synthesize multiple voices. Instead of creating one model per speaker, the model would learn a speaker embedding for each speaker.

The filelist format would change from single-speaker:

```text
wav_path|text
```

to multi-speaker:

```text
wav_path|speaker_index|text
```

Example:

```text
/prepared/wavs/Male/01332512906/a.wav|0|তার কথাগুলো শুনে বুঝলাম...
/prepared/wavs/Female/01800000001/b.wav|1|বাংলা ভাষার জন্য...
/prepared/wavs/Male/01700000001/c.wav|2|আজকের আবহাওয়া...
```

The original dataset speaker IDs would need a stable numeric mapping:

```json
{
  "01332512906": 0,
  "01800000001": 1,
  "01700000001": 2
}
```

The current implementation adds:

- a Bengali multi-speaker preparation script
- speaker filtering by minimum utterance count or minimum duration
- `speaker_id` to numeric `speaker_index` mapping
- saved `speaker_map.json`
- train/val/test filelists in `wav_path|speaker_index|text` format
- a Bengali multi-speaker config template and generated config
- training through `train_ms.py`
- inference that accepts a speaker index or original speaker ID

The config would use:

```json
"n_speakers": <number_of_speakers>
```

The generated config sets this automatically. Training uses:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ms.py -c configs/bengali_ms.generated.json -m bengali_ms
```

Recommended multi-speaker workflow:

1. Finish and listen to the single-speaker baseline.
2. Train a few separate single-speaker models to identify clean speakers.
3. Select speakers with enough clean, consistent audio.
4. Build the multi-speaker filelist and speaker map with `scripts/prepare_bengali_ms_dataset.py`.
5. Train with `train_ms.py`.
6. Evaluate whether each speaker identity is preserved during inference.

Multi-speaker training should be tested with a small clean speaker set first because it adds more failure points: unbalanced speakers, noisy speakers, speaker ID mistakes, and inference-time speaker selection.

For quick smoke tests, you can keep only one speaker or cap each speaker:

```bash
--speaker-ids 01332512906
--max-utterances-per-speaker 100
```

That verifies the multi-speaker code path, but it is not true multi-speaker training. True multi-speaker training requires at least two speakers in `speaker_map.json`.

## 16. Known Limitations And Next Steps

- No Bengali number/date normalization yet.
- No Bengali G2P or phoneme pipeline yet.
- Multi-speaker support is available, but speaker quality and balance still need manual review.
- No automatic audio quality scoring.
- No silence trimming yet.
- Future improvements can add richer Bengali normalization, better speaker selection, and better dataset validation.

## 17. Environment Snapshots

The folder `environment_snapshots/` records the package and GPU environment that was used successfully on the remote server.

Expected files:

```text
environment_snapshots/
  python_version.txt
  torch_cuda_version.txt
  requirements_server_working.txt
  conda_list_server_working.txt
  nvidia_smi_server_working.txt
```

Purpose of each file:

- `python_version.txt`: Python version used by the working environment.
- `torch_cuda_version.txt`: PyTorch version and CUDA runtime version.
- `requirements_server_working.txt`: exact `pip freeze` output from the working environment.
- `conda_list_server_working.txt`: full `conda list` output from the working environment.
- `nvidia_smi_server_working.txt`: GPU, driver, and CUDA driver information.

These snapshots are useful when reproducing the training setup or explaining the working server configuration. Do not replace the minimal `requirements.txt` with `pip freeze` unless you intentionally want a fully pinned server-specific environment.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

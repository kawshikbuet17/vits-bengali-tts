# Major Changes From Original VITS

This document gives a high-level summary of how this fork differs from the original `jaywalnut310/vits` repository.

The purpose of the fork is simple: keep the original VITS model and training flow mostly intact, but add runnable Bangla/Bengali single-speaker and multi-speaker TTS paths for a raw `.flac` + `.json` dataset.

## 1. What This Fork Adds

Original VITS expects data to already be in VITS-friendly form: WAV audio plus filelists.

This fork adds the Bengali dataset bridge:

```text
raw Bengali .flac + .json dataset
  -> prepared VITS filelists and WAV audio
  -> Bengali VITS training
  -> Bengali speech inference
```

Two Bengali paths are now available:

```text
single-speaker: wav_path|text               -> train.py
multi-speaker:  wav_path|speaker_index|text -> train_ms.py
```

## 2. Dataset Support

The Bengali dataset has a different structure from the datasets used in the original VITS examples.

Original VITS examples usually assume:

- audio is already WAV
- transcript text is already in filelists
- filelists are already prepared before training

The Bengali dataset starts as:

```text
TTS_Dataset/
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

Text is extracted from:

```text
annotation[*]["sentence"]
```

The preparation scripts convert this raw structure into the filelist structures expected by VITS.

## 3. Bengali Text Support

The original repository is mostly oriented around English examples and phoneme-based preprocessing.

This fork adds a simple Bengali character-based path:

- Bengali Unicode symbols are added to the symbol inventory.
- `bengali_cleaners` is added for Bengali text normalization.
- English-only assumptions are avoided for the Bengali baseline.
- Bengali text is represented directly as characters first, not phonemes.

This is intentionally minimal so the first baselines can run before adding more advanced Bengali normalization or G2P.

## 4. Training And Inference Additions

This fork adds:

- single-speaker Bengali dataset preparation
- multi-speaker Bengali dataset preparation
- single-speaker Bengali training config
- multi-speaker Bengali config template and generated config support
- single-speaker Bengali command-line inference
- multi-speaker Bengali command-line inference with speaker selection
- Bengali-specific documentation and sample raw dataset layout

The core model architecture is not broadly refactored.

## 5. Environment And Package Changes

The dependency setup was adjusted for the working server environment and modern GPU support.

Important high-level changes:

- PyTorch is installed separately based on the target GPU/CUDA version.
- The working server used PyTorch `2.11.0+cu128`.
- `requirements.txt` contains supporting Python package pins, but not the GPU-specific PyTorch wheel.
- `imageio-ffmpeg` is included so FLAC-to-WAV conversion can work inside the Python environment.
- `librosa`, `protobuf`, `tensorboard`, `numpy`, and related packages were updated/pinned to avoid version conflicts seen during setup.

The exact working server environment is recorded under:

```text
environment_snapshots/
```

## 6. Compatibility Fixes

A few practical compatibility fixes were added so the fork runs on the current server setup:

- `mel_processing.py` was adjusted for newer `librosa` and PyTorch STFT behavior.
- `train.py` and `train_ms.py` use a valid distributed training port.
- `monotonic_align` has build/fallback improvements.
- noisy dependency debug logs are suppressed so training logs stay readable.

These changes are practical runtime fixes, not a broad redesign of VITS.

## 7. Documentation Added

New documentation explains:

- what this project does
- input and output of the Bengali TTS pipeline
- original VITS dataset pattern vs Bengali raw dataset pattern
- which commands are for single-speaker and which are for multi-speaker
- how to prepare data, train, and run inference
- what changed from original VITS
- known limitations and next steps

Main docs:

```text
docs/TTS_VITS_Bengali_Implementation_Guide.md
docs/TTS_VITS_Beginner_Guide.md
docs/Major_Changes_From_Original_VITS.md
```

## 8. What Stayed Close To Original VITS

The fork intentionally preserves the original style of the repository:

- `train.py` is still used for single-speaker training.
- `train_ms.py` is used for multi-speaker training.
- original configs and filelists are kept.
- the VITS model structure is not rewritten.
- Bengali support is added as focused paths rather than a full refactor.

## 9. Current Status

The Bengali single-speaker path has reached the first successful end-to-end milestone:

- dataset preparation ran on the remote dataset
- Bengali filelists were generated
- preprocessing completed
- CUDA training started and saved checkpoints
- inference generated audible Bengali speech

The multi-speaker path has been added as a runnable code path and should be tested next on a small clean speaker subset before scaling to many speakers.

One-speaker multi-speaker runs can be used as a smoke test, but true multi-speaker training requires at least two speakers in the generated `speaker_map.json`.

## 10. Next Steps

Future work can improve the Bengali paths with:

- better speaker selection
- better speaker balancing for multi-speaker training
- Bengali number/date normalization
- optional Bengali phoneme or G2P support
- audio quality checks and silence trimming
- listening-based evaluation across checkpoints and speakers

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

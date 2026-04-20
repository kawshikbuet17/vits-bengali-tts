# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

This project prepares a raw Bangla/Bengali speech dataset and trains a VITS text-to-speech model. The current baseline uses Bengali text plus matching `.flac` audio/`.json` metadata for one selected speaker, and produces a trained model that can synthesize Bengali speech as `.wav` audio from new Bengali text. The same pipeline can be expanded later toward multi-speaker Bengali TTS.

## Bangla/Bengali TTS Fork Notice

This repository is a lightly modified fork of the original [`jaywalnut310/vits`](https://github.com/jaywalnut310/vits) codebase. It keeps the original VITS training and inference flow, but adds a runnable Bengali single-speaker baseline for a raw `.flac` + `.json` dataset.

For Bengali training, start with the complete step-by-step guide: [docs/TTS_VITS_Bengali_Implementation_Guide.md](docs/TTS_VITS_Bengali_Implementation_Guide.md).

For a concise summary of what changed from the original VITS repository, see [docs/Major_Changes_From_Original_VITS.md](docs/Major_Changes_From_Original_VITS.md).

Main Bengali additions:

- raw dataset preparation from recursive `Male/` and `Female/` folders
- `.flac` to VITS-compatible `.wav` conversion
- Bengali character-based text cleaning
- single-speaker Bengali config at `configs/bengali_base.json`
- Bengali inference helper at `scripts/infer_bengali.py`
- working server environment snapshots in `environment_snapshots/`

The original VITS README content is preserved below for reference.

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
0. Python >= 3.6 for the original repo. RTX 50-series GPU training needs Python >= 3.10 with a CUDA 12.8 PyTorch build.
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. Bengali character training does not require English speech tools such as espeak.
    1. `imageio-ffmpeg` provides an ffmpeg binary for Bengali audio preparation.
    1. Install PyTorch separately for your GPU, for example: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1`
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Optionally build Monotonic Alignment Search for faster training, and run preprocessing if you use your own datasets.
```sh
# Optional Cython-version Monotonoic Alignment Search. A Python fallback is available if this is skipped.
cd monotonic_align
python setup.py build_ext --inplace

# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base
```


## Inference Example
See [inference.ipynb](inference.ipynb)


## Bengali Single-Speaker Baseline

This fork also includes a minimal Bengali single-speaker setup for raw `.flac` + `.json` datasets. The original VITS training code is preserved: `scripts/prepare_bengali_dataset.py` converts the raw data into mono 22050 Hz WAV files and writes standard single-speaker VITS filelists in `wav|text` format.

The original VITS examples assume data has already been arranged into VITS-style filelists and WAV audio, for example LJ Speech style `wav_path|text` or VCTK multi-speaker style `wav_path|speaker_id|text`. The Bengali dataset starts in a different raw format: recursive `Male/` and `Female/` folders, speaker folders, optional date folders, matching `.flac` + `.json` files, and transcript text inside `annotation[*]["sentence"]`. The Bengali preparation script bridges that gap and creates the filelists VITS expects.

This baseline intentionally trains **one speaker at a time**. The `--speaker-id` value selects one voice from the larger raw dataset, and all other speakers are skipped for that run. To try another speaker, prepare the dataset again with a different `--speaker-id`, then preprocess and train under a different model name.

Multi-speaker Bengali VITS is a future direction. It will require multi-speaker filelists such as `wav_path|speaker_index|text`, a speaker mapping file, a multi-speaker config, and training through `train_ms.py`.

Full details are in [docs/TTS_VITS_Bengali_Implementation_Guide.md](docs/TTS_VITS_Bengali_Implementation_Guide.md).

```sh
# Prepare one speaker from the remote dataset
python scripts/prepare_bengali_dataset.py \
  --dataset-root /home/kawshik/TTS_Dataset \
  --output-root /home/kawshik/TTS_Dataset_vits_bengali_22050 \
  --speaker-id 01332512906 \
  --sample-rate 22050 \
  --val-ratio 0.02 \
  --test-ratio 0.02 \
  --min-duration 0.5 \
  --max-duration 11.5

# Clean Bengali text into .cleaned filelists
python preprocess.py \
  --text_index 1 \
  --text_cleaners bengali_cleaners \
  --filelists \
  filelists/bengali_audio_text_train_filelist.txt \
  filelists/bengali_audio_text_val_filelist.txt \
  filelists/bengali_audio_text_test_filelist.txt

# Train
python train.py -c configs/bengali_base.json -m bengali_base

# Infer from a trained generator checkpoint
python scripts/infer_bengali.py \
  --config configs/bengali_base.json \
  --checkpoint logs/bengali_base/G_<STEP>.pth \
  --text "তার কথাগুলো শুনে বুঝলাম বয়সের তুলনায় সে মানসিকতায় অনেক বড় হয়ে গিয়েছে।" \
  --output outputs/bengali_sample.wav
```

Customize `/home/kawshik/TTS_Dataset`, `/home/kawshik/TTS_Dataset_vits_bengali_22050`, `01332512906`, batch size, and checkpoint step on the remote server.

## Environment Snapshots

The folder `environment_snapshots/` stores exported details from the working remote server environment. These files are for reproducibility and debugging, not for direct training input.

Typical files include:

- `python_version.txt`
- `torch_cuda_version.txt`
- `requirements_server_working.txt`
- `conda_list_server_working.txt`
- `nvidia_smi_server_working.txt`

Use these files to confirm the Python, PyTorch, CUDA, package, GPU, and driver setup that successfully ran Bengali VITS training.

---

## Prepared By
**Kawshik Kumar Paul**  
Software Engineer | Researcher  
Department of Computer Science and Engineering (CSE)  
Bangladesh University of Engineering and Technology (BUET)  
**Email:** kawshikbuet17@gmail.com  

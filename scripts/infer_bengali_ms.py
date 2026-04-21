import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from scipy.io.wavfile import write

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols


def parse_args():
  parser = argparse.ArgumentParser(description="Run Bengali multi-speaker VITS inference.")
  parser.add_argument("--config", default="configs/bengali_ms.generated.json",
                      help="Path to Bengali multi-speaker VITS config")
  parser.add_argument("--checkpoint", required=True,
                      help="Path to generator checkpoint, e.g. logs/bengali_ms/G_100000.pth")
  parser.add_argument("--text", required=True,
                      help="Bengali text to synthesize")
  parser.add_argument("--output", default="outputs/bengali_ms_sample.wav",
                      help="Output wav path")
  parser.add_argument("--speaker-map", default=None,
                      help="Path to speaker_map.json from multi-speaker preparation")
  parser.add_argument("--speaker-id", default=None,
                      help="Original raw dataset speaker_id, looked up in --speaker-map")
  parser.add_argument("--speaker-index", type=int, default=None,
                      help="Numeric VITS speaker index; overrides --speaker-id")
  parser.add_argument("--noise-scale", type=float, default=0.667)
  parser.add_argument("--noise-scale-w", type=float, default=0.8)
  parser.add_argument("--length-scale", type=float, default=1.0)
  parser.add_argument("--max-len", type=int, default=None)
  parser.add_argument("--cpu", action="store_true",
                      help="Run inference on CPU instead of CUDA")
  return parser.parse_args()


def get_text(text, hps):
  text_norm = text_to_sequence(text, hps.data.text_cleaners)
  if hps.data.add_blank:
    text_norm = commons.intersperse(text_norm, 0)
  return torch.LongTensor(text_norm)


def resolve_speaker_index(args):
  if args.speaker_index is not None:
    return args.speaker_index
  if not args.speaker_id:
    raise RuntimeError("Provide --speaker-index or --speaker-id with --speaker-map.")
  if not args.speaker_map:
    raise RuntimeError("--speaker-map is required when using --speaker-id.")
  with open(args.speaker_map, "r", encoding="utf-8") as f:
    speaker_map = json.load(f)
  if args.speaker_id not in speaker_map:
    raise RuntimeError("speaker_id {} not found in {}".format(args.speaker_id, args.speaker_map))
  return int(speaker_map[args.speaker_id])


def main():
  args = parse_args()
  hps = utils.get_hparams_from_file(args.config)
  speaker_index = resolve_speaker_index(args)
  n_speakers = int(hps.data.n_speakers)
  if speaker_index < 0 or speaker_index >= n_speakers:
    raise RuntimeError("speaker index {} is outside config n_speakers={}".format(
        speaker_index, n_speakers))

  device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).to(device)
  net_g.eval()

  utils.load_checkpoint(args.checkpoint, net_g, None)

  text = get_text(args.text, hps)
  sid = torch.LongTensor([speaker_index]).to(device)
  with torch.no_grad():
    x = text.to(device).unsqueeze(0)
    x_lengths = torch.LongTensor([text.size(0)]).to(device)
    audio = net_g.infer(
        x,
        x_lengths,
        sid=sid,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        max_len=args.max_len)[0][0, 0].data.cpu().float().numpy()

  output_dir = os.path.dirname(args.output)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
  audio = np.clip(audio, -1.0, 1.0)
  write(args.output, hps.data.sampling_rate, (audio * 32767.0).astype(np.int16))
  print("Speaker index: {}".format(speaker_index))
  print("Wrote {}".format(args.output))


if __name__ == "__main__":
  main()

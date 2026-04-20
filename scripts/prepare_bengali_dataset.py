import argparse
import json
import os
import random
import shutil
import subprocess
import sys


def parse_args():
  parser = argparse.ArgumentParser(
      description="Prepare raw Bengali .flac/.json data for single-speaker VITS training.")
  parser.add_argument("--dataset-root", required=True,
                      help="Root of the raw dataset, e.g. /home/kawshik/TTS_Dataset")
  parser.add_argument("--output-root", required=True,
                      help="Directory where prepared wavs and prep reports are written")
  parser.add_argument("--filelists-dir", default="filelists",
                      help="Directory where VITS filelists are written")
  parser.add_argument("--prefix", default="bengali_audio_text",
                      help="Prefix for generated train/val/test filelists")
  parser.add_argument("--speaker-id", default=None,
                      help="Optional speaker_id to keep for single-speaker training")
  parser.add_argument("--sample-rate", type=int, default=22050,
                      help="Target sample rate for prepared wavs")
  parser.add_argument("--val-ratio", type=float, default=0.02,
                      help="Validation split ratio")
  parser.add_argument("--test-ratio", type=float, default=0.02,
                      help="Test split ratio")
  parser.add_argument("--seed", type=int, default=1234,
                      help="Random seed for splits")
  parser.add_argument("--min-duration", type=float, default=0.0,
                      help="Skip records whose JSON duration is below this value when present")
  parser.add_argument("--max-duration", type=float, default=0.0,
                      help="Skip records whose JSON duration is above this value when present; 0 disables")
  parser.add_argument("--min-text-chars", type=int, default=1,
                      help="Skip records with shorter extracted transcript text")
  parser.add_argument("--max-text-chars", type=int, default=250,
                      help="Skip records with longer extracted transcript text; 0 disables")
  parser.add_argument("--overwrite", action="store_true",
                      help="Recreate prepared wav files even if they already exist")
  parser.add_argument("--dry-run", action="store_true",
                      help="Scan and report without converting audio or writing filelists")
  return parser.parse_args()


def norm_path(path):
  return os.path.normpath(path.replace("\\", os.sep).replace("/", os.sep))


def read_json(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def collapse_whitespace(text):
  return " ".join(text.replace("|", " ").split())


def get_float(value):
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def extract_sentence(data):
  annotation = data.get("annotation")
  sentences = []
  if isinstance(annotation, list):
    sortable = []
    for idx, item in enumerate(annotation):
      if not isinstance(item, dict):
        continue
      sentence = item.get("sentence")
      if isinstance(sentence, str):
        sentence = collapse_whitespace(sentence)
        if sentence:
          start = get_float(item.get("start"))
          if start is None:
            start = float(idx)
          sortable.append((start, sentence))
    sortable.sort(key=lambda x: x[0])
    sentences = [sentence for _, sentence in sortable]

  if sentences:
    return collapse_whitespace(" ".join(sentences))

  for key in ("sentence", "transcript", "text"):
    value = data.get(key)
    if isinstance(value, str):
      value = collapse_whitespace(value)
      if value:
        return value
  return ""


def infer_speaker_id(json_path, data):
  speaker_id = data.get("speaker_id")
  if speaker_id is not None:
    return str(speaker_id)
  parts = os.path.normpath(json_path).split(os.sep)
  for part in reversed(parts):
    if part.isdigit() and len(part) >= 5:
      return part
  return None


def list_files(root, suffix):
  matches = []
  for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
      if filename.lower().endswith(suffix):
        matches.append(os.path.join(dirpath, filename))
  matches.sort()
  return matches


def index_audio_files(dataset_root):
  audio_files = list_files(dataset_root, ".flac")
  by_stem = {}
  by_name = {}
  for path in audio_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    name = os.path.basename(path)
    by_stem.setdefault(stem, []).append(path)
    by_name.setdefault(name, []).append(path)
  return audio_files, by_stem, by_name


def unique_lookup(mapping, key):
  values = mapping.get(key)
  if not values or len(values) != 1:
    return None
  return values[0]


def find_audio_path(dataset_root, json_path, data, audio_by_stem, audio_by_name):
  json_dir = os.path.dirname(json_path)
  json_stem = os.path.splitext(os.path.basename(json_path))[0]

  json_audio = data.get("path")
  if isinstance(json_audio, str) and json_audio.strip():
    rel = norm_path(json_audio.strip())
    candidates = [
        os.path.join(dataset_root, rel),
        os.path.join(json_dir, os.path.basename(rel)),
    ]
    for candidate in candidates:
      if os.path.isfile(candidate):
        return candidate
    found_by_name = unique_lookup(audio_by_name, os.path.basename(rel))
    if found_by_name:
      return found_by_name

  same_stem = os.path.join(json_dir, json_stem + ".flac")
  if os.path.isfile(same_stem):
    return same_stem

  speech_id = data.get("speech_id")
  if isinstance(speech_id, str) and speech_id.strip():
    found_by_stem = unique_lookup(audio_by_stem, speech_id.strip())
    if found_by_stem:
      return found_by_stem

  return unique_lookup(audio_by_stem, json_stem)


def relative_under(path, root):
  try:
    rel = os.path.relpath(path, root)
  except ValueError:
    rel = os.path.basename(path)
  if rel.startswith(".." + os.sep) or rel == "..":
    rel = os.path.basename(path)
  return rel


def prepared_wav_path(audio_path, dataset_root, output_root):
  rel = relative_under(audio_path, dataset_root)
  rel_no_ext = os.path.splitext(rel)[0] + ".wav"
  return os.path.join(output_root, "wavs", rel_no_ext)


def run_ffmpeg(input_path, output_path, sample_rate, overwrite):
  if os.path.exists(output_path) and not overwrite:
    return True, "exists"

  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    try:
      import imageio_ffmpeg
      ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
      return False, "ffmpeg not found; run `pip install -r requirements.txt` inside the active env. Detail: {}".format(exc)

  output_dir = os.path.dirname(output_path)
  if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

  command = [
      ffmpeg,
      "-hide_banner",
      "-loglevel", "error",
      "-y" if overwrite else "-n",
      "-i", input_path,
      "-ac", "1",
      "-ar", str(sample_rate),
      "-sample_fmt", "s16",
      output_path,
  ]
  process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  _, stderr = process.communicate()
  if process.returncode != 0:
    message = stderr.decode("utf-8", errors="replace").strip()
    return False, message or "ffmpeg failed"
  return True, "converted"


def should_keep_duration(data, min_duration, max_duration):
  duration = get_float(data.get("duration"))
  if duration is None:
    return True, ""
  if min_duration and duration < min_duration:
    return False, "duration_below_min"
  if max_duration and duration > max_duration:
    return False, "duration_above_max"
  return True, ""


def should_keep_text(text, min_chars, max_chars):
  text_len = len(text)
  if text_len < min_chars:
    return False, "text_too_short"
  if max_chars and text_len > max_chars:
    return False, "text_too_long"
  return True, ""


def split_records(records, val_ratio, test_ratio, seed):
  records = list(records)
  rng = random.Random(seed)
  rng.shuffle(records)
  total = len(records)
  if total == 0:
    return [], [], []

  n_test = int(round(total * test_ratio))
  n_val = int(round(total * val_ratio))
  if test_ratio > 0 and total >= 3:
    n_test = max(1, n_test)
  if val_ratio > 0 and total >= 3:
    n_val = max(1, n_val)
  if n_test + n_val >= total:
    overflow = n_test + n_val - total + 1
    reduce_val = min(n_val, overflow)
    n_val -= reduce_val
    overflow -= reduce_val
    if overflow:
      n_test = max(0, n_test - overflow)

  test = records[:n_test]
  val = records[n_test:n_test + n_val]
  train = records[n_test + n_val:]
  return train, val, test


def write_filelist(path, records):
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, "w", encoding="utf-8") as f:
    for record in records:
      f.write("{}|{}\n".format(record["wav_path"], record["text"]))


def write_skips(path, skipped):
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, "w", encoding="utf-8") as f:
    f.write("reason\tjson_path\tdetail\n")
    for item in skipped:
      f.write("{}\t{}\t{}\n".format(
          item.get("reason", ""),
          item.get("json_path", ""),
          item.get("detail", "").replace("\t", " ")))


def main():
  args = parse_args()
  dataset_root = os.path.abspath(args.dataset_root)
  output_root = os.path.abspath(args.output_root)
  filelists_dir = os.path.abspath(args.filelists_dir)

  if not os.path.isdir(dataset_root):
    raise RuntimeError("Dataset root does not exist: {}".format(dataset_root))

  json_files = list_files(dataset_root, ".json")
  audio_files, audio_by_stem, audio_by_name = index_audio_files(dataset_root)

  records = []
  skipped = []
  conversion_status = {}

  for json_path in json_files:
    try:
      data = read_json(json_path)
    except Exception as exc:
      skipped.append({"reason": "malformed_json", "json_path": json_path, "detail": str(exc)})
      continue

    if not isinstance(data, dict):
      skipped.append({"reason": "json_not_object", "json_path": json_path, "detail": ""})
      continue

    speaker_id = infer_speaker_id(json_path, data)
    if args.speaker_id and speaker_id != args.speaker_id:
      skipped.append({"reason": "speaker_mismatch", "json_path": json_path,
                      "detail": "found {}".format(speaker_id)})
      continue

    keep, reason = should_keep_duration(data, args.min_duration, args.max_duration)
    if not keep:
      skipped.append({"reason": reason, "json_path": json_path,
                      "detail": "duration={}".format(data.get("duration"))})
      continue

    text = extract_sentence(data)
    keep, reason = should_keep_text(text, args.min_text_chars, args.max_text_chars)
    if not keep:
      skipped.append({"reason": reason, "json_path": json_path,
                      "detail": "chars={}".format(len(text))})
      continue

    audio_path = find_audio_path(dataset_root, json_path, data, audio_by_stem, audio_by_name)
    if not audio_path:
      skipped.append({"reason": "missing_audio", "json_path": json_path, "detail": ""})
      continue

    wav_path = prepared_wav_path(audio_path, dataset_root, output_root)
    if not args.dry_run:
      ok, status = run_ffmpeg(audio_path, wav_path, args.sample_rate, args.overwrite)
      conversion_status[status] = conversion_status.get(status, 0) + 1
      if not ok:
        skipped.append({"reason": "audio_conversion_failed", "json_path": json_path,
                        "detail": "{}: {}".format(audio_path, status)})
        continue

    records.append({
        "wav_path": wav_path,
        "text": text,
        "json_path": json_path,
        "audio_path": audio_path,
        "speaker_id": speaker_id,
        "duration": data.get("duration"),
    })

  train, val, test = split_records(records, args.val_ratio, args.test_ratio, args.seed)

  filelists = {
      "train": os.path.join(filelists_dir, "{}_train_filelist.txt".format(args.prefix)),
      "val": os.path.join(filelists_dir, "{}_val_filelist.txt".format(args.prefix)),
      "test": os.path.join(filelists_dir, "{}_test_filelist.txt".format(args.prefix)),
  }

  if not args.dry_run:
    write_filelist(filelists["train"], train)
    write_filelist(filelists["val"], val)
    write_filelist(filelists["test"], test)
    report_dir = output_root
    if not os.path.exists(report_dir):
      os.makedirs(report_dir)
    skip_reasons = {}
    for item in skipped:
      reason = item.get("reason", "")
      skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    used_audio = set([record["audio_path"] for record in records])
    write_skips(os.path.join(report_dir, "bengali_prep_skipped.tsv"), skipped)
    report = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "filelists": filelists,
        "speaker_id": args.speaker_id,
        "sample_rate": args.sample_rate,
        "json_files_found": len(json_files),
        "audio_files_found": len(audio_files),
        "records_kept": len(records),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "skipped_count": len(skipped),
        "skip_reasons": skip_reasons,
        "unused_audio_count": len([path for path in audio_files if path not in used_audio]),
        "conversion_status": conversion_status,
    }
    with open(os.path.join(report_dir, "bengali_prep_report.json"), "w", encoding="utf-8") as f:
      json.dump(report, f, ensure_ascii=False, indent=2)

  print("JSON files found: {}".format(len(json_files)))
  print("FLAC files found: {}".format(len(audio_files)))
  print("Records kept: {}".format(len(records)))
  print("Train/val/test: {}/{}/{}".format(len(train), len(val), len(test)))
  print("Skipped: {}".format(len(skipped)))
  if args.dry_run:
    print("Dry run only; no audio or filelists were written.")
  else:
    print("Wrote filelists:")
    print("  train: {}".format(filelists["train"]))
    print("  val:   {}".format(filelists["val"]))
    print("  test:  {}".format(filelists["test"]))
    print("Report: {}".format(os.path.join(output_root, "bengali_prep_report.json")))


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print("ERROR: {}".format(exc), file=sys.stderr)
    sys.exit(1)

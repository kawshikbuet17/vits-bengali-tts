import argparse
import json
import os
import random
import sys

from prepare_bengali_dataset import (
  extract_sentence,
  find_audio_path,
  get_float,
  index_audio_files,
  infer_speaker_id,
  list_files,
  prepared_wav_path,
  read_json,
  run_ffmpeg,
  should_keep_duration,
  should_keep_text,
  write_skips,
)


def parse_args():
  parser = argparse.ArgumentParser(
      description="Prepare raw Bengali .flac/.json data for multi-speaker VITS training.")
  parser.add_argument("--dataset-root", required=True,
                      help="Root of the raw dataset, e.g. /home/kawshik/TTS_Dataset")
  parser.add_argument("--output-root", required=True,
                      help="Directory where prepared wavs, speaker map, and prep reports are written")
  parser.add_argument("--filelists-dir", default="filelists",
                      help="Directory where VITS filelists are written")
  parser.add_argument("--prefix", default="bengali_ms_audio_sid_text",
                      help="Prefix for generated train/val/test filelists")
  parser.add_argument("--speaker-ids", default=None,
                      help="Optional comma-separated raw speaker IDs to include")
  parser.add_argument("--speaker-ids-file", default=None,
                      help="Optional UTF-8 file containing one raw speaker ID per line")
  parser.add_argument("--sample-rate", type=int, default=22050,
                      help="Target sample rate for prepared wavs")
  parser.add_argument("--val-ratio", type=float, default=0.02,
                      help="Validation split ratio, applied per speaker when possible")
  parser.add_argument("--test-ratio", type=float, default=0.02,
                      help="Test split ratio, applied per speaker when possible")
  parser.add_argument("--seed", type=int, default=1234,
                      help="Random seed for splits and optional per-speaker caps")
  parser.add_argument("--min-duration", type=float, default=0.0,
                      help="Skip records whose JSON duration is below this value when present")
  parser.add_argument("--max-duration", type=float, default=0.0,
                      help="Skip records whose JSON duration is above this value when present; 0 disables")
  parser.add_argument("--min-text-chars", type=int, default=1,
                      help="Skip records with shorter extracted transcript text")
  parser.add_argument("--max-text-chars", type=int, default=250,
                      help="Skip records with longer extracted transcript text; 0 disables")
  parser.add_argument("--min-utterances-per-speaker", type=int, default=1,
                      help="Drop speakers with fewer kept utterances")
  parser.add_argument("--min-total-duration-per-speaker", type=float, default=0.0,
                      help="Drop speakers with less kept JSON duration in seconds; 0 disables")
  parser.add_argument("--max-utterances-per-speaker", type=int, default=0,
                      help="Cap utterances per speaker after filtering; 0 disables")
  parser.add_argument("--max-speakers", type=int, default=0,
                      help="Keep only the top N speakers by utterance count; 0 disables")
  parser.add_argument("--progress-interval", type=int, default=100,
                      help="Print audio conversion progress every N records; 0 disables")
  parser.add_argument("--speaker-map-out", default=None,
                      help="Path for speaker_map.json; default: <output-root>/speaker_map.json")
  parser.add_argument("--speaker-stats-out", default=None,
                      help="Path for speaker_stats.tsv; default: <output-root>/speaker_stats.tsv")
  parser.add_argument("--config-template", default="configs/bengali_ms.json",
                      help="Optional config template used to write --config-out")
  parser.add_argument("--config-out", default="configs/bengali_ms.generated.json",
                      help="Generated config with n_speakers and filelist paths updated; empty disables")
  parser.add_argument("--overwrite", action="store_true",
                      help="Recreate prepared wav files even if they already exist")
  parser.add_argument("--dry-run", action="store_true",
                      help="Scan and report without converting audio or writing filelists/config")
  return parser.parse_args()


def read_speaker_filter(args):
  speakers = set()
  if args.speaker_ids:
    for item in args.speaker_ids.split(","):
      item = item.strip()
      if item:
        speakers.add(item)
  if args.speaker_ids_file:
    with open(args.speaker_ids_file, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
          speakers.add(line)
  return speakers


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


def group_by_speaker(records):
  grouped = {}
  for record in records:
    grouped.setdefault(record["speaker_id"], []).append(record)
  return grouped


def duration_sum(records):
  total = 0.0
  for record in records:
    duration = get_float(record.get("duration"))
    if duration is not None:
      total += duration
  return total


def filter_speakers(grouped, args):
  kept = {}
  dropped = {}
  for speaker_id, records in grouped.items():
    total_duration = duration_sum(records)
    if len(records) < args.min_utterances_per_speaker:
      dropped[speaker_id] = "utterances_below_min"
      continue
    if args.min_total_duration_per_speaker and total_duration < args.min_total_duration_per_speaker:
      dropped[speaker_id] = "duration_below_min"
      continue
    kept[speaker_id] = records

  if args.max_speakers and len(kept) > args.max_speakers:
    ranked = sorted(kept.items(), key=lambda item: (-len(item[1]), item[0]))
    keep_ids = set([speaker_id for speaker_id, _ in ranked[:args.max_speakers]])
    for speaker_id in list(kept.keys()):
      if speaker_id not in keep_ids:
        dropped[speaker_id] = "outside_max_speakers"
        del kept[speaker_id]

  if args.max_utterances_per_speaker:
    rng = random.Random(args.seed)
    for speaker_id, records in list(kept.items()):
      if len(records) > args.max_utterances_per_speaker:
        records = list(records)
        rng.shuffle(records)
        kept[speaker_id] = records[:args.max_utterances_per_speaker]

  return kept, dropped


def make_speaker_map(grouped):
  return {
      speaker_id: index
      for index, speaker_id in enumerate(sorted(grouped.keys()))
  }


def write_ms_filelist(path, records):
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, "w", encoding="utf-8") as f:
    for record in records:
      f.write("{}|{}|{}\n".format(record["wav_path"], record["speaker_index"], record["text"]))


def write_speaker_stats(path, grouped, speaker_map):
  directory = os.path.dirname(path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, "w", encoding="utf-8") as f:
    f.write("speaker_id\tspeaker_index\tutterances\ttotal_duration_seconds\n")
    for speaker_id in sorted(grouped.keys()):
      records = grouped[speaker_id]
      f.write("{}\t{}\t{}\t{:.3f}\n".format(
          speaker_id, speaker_map[speaker_id], len(records), duration_sum(records)))


def write_generated_config(template_path, output_path, filelists, n_speakers):
  if not output_path:
    return None
  with open(template_path, "r", encoding="utf-8") as f:
    config = json.load(f)
  config["data"]["training_files"] = filelists["train"] + ".cleaned"
  config["data"]["validation_files"] = filelists["val"] + ".cleaned"
  config["data"]["test_files"] = filelists["test"] + ".cleaned"
  config["data"]["n_speakers"] = n_speakers
  directory = os.path.dirname(output_path)
  if directory and not os.path.exists(directory):
    os.makedirs(directory)
  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
  return output_path


def main():
  args = parse_args()
  dataset_root = os.path.abspath(args.dataset_root)
  output_root = os.path.abspath(args.output_root)
  filelists_dir = os.path.abspath(args.filelists_dir)
  speaker_filter = read_speaker_filter(args)

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
    if not speaker_id:
      skipped.append({"reason": "missing_speaker_id", "json_path": json_path, "detail": ""})
      continue
    if speaker_filter and speaker_id not in speaker_filter:
      skipped.append({"reason": "speaker_not_selected", "json_path": json_path,
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
    records.append({
        "wav_path": wav_path,
        "text": text,
        "json_path": json_path,
        "audio_path": audio_path,
        "speaker_id": speaker_id,
        "duration": data.get("duration"),
    })

  grouped_initial = group_by_speaker(records)
  grouped, dropped_speakers = filter_speakers(grouped_initial, args)
  speaker_map = make_speaker_map(grouped)

  filtered_records = []
  for speaker_id in sorted(grouped.keys()):
    for record in grouped[speaker_id]:
      item = dict(record)
      item["speaker_index"] = speaker_map[speaker_id]
      filtered_records.append(item)

  dropped_ids = set(dropped_speakers.keys())
  skipped_for_speaker_filter = [
      {"reason": "speaker_filtered_after_stats", "json_path": record["json_path"],
       "detail": "{}: {}".format(record["speaker_id"], dropped_speakers[record["speaker_id"]])}
      for record in records
      if record["speaker_id"] in dropped_ids
  ]
  skipped.extend(skipped_for_speaker_filter)

  converted_records = []
  total_to_convert = len(filtered_records)
  for idx, record in enumerate(filtered_records, 1):
    if args.progress_interval and (idx == 1 or idx % args.progress_interval == 0 or idx == total_to_convert):
      print("Converting audio {}/{} for speaker {}".format(
          idx, total_to_convert, record["speaker_id"]))
    if not args.dry_run:
      ok, status = run_ffmpeg(record["audio_path"], record["wav_path"], args.sample_rate, args.overwrite)
      conversion_status[status] = conversion_status.get(status, 0) + 1
      if not ok:
        skipped.append({"reason": "audio_conversion_failed", "json_path": record["json_path"],
                        "detail": "{}: {}".format(record["audio_path"], status)})
        continue
    converted_records.append(record)

  grouped = group_by_speaker(converted_records)
  speaker_map = make_speaker_map(grouped)
  if not speaker_map:
    raise RuntimeError("No multi-speaker records were kept. Check speaker filters, duration/text filters, and audio conversion.")
  for record in converted_records:
    record["speaker_index"] = speaker_map[record["speaker_id"]]

  train = []
  val = []
  test = []
  for speaker_id in sorted(grouped.keys()):
    spk_train, spk_val, spk_test = split_records(
        [dict(record, speaker_index=speaker_map[speaker_id]) for record in grouped[speaker_id]],
        args.val_ratio,
        args.test_ratio,
        args.seed)
    train.extend(spk_train)
    val.extend(spk_val)
    test.extend(spk_test)

  rng = random.Random(args.seed)
  rng.shuffle(train)
  rng.shuffle(val)
  rng.shuffle(test)

  filelists = {
      "train": os.path.join(filelists_dir, "{}_train_filelist.txt".format(args.prefix)),
      "val": os.path.join(filelists_dir, "{}_val_filelist.txt".format(args.prefix)),
      "test": os.path.join(filelists_dir, "{}_test_filelist.txt".format(args.prefix)),
  }
  speaker_map_out = args.speaker_map_out or os.path.join(output_root, "speaker_map.json")
  speaker_stats_out = args.speaker_stats_out or os.path.join(output_root, "speaker_stats.tsv")

  generated_config = None
  if not args.dry_run:
    write_ms_filelist(filelists["train"], train)
    write_ms_filelist(filelists["val"], val)
    write_ms_filelist(filelists["test"], test)
    if not os.path.exists(output_root):
      os.makedirs(output_root)
    with open(speaker_map_out, "w", encoding="utf-8") as f:
      json.dump(speaker_map, f, ensure_ascii=False, indent=2)
      f.write("\n")
    write_speaker_stats(speaker_stats_out, grouped, speaker_map)
    write_skips(os.path.join(output_root, "bengali_ms_prep_skipped.tsv"), skipped)
    if args.config_out:
      generated_config = write_generated_config(args.config_template, args.config_out, filelists, len(speaker_map))

    skip_reasons = {}
    for item in skipped:
      reason = item.get("reason", "")
      skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    used_audio = set([record["audio_path"] for record in converted_records])
    report = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "filelists": filelists,
        "speaker_map": speaker_map_out,
        "speaker_stats": speaker_stats_out,
        "config_out": generated_config,
        "sample_rate": args.sample_rate,
        "json_files_found": len(json_files),
        "audio_files_found": len(audio_files),
        "records_before_speaker_stats_filter": len(records),
        "records_kept": len(converted_records),
        "speakers_kept": len(speaker_map),
        "speakers_dropped": dropped_speakers,
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "skipped_count": len(skipped),
        "skip_reasons": skip_reasons,
        "unused_audio_count": len([path for path in audio_files if path not in used_audio]),
        "conversion_status": conversion_status,
    }
    with open(os.path.join(output_root, "bengali_ms_prep_report.json"), "w", encoding="utf-8") as f:
      json.dump(report, f, ensure_ascii=False, indent=2)

  print("JSON files found: {}".format(len(json_files)))
  print("FLAC files found: {}".format(len(audio_files)))
  print("Speakers kept: {}".format(len(speaker_map)))
  if len(speaker_map) == 1:
    print("WARNING: only one speaker was kept; this will run but is not a true multi-speaker dataset.")
  print("Records kept: {}".format(len(converted_records)))
  print("Train/val/test: {}/{}/{}".format(len(train), len(val), len(test)))
  print("Skipped: {}".format(len(skipped)))
  if args.dry_run:
    print("Dry run only; no audio, filelists, speaker map, or config were written.")
  else:
    print("Wrote filelists:")
    print("  train: {}".format(filelists["train"]))
    print("  val:   {}".format(filelists["val"]))
    print("  test:  {}".format(filelists["test"]))
    print("Speaker map: {}".format(speaker_map_out))
    print("Speaker stats: {}".format(speaker_stats_out))
    if generated_config:
      print("Generated config: {}".format(generated_config))
    print("Report: {}".format(os.path.join(output_root, "bengali_ms_prep_report.json")))


if __name__ == "__main__":
  try:
    main()
  except Exception as exc:
    print("ERROR: {}".format(exc), file=sys.stderr)
    sys.exit(1)

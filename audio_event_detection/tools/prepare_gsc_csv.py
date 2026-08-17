#!/usr/bin/env python3
"""Prepare Google Speech Commands CSV files for ST Model Zoo Services."""

from __future__ import annotations

import argparse
import array
import csv
import random
import shutil
import sys
import wave
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "This script requires PyYAML. Run it in the same Python environment "
        "used for ST Model Zoo training."
    ) from exc

try:
    import librosa
except ImportError:
    librosa = None


__version__ = "2.6.0"
CSV_FIELDS = ("filename", "category")
SPECIAL_CLASSES = ("unknown", "silence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create balanced ST CSV splits from Google Speech Commands."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="ST Model Zoo YAML configuration containing dataset.class_names.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Speech Commands root (default: script directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="CSV/report output directory (default: dataset root).",
    )
    parser.add_argument(
        "--quantization-samples-per-class",
        type=int,
        default=100,
        help="Training samples per class used in quantization.csv.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Parallel workers used for Librosa validation (default: 8).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser.parse_args()


def load_config(config_path: Path) -> tuple[list[str], dict[str, Any], int]:
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    try:
        configured_classes = list(config["dataset"]["class_names"])
        preprocessing = config["preprocessing"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "YAML must contain dataset.class_names and preprocessing sections."
        ) from exc

    if not configured_classes or not all(
        isinstance(name, str) and name.strip() for name in configured_classes
    ):
        raise ValueError("dataset.class_names must be a non-empty list of strings.")

    effective_classes = list(dict.fromkeys(name.strip() for name in configured_classes))
    for special_class in SPECIAL_CLASSES:
        if special_class not in effective_classes:
            effective_classes.append(special_class)

    gsc_default = bool(preprocessing.get("gsc_default", False))
    settings: dict[str, Any] = {
        "gsc_default": gsc_default,
        "target_rate": int(preprocessing.get("target_rate", 16000)),
        "reserved_background_files": list(
            preprocessing.get("reserved_background_files", [])
        ),
    }
    if not gsc_default:
        settings.update(
            {
                "top_db": int(preprocessing["top_db"]),
                "frame_length": int(preprocessing["frame_length"]),
                "hop_length": int(preprocessing["hop_length"]),
            }
        )
    elif not settings["reserved_background_files"]:
        raise ValueError(
            "GSC preprocessing must reserve at least one background WAV using "
            "preprocessing.reserved_background_files."
        )
    seed = int(config.get("dataset", {}).get("seed", 120))
    return effective_classes, settings, seed


def read_split_list(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split list: {path}")
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def discover_clips(dataset_root: Path) -> dict[str, str]:
    clips: dict[str, str] = {}
    for wav_path in sorted(dataset_root.glob("*/*.wav")):
        category = wav_path.parent.name
        if category.startswith("_"):
            continue
        clips[wav_path.relative_to(dataset_root).as_posix()] = category
    return clips


def survives_st_silence_removal(
    path: Path, preprocessing: dict[str, Any]
) -> tuple[bool, str]:
    # GSC preprocessing preserves silence and fixes length with zero padding.
    # Validate its expected PCM input without requiring Librosa in this utility.
    if preprocessing.get("gsc_default", False):
        try:
            with wave.open(str(path), "rb") as wav_file:
                if wav_file.getnframes() == 0:
                    return False, "WAV frame count is zero"
                if wav_file.getnchannels() != 1:
                    return False, "GSC WAV is not mono"
                if wav_file.getframerate() != preprocessing["target_rate"]:
                    return False, "GSC WAV has an unexpected sample rate"
        except Exception as exc:
            return False, f"decode error: {exc}"
        return True, ""

    if librosa is None:
        raise RuntimeError(
            "Librosa is required when preparing data for ST silence-removal preprocessing"
        )

    try:
        waveform, _ = librosa.load(path, sr=preprocessing["target_rate"])
    except Exception as exc:  # Report corrupt or unsupported audio as eliminated.
        return False, f"decode error: {exc}"

    if len(waveform) == 0:
        return False, "decoded waveform length is zero"

    try:
        intervals = librosa.effects.split(
            waveform,
            top_db=preprocessing["top_db"],
            frame_length=preprocessing["frame_length"],
            hop_length=preprocessing["hop_length"],
        )
    except Exception as exc:
        return False, f"silence-removal error: {exc}"

    remaining_length = sum(max(0, int(end) - int(start)) for start, end in intervals)
    if remaining_length == 0:
        return False, "LibrosaSilenceRemovalPipeline would produce zero-length wave"
    return True, ""


def validation_task(
    task: tuple[str, dict[str, Any]]
) -> tuple[bool, str]:
    path, preprocessing = task
    return survives_st_silence_removal(Path(path), preprocessing)


def validate_rows(
    rows: list[tuple[str, str]],
    dataset_root: Path,
    preprocessing: dict[str, Any],
    eliminated: list[dict[str, str]],
    executor: ProcessPoolExecutor,
) -> list[tuple[str, str]]:
    valid: list[tuple[str, str]] = []
    tasks = [(str(dataset_root / filename), preprocessing) for filename, _ in rows]
    results = executor.map(validation_task, tasks, chunksize=32)
    for (filename, category), (accepted, reason) in zip(rows, results):
        path = dataset_root / filename
        if accepted:
            valid.append((filename, category))
        else:
            eliminated.append(
                {
                    "category": category,
                    "filename": filename,
                    "path": str(path.resolve()),
                    "reason": reason,
                }
            )
    return valid


def select_unknown_rows(
    candidates: list[tuple[str, str]],
    desired_count: int,
    dataset_root: Path,
    preprocessing: dict[str, Any],
    eliminated: list[dict[str, str]],
    rng: random.Random,
    executor: ProcessPoolExecutor,
) -> list[tuple[str, str]]:
    by_source_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in candidates:
        by_source_class[row[1]].append(row)
    for rows in by_source_class.values():
        rng.shuffle(rows)

    # Give every non-target source class the same quota. Randomly assign the
    # unavoidable remainder, so no alphabetically early classes are favoured.
    source_classes = sorted(by_source_class)
    if not source_classes:
        raise RuntimeError("No non-target classes are available for unknown samples.")
    base_quota, remainder = divmod(desired_count, len(source_classes))
    remainder_classes = set(rng.sample(source_classes, remainder))
    quotas = {
        source_class: base_quota + (source_class in remainder_classes)
        for source_class in source_classes
    }

    selected: list[tuple[str, str]] = []
    for source_class in source_classes:
        quota = quotas[source_class]
        accepted_for_class = 0
        rows = by_source_class[source_class]
        batch_size = max(32, min(256, quota * 2))
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            tasks = [(str(dataset_root / filename), preprocessing) for filename, _ in batch]
            results = executor.map(validation_task, tasks, chunksize=32)
            for (filename, _), (accepted, reason) in zip(batch, results):
                if accepted:
                    selected.append((filename, "unknown"))
                    accepted_for_class += 1
                else:
                    path = dataset_root / filename
                    eliminated.append(
                        {
                            "category": "unknown",
                            "filename": filename,
                            "path": str(path.resolve()),
                            "reason": reason,
                        }
                    )
                if accepted_for_class == quota:
                    break
            if accepted_for_class == quota:
                break
        if accepted_for_class < quota:
            raise RuntimeError(
                f"Only {accepted_for_class} valid clips were available for unknown "
                f"source class {source_class!r}; its balanced quota is {quota}."
            )

    if len(selected) < desired_count:
        raise RuntimeError(
            f"Only {len(selected)} valid unknown clips were available; "
            f"{desired_count} were requested."
        )
    return sorted(selected)


def read_pcm_wav(path: Path, required_rate: int) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(path), "rb") as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
    if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != required_rate:
        raise ValueError(
            f"Background WAV must be mono 16-bit PCM at {required_rate} Hz: {path}"
        )
    return params, frames


def scale_pcm16(frames: bytes, gain: float) -> bytes:
    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    for index, sample in enumerate(samples):
        samples[index] = round(sample * gain)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


def generate_background_rows(
    dataset_root: Path,
    split_name: str,
    source_paths: list[Path],
    desired_count: int,
    category: str,
    gain_range: tuple[float, float],
    preprocessing: dict[str, Any],
    eliminated: list[dict[str, str]],
    rng: random.Random,
) -> list[tuple[str, str]]:
    output_folder = dataset_root / "_st_generated" / category / split_name
    output_folder.mkdir(parents=True, exist_ok=True)
    sample_rate = preprocessing["target_rate"]
    bytes_per_sample = 2
    window_bytes = sample_rate * bytes_per_sample
    sources: list[tuple[Path, bytes]] = []
    for source_path in source_paths:
        _, frames = read_pcm_wav(source_path, sample_rate)
        if len(frames) >= window_bytes:
            sources.append((source_path, frames))
    if not sources:
        raise RuntimeError(f"No usable background-noise source for {split_name}.")

    # Allocate an equal quota to each background WAV. The randomly assigned
    # remainder prevents a persistent filename-order bias.
    base_quota, remainder = divmod(desired_count, len(sources))
    remainder_sources = set(rng.sample(range(len(sources)), remainder))
    rows: list[tuple[str, str]] = []
    for source_index, (source_path, frames) in enumerate(sources):
        quota = base_quota + (source_index in remainder_sources)
        accepted_for_source = 0
        attempts = 0
        max_attempts = max(20, quota * 10)
        while accepted_for_source < quota and attempts < max_attempts:
            max_start_sample = len(frames) // bytes_per_sample - sample_rate
            start_sample = rng.randint(0, max_start_sample) if max_start_sample else 0
            start = start_sample * bytes_per_sample
            clip = frames[start : start + window_bytes]
            gain = rng.uniform(*gain_range)
            clip = scale_pcm16(clip, gain)
            filename = (
                f"{category}_{len(rows):05d}_{source_path.stem}_"
                f"{start_sample:08d}.wav"
            )
            output_path = output_folder / filename
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(bytes_per_sample)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(clip)

            relative_name = output_path.relative_to(dataset_root).as_posix()
            accepted, reason = survives_st_silence_removal(output_path, preprocessing)
            if accepted:
                rows.append((relative_name, category))
                accepted_for_source += 1
            else:
                eliminated.append(
                    {
                        "category": category,
                        "filename": relative_name,
                        "path": str(output_path.resolve()),
                        "reason": reason,
                    }
                )
                output_path.unlink(missing_ok=True)
            attempts += 1
        if accepted_for_source < quota:
            raise RuntimeError(
                f"Only {accepted_for_source} valid {category} clips could be generated "
                f"from {source_path}; its balanced quota is {quota}."
            )

    if len(rows) < desired_count:
        raise RuntimeError(
            f"Only {len(rows)} valid {category} clips could be generated for "
            f"{split_name}; {desired_count} were requested."
        )
    return sorted(rows)


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        writer.writerows(sorted(rows))


def balanced_quantization_rows(
    training_rows: list[tuple[str, str]], samples_per_class: int, rng: random.Random
) -> list[tuple[str, str]]:
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in training_rows:
        by_class[row[1]].append(row)
    selected: list[tuple[str, str]] = []
    for category in sorted(by_class):
        candidates = by_class[category][:]
        rng.shuffle(candidates)
        if len(candidates) < samples_per_class:
            raise RuntimeError(
                f"Class {category!r} has only {len(candidates)} training samples; "
                f"quantization requires {samples_per_class}."
            )
        selected.extend(candidates[:samples_per_class])
    return sorted(selected)


def write_report(
    path: Path,
    config_path: Path,
    dataset_root: Path,
    effective_classes: list[str],
    preprocessing: dict[str, Any],
    outputs: dict[str, list[tuple[str, str]]],
    eliminated: list[dict[str, str]],
) -> None:
    lines = [
        "ST Speech Commands CSV preparation report",
        "=========================================",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Script version: {__version__}",
        f"Python version: {sys.version.split()[0]}",
        f"YAML config: {config_path}",
        f"Dataset root: {dataset_root}",
        f"Effective class_names: {effective_classes}",
        f"Preprocessing: {preprocessing}",
        "",
    ]
    for filename, rows in outputs.items():
        counts = Counter(category for _, category in rows)
        lines.append(f"{filename}: {len(rows)} samples")
        for category in effective_classes:
            lines.append(f"  {category:<16} {counts.get(category, 0):>6}")
        lines.append("")

    lines.extend(
        [
            f"Eliminated files: {len(eliminated)}",
            "Files below were rejected because decoding or the configured "
            "LibrosaSilenceRemovalPipeline check produced no usable samples.",
        ]
    )
    if eliminated:
        for item in eliminated:
            lines.append(
                f"  class={item['category']} filename={item['filename']} "
                f"path={item['path']} reason={item['reason']}"
            )
    else:
        lines.append("  None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_counts(name: str, rows: list[tuple[str, str]]) -> None:
    counts = Counter(category for _, category in rows)
    print(f"\n{name}: {len(rows)} samples, {len(counts)} classes")
    for category in sorted(counts):
        print(f"  {category:<16} {counts[category]:>6}")


def main() -> int:
    args = parse_args()
    if args.quantization_samples_per_class < 1:
        raise ValueError("--quantization-samples-per-class must be at least 1")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")

    dataset_root = args.dataset_root.resolve()
    output_dir = (args.output_dir or dataset_root).resolve()
    config_path = args.config.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_classes, preprocessing, config_seed = load_config(config_path)
    seed = args.seed if args.seed is not None else config_seed
    core_classes = [name for name in effective_classes if name not in SPECIAL_CLASSES]
    rng = random.Random(seed)

    validation_names = read_split_list(dataset_root / "validation_list.txt")
    testing_names = read_split_list(dataset_root / "testing_list.txt")
    if validation_names & testing_names:
        raise ValueError("validation_list.txt and testing_list.txt overlap")

    clips = discover_clips(dataset_root)
    missing = (validation_names | testing_names) - clips.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed WAV files are missing")

    split_names = {
        "training": set(clips) - validation_names - testing_names,
        "validation": validation_names,
        "testing": testing_names,
    }
    eliminated: list[dict[str, str]] = []
    prepared: dict[str, list[tuple[str, str]]] = {}

    generated_root = (dataset_root / "_st_generated").resolve()
    if generated_root.exists():
        if generated_root.parent != dataset_root:
            raise RuntimeError(f"Unsafe generated-data path: {generated_root}")
        shutil.rmtree(generated_root)

    background_sources = sorted((dataset_root / "_background_noise_").glob("*.wav"))
    if not background_sources:
        raise RuntimeError("No WAV files found in _background_noise_")

    reserved_names = {
        Path(name).name for name in preprocessing.get("reserved_background_files", [])
    }
    available_background_names = {path.name for path in background_sources}
    missing_reserved = reserved_names - available_background_names
    if missing_reserved:
        raise FileNotFoundError(
            f"Reserved background WAV files are missing: {sorted(missing_reserved)}"
        )
    if preprocessing.get("gsc_default", False):
        reserved_background_sources = [
            path for path in background_sources if path.name in reserved_names
        ]
        training_background_sources = [
            path for path in background_sources if path.name not in reserved_names
        ]
        if not reserved_background_sources:
            raise RuntimeError("At least one background WAV must be reserved")
        if not training_background_sources:
            raise RuntimeError("At least one background WAV must remain for training")
        print(
            "Training-only background WAVs: "
            f"{[path.name for path in training_background_sources]}"
        )
        print(
            "Validation/test-only background WAVs: "
            f"{[path.name for path in reserved_background_sources]}"
        )
    else:
        training_background_sources = background_sources
        reserved_background_sources = background_sources

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for split_name, filenames in split_names.items():
            target_candidates = sorted(
                (filename, clips[filename])
                for filename in filenames
                if clips[filename] in core_classes
            )
            target_rows = validate_rows(
                target_candidates, dataset_root, preprocessing, eliminated, executor
            )
            target_counts = Counter(category for _, category in target_rows)
            absent = set(core_classes) - set(target_counts)
            if absent:
                raise RuntimeError(f"Classes missing from {split_name}: {sorted(absent)}")
            balance_count = int(median(target_counts.values()))

            unknown_candidates = sorted(
                (filename, clips[filename])
                for filename in filenames
                if clips[filename] not in core_classes
            )
            # Unknown consists exclusively of excluded Speech Commands words.
            # select_unknown_rows assigns an even quota to every excluded
            # folder and randomly selects recordings within each folder.
            unknown_rows = select_unknown_rows(
                unknown_candidates,
                balance_count,
                dataset_root,
                preprocessing,
                eliminated,
                rng,
                executor,
            )
            # Strict GSC mode keeps background sources disjoint. Training and
            # quantization silence use training-only WAVs, while validation and
            # testing silence use only explicitly reserved WAVs.
            silence_sources = (
                training_background_sources
                if split_name == "training"
                else reserved_background_sources
            )
            silence_rows = generate_background_rows(
                dataset_root,
                split_name,
                silence_sources,
                balance_count,
                "silence",
                (1.0, 1.0),
                preprocessing,
                eliminated,
                rng,
            )
            prepared[split_name] = sorted(
                target_rows
                + unknown_rows
                + silence_rows
            )

    quantization_rows = balanced_quantization_rows(
        prepared["training"], args.quantization_samples_per_class, rng
    )
    outputs = {
        "training.csv": prepared["training"],
        "validation.csv": prepared["validation"],
        "testing.csv": prepared["testing"],
        "evaluation.csv": prepared["testing"],
        "quantization.csv": quantization_rows,
    }
    for filename, rows in outputs.items():
        write_csv(output_dir / filename, rows)

    report_path = output_dir / "prepare_st_csv_report.txt"
    write_report(
        report_path,
        config_path,
        dataset_root,
        effective_classes,
        preprocessing,
        outputs,
        eliminated,
    )

    print(f"prepare_st_csv.py version {__version__}")
    print(f"Effective classes: {effective_classes}")
    for filename, rows in outputs.items():
        print_counts(filename, rows)
    print(f"\nEliminated files: {len(eliminated)}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

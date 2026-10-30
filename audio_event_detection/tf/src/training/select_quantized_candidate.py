import csv
import json
import math
import shutil
from pathlib import Path

import mlflow
import tensorflow as tf
from hydra.core.hydra_config import HydraConfig

from audio_event_detection.tf.src.utils import AED_CUSTOM_OBJECTS
from audio_event_detection.tf.src.quantization.tflite_quantizer import (
    TFLitePTQQuantizer,
)
from audio_event_detection.tf.src.evaluation.tflite_evaluator import (
    AEDTFliteEvaluator,
)


def select_quantized_candidate(cfg, dataloaders):
    if (
        cfg.quantization.quantizer != "TFlite_converter"
        or cfg.quantization.quantization_type != "PTQ"
    ):
        raise ValueError("This selector supports TFLite PTQ only.")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = (
        run_dir / cfg.general.saved_models_dir / "candidates"
    )
    manifest = json.loads(
        (checkpoint_dir / "manifest.json").read_text(encoding="utf-8")
    )
    candidates = manifest["candidates"]
    if not candidates:
        raise ValueError("No candidate checkpoints found.")

    export_root = run_dir / cfg.quantization.export_dir
    export_root.mkdir(parents=True, exist_ok=True)

    # BaseAEDEvaluator normally prefers test_ds. Override that for selection.
    selection_loaders = dict(dataloaders)
    selection_loaders["test_ds"] = None
    selection_loaders["test_clip_labels"] = None

    results = []

    for candidate in candidates:
        candidate_id = f"epoch_{candidate['epoch']:04d}"
        candidate_dir = export_root / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)

        wrapper = tf.keras.models.load_model(
            checkpoint_dir / candidate["checkpoint"],
            custom_objects=AED_CUSTOM_OBJECTS,
            compile=False,
        )

        # Same extraction used by AEDTrainer._save_checkpoints().
        float_model = wrapper.layers[-1]

        quantizer = TFLitePTQQuantizer(
            cfg=cfg,
            model=float_model,
            dataloaders=selection_loaders,
        )

        # Prevent candidates from overwriting each other's TFLite files.
        quantizer.export_dir = str(candidate_dir.relative_to(run_dir))
        quantized = quantizer.quantize()

        evaluator = AEDTFliteEvaluator(
            cfg=cfg,
            model=quantized,
            dataloaders=selection_loaders,
        )

        # Keep confusion matrices and logs separate for each candidate.
        evaluator.output_dir = str(candidate_dir)

        # Avoid mixing candidate metrics in the parent MLflow run.
        with mlflow.start_run(run_name=candidate_id, nested=True):
            mlflow.log_param("source_epoch", candidate["epoch"])
            measured = evaluator.evaluate()

        if isinstance(measured, tuple):
            patch_accuracy, clip_accuracy = measured
        else:
            patch_accuracy, clip_accuracy = measured, None

        patch_accuracy = float(patch_accuracy)
        if not math.isfinite(patch_accuracy):
            raise ValueError(f"Invalid accuracy for {candidate_id}.")

        results.append({
            **candidate,
            "quant_val_patch_accuracy": patch_accuracy,
            "quant_val_clip_accuracy": (
                None if clip_accuracy is None else float(clip_accuracy)
            ),
            "tflite_path": str(quantized.model_path),
        })

        del evaluator, quantized, quantizer, float_model, wrapper

    # Use full-precision scores; round only when displaying them.
    results.sort(key=lambda row: (
        -row["quant_val_patch_accuracy"],
        row["val_loss"],
        row["epoch"],
    ))

    winner = results[0]
    for row in results:
        row["selected"] = row["epoch"] == winner["epoch"]

    with (export_root / "candidate_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    (export_root / "candidate_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # Copy the exact evaluated bytes; do not quantize the winner again.
    final_path = export_root / "quantized_model.tflite"
    temporary = export_root / "selected_model.tmp"
    shutil.copyfile(winner["tflite_path"], temporary)
    temporary.replace(final_path)

    (export_root / "selection.json").write_text(
        json.dumps({
            "criterion": "quantized validation patch accuracy",
            "tie_breakers": ["float validation loss", "earlier epoch"],
            "winner": winner,
            "final_model": str(final_path),
        }, indent=2),
        encoding="utf-8",
    )

    print("\nEpoch | Float val acc | Float val loss | Quant val acc | Selected")
    for row in results:
        print(
            f"{row['epoch']:5d} | "
            f"{row['val_accuracy'] * 100:12.3f}% | "
            f"{row['val_loss']:14.6f} | "
            f"{row['quant_val_patch_accuracy'] * 100:12.3f}% | "
            f"{'*' if row['selected'] else ''}"
        )

    mlflow.log_metric(
        "selected_quant_val_patch_acc",
        winner["quant_val_patch_accuracy"] * 100,
    )
    mlflow.log_param("selected_checkpoint_epoch", winner["epoch"])

    selected = tf.lite.Interpreter(model_path=str(final_path))
    selected.allocate_tensors()
    setattr(selected, "model_path", str(final_path))
    return selected
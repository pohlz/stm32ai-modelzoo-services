#!/usr/bin/env python3
"""Wait for an NVIDIA GPU, then run a command with that GPU selected."""

import argparse
import csv
import math
import os
import re
import subprocess
import sys
import time


def memory_mib(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(M|MiB|G|GiB)?", value, re.I)
    if not match:
        raise argparse.ArgumentTypeError("use MiB (e.g. 8192) or GiB (e.g. 8G)")
    amount = float(match[1]) * (1024 if (match[2] or "").lower() in ("g", "gib") else 1)
    if not math.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError("memory must be positive")
    return math.ceil(amount)


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def query(fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", "--query-" + kind + "=" + fields,
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True, timeout=30,
    )
    return [[cell.strip() for cell in row]
            for row in csv.reader(result.stdout.splitlines()) if row]


def choose_gpu(gpus, busy, memory):
    eligible = [gpu for gpu in gpus
                if (gpu["free"] >= memory if memory is not None
                    else gpu["uuid"] not in busy and gpu["util"] <= 1)]
    return max(eligible, key=lambda gpu: gpu["free"], default=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        "Example: python3 wait_gpu.py --memory 12G -- python3 train.py")
    parser.add_argument("--memory", type=memory_mib,
                        help="minimum free VRAM; default unit MiB, G/GiB = 1024 MiB")
    parser.add_argument("--interval", type=positive_seconds, default=10,
                        help="poll interval in seconds (default: 10)")
    parser.add_argument("--timeout", type=positive_seconds,
                        help="stop waiting after this many seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")

    start = time.monotonic()
    while True:
        if args.timeout is not None and time.monotonic() - start >= args.timeout:
            print("Timed out waiting for a GPU.", file=sys.stderr)
            return 124
        # Only query utilization in idle mode; memory mode does not need it.
        fields = "index,uuid,memory.free,memory.total"
        if args.memory is None:
            fields += ",utilization.gpu"
        gpus = []
        for row in query(fields):
            gpus.append(dict(index=row[0], uuid=row[1], free=int(row[2]),
                             total=int(row[3]), util=int(row[4]) if len(row) > 4 else 0))
        if not gpus:
            raise RuntimeError("nvidia-smi reports no visible GPUs")
        if args.memory is not None and all(gpu["total"] < args.memory for gpu in gpus):
            raise RuntimeError("requested memory exceeds every visible GPU's total memory")
        busy = {row[0] for row in query("gpu_uuid", "compute-apps")} if args.memory is None else set()
        selected = choose_gpu(gpus, busy, args.memory)
        if selected:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
            env["WAIT_GPU_UUID"] = selected["uuid"]
            env["WAIT_GPU_INDEX"] = selected["index"]
            print(f"Starting on GPU {selected['index']} ({selected['uuid']}), "
                  f"{selected['free']} MiB free", file=sys.stderr, flush=True)
            # Replace this launcher so signals and the exit code reach the command.
            os.execvpe(command[0], command, env)
        status = ", ".join(f"GPU {gpu['index']}: {gpu['free']} MiB free" for gpu in gpus)
        print("Waiting: " + status, file=sys.stderr, flush=True)
        delay = args.interval
        if args.timeout is not None:
            delay = min(delay, max(0, args.timeout - (time.monotonic() - start)))
        time.sleep(delay)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"wait_gpu: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        sys.exit(1)

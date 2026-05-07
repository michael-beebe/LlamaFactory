# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""High-resolution per-step timing wrapper for the LlamaFactory v1 SFT trainer.

Emits one JSONL row per training step to ``$BENCH_TIMING_PATH`` containing
(step, dt_sec, loss, grad_norm) using a ``time.perf_counter()`` measured
on rank 0 only. Bypasses the 1-second timestamp resolution of LlamaFactory's
human-readable log line.

Usage (driven by run.sh):
    BENCH_TIMING_PATH=/path/to/timings.jsonl \\
    torchrun ... -m michaelbeebe.bench.timing_runner <config.yaml> [overrides]
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# Make sure the LlamaFactory src/ is on sys.path. This file lives at
# <repo>/michaelbeebe/bench/timing_runner.py — repo root is two levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llamafactory.v1.utils.callbacks.trainer_callback import TrainerCallback  # noqa: E402


class StepTimingCallback(TrainerCallback):
    """Record perf_counter step times to JSONL on rank 0."""

    def __init__(self, out_path: str):
        self.out_path = out_path
        self._t0 = None
        self._fh = None
        self._rank = int(os.environ.get("RANK", "0"))

    def _open(self):
        if self._fh is None and self._rank == 0:
            self._fh = open(self.out_path, "w", buffering=1)  # line-buffered

    def on_train_begin(self, args, state, **kwargs):
        self._open()

    def on_step_begin(self, args, state, **kwargs):
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, **kwargs):
        if self._rank != 0 or self._t0 is None or self._fh is None:
            return
        dt = time.perf_counter() - self._t0
        row = {
            "step": int(state.global_step),
            "dt_sec": float(dt),
            "loss": float(state.loss),
            "grad_norm": float(state.grad_norm),
            "lr": float(state.learning_rate),
        }
        self._fh.write(json.dumps(row) + "\n")

    def on_train_end(self, args, state, **kwargs):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _main():
    # Mimic the v1 sft_trainer entrypoint but inject our callback before
    # calling fit(). We intentionally re-import inside main so the polyfill
    # in llamafactory.v1.__init__ runs first.
    from llamafactory.v1.accelerator.interface import DistributedInterface
    from llamafactory.v1.config import get_args
    from llamafactory.v1.core.data_engine import DataEngine
    from llamafactory.v1.core.model_engine import ModelEngine
    from llamafactory.v1.trainers.sft_trainer import SFTTrainer

    timing_path = os.environ.get("BENCH_TIMING_PATH")
    if not timing_path:
        raise SystemExit("BENCH_TIMING_PATH env var must be set.")

    model_args, data_args, training_args, _ = get_args()
    DistributedInterface(training_args.dist_config)
    train_dataset = DataEngine(data_args.train_dataset)
    model_engine = ModelEngine(model_args, is_train=True)
    trainer = SFTTrainer(
        args=training_args,
        model=model_engine.model,
        renderer=model_engine.renderer,
        train_dataset=train_dataset,
        callbacks=[StepTimingCallback(timing_path)],
    )
    trainer.fit()
    trainer.save_model()
    DistributedInterface().destroy()


if __name__ == "__main__":
    _main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tuned-config loader for ``kernel_unified_attention`` (vLLM v0.21.0).

Mirrors the loader pattern used by LoRA ops
(``vllm/lora/ops/triton_ops/utils.py:get_lora_op_configs``):

  - One JSON file per (GPU, sliding/full × prefill/decode) variant under
    ``$VLLM_TUNED_CONFIG_FOLDER`` (the same env var LoRA ops use).
  - Filename: ``{gpu_name}_UNIFIED_ATTENTION_{SLIDING|FULL}_{PREFILL|DECODE}.json``.
  - JSON traversal:
        head_size -> block_size -> num_queries_per_kv -> sliding_window
            -> batch_size -> dtype -> "2D"|"3D" -> {TILE_SIZE, num_warps, num_stages}
  - Strict on every dim except ``batch_size``, where we fall back to the
    nearest-neighbor int key (mirrors ``get_lora_op_configs``).

On miss the function returns ``None`` and the caller MUST preserve
upstream defaults bit-for-bit.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _variant_tag(*, sliding_window: int, is_3d: bool) -> str:
    sw = "SLIDING" if int(sliding_window) > 0 else "FULL"
    pd = "DECODE" if bool(is_3d) else "PREFILL"
    return f"{sw}_{pd}"


def _config_filename(gpu_name: str, *, sliding_window: int,
                     is_3d: bool) -> str:
    tag = _variant_tag(sliding_window=sliding_window, is_3d=is_3d)
    return f"{gpu_name}_UNIFIED_ATTENTION_{tag}.json"


@lru_cache(maxsize=8)
def _load_config_file(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            "unified_attention: failed to load tuned config %s: %s", path, e,
        )
        return None


def _gpu_name() -> str:
    return torch.cuda.get_device_name(0).replace(" ", "_")


def get_unified_attention_config(
    *,
    head_size: int,
    block_size: int,
    num_queries_per_kv: int,
    sliding_window: int,
    batch_size: int,
    dtype: str,
    is_3d: bool,
) -> dict[str, Any] | None:
    """Return the leaf launch-knob dict for one unified_attention shape,
    or ``None`` if no tuned config is available.

    On hit returns ``{"TILE_SIZE": int, "num_warps": int, "num_stages": int}``.
    On miss returns ``None`` and the caller preserves upstream defaults.
    """
    configs_folder = envs.VLLM_TUNED_CONFIG_FOLDER
    if not configs_folder:
        return None

    gpu_name = _gpu_name()
    fname = _config_filename(
        gpu_name, sliding_window=sliding_window, is_3d=is_3d,
    )
    config_path = os.path.join(configs_folder, fname)
    config_data = _load_config_file(config_path)
    if config_data is None:
        return None

    # Strict traversal: head_size -> block_size -> num_queries_per_kv
    #                   -> sliding_window
    strict_prefix = [
        str(int(head_size)),
        str(int(block_size)),
        str(int(num_queries_per_kv)),
        str(int(sliding_window)),
    ]
    for k in strict_prefix:
        if not isinstance(config_data, dict):
            return None
        if k not in config_data:
            return None
        config_data = config_data[k]

    # Nearest-neighbor fallback at batch_size only.
    if not isinstance(config_data, dict) or not config_data:
        return None
    bs_target = int(batch_size)
    bs_key = str(bs_target)
    if bs_key not in config_data:
        try:
            int_keys = [int(k) for k in config_data.keys()]
        except ValueError:
            return None
        if not int_keys:
            return None
        nearest = min(int_keys, key=lambda x: abs(x - bs_target))
        bs_key = str(nearest)
    config_data = config_data[bs_key]

    # Strict traversal: dtype -> "2D"|"3D"
    is_3d_str = "3D" if is_3d else "2D"
    for k in [str(dtype), is_3d_str]:
        if not isinstance(config_data, dict):
            return None
        if k not in config_data:
            return None
        config_data = config_data[k]

    if not isinstance(config_data, dict):
        return None
    return config_data

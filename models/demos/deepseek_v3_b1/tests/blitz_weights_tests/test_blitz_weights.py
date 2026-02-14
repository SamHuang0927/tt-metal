# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Device test for overlapped (fused) weight extraction.

Uses an NCRISC kernel to copy each sub-tensor out of the fused overlapped
tensor, then verifies the extracted data matches the independently
preprocessed reference (after bfp8 round-trip).

Tests all three constituents of get_tt_q_ab_proj_and_kv_a_proj_weights:
  - q_a_proj (packed)
  - q_b_proj (shuffled + tile-reshaped)
  - kv_a_proj (shard-reordered)
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.blitz_decode_weights import _KV_A_PROJ_SHARD_ORDER, BlitzDecodeWeights
from models.demos.deepseek_v3_b1.tests.blitz_weights_tests.op import CopyToOutput
from models.demos.deepseek_v3_b1.utils import shuffle_weights_for_interleaved_qnope_qrope

# ---------------------------------------------------------------------------
# Constants (mirror the values in get_tt_q_ab_proj_and_kv_a_proj_weights)
# ---------------------------------------------------------------------------
GRID_X, GRID_Y = 12, 8
KV_GRID_X, KV_GRID_Y = 9, 2
KV_GRID_OFFSET = (0, 8)

Q_AB_NUM_CORES = GRID_X * GRID_Y
KV_NUM_CORES = KV_GRID_X * KV_GRID_Y

NUM_QNOPE_HEADS = 64
NUM_QROPE_HEADS = 64
QNOPE_HEAD_DIM = 128
QROPE_HEAD_DIM = 64
HEADS_PER_ROW = 8

Q_A_PROJ_SHAPE = (7168, 1536)
Q_B_PROJ_SHAPE = (1536, 12288)
KV_A_PROJ_SHAPE = (7168, 576)

TILE_H, TILE_W = 32, 32

PACKED_H = Q_A_PROJ_SHAPE[0] // 2  # 3584
PACKED_W = Q_A_PROJ_SHAPE[1] * 2  # 3072
PACKED_SHARD_W = PACKED_W // Q_AB_NUM_CORES  # 32

SHUFFLED_SHARD_W = Q_B_PROJ_SHAPE[1] // Q_AB_NUM_CORES  # 128
TILE_RESHAPED_H = Q_B_PROJ_SHAPE[0] * SHUFFLED_SHARD_W // PACKED_SHARD_W  # 6144
TILE_RESHAPED_W = PACKED_W  # 3072

FUSED_SHARD_H = PACKED_H + TILE_RESHAPED_H  # 9728
FUSED_SHARD_W = PACKED_SHARD_W  # 32

Q_A_TILES_PER_SHARD = PACKED_H // TILE_H  # 112
Q_B_TILES_PER_SHARD = TILE_RESHAPED_H // TILE_H  # 192
KV_TILES_PER_SHARD = KV_A_PROJ_SHAPE[0] // TILE_H  # 224

KV_H = KV_A_PROJ_SHAPE[0]  # 7168
KV_W = KV_A_PROJ_SHAPE[1]  # 576
KV_SHARD_W = KV_W // KV_NUM_CORES  # 32


# ---------------------------------------------------------------------------
# Core range helpers
# ---------------------------------------------------------------------------
def _q_ab_core_range_set():
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))})


def _kv_core_range_set():
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(KV_GRID_OFFSET[0], KV_GRID_OFFSET[1]),
                ttnn.CoreCoord(
                    KV_GRID_OFFSET[0] + KV_GRID_X - 1,
                    KV_GRID_OFFSET[1] + KV_GRID_Y - 1,
                ),
            )
        }
    )


# ---------------------------------------------------------------------------
# Reference helpers (independently preprocess for comparison)
# ---------------------------------------------------------------------------
def _pack_q_a(weights):
    """Pack (H, W) -> (H/2, 2W) by interleaving K-halves."""
    H, W = weights.shape
    return weights.reshape(2, H // 2, W).permute(1, 0, 2).reshape(H // 2, 2 * W)


def _shuffle_q_b(weights):
    """Shuffle q_b_proj for interleaved Qnope/Qrope layout."""
    return shuffle_weights_for_interleaved_qnope_qrope(
        weights,
        num_qnope_heads=NUM_QNOPE_HEADS,
        num_qrope_heads=NUM_QROPE_HEADS,
        qnope_head_dim=QNOPE_HEAD_DIM,
        qrope_head_dim=QROPE_HEAD_DIM,
        heads_per_row=HEADS_PER_ROW,
    )


def _reorder_kv_a(weights):
    """Reorder kv_a_proj shards according to _KV_A_PROJ_SHARD_ORDER."""
    kv_h, kv_w = weights.shape
    shards = weights.reshape(kv_h, KV_NUM_CORES, KV_SHARD_W)
    return shards[:, _KV_A_PROJ_SHARD_ORDER, :].reshape(kv_h, kv_w)


def _build_tile_reshaped_q_b(shuffled):
    """Tile-reshape each shard of shuffled q_b from (1536,128) to (6144,32)."""
    out = torch.zeros(TILE_RESHAPED_H, TILE_RESHAPED_W, dtype=shuffled.dtype)
    for i in range(Q_AB_NUM_CORES):
        shard = shuffled[:, i * SHUFFLED_SHARD_W : (i + 1) * SHUFFLED_SHARD_W]
        reshaped = BlitzDecodeWeights._tile_reshape(
            shard,
            src_shape=(Q_B_PROJ_SHAPE[0], SHUFFLED_SHARD_W),
            dst_shape=(TILE_RESHAPED_H, FUSED_SHARD_W),
        )
        out[:, i * FUSED_SHARD_W : (i + 1) * FUSED_SHARD_W] = reshaped
    return out


# ---------------------------------------------------------------------------
# Device tensor helpers
# ---------------------------------------------------------------------------
def _create_output_device_tensor(height, width, core_range_set, device):
    """Allocate a zeroed output tensor on device."""
    num_cores = core_range_set.num_cores()
    shard_w = width // num_cores
    shard_spec = ttnn.ShardSpec(core_range_set, (height, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    return ttnn.from_torch(
        torch.zeros(height, width, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )


def _bfp8_reference(torch_data, core_range_set, device):
    """Send torch data through bfp8 round-trip on device to get ground truth."""
    height, width = torch_data.shape
    num_cores = core_range_set.num_cores()
    shard_w = width // num_cores
    shard_spec = ttnn.ShardSpec(core_range_set, (height, shard_w), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
    tt = ttnn.from_torch(
        torch_data,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    return ttnn.to_torch(tt)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_q_ab_proj_and_kv_a_proj_overlap(device):
    """Verify all three constituents of the q_ab_proj + kv_a_proj overlap.

    Creates the fused tensor once via BlitzDecodeWeights, then extracts
    each sub-tensor with CopyToOutput and checks it against an
    independently preprocessed + bfp8 round-tripped reference.
    """
    torch.manual_seed(42)
    q_a_raw = torch.randn(Q_A_PROJ_SHAPE, dtype=torch.bfloat16)
    q_b_raw = torch.randn(Q_B_PROJ_SHAPE, dtype=torch.bfloat16)
    kv_raw = torch.randn(KV_A_PROJ_SHAPE, dtype=torch.bfloat16)

    bdw = BlitzDecodeWeights(device)
    fused_tt = bdw.get_tt_q_ab_proj_and_kv_a_proj_weights(q_a_raw, q_b_raw, kv_raw)

    q_ab_crs = _q_ab_core_range_set()
    kv_crs = _kv_core_range_set()

    # -- q_a_proj (packed) -------------------------------------------------
    q_a_out = _create_output_device_tensor(PACKED_H, PACKED_W, q_ab_crs, device)
    q_a_result = ttnn.to_torch(CopyToOutput.op(fused_tt, q_a_out, tile_offset=0, num_tiles_to_copy=Q_A_TILES_PER_SHARD))
    q_a_ref = _bfp8_reference(_pack_q_a(q_a_raw), q_ab_crs, device)
    assert torch.equal(q_a_result, q_a_ref), "q_a_proj extraction mismatch"
    logger.info("q_a_proj extraction passed")

    # -- q_b_proj (shuffled + tile-reshaped) -------------------------------
    q_b_out = _create_output_device_tensor(TILE_RESHAPED_H, TILE_RESHAPED_W, q_ab_crs, device)
    q_b_result = ttnn.to_torch(
        CopyToOutput.op(fused_tt, q_b_out, tile_offset=Q_A_TILES_PER_SHARD, num_tiles_to_copy=Q_B_TILES_PER_SHARD)
    )
    q_b_ref = _bfp8_reference(_build_tile_reshaped_q_b(_shuffle_q_b(q_b_raw)), q_ab_crs, device)
    assert torch.equal(q_b_result, q_b_ref), "q_b_proj extraction mismatch"
    logger.info("q_b_proj extraction passed")

    # -- kv_a_proj (shard-reordered) ---------------------------------------
    kv_out = _create_output_device_tensor(KV_H, KV_W, kv_crs, device)
    kv_result = ttnn.to_torch(CopyToOutput.op(fused_tt, kv_out, tile_offset=0, num_tiles_to_copy=KV_TILES_PER_SHARD))
    kv_ref = _bfp8_reference(_reorder_kv_a(kv_raw), kv_crs, device)
    assert torch.equal(kv_result, kv_ref), "kv_a_proj extraction mismatch"
    logger.info("kv_a_proj extraction passed")

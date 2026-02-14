# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Blitz decode weight overlapping infrastructure.

"Overlapping" means fusing multiple weight tensors into a single
width-sharded tensor so they share the same L1 base address on each core.
For each core the individual shards are stitched together vertically,
with tile-reshape applied when shard widths differ, producing one
contiguous buffer per core.  Kernels locate each sub-weight at a known
row offset within the fused shard.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_b1.utils import shuffle_weights_for_interleaved_qnope_qrope

# kv_a_proj shard reorder: places the Knope-rope boundary shard so that
# the physical core layout matches the logical split expected by the
# KV cache branch kernel.
_KV_A_PROJ_SHARD_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 16, 8, 9, 10, 11, 12, 13, 14, 15, 17]


class BlitzDecodeWeights:
    """Fuses weight tensors to share the same L1 base address per core.

    Methods take raw torch weight tensors, apply any required preprocessing
    (packing, shuffling), stitch per-core shards, and return the result as
    a device-resident ttnn.Tensor with WIDTH_SHARDED placement.

    Args:
        device: The ttnn device (or MeshDevice) to place tensors on.
    """

    def __init__(self, device) -> None:
        self._device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tt_q_ab_proj_and_kv_a_proj_weights(
        self,
        q_a_proj_weights: torch.Tensor,
        q_b_proj_weights: torch.Tensor,
        kv_a_proj_weights: torch.Tensor,
    ) -> ttnn.Tensor:
        """Fuse q_a_proj, q_b_proj, and kv_a_proj into one WIDTH_SHARDED tensor.

        The fused tensor spans two core regions that share the same shard
        width, giving every core a single base address:

        * **Top region** (8x12 = 96 cores): q_a_proj packed + q_b_proj
          shuffled, stitched per core.
        * **Bottom region** (2x9 = 18 cores at offset (8,0)): kv_a_proj
          with shard reordering, zero-padded to the same shard height as
          the top region.

        Preprocessing applied internally:

        * q_a_proj (7168, 1536) is packed to (3584, 3072).
        * q_b_proj (1536, 12288) is shuffled for interleaved Qnope/Qrope.
        * kv_a_proj (7168, 576) shards are reordered to match the KV cache
          branch core layout.

        Layout::

            -- Top 96 cores (8x12) --
            q_a_proj packed  : (3584, 3072)  -> shard (3584,  32)
            q_b_proj shuffled: (1536, 12288) -> shard (1536, 128)
              tile-reshaped to                        (6144,  32)
            fused shard      : (9728,   32)

            -- Bottom 18 cores (2x9) --
            kv_a_proj        : (7168,  576)  -> shard (7168,  32)
              zero-padded to                          (9728,  32)

            combined tensor  : (9728, 3648)   114 cores x 32

        Args:
            q_a_proj_weights: Raw q_a_proj tensor, must be (7168, 1536).
            q_b_proj_weights: Raw (unshuffled) q_b_proj tensor, must be
                (1536, 12288).
            kv_a_proj_weights: Raw kv_a_proj tensor, must be (7168, 576).

        Returns:
            A device-resident ttnn.Tensor with WIDTH_SHARDED memory config
            spanning both core regions.
        """
        # -- Constants --------------------------------------------------
        GRID_X, GRID_Y = 12, 8  # q_a/q_b proj: 8 rows x 12 cols
        KV_GRID_X, KV_GRID_Y = 9, 2  # kv_a_proj:    2 rows x 9 cols
        KV_GRID_OFFSET = (0, 8)  # kv_a_proj starts at row 8, col 0

        # -- Validate device grid ----------------------------------------
        device_grid = self._device.compute_with_storage_grid_size()
        required_rows = KV_GRID_OFFSET[1] + KV_GRID_Y  # 10
        required_cols = GRID_X  # 12
        assert device_grid.y >= required_rows, (
            f"Device grid needs at least {required_rows} rows, " f"got {device_grid.y}"
        )
        assert device_grid.x >= required_cols, (
            f"Device grid needs at least {required_cols} cols, " f"got {device_grid.x}"
        )
        NUM_QNOPE_HEADS = 64
        NUM_QROPE_HEADS = 64
        QNOPE_HEAD_DIM = 128
        QROPE_HEAD_DIM = 64
        HEADS_PER_ROW = 8
        KNOPE_DIM = 512
        KROPE_DIM = 64

        EXPECTED_Q_B_PROJ_WIDTH = NUM_QNOPE_HEADS * QNOPE_HEAD_DIM + NUM_QROPE_HEADS * QROPE_HEAD_DIM  # 12288
        EXPECTED_KV_WIDTH = KNOPE_DIM + KROPE_DIM  # 576

        # -- Validate raw input shapes ----------------------------------
        assert q_a_proj_weights.shape == (
            7168,
            1536,
        ), f"q_a_proj_weights must be (7168, 1536), got {tuple(q_a_proj_weights.shape)}"
        assert q_b_proj_weights.shape == (1536, EXPECTED_Q_B_PROJ_WIDTH), (
            f"q_b_proj_weights must be (1536, {EXPECTED_Q_B_PROJ_WIDTH}), " f"got {tuple(q_b_proj_weights.shape)}"
        )
        assert kv_a_proj_weights.shape == (7168, EXPECTED_KV_WIDTH), (
            f"kv_a_proj_weights must be (7168, {EXPECTED_KV_WIDTH}), " f"got {tuple(kv_a_proj_weights.shape)}"
        )

        q_ab_num_cores = GRID_X * GRID_Y
        kv_num_cores = KV_GRID_X * KV_GRID_Y

        # -- Step 1: pack q_a_proj  (H, W) -> (H/2, 2W) ----------------
        H, W = q_a_proj_weights.shape
        packed = q_a_proj_weights.reshape(2, H // 2, W).permute(1, 0, 2).reshape(H // 2, 2 * W)

        # -- Step 2: shuffle q_b_proj for interleaved Qnope/Qrope ------
        shuffled = shuffle_weights_for_interleaved_qnope_qrope(
            q_b_proj_weights,
            num_qnope_heads=NUM_QNOPE_HEADS,
            num_qrope_heads=NUM_QROPE_HEADS,
            qnope_head_dim=QNOPE_HEAD_DIM,
            qrope_head_dim=QROPE_HEAD_DIM,
            heads_per_row=HEADS_PER_ROW,
        )

        # -- Step 3: stitch q_a + q_b per-core shards -------------------
        q_ab_fused, q_ab_shard_shape = BlitzDecodeWeights._stitch_width_sharded(packed, shuffled, q_ab_num_cores)
        fused_shard_h, target_w = q_ab_shard_shape

        # -- Step 4: reorder kv_a_proj shards ---------------------------
        kv_h, kv_w = kv_a_proj_weights.shape
        kv_shard_w = kv_w // kv_num_cores
        assert kv_shard_w == target_w, (
            f"kv_a_proj shard width ({kv_shard_w}) must equal q_ab fused " f"shard width ({target_w})"
        )

        kv_shards = kv_a_proj_weights.reshape(kv_h, kv_num_cores, kv_shard_w)
        kv_reordered = kv_shards[:, _KV_A_PROJ_SHARD_ORDER, :].reshape(kv_h, kv_w)

        # -- Step 5: pad kv shards to fused shard height ----------------
        kv_padded = torch.zeros(fused_shard_h, kv_w, dtype=kv_a_proj_weights.dtype)
        kv_padded[:kv_h, :] = kv_reordered

        # -- Step 6: concatenate q_ab and kv along width ----------------
        total_cores = q_ab_num_cores + kv_num_cores
        combined = torch.cat([q_ab_fused, kv_padded], dim=1)
        assert combined.shape == (fused_shard_h, target_w * total_cores)

        # -- Step 7: place on device as WIDTH_SHARDED -------------------
        q_ab_grid = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1),
        )
        kv_grid = ttnn.CoreRange(
            ttnn.CoreCoord(KV_GRID_OFFSET[0], KV_GRID_OFFSET[1]),
            ttnn.CoreCoord(
                KV_GRID_OFFSET[0] + KV_GRID_X - 1,
                KV_GRID_OFFSET[1] + KV_GRID_Y - 1,
            ),
        )
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({q_ab_grid, kv_grid}),
            (fused_shard_h, target_w),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        return ttnn.from_torch(
            combined,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stitch_width_sharded(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        num_cores: int,
        tile_h: int = 32,
        tile_w: int = 32,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Stitch two width-sharded tensors into one fused tensor.

        For every core the two shards are concatenated vertically.  When
        the shard widths differ, the wider shard is tile-reshaped to match
        the narrower one (preserving tile ordering) so the concatenation
        is well-defined.

        Args:
            tensor1: First weight tensor (H1, W1).
            tensor2: Second weight tensor (H2, W2).
            num_cores: Total cores in the width-sharded grid.
            tile_h: Tile height (default 32).
            tile_w: Tile width (default 32).

        Returns:
            (fused_tensor, shard_shape) ready for WIDTH_SHARDED
            placement on num_cores cores.
        """
        H1, W1 = tensor1.shape
        H2, W2 = tensor2.shape

        shard_w1 = W1 // num_cores
        shard_w2 = W2 // num_cores

        # Use the narrower shard width as target; tile-reshape the wider.
        if shard_w1 <= shard_w2:
            target_w = shard_w1
            narrow, wide = tensor1, tensor2
            narrow_h, wide_h = H1, H2
            narrow_sw, wide_sw = shard_w1, shard_w2
        else:
            target_w = shard_w2
            narrow, wide = tensor2, tensor1
            narrow_h, wide_h = H2, H1
            narrow_sw, wide_sw = shard_w2, shard_w1

        # Height of each wide shard after tile-reshape to target_w
        reshaped_h = wide_h * wide_sw // target_w

        fused_shard_h = narrow_h + reshaped_h
        fused = torch.zeros(fused_shard_h, target_w * num_cores, dtype=tensor1.dtype)

        for core_idx in range(num_cores):
            col_start = core_idx * target_w
            col_end = col_start + target_w

            # Narrow shard: already at target width, just copy.
            n_start = core_idx * narrow_sw
            n_end = n_start + narrow_sw
            fused[:narrow_h, col_start:col_end] = narrow[:, n_start:n_end]

            # Wide shard: tile-reshape from (wide_h, wide_sw) to
            # (reshaped_h, target_w), then copy.
            w_start = core_idx * wide_sw
            w_end = w_start + wide_sw
            w_shard = wide[:, w_start:w_end]
            w_reshaped = BlitzDecodeWeights._tile_reshape(
                w_shard,
                src_shape=(wide_h, wide_sw),
                dst_shape=(reshaped_h, target_w),
                tile_h=tile_h,
                tile_w=tile_w,
            )
            fused[narrow_h:, col_start:col_end] = w_reshaped

        shard_shape = (fused_shard_h, target_w)
        return fused, shard_shape

    @staticmethod
    def _tile_reshape(
        tensor: torch.Tensor,
        src_shape: tuple[int, int],
        dst_shape: tuple[int, int],
        tile_h: int = 32,
        tile_w: int = 32,
    ) -> torch.Tensor:
        """Reshape a 2-D tensor while preserving row-major tile ordering.

        Data is stored as a grid of (tile_h x tile_w) tiles in row-major
        order.  A naive torch.reshape changes which values land in each
        tile.  This helper keeps every tile's contents unchanged by:

        1. Splitting into the source tile grid.
        2. Flattening to a 1-D tile sequence (row-major).
        3. Re-gridding into the destination tile dimensions.

        Total tile count must be identical for source and destination.
        """
        src_h, src_w = src_shape
        dst_h, dst_w = dst_shape
        src_tr, src_tc = src_h // tile_h, src_w // tile_w
        dst_tr, dst_tc = dst_h // tile_h, dst_w // tile_w
        assert src_tr * src_tc == dst_tr * dst_tc, f"Tile count mismatch: {src_tr * src_tc} vs {dst_tr * dst_tc}"
        # (H, W) -> (tile_rows, tile_h, tile_cols, tile_w)
        #         -> (tile_rows, tile_cols, tile_h, tile_w)
        tiles = tensor.reshape(src_tr, tile_h, src_tc, tile_w).permute(0, 2, 1, 3)
        # Flatten to 1-D tile sequence, re-grid to destination layout
        tiles = tiles.reshape(-1, tile_h, tile_w).reshape(dst_tr, dst_tc, tile_h, tile_w)
        # (dst_tr, dst_tc, tile_h, tile_w) -> (dst_H, dst_W)
        return tiles.permute(0, 2, 1, 3).reshape(dst_h, dst_w)

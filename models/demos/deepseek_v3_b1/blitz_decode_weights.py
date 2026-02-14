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

import numpy as np
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

    def get_tt_o_proj_and_gate_mm_weights(
        self,
        o_proj_weights: torch.Tensor,
        gate_mm_weights: torch.Tensor,
    ) -> ttnn.Tensor:
        """Fuse o_proj (BFP8) and gate_mm_weights (BFP16) into one WIDTH_SHARDED tensor.

        Both sub-tensors share the same L1 base address but live on
        separate, non-overlapping core regions.  Because BFP8 and BFP16
        tiles have different byte sizes (1088 vs 2048 bytes per 32x32
        tile), the standard ``ttnn.from_torch`` path — which applies a
        single dtype to the whole tensor — cannot be used.

        Instead, this method:

        1. Converts each sub-tensor to its target BFP format and tilizes
           it entirely on the host using vectorised numpy.
        2. Packs per-shard raw bytes: o_proj shards in BFP8, gate_mm
           shards in BFP16, both zero-padded to the same byte size.
        3. Sends the combined buffer as a UINT32 raw-byte container with
           WIDTH_SHARDED placement, so the allocator gives every core the
           same base address.

        Kernels locate each sub-tensor by a **byte offset** (not a tile
        offset) that is passed as a compile-time arg, and bind separate
        circular buffers with the correct data format (Bfp8_b or
        Float16_b) at the computed address.

        Layout::

            -- o_proj region: 112 cores (matmul2 grid) --
            o_proj (8192, 7168) as bfloat8_b
              shard (8192, 64) = 512 tiles x 1088 B = 557 056 B

            -- gate_mm region: 8 cores (row 9, cols 0-7) --
            gate_mm (7168, 256) as bfloat16
              shard (7168, 32) = 224 tiles x 2048 B = 458 752 B
              zero-padded to 557 056 B

            combined: 120 cores, shard = 139 264 UINT32 (557 056 B)

        Args:
            o_proj_weights:   Raw o_proj tensor, shape (8192, 7168).
            gate_mm_weights:  Raw gate_mm tensor, shape (7168, 256).

        Returns:
            A device-resident ttnn.Tensor (UINT32, ROW_MAJOR, WIDTH_SHARDED)
            spanning both core regions.  The tensor is a raw byte container;
            kernels interpret the bytes via CB data-format settings.
        """
        # -- Tile / format constants ------------------------------------
        TILE_H, TILE_W = 32, 32
        BFP8_TILE_BYTES = 1088  # (256*4) + (16*4)  for 32x32
        BFP16_TILE_BYTES = 2048  # 1024 * 2          for 32x32

        # -- Core grid constants ----------------------------------------
        #  o_proj: 112 cores in matmul2 layout (rows 0-7 full 13 cols + row 8 cols 0-7)
        O_PROJ_NUM_CORES = 112
        O_PROJ_SHARD_W = 64  # 7168 / 112
        #  gate_mm: 8 cores on row 9 (non-overlapping with o_proj)
        GATE_MM_NUM_CORES = 8
        GATE_MM_SHARD_W = 32  # 256 / 8

        # -- Validate shapes --------------------------------------------
        assert o_proj_weights.shape == (8192, 7168), f"o_proj must be (8192, 7168), got {tuple(o_proj_weights.shape)}"
        assert gate_mm_weights.shape == (7168, 256), f"gate_mm must be (7168, 256), got {tuple(gate_mm_weights.shape)}"

        o_H, o_W = o_proj_weights.shape
        g_H, g_W = gate_mm_weights.shape

        # -- Compute per-shard byte sizes --------------------------------
        o_tiles_per_shard = (o_H // TILE_H) * (O_PROJ_SHARD_W // TILE_W)  # 256*2 = 512
        g_tiles_per_shard = (g_H // TILE_H) * (GATE_MM_SHARD_W // TILE_W)  # 224*1 = 224

        o_shard_bytes = o_tiles_per_shard * BFP8_TILE_BYTES  # 557 056
        g_shard_bytes = g_tiles_per_shard * BFP16_TILE_BYTES  # 458 752
        max_shard_bytes = max(o_shard_bytes, g_shard_bytes)  # 557 056
        assert max_shard_bytes % 4 == 0, "shard bytes must be UINT32-aligned"

        # -- Pack shards ------------------------------------------------
        packed = bytearray()

        # o_proj shards (BFP8_b)
        for i in range(O_PROJ_NUM_CORES):
            shard_data = o_proj_weights[:, i * O_PROJ_SHARD_W : (i + 1) * O_PROJ_SHARD_W].contiguous()
            shard_raw = BlitzDecodeWeights._tilize_and_pack_bfp8(shard_data, TILE_H, TILE_W)
            assert len(shard_raw) == o_shard_bytes
            packed.extend(shard_raw)
            packed.extend(b"\x00" * (max_shard_bytes - o_shard_bytes))

        # gate_mm shards (bfloat16)
        for i in range(GATE_MM_NUM_CORES):
            shard_data = gate_mm_weights[:, i * GATE_MM_SHARD_W : (i + 1) * GATE_MM_SHARD_W].contiguous()
            shard_raw = BlitzDecodeWeights._tilize_and_pack_bfloat16(shard_data, TILE_H, TILE_W)
            assert len(shard_raw) == g_shard_bytes
            packed.extend(shard_raw)
            packed.extend(b"\x00" * (max_shard_bytes - g_shard_bytes))

        # -- Build UINT32 tensor on device ------------------------------
        total_cores = O_PROJ_NUM_CORES + GATE_MM_NUM_CORES
        uint32_per_shard = max_shard_bytes // 4

        raw_data = torch.frombuffer(bytes(packed), dtype=torch.int32).clone()

        o_proj_grid_1 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(12, 7))  # 104
        o_proj_grid_2 = ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 8))  # 8
        gate_mm_grid = ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(7, 9))  # 8

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({o_proj_grid_1, o_proj_grid_2, gate_mm_grid}),
            (1, uint32_per_shard),
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        return ttnn.from_torch(
            raw_data.reshape(1, uint32_per_shard * total_cores),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self._device,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._device),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tilize_and_pack_bfp8(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
        """Tilize a 2-D tensor and pack as BFP8_b raw bytes.

        Produces the exact byte layout the hardware expects:
        ``[16 exponent uint32 words][256 mantissa uint32 words]`` per tile,
        with tiles in row-major order across the tensor.

        BFP8_b format (per 16-element block):
        * 1 shared exponent byte (max bfloat16 exponent in the block)
        * 16 mantissa bytes, each = ``sign(1) | mantissa(7)``

        Tile layout: 4 faces (16x16) in order
        face0 (rows 0-15, cols 0-15), face1 (rows 0-15, cols 16-31),
        face2 (rows 16-31, cols 0-15), face3 (rows 16-31, cols 16-31).
        Each face row of 16 elements forms one BFP8 block.
        """
        H, W = data_2d.shape
        face_h, face_w = tile_h // 2, tile_w // 2
        tr, tc = H // tile_h, W // tile_w
        num_tiles = tr * tc

        data_np = data_2d.contiguous().float().numpy()

        # Reshape into tile grid -> (tr, tc, tile_h, tile_w)
        tiles = data_np.reshape(tr, tile_h, tc, tile_w).transpose(0, 2, 1, 3)
        tiles = tiles.reshape(num_tiles, tile_h, tile_w)

        # Extract 4 faces per tile -> face-ordered (N, 1024)
        face_ordered = np.concatenate(
            [
                tiles[:, :face_h, :face_w].reshape(num_tiles, -1),
                tiles[:, :face_h, face_w:].reshape(num_tiles, -1),
                tiles[:, face_h:, :face_w].reshape(num_tiles, -1),
                tiles[:, face_h:, face_w:].reshape(num_tiles, -1),
            ],
            axis=1,
        )

        # Reshape into BFP8 blocks: (N, 64 blocks, 16 elements)
        blocks = face_ordered.reshape(num_tiles, 64, 16)

        # --- bfloat16 field extraction (vectorised) ---
        float_bits = blocks.view(np.uint32)
        bf16_bits = (float_bits >> 16).astype(np.uint16)

        signs = ((bf16_bits >> 15) & 1).astype(np.uint8)
        exponents = ((bf16_bits >> 7) & 0xFF).astype(np.int32)
        mantissa7 = (bf16_bits & 0x7F).astype(np.int32)

        # Implicit leading 1; zero for denormals (exp==0)
        explicit_mant = np.where(exponents == 0, 0, (1 << 7) | mantissa7)

        # Shared exponent = max in each 16-element block
        shared_exp = np.max(exponents, axis=2)  # (N, 64)

        # Shift mantissa by delta
        delta = shared_exp[:, :, np.newaxis] - exponents
        shifted = explicit_mant >> delta
        packed_mant = ((signs << 7) | (shifted & 0x7F)).astype(np.uint8)

        # Assemble per-tile bytes: [exp words][mant words]
        exp_bytes = shared_exp.astype(np.uint8)  # (N, 64)
        exp_words = exp_bytes.view(np.uint32).reshape(num_tiles, 16)

        mant_words = packed_mant.reshape(num_tiles, 1024).view(np.uint32).reshape(num_tiles, 256)

        tile_words = np.concatenate([exp_words, mant_words], axis=1)  # (N, 272)
        return tile_words.tobytes()

    @staticmethod
    def _tilize_and_pack_bfloat16(data_2d: torch.Tensor, tile_h: int = 32, tile_w: int = 32) -> bytes:
        """Tilize a 2-D tensor and pack as bfloat16 (Float16_b) raw bytes.

        Each tile is 2048 bytes: 1024 elements x 2 bytes, stored in
        face order (face0, face1, face2, face3), row-major within each
        face.  bfloat16 is the top 16 bits of IEEE-754 float32.
        """
        H, W = data_2d.shape
        face_h, face_w = tile_h // 2, tile_w // 2
        tr, tc = H // tile_h, W // tile_w
        num_tiles = tr * tc

        data_np = data_2d.contiguous().float().numpy()

        tiles = data_np.reshape(tr, tile_h, tc, tile_w).transpose(0, 2, 1, 3)
        tiles = tiles.reshape(num_tiles, tile_h, tile_w)

        face_ordered = np.concatenate(
            [
                tiles[:, :face_h, :face_w].reshape(num_tiles, -1),
                tiles[:, :face_h, face_w:].reshape(num_tiles, -1),
                tiles[:, face_h:, :face_w].reshape(num_tiles, -1),
                tiles[:, face_h:, face_w:].reshape(num_tiles, -1),
            ],
            axis=1,
        )  # (N, 1024)

        # bfloat16 = top 16 bits of float32
        float_bits = face_ordered.view(np.uint32)
        bf16_bits = (float_bits >> 16).astype(np.uint16)
        return bf16_bits.tobytes()

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

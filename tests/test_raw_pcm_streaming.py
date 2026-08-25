"""Unit tests for the raw-PCM streaming helpers in ``pocket_tts.data.audio``.

These exercise the format conversion and resampling primitives without
loading the TTS model, so the suite stays fast and CI-friendly.
"""

import io

import numpy as np
import torch

from pocket_tts.data.audio import chunk_to_pcm_bytes, downsample_audio_chunk, stream_raw_pcm_chunks


class _PersistentBuffer(io.BytesIO):
    """BytesIO that survives close() — the streaming helper uses ``with file_like:``
    to signal end-of-stream to its consumer (a queue in production), but tests
    want to inspect the bytes after the helper returns.
    """

    def close(self):  # noqa: D401
        pass

    def hard_close(self):
        super().close()


def test_chunk_to_pcm_bytes_quantizes_and_serializes_little_endian():
    chunk = torch.tensor([0.0, 1.0, -1.0, 0.5, -0.5], dtype=torch.float32)
    pcm = chunk_to_pcm_bytes(chunk)

    # 5 samples × 2 bytes each
    assert len(pcm) == 10

    decoded = np.frombuffer(pcm, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == 32767  # +1.0 → max
    assert decoded[2] == -32767  # -1.0 → min (mirrored, not -32768)
    assert decoded[3] == int(0.5 * 32767)
    assert decoded[4] == int(-0.5 * 32767)


def test_chunk_to_pcm_bytes_clamps_out_of_range():
    chunk = torch.tensor([2.0, -2.0, 1.5], dtype=torch.float32)
    decoded = np.frombuffer(chunk_to_pcm_bytes(chunk), dtype="<i2")
    assert decoded[0] == 32767
    assert decoded[1] == -32767
    assert decoded[2] == 32767


def test_downsample_24k_to_8k_yields_3x_fewer_samples():
    # 600 samples at 24 kHz = 25 ms of audio → 200 samples at 8 kHz
    chunk = torch.zeros(600, dtype=torch.float32)
    out = downsample_audio_chunk(chunk, from_rate=24000, to_rate=8000)
    assert out.numel() == 200


def test_downsample_no_op_when_rates_equal():
    chunk = torch.randn(128, dtype=torch.float32)
    out = downsample_audio_chunk(chunk, from_rate=24000, to_rate=24000)
    # Same tensor returned unchanged when rates match — the early return path.
    assert torch.equal(out, chunk)


def test_downsample_preserves_dc_offset_within_passband():
    # A DC signal (constant) should remain ~constant after downsampling.
    chunk = torch.full((600,), 0.5, dtype=torch.float32)
    out = downsample_audio_chunk(chunk, from_rate=24000, to_rate=8000)
    # Allow some edge-effect tolerance from the polyphase FIR.
    mid = out[20:-20]
    assert torch.allclose(mid, torch.full_like(mid, 0.5), atol=0.05)


def test_stream_raw_pcm_chunks_writes_concatenated_bytes_without_header():
    buf = _PersistentBuffer()
    chunks = [
        torch.tensor([0.0, 0.5, -0.5], dtype=torch.float32),
        torch.tensor([1.0, -1.0], dtype=torch.float32),
    ]

    # source == target → no resample
    stream_raw_pcm_chunks(buf, iter(chunks), source_sample_rate=24000, target_sample_rate=None)

    raw = buf.getvalue()
    # 5 samples × 2 bytes — no RIFF header (44 bytes) and no trailing silence padding.
    assert len(raw) == 10

    decoded = np.frombuffer(raw, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == int(0.5 * 32767)
    assert decoded[2] == int(-0.5 * 32767)
    assert decoded[3] == 32767
    assert decoded[4] == -32767


def test_stream_raw_pcm_chunks_resamples_when_target_differs():
    buf = _PersistentBuffer()
    # 600 samples × 2 chunks = 1200 samples @ 24 kHz → 400 samples @ 8 kHz → 800 bytes.
    chunks = [torch.zeros(600, dtype=torch.float32), torch.zeros(600, dtype=torch.float32)]

    stream_raw_pcm_chunks(buf, iter(chunks), source_sample_rate=24000, target_sample_rate=8000)

    raw = buf.getvalue()
    assert len(raw) == 800

"""
Audio IO methods are defined in this module (info, read, write),
We rely on av library for faster read when possible, otherwise on torchaudio.
"""

import logging
import math
import os
import sys
import wave
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from typing_extensions import TypeIs

logger = logging.getLogger(__name__)

FIRST_CHUNK_LENGTH_SECONDS = float(os.environ.get("FIRST_CHUNK_LENGTH_SECONDS", "0"))


def audio_read(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Read audio file. WAV uses built-in wave module; other formats require soundfile."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        # Use built-in wave module for WAV files
        with wave.open(str(filepath), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if sample_width != 2:
                return _audio_read_with_soundfile(filepath)
            raw_data = wav_file.readframes(-1)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return torch.from_numpy(samples).unsqueeze(0), sample_rate

    return _audio_read_with_soundfile(filepath)


def _audio_read_with_soundfile(filepath: Path) -> tuple[torch.Tensor, int]:
    # For non-WAV and non-16-bit WAV formats, use soundfile (optional dependency)
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV or non-16-bit WAV audio files. "
            "Install with: `pip install soundfile` or `uvx --with soundfile`"
        ) from e

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    return wav, sample_rate


def chunk_to_pcm_bytes(audio_chunk: torch.Tensor) -> bytes:
    """Convert a float32 audio chunk in [-1, 1] to int16 little-endian PCM bytes.

    Shared by the WAV writer and the raw-PCM streaming path so the
    quantization rule lives in exactly one place.
    """
    chunk_int16 = (audio_chunk.clamp(-1, 1) * 32767).short()
    return chunk_int16.detach().cpu().numpy().tobytes()


class StreamingWAVWriter:
    """WAV writer using Python's standard library wave module."""

    def __init__(self, output_stream: BinaryIO, sample_rate: int):
        self.output_stream = output_stream
        self.sample_rate = sample_rate
        self.wave_writer: wave.Wave_write | None = None
        self.first_chunk_buffer: list[bytes] | None = []
        self.is_seekable = _is_seekable(output_stream)

    @property
    def _writer(self) -> wave.Wave_write:
        if self.wave_writer is None:
            raise RuntimeError("write_header() must be called before writing audio")
        return self.wave_writer

    def write_header(self, sample_rate: int):
        """Initialize WAV writer with header."""
        # For stdout streaming, we need to handle the unseekable stream case
        # The wave module supports unseekable streams since Python 3.4
        # Closed with the underlying stream by the caller's `with f:` block.
        self.wave_writer = wave.open(self.output_stream, "wb")  # noqa: SIM115
        self.wave_writer.setnchannels(1)  # Mono
        self.wave_writer.setsampwidth(2)  # 16-bit
        self.wave_writer.setframerate(sample_rate)
        if not self.is_seekable:
            self.wave_writer.setnframes(1_000_000_000)

    def write_pcm_data(self, audio_chunk: torch.Tensor):
        """Write PCM data using wave module."""
        chunk_bytes = chunk_to_pcm_bytes(audio_chunk)

        if self.first_chunk_buffer is not None:
            self.first_chunk_buffer.append(chunk_bytes)
            total_length = sum(len(c) for c in self.first_chunk_buffer)
            target_length = (
                int(self.sample_rate * FIRST_CHUNK_LENGTH_SECONDS) * 2
            )  # 2 bytes per sample
            if total_length < target_length:
                return
            self._flush()
            return

        # Use writeframesraw to avoid frame count validation for streaming
        self._writer.writeframesraw(chunk_bytes)

    def _flush(self):
        if self.first_chunk_buffer is not None:
            self._writer.writeframesraw(b"".join(self.first_chunk_buffer))
            self.first_chunk_buffer = None

    def finalize(self):
        """Close the wave writer."""
        self._flush()

        # Let's add 200ms of silence to ensure proper playback
        silence_duration_sec = 0.2
        num_silence_samples = int(self.sample_rate * silence_duration_sec)

        writer = self._writer
        writer.writeframesraw(bytes(num_silence_samples * 2))

        if not self.is_seekable:
            # do not update the header for unseekable streams
            writer._patchheader = lambda: None  # ty: ignore[unresolved-attribute]
        writer.close()


def downsample_audio_chunk(audio_chunk: torch.Tensor, from_rate: int, to_rate: int) -> torch.Tensor:
    """Resample a streaming audio chunk via ``scipy.signal.resample_poly``.

    Per-chunk polyphase resampling introduces small discontinuities at chunk
    boundaries (the FIR's state doesn't carry across chunks). For telephony
    pipelines whose downstream encoder/carrier already band-limits to ~3.4 kHz
    these artifacts are inaudible. For full-fidelity offline use, resample the
    whole signal before chunking.
    """
    if from_rate == to_rate:
        return audio_chunk

    from scipy.signal import resample_poly

    gcd = math.gcd(int(from_rate), int(to_rate))
    up = int(to_rate) // gcd
    down = int(from_rate) // gcd

    wav_np = audio_chunk.detach().cpu().numpy()
    resampled_np = resample_poly(wav_np, up, down, axis=-1)
    return torch.from_numpy(resampled_np).to(audio_chunk.device).to(audio_chunk.dtype)


def stream_raw_pcm_chunks(
    file_like: BinaryIO,
    audio_chunks: Iterator[torch.Tensor],
    source_sample_rate: int,
    target_sample_rate: int | None = None,
) -> None:
    """Stream audio chunks as raw int16 little-endian PCM bytes — no header, no padding.

    Mono, 16-bit signed little-endian. The sample rate is determined by the
    model (``source_sample_rate``) and must be communicated out-of-band since
    raw PCM carries no format metadata.

    If ``target_sample_rate`` is given and differs from the source rate, each
    chunk is resampled in-line via ``downsample_audio_chunk``. This is the
    intended low-latency path for telephony bridges that want 8 kHz PCM
    delivered directly (skipping a client-side resampler / ffmpeg).

    Counterpart to ``stream_audio_chunks`` for callers that want to skip the
    WAV envelope.
    """
    needs_resample = target_sample_rate is not None and target_sample_rate != source_sample_rate
    with file_like:
        for chunk in audio_chunks:
            if needs_resample:
                chunk = downsample_audio_chunk(chunk, source_sample_rate, target_sample_rate)
            file_like.write(chunk_to_pcm_bytes(chunk))


def is_file_like(obj: object) -> TypeIs[BinaryIO]:
    """Check if object has basic file-like methods."""
    return all(hasattr(obj, attr) for attr in ["write", "close"])


def _is_seekable(obj: object) -> bool:
    seekable = getattr(obj, "seekable", None)
    if seekable is not None:
        try:
            return bool(seekable())
        except OSError:
            return False
    return all(hasattr(obj, attr) for attr in ("seek", "tell"))


def stream_audio_chunks(
    path: str | Path | BinaryIO | None, audio_chunks: Iterator[torch.Tensor], sample_rate: int
):
    """Stream audio chunks to a WAV file or stdout, optionally playing them."""
    f: BinaryIO | nullcontext[None]
    if path == "-":
        f = sys.stdout.buffer
    elif path is None:
        f = nullcontext()
    elif is_file_like(path):
        f = path
    else:
        f = open(path, "wb")  # noqa: SIM115  -- closed by the `with f:` below

    with f:
        if path is not None:
            assert not isinstance(f, nullcontext)
            writer = StreamingWAVWriter(f, sample_rate)
            writer.write_header(sample_rate)

        for chunk in audio_chunks:
            # Then write to file
            if path is not None:
                writer.write_pcm_data(chunk)

        if path is not None:
            writer.finalize()

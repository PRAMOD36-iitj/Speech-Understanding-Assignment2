"""
Part III: Zero-Shot Cross-Lingual Voice Cloning (TTS)
Tasks:
  3.1  Voice Embedding Extraction (x-vector / d-vector)
  3.2  Prosody Warping via DTW (F0 + Energy mapping)
  3.3  Synthesis with VITS / YourTTS (22.05 kHz+)
All code: Python + PyTorch only
"""

import os
import json
import math
import re
import unicodedata
import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.interpolate as interp
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional, Tuple
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────
# Utility: Audio I/O
# ─────────────────────────────────────────────────────────────────

def load_audio_mono(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    return audio.astype(np.float32), sr


def ensure_min_tts_sample_rate(requested_sr: int, minimum_sr: int = 22050) -> int:
    """Part III requires output audio at 22.05 kHz or higher."""
    return max(int(requested_sr), int(minimum_sr))


def prepare_exact_reference_audio(
    audio: np.ndarray,
    sr: int,
    target_duration_s: float = 60.0,
) -> np.ndarray:
    """
    Task 3.1 requires exactly 60 seconds of the student's voice.
    We keep the first target_duration_s seconds and fail if the recording
    is too short to satisfy the assignment contract.
    """
    target_samples = int(round(target_duration_s * sr))
    if len(audio) < target_samples:
        raise ValueError(
            f"Student reference is {len(audio)/sr:.2f}s, but Part III requires "
            f"at least {target_duration_s:.2f}s."
        )
    ref = audio[:target_samples]
    ref = ref / (np.max(np.abs(ref)) + 1e-8)
    return ref.astype(np.float32)


def backend_supports_speecht5() -> bool:
    """Check whether the pretrained SpeechT5 stack is importable."""
    try:
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan  # noqa: F401
        from speechbrain.inference.classifiers import EncoderClassifier  # noqa: F401
        from speechbrain.utils.fetching import LocalStrategy  # noqa: F401
        return True
    except Exception:
        return False


_SMALL_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS_NUMBER_WORDS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}


def integer_to_words(value: int) -> str:
    """Compact integer-to-words conversion for TTS prompt cleanup."""
    value = int(value)
    if value < 0:
        return f"minus {integer_to_words(-value)}"
    if value < 20:
        return _SMALL_NUMBER_WORDS[value]
    if value < 100:
        tens = (value // 10) * 10
        remainder = value % 10
        return _TENS_NUMBER_WORDS[tens] if remainder == 0 else f"{_TENS_NUMBER_WORDS[tens]} {_SMALL_NUMBER_WORDS[remainder]}"
    if value < 1000:
        head = value // 100
        remainder = value % 100
        prefix = f"{_SMALL_NUMBER_WORDS[head]} hundred"
        return prefix if remainder == 0 else f"{prefix} {integer_to_words(remainder)}"
    if value < 1_000_000:
        head = value // 1000
        remainder = value % 1000
        prefix = f"{integer_to_words(head)} thousand"
        return prefix if remainder == 0 else f"{prefix} {integer_to_words(remainder)}"
    if value < 1_000_000_000:
        head = value // 1_000_000
        remainder = value % 1_000_000
        prefix = f"{integer_to_words(head)} million"
        return prefix if remainder == 0 else f"{prefix} {integer_to_words(remainder)}"
    return str(value)


def numeric_token_to_words(token: str) -> str:
    """Expand digits into words for more stable SpeechT5 pronunciation."""
    token = token.strip()
    if not token:
        return ""
    clean = token.replace(",", "")
    if "." in clean:
        whole, frac = clean.split(".", 1)
        if whole and frac and whole.isdigit() and frac.isdigit():
            frac_words = " ".join(_SMALL_NUMBER_WORDS[int(ch)] for ch in frac)
            return f"{integer_to_words(int(whole))} point {frac_words}"
    if clean.isdigit():
        return integer_to_words(int(clean))
    return token


def romanize_for_tts(text: str) -> str:
    """
    Convert Gondi romanization with diacritics into ASCII-friendly text that
    pretrained English-centric TTS tokenizers can process more reliably.
    """
    replacements = {
        "ā": "aa", "ī": "ii", "ū": "uu", "ē": "ee", "ō": "oo",
        "ṛ": "ri", "ṝ": "rii", "ḍ": "d", "ṭ": "t", "ṇ": "n",
        "ṅ": "ng", "ñ": "ny", "ṃ": "m", "ṁ": "m", "ḥ": "h",
        "ś": "sh", "ṣ": "sh", "ç": "c",
    }
    normalized = text or ""
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst).replace(src.upper(), dst.upper())
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("’", "'").replace("`", "'")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("@", " at ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("...", ". ")
    normalized = re.sub(r"(?<=\w)-(?=\w)", " ", normalized)
    normalized = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"{numeric_token_to_words(m.group(1))} percent", normalized)
    normalized = re.sub(r"\b\d+\.\d+\b", lambda m: numeric_token_to_words(m.group(0)), normalized)
    normalized = re.sub(r"\b\d+\b", lambda m: numeric_token_to_words(m.group(0)), normalized)
    normalized = re.sub(r"[^A-Za-z0-9 ,.;:?!'\-]", " ", normalized)
    normalized = re.sub(r"\s*-\s*", " ", normalized)
    normalized = re.sub(r"([,;:?!])", r" \1 ", normalized)
    normalized = re.sub(r"\.(?=\w)", ". ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized and normalized[-1] not in ".?!":
        normalized = f"{normalized}."
    return normalized


def apply_segment_fade(audio: np.ndarray, sr: int, fade_ms: float = 15.0) -> np.ndarray:
    """Reduce boundary clicks without changing segment duration."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 4:
        return audio
    fade_samples = min(int(round(sr * fade_ms / 1000.0)), max(1, len(audio) // 8))
    if fade_samples <= 1:
        return audio
    envelope = np.ones(len(audio), dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    envelope[:fade_samples] = ramp
    envelope[-fade_samples:] = ramp[::-1]
    return (audio * envelope).astype(np.float32)


def finalize_output_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Light cleanup for the concatenated lecture waveform."""
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if len(audio) == 0:
        return audio
    audio = audio - float(np.mean(audio))
    if len(audio) > sr // 2:
        sos = signal.butter(2, 40.0, btype="highpass", fs=sr, output="sos")
        audio = signal.sosfiltfilt(sos, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-8
    return (0.97 * audio / peak).astype(np.float32)


def build_part3_run_diagnostics(
    *,
    mcd_proxy_db: float,
    mcd_target_db: float,
    reused_content: bool,
    sample_rate_hz: int,
    output_duration_s: float,
    target_duration_s: float,
) -> Dict[str, object]:
    """
    Build a clear status summary so Part III outputs explicitly show whether
    the next priority is improving synthesis quality or improving upstream
    transcript/translation coverage to avoid repeated content.
    """
    quality_needs_improvement = mcd_proxy_db >= mcd_target_db
    content_needs_full_length_source = reused_content
    sample_rate_requirement_met = sample_rate_hz >= 22050
    duration_requirement_met = abs(output_duration_s - target_duration_s) < 0.25

    if quality_needs_improvement and content_needs_full_length_source:
        next_step = (
            "Improve Part III audio quality further, and fix Part II/Part I so the "
            "10-minute Part III lecture has real full-length content instead of repeated segments."
        )
    elif quality_needs_improvement:
        next_step = "Improve Part III audio quality further."
    elif content_needs_full_length_source:
        next_step = (
            "Fix Part II/Part I so the 10-minute Part III lecture has real full-length "
            "content instead of repeated segments."
        )
    else:
        next_step = "Part III structure looks aligned; keep validating content and quality."

    status_lines = [
        f"Part III quality target (<{mcd_target_db:.1f} dB) met: {not quality_needs_improvement}",
        f"Part III uses repeated content to fill lecture target: {content_needs_full_length_source}",
        f"Part III sample-rate requirement met: {sample_rate_requirement_met}",
        f"Part III duration target met: {duration_requirement_met}",
        f"Next recommended focus: {next_step}",
    ]
    return {
        "quality_needs_improvement": quality_needs_improvement,
        "content_needs_full_length_source": content_needs_full_length_source,
        "sample_rate_requirement_met": sample_rate_requirement_met,
        "duration_requirement_met": duration_requirement_met,
        "next_step_recommendation": next_step,
        "status_lines": status_lines,
    }


def print_part3_run_diagnostics(diagnostics: Dict[str, object]) -> None:
    """Print a short, readable Part III status block after synthesis."""
    print("\n[Status] Part III run summary")
    for line in diagnostics.get("status_lines", []):
        print(f"  - {line}")


# ─────────────────────────────────────────────────────────────────
# Task 3.1  Speaker Embedding Extraction (d-vector / x-vector)
# ─────────────────────────────────────────────────────────────────

class SpeakerEncoder(nn.Module):
    """
    LSTM-based d-vector speaker encoder.
    Architecture: 3-layer LSTM → attention pooling → L2-normalized embedding.
    Input:  (B, T, n_mels) log-Mel frames
    Output: (B, emb_dim) L2-normalised d-vector

    For production use, replace with a pretrained ECAPA-TDNN or
    ResNet34-SE from speechbrain/wespeaker — the interface is identical.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 768,
        num_layers: int = 3,
        emb_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        # Attention pooling (self-attention over T)
        self.attn_fc = nn.Linear(hidden_size, 1)
        # Projection to d-vector
        self.proj = nn.Linear(hidden_size, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_mels) → (B, emb_dim)"""
        out, _ = self.lstm(x)        # (B, T, H)
        attn = torch.softmax(self.attn_fc(out), dim=1)   # (B, T, 1)
        pooled = (out * attn).sum(dim=1)                 # (B, H)
        emb = self.norm(self.proj(pooled))               # (B, emb_dim)
        return F.normalize(emb, p=2, dim=-1)             # L2 norm


def extract_mel_features(
    audio: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
) -> torch.Tensor:
    """Return (1, T, n_mels) log-Mel tensor."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        fmin=0, fmax=8000,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = torch.tensor(log_mel.T, dtype=torch.float32).unsqueeze(0)  # (1,T,80)
    return log_mel


def extract_speaker_embedding(
    audio: np.ndarray,
    sr: int,
    model: SpeakerEncoder,
    chunk_sec: float = 3.0,
) -> np.ndarray:
    """
    Extract a d-vector from student reference audio.
    Averages embeddings over non-overlapping 3-second chunks for stability.
    """
    model.eval()
    chunk_samples = int(chunk_sec * sr)
    embeddings = []

    for start in range(0, len(audio), chunk_samples):
        chunk = audio[start: start + chunk_samples]
        if len(chunk) < sr // 4:   # skip very short tail
            continue
        mel = extract_mel_features(chunk, sr=sr)
        with torch.no_grad():
            emb = model(mel)         # (1, emb_dim)
        embeddings.append(emb.squeeze(0).cpu().numpy())

    if not embeddings:
        raise ValueError("Audio too short to extract embedding.")

    mean_emb = np.mean(embeddings, axis=0)
    # Re-normalise
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    return mean_emb   # (emb_dim,)


# ─────────────────────────────────────────────────────────────────
# Task 3.2  Prosody Warping via Dynamic Time Warping (DTW)
# ─────────────────────────────────────────────────────────────────

def extract_f0(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 600.0,
) -> np.ndarray:
    """
    Extract fundamental frequency (F0) using the PYIN algorithm.
    Voiced frames have F0 > 0; unvoiced frames are 0.
    Returns F0 contour array of shape (T,).
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0,
    )
    f0 = np.where(voiced_flag, f0, 0.0)
    return f0.astype(np.float32)


def extract_energy(
    audio: np.ndarray,
    sr: int,
    hop_length: int = 256,
    n_fft: int = 1024,
) -> np.ndarray:
    """
    Extract RMS energy contour (per frame).
    Returns energy array of shape (T,).
    """
    energy = librosa.feature.rms(
        y=audio, frame_length=n_fft, hop_length=hop_length
    )[0]
    return energy.astype(np.float32)


def normalize_translated_segments(
    translated_segments: List[Dict],
    default_duration_s: float = 8.0,
) -> List[Dict]:
    """
    Ensure every translated segment has usable timing. If timestamps are
    missing or invalid, segments are laid out sequentially with a default
    duration so Part III can still build a lecture timeline.
    """
    normalized: List[Dict] = []
    cursor = 0.0
    for seg in translated_segments:
        seg_copy = dict(seg)
        start = seg_copy.get("start")
        end = seg_copy.get("end")
        valid_times = isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start
        if valid_times:
            start = float(start)
            end = float(end)
        else:
            start = cursor
            end = start + default_duration_s
        cursor = end
        seg_copy["start"] = round(start, 3)
        seg_copy["end"] = round(end, 3)
        normalized.append(seg_copy)
    return normalized


def expand_segments_to_target_duration(
    translated_segments: List[Dict],
    target_duration_s: float,
    default_duration_s: float = 8.0,
) -> List[Dict]:
    """
    Build a lecture schedule that fills the requested duration. When earlier
    stages only provide a short translated excerpt, we cycle that excerpt to
    produce a full Part III lecture-length output while keeping segment-level
    timing explicit.
    """
    base_segments = normalize_translated_segments(
        translated_segments,
        default_duration_s=default_duration_s,
    )
    if not base_segments:
        return []

    first_start = float(base_segments[0]["start"])
    last_end = float(base_segments[-1]["end"])
    timeline_coverage = max(0.0, last_end - first_start)
    if timeline_coverage >= target_duration_s - 1e-6:
        scheduled: List[Dict] = []
        for base_idx, seg in enumerate(base_segments):
            start = max(0.0, float(seg["start"]) - first_start)
            end = min(target_duration_s, float(seg["end"]) - first_start)
            if end <= start:
                continue
            seg_copy = dict(seg)
            seg_copy["start"] = round(start, 3)
            seg_copy["end"] = round(end, 3)
            seg_copy["source_segment_index"] = base_idx
            seg_copy["cycle_index"] = 0
            seg_copy["reused_content"] = False
            scheduled.append(seg_copy)
            if end >= target_duration_s - 1e-6:
                break
        return scheduled

    expanded: List[Dict] = []
    cursor = 0.0
    cycle_idx = 0
    while cursor < target_duration_s - 1e-6:
        for base_idx, seg in enumerate(base_segments):
            seg_duration = max(float(seg["end"]) - float(seg["start"]), 0.1)
            start = cursor
            end = min(target_duration_s, start + seg_duration)
            seg_copy = dict(seg)
            seg_copy["start"] = round(start, 3)
            seg_copy["end"] = round(end, 3)
            seg_copy["source_segment_index"] = base_idx
            seg_copy["cycle_index"] = cycle_idx
            seg_copy["reused_content"] = cycle_idx > 0
            expanded.append(seg_copy)
            cursor = end
            if cursor >= target_duration_s - 1e-6:
                break
        cycle_idx += 1
    return expanded


class DTWProsodyWarper:
    """
    Dynamic Time Warping-based prosody mapper.

    Given:
      - source prosody (professor's F0 + energy)
      - synthesised prosody (flat or initial TTS output)
    Produces a time-warped prosody that preserves the teaching style of
    the professor while adapting to the synthesised speech duration.

    Mathematical formulation:
      DTW finds the optimal monotone alignment path W* = {(i,j)} that
      minimises the accumulated cost:
        D(i,j) = d(f0_src[i], f0_syn[j])
                + min(D(i-1,j), D(i,j-1), D(i-1,j-1))
      where d(·,·) is a combined F0+energy distance:
        d(a,b) = |log(a_f0+ε) - log(b_f0+ε)| + γ·|a_E - b_E|

    The warped F0 is then re-applied to the synthesised waveform by
    modifying the PSOLA (Pitch Synchronous Overlap Add) parameters.
    """

    def __init__(self, energy_weight: float = 0.3):
        self.energy_weight = energy_weight

    def _dtw_distance_matrix(
        self,
        f0_a: np.ndarray,
        energy_a: np.ndarray,
        f0_b: np.ndarray,
        energy_b: np.ndarray,
    ) -> np.ndarray:
        """Compute combined F0+energy cost matrix."""
        Na, Nb = len(f0_a), len(f0_b)
        cost = np.zeros((Na, Nb), dtype=np.float32)

        log_f0_a = np.log(f0_a + 1.0)
        log_f0_b = np.log(f0_b + 1.0)

        for i in range(Na):
            for j in range(Nb):
                f0_dist = abs(log_f0_a[i] - log_f0_b[j])
                e_dist = abs(energy_a[i] - energy_b[j])
                cost[i, j] = f0_dist + self.energy_weight * e_dist

        return cost

    def _dtw_fast(
        self,
        f0_a: np.ndarray,
        energy_a: np.ndarray,
        f0_b: np.ndarray,
        energy_b: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Fast vectorized DTW with traceback.
        Returns accumulated cost matrix and optimal path.
        """
        Na, Nb = len(f0_a), len(f0_b)
        log_f0_a = np.log(f0_a + 1.0).reshape(-1, 1)
        log_f0_b = np.log(f0_b + 1.0).reshape(1, -1)
        energy_a = energy_a.reshape(-1, 1)
        energy_b = energy_b.reshape(1, -1)

        local_cost = (
            np.abs(log_f0_a - log_f0_b)
            + self.energy_weight * np.abs(energy_a - energy_b)
        )  # (Na, Nb)

        # Accumulated cost
        D = np.full((Na, Nb), np.inf, dtype=np.float32)
        D[0, 0] = local_cost[0, 0]

        for i in range(1, Na):
            D[i, 0] = D[i - 1, 0] + local_cost[i, 0]
        for j in range(1, Nb):
            D[0, j] = D[0, j - 1] + local_cost[0, j]
        for i in range(1, Na):
            for j in range(1, Nb):
                D[i, j] = local_cost[i, j] + min(
                    D[i - 1, j],
                    D[i, j - 1],
                    D[i - 1, j - 1],
                )

        # Traceback
        path = [(Na - 1, Nb - 1)]
        i, j = Na - 1, Nb - 1
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                best = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
                if best == 0:
                    i -= 1
                elif best == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1
            path.append((i, j))

        path.reverse()
        return D, path

    def warp_prosody(
        self,
        f0_source: np.ndarray,       # professor's F0 (T_src,)
        energy_source: np.ndarray,   # professor's energy (T_src,)
        f0_target: np.ndarray,       # synthesised speech F0 (T_tgt,)
        energy_target: np.ndarray,   # synthesised energy (T_tgt,)
        target_audio: np.ndarray,    # synthesised waveform
        sr: int,
        hop_length: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply DTW-warped professor prosody to synthesised speech.

        Returns:
          warped_f0      (T_tgt,)  - warped F0 contour
          warped_energy  (T_tgt,)  - warped energy contour
          warped_audio   (samples,) - prosody-modified audio
        """
        # Align source length to target via DTW
        _, path = self._dtw_fast(
            f0_source, energy_source,
            f0_target, energy_target,
        )

        # Build mapping: target frame → warped source F0/energy
        tgt_to_src_f0 = np.zeros(len(f0_target))
        tgt_to_src_e = np.zeros(len(energy_target))
        count = np.zeros(len(f0_target))

        for src_i, tgt_j in path:
            if tgt_j < len(f0_target):
                tgt_to_src_f0[tgt_j] += f0_source[src_i]
                tgt_to_src_e[tgt_j] += energy_source[src_i]
                count[tgt_j] += 1

        count = np.where(count == 0, 1, count)
        warped_f0 = tgt_to_src_f0 / count
        warped_energy = tgt_to_src_e / count

        # Smooth warped contours
        warped_f0 = self._smooth(warped_f0, window=7)
        warped_energy = self._smooth(warped_energy, window=7)

        # Apply warped F0 to audio via PSOLA-inspired pitch shifting
        warped_audio = self._apply_f0_warping(
            target_audio, f0_target, warped_f0, sr, hop_length
        )

        # Apply energy envelope scaling
        warped_audio = self._apply_energy_scaling(
            warped_audio, energy_target, warped_energy, sr, hop_length
        )

        return warped_f0, warped_energy, warped_audio

    @staticmethod
    def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
        """Hanning-windowed moving average smoothing."""
        if len(arr) < window:
            return arr
        kernel = np.hanning(window)
        kernel /= kernel.sum()
        return np.convolve(arr, kernel, mode="same")

    @staticmethod
    def _apply_f0_warping(
        audio: np.ndarray,
        original_f0: np.ndarray,
        warped_f0: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> np.ndarray:
        """
        Segment-level pitch shifting to match median warped F0 contour.
        Avoids frame-by-frame phase vocoder artifacts (clicking/phasiness).
        """
        valid = (original_f0 > 50) & (warped_f0 > 50)
        if not np.any(valid):
            return audio
        
        ratio = np.median(warped_f0[valid]) / np.median(original_f0[valid])
        n_steps = 12 * math.log2(max(ratio, 0.5))
        n_steps = np.clip(n_steps, -4, 4)  # keep the shift gentle and realistic
        
        try:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=float(n_steps))
        except Exception:
            return audio

    @staticmethod
    def _apply_energy_scaling(
        audio: np.ndarray,
        original_energy: np.ndarray,
        warped_energy: np.ndarray,
        sr: int,
        hop_length: int,
    ) -> np.ndarray:
        """
        Scale each frame's amplitude so its energy matches the warped target.
        Gain = sqrt(E_warped / E_original), clipped to prevent noise blowup.
        """
        output = audio.copy()
        n_frames = min(len(original_energy), len(warped_energy))

        for t in range(n_frames):
            start = t * hop_length
            end = min(start + hop_length, len(audio))
            orig_e = original_energy[t] + 1e-8
            warp_e = warped_energy[t] + 1e-8
            
            # Do not amplify silence
            if orig_e > 1e-3:
                gain = np.clip(np.sqrt(warp_e / orig_e), 0.5, 2.0)
            else:
                gain = 1.0
                
            output[start:end] = output[start:end] * gain

        return output


# ─────────────────────────────────────────────────────────────────
# Task 3.3  Zero-Shot Voice Cloning Synthesis (VITS / YourTTS)
# ─────────────────────────────────────────────────────────────────

class GondiPhonemeVocabulary:
    """
    Minimal phoneme vocabulary for Gondi TTS.
    Maps IPA symbols to integer IDs for the VITS text encoder.
    """

    # Gondi phoneme inventory (Dravidian + borrowings)
    PHONEMES = [
        "<pad>", "<bos>", "<eos>", "<unk>", " ",
        # Vowels
        "ə", "a", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː", "oː", "ɔː",
        "ɑː", "æ", "ɛ", "ɒ",
        # Nasalized
        "ã", "ĩː", "ũː",
        # Consonants – plosives
        "p", "pʰ", "b", "bʱ",
        "t̪", "t̪ʰ", "d̪", "d̪ʱ",
        "ʈ", "ʈʰ", "ɖ", "ɖʱ",
        "k", "kʰ", "ɡ", "ɡʱ",
        # Affricates
        "tʃ", "tʃʰ", "dʒ", "dʒʱ",
        # Nasals
        "m", "n", "ɳ", "ɲ", "ŋ",
        # Fricatives
        "f", "s", "z", "ʃ", "ʂ", "ɦ", "h", "x",
        # Liquids & glides
        "l", "ɭ", "ɾ", "ɽ", "j", "ʋ", "w",
        # Boundary marker
        "|",
    ]

    def __init__(self):
        self.ph2id = {ph: i for i, ph in enumerate(self.PHONEMES)}
        self.id2ph = {i: ph for ph, i in self.ph2id.items()}

    def encode(self, ipa_string: str) -> List[int]:
        """
        Convert an IPA string to a list of token IDs.
        Greedy longest-match tokenisation.
        """
        ids = [self.ph2id["<bos>"]]
        i = 0
        while i < len(ipa_string):
            matched = False
            # Try longest match (up to 4 chars for digraphs/diacritics)
            for length in [4, 3, 2, 1]:
                chunk = ipa_string[i: i + length]
                if chunk in self.ph2id:
                    ids.append(self.ph2id[chunk])
                    i += length
                    matched = True
                    break
            if not matched:
                ids.append(self.ph2id.get("<unk>", 3))
                i += 1
        ids.append(self.ph2id["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id2ph.get(i, "?") for i in ids
                       if i not in [0, 1, 2])


class VITSTextEncoder(nn.Module):
    """
    Lightweight VITS-style text encoder.
    Input:  integer token sequence (B, L)
    Output: (B, L, d_model) hidden representations + (B, d_model) global
    Architecture: Embedding → 4× Transformer layers → Linear projection
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        nhead: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, tokens: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        mask = None
        if lengths is not None:
            mask = torch.arange(tokens.size(1), device=tokens.device)[None] >= lengths[:, None]
        x = self.pos_enc(self.embed(tokens))
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DurationPredictor(nn.Module):
    """Predicts phoneme durations (frames) for upsampling text to mel-time."""

    def __init__(self, d_model: int = 192, hidden: int = 256, kernel: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, hidden, kernel, padding=kernel // 2)
        self.norm1 = nn.LayerNorm(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2)
        self.norm2 = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) → durations (B, L) in log-scale."""
        h = F.relu(self.norm1(self.conv1(x.transpose(1, 2)).transpose(1, 2)))
        h = F.relu(self.norm2(self.conv2(h.transpose(1, 2)).transpose(1, 2)))
        return F.softplus(self.proj(h).squeeze(-1))   # (B, L), strictly positive


class SpeakerConditionedDecoder(nn.Module):
    """
    Mel-spectrogram decoder conditioned on speaker embedding.
    Converts text-encoded, duration-upsampled representation to 80-band mel.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_mels: int = 80,
        spk_emb_dim: int = 256,
        nhead: int = 2,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spk_proj = nn.Linear(spk_emb_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mel_proj = nn.Linear(d_model, n_mels)

    def forward(
        self,
        hidden: torch.Tensor,    # (B, T, d_model) — upsampled text
        spk_emb: torch.Tensor,   # (B, spk_emb_dim) or (1, spk_emb_dim)
    ) -> torch.Tensor:
        """Returns mel spectrogram (B, T, n_mels)."""
        # Broadcast speaker embedding across time
        spk = self.spk_proj(spk_emb).unsqueeze(1)   # (B, 1, d_model)
        x = hidden + spk                             # FiLM-lite conditioning
        x = self.decoder(x)
        return self.mel_proj(x)                      # (B, T, n_mels)


class GriffinLimVocoder:
    """
    Griffin-Lim vocoder: converts mel spectrogram → waveform.
    Requires NO neural network; deterministic inversion via iterative STFT.
    For production, replace with a pretrained HiFi-GAN neural vocoder.
    """

    def __init__(
        self,
        sr: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        n_iter: int = 64,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_iter = n_iter
        # Mel filterbank (inverse)
        self.mel_fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=8000
        )  # (n_mels, 1+n_fft//2)

    def mel_to_linear(self, mel_db: np.ndarray) -> np.ndarray:
        """Invert mel-scale via pseudo-inverse of filterbank."""
        mel_power = librosa.db_to_power(mel_db)        # (T, n_mels)
        mel_power = mel_power.T                         # (n_mels, T)
        # Pseudo-inverse (Moore-Penrose)
        mel_fb_pinv = np.linalg.pinv(self.mel_fb)      # (1+n_fft//2, n_mels)
        linear = np.maximum(mel_fb_pinv @ mel_power, 0.0)  # (F, T)
        return linear

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """
        mel: (T, n_mels) log-mel in dB
        Returns: waveform (samples,) at self.sr
        """
        linear = self.mel_to_linear(mel)                # (F, T)
        audio = librosa.griffinlim(
            linear,
            n_iter=self.n_iter,
            hop_length=self.hop_length,
            win_length=self.n_fft,
        )
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)


class SpeechT5ZeroShotBackend:
    """
    Pretrained zero-shot voice cloning backend built from:
      - SpeechT5 text-to-speech + HiFi-GAN vocoder
      - SpeechBrain ECAPA speaker encoder

    This is the preferred Part III backend because it uses pretrained models
    instead of the untrained assignment-internal stack.
    """

    def __init__(
        self,
        target_sr: int = 22050,
        device: Optional[str] = None,
        tts_model_name: str = "microsoft/speecht5_tts",
        vocoder_name: str = "microsoft/speecht5_hifigan",
        speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
    ):
        self.target_sr = target_sr
        self.output_sr = 16000
        self.tts_model_name = tts_model_name
        self.vocoder_name = vocoder_name
        self.speaker_model_name = speaker_model_name
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.speaker_embedding_dim = 512

    def _load(self):
        if self.loaded:
            return

        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.processor = SpeechT5Processor.from_pretrained(self.tts_model_name)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.tts_model_name).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.vocoder_name).to(self.device)
        self.model.eval()
        self.vocoder.eval()

        run_opts = {"device": self.device}
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source=self.speaker_model_name,
            savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb"),
            local_strategy=LocalStrategy.COPY,
            run_opts=run_opts,
        )
        self.speaker_embedding_dim = int(self.model.config.speaker_embedding_dim)
        self.loaded = True

    def _adapt_speaker_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        expected = self.speaker_embedding_dim
        if emb.shape[-1] < expected:
            emb = F.pad(emb, (0, expected - emb.shape[-1]))
        elif emb.shape[-1] > expected:
            emb = emb[..., :expected]
        return F.normalize(emb, p=2, dim=-1)

    def extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        self._load()
        wave = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        if sr != self.output_sr:
            wave = torchaudio.functional.resample(wave, sr, self.output_sr)
        wave = wave.to(self.device)
        with torch.no_grad():
            emb = self.speaker_encoder.encode_batch(wave).squeeze(1)
            emb = self._adapt_speaker_embedding(emb)
        return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def synthesise_segment(self, text: str, speaker_embedding: np.ndarray) -> np.ndarray:
        self._load()
        prompt = romanize_for_tts(text)
        if not prompt:
            prompt = "lecture"

        inputs = self.processor(text=prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        speaker = torch.tensor(speaker_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            speech = self.model.generate_speech(
                input_ids,
                speaker,
                vocoder=self.vocoder,
            )

        audio = speech.detach().cpu().numpy().astype(np.float32)
        if self.output_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=self.output_sr, target_sr=self.target_sr)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)


class ZeroShotVoiceCloner:
    """
    Full zero-shot cross-lingual voice cloning pipeline.

    Given:
      - Gondi IPA transcript (from Part II)
      - Student 60s reference audio
      - Professor lecture audio (for prosody)
    Produces:
      - output_LRL_cloned.wav at ≥22.05 kHz
    """

    def __init__(
        self,
        spk_emb_dim: int = 256,
        d_model: int = 192,
        n_mels: int = 80,
        sr: int = 22050,
        backend: str = "auto",
    ):
        self.sr = sr
        self.requested_backend = backend
        self.backend_name = self._resolve_backend(backend)
        self.dtw_warper = DTWProsodyWarper(energy_weight=0.3)
        self.vocoder = GriffinLimVocoder(sr=sr, n_mels=n_mels)

        self.vocab = GondiPhonemeVocabulary()
        self.pretrained_backend: Optional[SpeechT5ZeroShotBackend] = None
        self.speaker_encoder = None
        self.text_encoder = None
        self.duration_predictor = None
        self.decoder = None

        if self.backend_name == "speecht5":
            self.pretrained_backend = SpeechT5ZeroShotBackend(target_sr=sr)
            self.speaker_embedding_dim = self.pretrained_backend.speaker_embedding_dim
        else:
            vocab_size = len(self.vocab.PHONEMES)
            self.speaker_encoder = SpeakerEncoder(
                n_mels=n_mels, hidden_size=768, num_layers=3, emb_dim=spk_emb_dim
            )
            self.text_encoder = VITSTextEncoder(
                vocab_size=vocab_size, d_model=d_model, nhead=2, num_layers=4
            )
            self.duration_predictor = DurationPredictor(d_model=d_model)
            self.decoder = SpeakerConditionedDecoder(
                d_model=d_model, n_mels=n_mels, spk_emb_dim=spk_emb_dim
            )

            # Evaluation mode (no training in this assignment pipeline)
            self.speaker_encoder.eval()
            self.text_encoder.eval()
            self.duration_predictor.eval()
            self.decoder.eval()
            self.speaker_embedding_dim = spk_emb_dim

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "speecht5":
            if not backend_supports_speecht5():
                raise RuntimeError("SpeechT5 backend requested but speechbrain/transformers support is unavailable.")
            return "speecht5"
        if backend == "auto":
            return "speecht5" if backend_supports_speecht5() else "internal"
        return "internal"

    def extract_reference_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.backend_name == "speecht5":
            assert self.pretrained_backend is not None
            return self.pretrained_backend.extract_speaker_embedding(audio, sr)
        if self.speaker_encoder is None:
            raise RuntimeError("Internal speaker encoder is not initialised.")
        return extract_speaker_embedding(audio, sr, self.speaker_encoder, chunk_sec=3.0)

    @staticmethod
    def _segment_prompt(seg: Dict) -> str:
        for key in ["gondi", "original", "text", "gondi_ipa"]:
            value = seg.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    @staticmethod
    def _is_vowel_phoneme(phoneme: str) -> bool:
        return phoneme in {
            "ə", "a", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː",
            "oː", "ɔː", "ɑː", "æ", "ɛ", "ɒ", "ã", "ĩː", "ũː",
        }

    def _duration_weights(self, token_ids: List[int]) -> np.ndarray:
        """
        Hand-tuned phoneme duration priors used when target segment timing
        is known. This avoids depending on the untrained duration predictor
        for overall lecture pacing.
        """
        weights = []
        for token_id in token_ids:
            phoneme = self.vocab.id2ph.get(token_id, "<unk>")
            if phoneme in {"<pad>", "<bos>", "<eos>"}:
                weight = 0.25
            elif phoneme == " ":
                weight = 0.35
            elif phoneme == "|":
                weight = 0.6
            elif self._is_vowel_phoneme(phoneme):
                weight = 1.8 if "ː" in phoneme else 1.4
            elif phoneme in {"m", "n", "ŋ", "ɲ", "ɳ", "l", "ɭ", "ɾ", "ɽ", "j", "ʋ", "w"}:
                weight = 1.1
            elif phoneme in {"s", "ʃ", "ʂ", "h", "ɦ", "f", "z", "x"}:
                weight = 0.9
            else:
                weight = 1.0
            weights.append(weight)
        return np.asarray(weights, dtype=np.float32)

    def _durations_from_target_duration(
        self,
        token_ids: List[int],
        target_duration_s: float,
        hop_length: int,
    ) -> torch.Tensor:
        """
        Convert a target segment duration into per-token frame counts.
        The sum is forced to match the requested duration as closely as
        possible while keeping each token at least one frame long.
        """
        total_frames = max(
            len(token_ids),
            int(round(target_duration_s * self.sr / hop_length)),
        )
        weights = self._duration_weights(token_ids)
        weight_sum = float(weights.sum()) + 1e-8
        raw = weights / weight_sum * total_frames
        dur_int = np.maximum(np.floor(raw).astype(np.int64), 1)

        frame_delta = total_frames - int(dur_int.sum())
        if frame_delta > 0:
            order = np.argsort(-(raw - dur_int))
            for step in range(frame_delta):
                dur_int[order[step % len(order)]] += 1
        elif frame_delta < 0:
            while frame_delta < 0:
                removable = np.where(dur_int > 1)[0]
                if len(removable) == 0:
                    break
                order = removable[np.argsort(raw[removable] - dur_int[removable])]
                for idx in order:
                    if frame_delta == 0:
                        break
                    if dur_int[idx] > 1:
                        dur_int[idx] -= 1
                        frame_delta += 1

        return torch.tensor(dur_int, dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def _decoder_output_to_mel_db(mel: torch.Tensor) -> torch.Tensor:
        """
        Map unconstrained decoder activations into a realistic log-mel dB
        range expected by Griffin-Lim inversion.
        """
        return torch.sigmoid(mel) * 80.0 - 80.0

    def _match_duration(self, audio: np.ndarray, target_duration_s: Optional[float]) -> np.ndarray:
        """Trim/pad a segment to the requested duration without time-stretching."""
        if target_duration_s is None or target_duration_s <= 0.0 or len(audio) < 2:
            return audio.astype(np.float32)

        target_samples = max(1, int(round(target_duration_s * self.sr)))
        if len(audio) > target_samples:
            # Gentle fade out if trimming
            fade_len = min(1024, target_samples)
            if fade_len > 0:
                audio[target_samples - fade_len:target_samples] *= np.linspace(1, 0, fade_len)
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        return audio.astype(np.float32)

    def _upsample_hidden(
        self, hidden: torch.Tensor, durations: torch.Tensor
    ) -> torch.Tensor:
        """
        Repeat each phoneme hidden state according to predicted duration.
        Implements the length regulator in VITS / FastSpeech.
        hidden:    (1, L, d_model)
        durations: (1, L)
        Returns:   (1, T, d_model) where T = sum(durations)
        """
        dur_int = torch.clamp(torch.round(durations), min=1).long().squeeze(0)  # (L,)
        upsampled = torch.repeat_interleave(
            hidden.squeeze(0), dur_int, dim=0
        )  # (T, d_model)
        return upsampled.unsqueeze(0)   # (1, T, d_model)

    @torch.no_grad()
    def synthesise_segment(
        self,
        ipa_text: str,
        spk_embedding: np.ndarray,
        target_duration_s: Optional[float] = None,
        hop_length: int = 256,
        surface_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesise one Gondi IPA segment using the student's voice embedding.
        Returns waveform (samples,).
        """
        if self.backend_name == "speecht5":
            assert self.pretrained_backend is not None
            prompt = surface_text or ipa_text
            audio = self.pretrained_backend.synthesise_segment(prompt, spk_embedding)
            return self._match_duration(audio, target_duration_s)

        # Encode text
        token_ids = self.vocab.encode(ipa_text)
        tokens = torch.tensor([token_ids], dtype=torch.long)   # (1, L)
        hidden = self.text_encoder(tokens)                      # (1, L, d_model)

        # Prefer timestamp-driven durations when segment timing is available.
        if target_duration_s is not None and target_duration_s > 0.0:
            durations = self._durations_from_target_duration(
                token_ids,
                target_duration_s=target_duration_s,
                hop_length=hop_length,
            )
        else:
            durations = self.duration_predictor(hidden) * 5.0   # (1, L)

        # Upsample
        upsampled = self._upsample_hidden(hidden, durations)    # (1, T, d_model)

        # Decode with speaker embedding
        spk = torch.tensor(spk_embedding, dtype=torch.float32).unsqueeze(0)  # (1, emb_dim)
        mel = self.decoder(upsampled, spk)                       # (1, T, n_mels)
        mel = self._decoder_output_to_mel_db(mel)
        mel_np = mel.squeeze(0).cpu().numpy()                    # (T, n_mels)

        # Invert mel → waveform
        audio = self.vocoder(mel_np)
        return self._match_duration(audio, target_duration_s)

    def synthesise_full_lecture(
        self,
        translated_segments: List[Dict],
        spk_embedding: np.ndarray,
        prof_audio: np.ndarray,
        prof_sr: int,
        hop_length: int = 256,
    ) -> np.ndarray:
        """
        Synthesise all Gondi segments and concatenate into a full lecture.
        Apply DTW prosody warping from the professor.
        """
        # Extract professor prosody once
        print("  Extracting professor F0 and energy...")
        prof_f0 = extract_f0(prof_audio, prof_sr, hop_length=hop_length)
        prof_energy = extract_energy(prof_audio, prof_sr, hop_length=hop_length)

        all_audio = []
        rendered_segments = []
        chunk_cache: Dict[Tuple[str, float], np.ndarray] = {}
        chunk_prosody_cache: Dict[Tuple[str, float], Tuple[np.ndarray, np.ndarray]] = {}
        progress = tqdm(
            translated_segments,
            desc="  Part III synthesis",
            unit="seg",
            dynamic_ncols=True,
            leave=False,
        )
        cache_hits = 0
        for i, seg in enumerate(progress):
            ipa_text = seg.get("gondi_ipa", seg.get("gondi", ""))
            surface_text = self._segment_prompt(seg)
            if self.backend_name == "speecht5":
                if not surface_text.strip():
                    continue
            elif not ipa_text.strip():
                continue

            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
            seg_duration = seg_end - seg_start if seg_end > seg_start else None
            preview_source = surface_text if self.backend_name == "speecht5" else ipa_text
            preview = preview_source[:40].encode("ascii", errors="replace").decode("ascii")
            cache_duration = round(seg_duration or -1.0, 3)
            cache_text = surface_text if self.backend_name == "speecht5" else ipa_text
            cache_key = (cache_text, cache_duration)

            print(f"  Segment {i+1}/{len(translated_segments)}: "
                  f"{preview}...")
            if cache_key in chunk_cache:
                chunk = chunk_cache[cache_key].copy()
                chunk_f0, chunk_energy = chunk_prosody_cache[cache_key]
                cache_hits += 1
            else:
                chunk = self.synthesise_segment(
                    ipa_text,
                    spk_embedding,
                    target_duration_s=seg_duration,
                    hop_length=hop_length,
                    surface_text=surface_text,
                )
                # DTW prosody warping operates on the synthesised segment's
                # local contour, so cache the pre-warp audio and features for
                # repeated lecture cycles.
                chunk_f0 = extract_f0(chunk, self.sr, hop_length=hop_length)
                chunk_energy = extract_energy(chunk, self.sr, hop_length=hop_length)
                chunk_cache[cache_key] = chunk.copy()
                chunk_prosody_cache[cache_key] = (chunk_f0, chunk_energy)

            # Prefer real segment timestamps when available; otherwise fall
            # back to proportional slicing across the professor reference.
            if seg_duration is not None:
                src_start = max(0, int(round(seg_start * prof_sr / hop_length)))
                src_end = min(len(prof_f0), int(round(seg_end * prof_sr / hop_length)))
            else:
                src_start = int(len(prof_f0) * (i / max(len(translated_segments), 1)))
                src_end = int(len(prof_f0) * ((i + 1) / max(len(translated_segments), 1)))
            prof_f0_seg = prof_f0[src_start:src_end]
            prof_energy_seg = prof_energy[src_start:src_end]

            # Skip warping if arrays are too short
            if len(prof_f0_seg) > 10 and len(chunk_f0) > 10:
                _, _, chunk = self.dtw_warper.warp_prosody(
                    prof_f0_seg, prof_energy_seg,
                    chunk_f0, chunk_energy,
                    chunk, self.sr, hop_length,
                )

            # Restore segment padding to guarantee exact DTW time alignment
            chunk = self._match_duration(chunk, seg_duration)
            chunk = apply_segment_fade(chunk, self.sr)
            all_audio.append(chunk)
            rendered_segments.append(seg)
            progress.set_postfix({
                "backend": self.backend_name,
                "cached": cache_hits,
                "done_s": round(sum(len(a) for a in all_audio) / self.sr, 1),
            })
        progress.close()

        if not all_audio:
            return np.zeros(self.sr, dtype=np.float32)

        # Preserve timestamp gaps when available; otherwise fall back to a
        # short constant pause between segments.
        timed_segments = all("start" in seg and "end" in seg for seg in rendered_segments)
        if timed_segments:
            target_end = max(float(seg.get("end", 0.0)) for seg in translated_segments)
            total_samples = max(1, int(round(target_end * self.sr)))
            full_audio = np.zeros(total_samples, dtype=np.float32)
            overlap_weight = np.zeros(total_samples, dtype=np.float32)
            for seg, audio_chunk in zip(rendered_segments, all_audio):
                seg_start = max(0.0, float(seg.get("start", 0.0)))
                start_sample = int(round(seg_start * self.sr))
                end_sample = min(total_samples, start_sample + len(audio_chunk))
                if end_sample <= start_sample:
                    continue
                chunk = audio_chunk[: end_sample - start_sample]
                full_audio[start_sample:end_sample] += chunk
                overlap_weight[start_sample:end_sample] += 1.0
            nonzero = overlap_weight > 0
            full_audio[nonzero] /= overlap_weight[nonzero]
        else:
            silence = np.zeros(int(0.05 * self.sr), dtype=np.float32)
            full_audio = all_audio[0]
            for a in all_audio[1:]:
                full_audio = np.concatenate([full_audio, silence, a])

        return finalize_output_audio(full_audio, self.sr)


# ─────────────────────────────────────────────────────────────────
# MCD Evaluation
# ─────────────────────────────────────────────────────────────────

def compute_mcd(
    ref_audio: np.ndarray,
    syn_audio: np.ndarray,
    sr: int,
    n_mfcc: int = 24,
    hop_length: int = 256,
    n_fft: int = 1024,
) -> float:
    """
    Mel-Cepstral Distortion (MCD) between reference and synthesised speech.
    MCD = (10 / ln 10) * sqrt(2 * sum_{k=1}^{K} (mc_ref[k] - mc_syn[k])^2)
    K = n_mfcc - 1  (exclude 0th coefficient)
    Lower is better; target < 8.0 dB.
    """
    def get_mfcc(audio):
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc,
            hop_length=hop_length, n_fft=n_fft,
        )
        return mfcc[1:]   # drop 0th coeff; (K, T)

    mfcc_ref = get_mfcc(ref_audio)   # (K, T_ref)
    mfcc_syn = get_mfcc(syn_audio)   # (K, T_syn)

    # DTW align
    T_ref = mfcc_ref.shape[1]
    T_syn = mfcc_syn.shape[1]
    D = np.full((T_ref, T_syn), np.inf)
    D[0, 0] = np.sum((mfcc_ref[:, 0] - mfcc_syn[:, 0]) ** 2)

    for i in range(1, T_ref):
        D[i, 0] = D[i-1, 0] + np.sum((mfcc_ref[:, i] - mfcc_syn[:, 0]) ** 2)
    for j in range(1, T_syn):
        D[0, j] = D[0, j-1] + np.sum((mfcc_ref[:, 0] - mfcc_syn[:, j]) ** 2)
    for i in range(1, T_ref):
        for j in range(1, T_syn):
            cost = np.sum((mfcc_ref[:, i] - mfcc_syn[:, j]) ** 2)
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])

    # Traceback to get aligned pairs
    i, j = T_ref - 1, T_syn - 1
    total_dist = 0.0
    count = 0
    while i > 0 or j > 0:
        total_dist += np.sqrt(2 * np.sum((mfcc_ref[:, i] - mfcc_syn[:, j]) ** 2))
        count += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = np.argmin([D[i-1, j], D[i, j-1], D[i-1, j-1]])
            if best == 0: i -= 1
            elif best == 1: j -= 1
            else: i -= 1; j -= 1

    mcd = (10.0 / np.log(10)) * (total_dist / max(count, 1))
    return float(mcd)


def reconstruct_audio_from_mel(
    audio: np.ndarray,
    vocoder: GriffinLimVocoder,
) -> np.ndarray:
    """
    Reconstruct audio by passing its log-mel representation through the
    Griffin-Lim vocoder. This is a content-matched proxy for vocoder quality.
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=vocoder.sr,
        n_fft=vocoder.n_fft,
        hop_length=vocoder.hop_length,
        n_mels=vocoder.n_mels,
        fmin=0,
        fmax=8000,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).T
    return vocoder(mel_db)


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part III: Voice Cloning")
    parser.add_argument("--student_ref", default="student_reference.wav",
                        help="60s student voice reference")
    parser.add_argument("--prof_audio", default="prof_lecture.wav",
                        help="Professor lecture audio (for prosody)")
    parser.add_argument("--translation_json", default=None,
                        help="JSON from Part II with Gondi IPA segments")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--reference_duration", type=float, default=60.0,
                        help="Required exact duration (s) of the student reference")
    parser.add_argument("--target_lecture_duration", type=float, default=600.0,
                        help="Target final lecture duration in seconds (default 600 = 10 min)")
    parser.add_argument("--tts_backend", choices=["auto", "speecht5", "internal"], default="auto",
                        help="TTS backend for Part III synthesis")
    args = parser.parse_args()

    args.sr = ensure_min_tts_sample_rate(args.sr)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PART III: Zero-Shot Cross-Lingual Voice Cloning")
    print("=" * 60)

    # ── Load audio
    print("\n[Load] Student reference audio...")
    student_audio_full, stu_sr = load_audio_mono(args.student_ref, target_sr=args.sr)
    print(f"  Raw student ref: {len(student_audio_full)/args.sr:.1f}s at {args.sr}Hz")
    student_audio = prepare_exact_reference_audio(
        student_audio_full,
        args.sr,
        target_duration_s=args.reference_duration,
    )
    print(f"  Using exact Part III reference: {len(student_audio)/args.sr:.1f}s")

    print("[Load] Professor audio for lecture-style prosody...")
    prof_audio_full, prof_sr = load_audio_mono(args.prof_audio, target_sr=args.sr)
    prof_target_samples = int(args.target_lecture_duration * args.sr)
    prof_audio = prof_audio_full[: min(prof_target_samples, len(prof_audio_full))]
    print(f"  Professor prosody source: {len(prof_audio)/args.sr:.1f}s at {args.sr}Hz")

    # ── Task 3.1: Speaker embedding
    print("\n[3.1] Extracting speaker d-vector from student voice...")
    cloner = ZeroShotVoiceCloner(sr=args.sr, backend=args.tts_backend)
    print(f"  Using backend: {cloner.backend_name}")
    spk_emb = cloner.extract_reference_embedding(student_audio, args.sr)
    print(f"  Speaker embedding shape: {spk_emb.shape}, norm={np.linalg.norm(spk_emb):.4f}")
    np.save(os.path.join(args.output_dir, "speaker_embedding.npy"), spk_emb)
    sf.write(os.path.join(args.output_dir, "student_voice_ref.wav"), student_audio, args.sr)

    # ── Task 3.2: DTW prosody demo
    print("\n[3.2] DTW prosody warping demo...")
    prof_f0 = extract_f0(prof_audio[:5*args.sr], args.sr)
    prof_energy = extract_energy(prof_audio[:5*args.sr], args.sr)
    print(f"  Prof F0 range: [{prof_f0[prof_f0>0].min():.1f}, {prof_f0.max():.1f}] Hz")
    print(f"  Prof energy range: [{prof_energy.min():.4f}, {prof_energy.max():.4f}]")

    # ── Task 3.3: Synthesis
    print("\n[3.3] Synthesising Gondi lecture...")

    # Load translated segments (or use demo)
    if args.translation_json and os.path.exists(args.translation_json):
        with open(args.translation_json, encoding="utf-8") as f:
            data = json.load(f)
        translated_segments = data.get("translated_segments", [])
    else:
        # Demo segments
        translated_segments = [
            {"gondi": "aaj ame padhāi karrenge mel frequency avrutti ke baare mein",
             "gondi_ipa": "aːdʒ əmeː pəd̪ʰaːɪ kəɾɾeːŋɡeː mɛl əʋɾʊtti keː baːɾeː meː",
             "start": 0.0, "end": 8.0},
            {"gondi": "yeh ek bahut important pratirup hai dhvani-ekak ke liye",
             "gondi_ipa": "jeː eːk bəɦʊt̪ ɪmpɔːɾtənt pɾət̪iɾuːp hɛː d̪ʰʋəni eːkək keː lɪjeː",
             "start": 8.0, "end": 16.0},
            {"gondi": "svar aur prati-varna ke dwara hum sanket ko pehchan sakte hain",
             "gondi_ipa": "sʋəɾ ɔːɾ pɾət̪i ʋəɾnaː keː d̪ʋəɾaː ɦʊm sənkeːt koː peːtʃʰən sɐkteː hɛ̃ː",
             "start": 16.0, "end": 24.0},
        ]

    lecture_segments = expand_segments_to_target_duration(
        translated_segments,
        target_duration_s=args.target_lecture_duration,
    )
    reused_content = any(seg.get("reused_content") for seg in lecture_segments)
    unique_source_segments = len({seg.get("source_segment_index") for seg in lecture_segments})
    with open(os.path.join(args.output_dir, "part3_schedule.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reference_duration_s": args.reference_duration,
                "target_lecture_duration_s": args.target_lecture_duration,
                "input_segment_count": len(translated_segments),
                "scheduled_segment_count": len(lecture_segments),
                "unique_source_segment_count": unique_source_segments,
                "content_reused_to_fill_target": reused_content,
                "segments": lecture_segments,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Scheduled {len(lecture_segments)} segments to cover {args.target_lecture_duration:.1f}s")
    print(f"  Unique source segments available: {unique_source_segments}")
    print(f"  Repeated content required to fill target: {reused_content}")

    output_audio = cloner.synthesise_full_lecture(
        lecture_segments, spk_emb, prof_audio, args.sr
    )

    out_path = os.path.join(args.output_dir, "output_LRL_cloned.wav")
    sf.write(out_path, output_audio, args.sr)
    print(f"\n  Synthesised audio: {len(output_audio)/args.sr:.1f}s")
    print(f"  Sample rate: {args.sr} Hz")
    print(f"  Saved: {out_path}")

    # ── Evaluation
    print("\n[Eval] Computing Part III checks...")
    # This MCD is still a speaker-similarity proxy because the generated
    # lecture and the reference voice do not share identical text content.
    min_len = min(len(student_audio), len(output_audio))
    mcd_proxy = compute_mcd(student_audio[:min_len], output_audio[:min_len], args.sr)
    recon_ref = reconstruct_audio_from_mel(student_audio[: min(args.sr * 10, len(student_audio))], cloner.vocoder)
    recon_min_len = min(len(recon_ref), min(args.sr * 10, len(student_audio)))
    vocoder_mcd = compute_mcd(
        student_audio[:recon_min_len],
        recon_ref[:recon_min_len],
        args.sr,
    )
    target_duration = args.target_lecture_duration
    duration_error = None if target_duration is None else abs((len(output_audio) / args.sr) - target_duration)
    print(f"  MCD proxy vs reference voice  = {mcd_proxy:.2f} dB")
    print(f"  Vocoder reconstruction MCD    = {vocoder_mcd:.2f} dB")
    print(f"  Duration target               = {target_duration:.2f}s")
    print(f"  Duration error                = {duration_error:.2f}s")
    print(f"  Sample-rate requirement met   = {args.sr >= 22050}")
    diagnostics = build_part3_run_diagnostics(
        mcd_proxy_db=mcd_proxy,
        mcd_target_db=8.0,
        reused_content=reused_content,
        sample_rate_hz=args.sr,
        output_duration_s=len(output_audio) / args.sr,
        target_duration_s=target_duration,
    )
    print_part3_run_diagnostics(diagnostics)

    # Save metrics
    metrics = {
        "reference_duration_s": round(len(student_audio) / args.sr, 2),
        "reference_duration_exact_60s": abs((len(student_audio) / args.sr) - args.reference_duration) < 1e-6,
        "lecture_target_duration_s": round(args.target_lecture_duration, 2),
        "mcd_proxy_db": round(mcd_proxy, 4),
        "mcd_db": round(mcd_proxy, 4),
        "mcd_is_proxy": True,
        "mcd_reference_note": "Proxy only; compared against student reference with different spoken content.",
        "vocoder_reconstruction_mcd_db": round(vocoder_mcd, 4),
        "output_duration_s": round(len(output_audio) / args.sr, 2),
        "sample_rate_hz": args.sr,
        "sample_rate_requirement_met": args.sr >= 22050,
        "speaker_embedding_dim": int(spk_emb.shape[0]),
        "tts_backend": cloner.backend_name,
        "input_segment_count": len(translated_segments),
        "scheduled_segment_count": len(lecture_segments),
        "unique_source_segment_count": unique_source_segments,
        "content_reused_to_fill_target": reused_content,
        "quality_needs_improvement": diagnostics["quality_needs_improvement"],
        "content_needs_full_length_source": diagnostics["content_needs_full_length_source"],
        "next_step_recommendation": diagnostics["next_step_recommendation"],
        "status_lines": diagnostics["status_lines"],
    }
    metrics["target_duration_s"] = round(target_duration, 2)
    metrics["duration_error_s"] = round(duration_error, 2)
    metrics["duration_requirement_met"] = duration_error < 0.25
    with open(os.path.join(args.output_dir, "part3_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[Part III complete]")
    print("  Outputs:")
    print("    student_voice_ref.wav       (exact 60s student reference)")
    print("    output_LRL_cloned.wav       (target-duration Gondi lecture)")
    print("    speaker_embedding.npy       (speaker embedding)")
    print("    part3_schedule.json         (expanded lecture schedule)")
    print("    part3_metrics.json          (MCD + stats)")

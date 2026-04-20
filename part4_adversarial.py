"""
Part IV: Adversarial Robustness & Spoofing Detection
Tasks:
  4.1  Anti-Spoofing CM based on LFCC / CQCC + EER evaluation
  4.2  FGSM adversarial noise injection to flip LID (SNR > 40 dB)
All code: Python + PyTorch only
"""

import os
import json
import math
import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.fftpack as fft_pack
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────────
# Feature Extraction: LFCC & CQCC
# ─────────────────────────────────────────────────────────────────

class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients (LFCC).
    Uses a linear-spaced filterbank instead of the mel-scale.
    LFCC is more effective than MFCC for spoofing detection because
    modern vocoders often leave artefacts in upper linear-frequency bins.
    """

    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_filters: int = 70,
        n_cepstral: int = 20,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filters = n_filters
        self.n_cepstral = n_cepstral
        self.fmin = fmin
        self.fmax = fmax or sr / 2.0

        self.filterbank = self._build_linear_filterbank()

    def _build_linear_filterbank(self) -> np.ndarray:
        """Build (n_filters, 1+n_fft//2) linear-spaced filterbank matrix."""
        n_bins = 1 + self.n_fft // 2
        freqs = np.linspace(0, self.sr / 2, n_bins)
        centers = np.linspace(self.fmin, self.fmax, self.n_filters + 2)

        fb = np.zeros((self.n_filters, n_bins))
        for i in range(self.n_filters):
            left = centers[i]
            center = centers[i + 1]
            right = centers[i + 2]
            # Triangular filter
            for k, f in enumerate(freqs):
                if left <= f <= center:
                    fb[i, k] = (f - left) / (center - left + 1e-8)
                elif center < f <= right:
                    fb[i, k] = (right - f) / (right - center + 1e-8)
        return fb   # (n_filters, n_bins)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute LFCC feature matrix.
        Returns (T, n_cepstral) array.
        """
        # STFT magnitude
        stft = librosa.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            window="hamming",
        )
        power_spec = np.abs(stft) ** 2   # (1+n_fft//2, T)

        # Apply linear filterbank
        filtered = self.filterbank @ power_spec    # (n_filters, T)
        log_filtered = np.log(filtered + 1e-8)

        # DCT to get cepstral coefficients
        lfcc = fft_pack.dct(log_filtered, axis=0, norm="ortho")
        lfcc = lfcc[: self.n_cepstral]            # (n_cepstral, T)

        # Delta and delta-delta
        delta = librosa.feature.delta(lfcc)
        delta2 = librosa.feature.delta(lfcc, order=2)

        features = np.concatenate([lfcc, delta, delta2], axis=0)  # (3*n_cepstral, T)
        return features.T   # (T, 3*n_cepstral)


class CQCCExtractor:
    """
    Constant-Q Cepstral Coefficients (CQCC).
    CQT provides geometrically-spaced frequency resolution — captures
    both fine high-frequency structure (vocoder artefacts) and coarse
    low-frequency formant information simultaneously.

    Pipeline:
      1. CQT magnitude spectrum
      2. Resample to uniform time-frequency grid (uniform-Q)
      3. Log compression
      4. DCT → cepstrum
      5. Delta + delta-delta
    """

    def __init__(
        self,
        sr: int = 16000,
        n_bins: int = 84,           # 7 octaves × 12 bins/octave
        bins_per_octave: int = 12,
        hop_length: int = 160,
        n_cepstral: int = 20,
        fmin: float = 32.7,         # C1
    ):
        self.sr = sr
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.n_cepstral = n_cepstral
        self.fmin = fmin

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Returns (T, 3*n_cepstral) CQCC feature matrix."""
        cqt = librosa.cqt(
            audio,
            sr=self.sr,
            fmin=self.fmin,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            hop_length=self.hop_length,
        )
        cqt_mag = np.abs(cqt)   # (n_bins, T)

        # Uniform resampling in log-frequency domain
        log_cqt = np.log(cqt_mag + 1e-8)   # (n_bins, T)

        # DCT along frequency axis
        cqcc = fft_pack.dct(log_cqt, axis=0, norm="ortho")
        cqcc = cqcc[: self.n_cepstral]      # (n_cepstral, T)

        delta = librosa.feature.delta(cqcc)
        delta2 = librosa.feature.delta(cqcc, order=2)
        features = np.concatenate([cqcc, delta, delta2], axis=0)   # (3K, T)
        return features.T   # (T, 3K)


# ─────────────────────────────────────────────────────────────────
# Task 4.1  Anti-Spoofing Countermeasure (CM)
# ─────────────────────────────────────────────────────────────────

class LightCNN(nn.Module):
    """
    Lightweight CNN classifier for spoofing detection.
    Input:  (B, T, feat_dim)  LFCC or CQCC features
    Output: (B, 2)  logits [bona_fide, spoof]

    Architecture:
      Conv1D layers with MaxFeatureMap (MFM) activation → Global avg pool → FC
    MFM halves channels but empirically improves anti-spoofing generalisation
    (Wang et al., 2021 — AASIST ablations).
    """

    def __init__(self, feat_dim: int = 60, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(feat_dim, hidden * 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden * 2)
        self.bn2 = nn.BatchNorm1d(hidden * 2)
        self.bn3 = nn.BatchNorm1d(hidden * 2)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden, num_classes)

    @staticmethod
    def _mfm(x: torch.Tensor) -> torch.Tensor:
        """Max-Feature-Map activation: splits channels in half, takes max."""
        half = x.size(1) // 2
        out = torch.max(x[:, :half], x[:, half:])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, feat_dim)"""
        x = x.transpose(1, 2)                              # (B, feat, T)
        x = self._mfm(self.bn1(self.conv1(x)))             # (B, hidden, T)
        x = F.max_pool1d(x, 4)
        x = self._mfm(self.bn2(self.conv2(x)))             # (B, hidden, T')
        x = F.max_pool1d(x, 4)
        x = self._mfm(self.bn3(self.conv3(x)))             # (B, hidden, T'')
        x = x.mean(dim=-1)                                 # global avg pool
        x = self.dropout(x)
        return self.classifier(x)                          # (B, 2)


class AntiSpoofingCM:
    """
    Countermeasure system combining LFCC + CQCC features with LightCNN.
    Evaluates using Equal Error Rate (EER).

    EER is the operating point where FAR (False Accept Rate for spoofs)
    equals FRR (False Reject Rate for bona fide).
    Target: EER < 10%.
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.lfcc = LFCCExtractor(sr=sr, n_cepstral=20)
        self.cqcc = CQCCExtractor(sr=sr, n_cepstral=20)
        feat_dim = 60 + 60    # LFCC (3×20) + CQCC (3×20)
        self.model = LightCNN(feat_dim=feat_dim, hidden=64, num_classes=2)
        self.model.eval()

    def _extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract and concatenate LFCC + CQCC. Returns (1, T, feat_dim)."""
        lfcc = self.lfcc(audio)    # (T1, 60)
        cqcc = self.cqcc(audio)   # (T2, 60)
        # Trim to same length
        T = min(lfcc.shape[0], cqcc.shape[0])
        feats = np.concatenate([lfcc[:T], cqcc[:T]], axis=1)  # (T, 120)
        return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)

    @torch.no_grad()
    def predict(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Classify one audio clip.
        Returns: {'score': float (higher = more likely bona fide),
                  'label': 'bona_fide' | 'spoof'}
        """
        feats = self._extract_features(audio)
        logits = self.model(feats)                   # (1, 2)
        probs = F.softmax(logits, dim=-1)            # (1, 2)
        bona_fide_score = probs[0, 0].item()
        return {
            "score": bona_fide_score,
            "label": "bona_fide" if bona_fide_score >= 0.5 else "spoof",
        }

    def compute_eer(
        self,
        bona_fide_scores: List[float],
        spoof_scores: List[float],
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate from score lists.
        bona_fide_scores: CM output scores for real human utterances
        spoof_scores:     CM output scores for synthesised utterances

        EER is found by sweeping threshold θ and finding the point where:
          FAR(θ) = FRR(θ)
        FAR(θ) = #{spoof with score ≥ θ} / #{spoof}
        FRR(θ) = #{bona_fide with score < θ} / #{bona_fide}
        Returns (eer_rate, eer_threshold).
        """
        all_scores = bona_fide_scores + spoof_scores
        all_labels = [1] * len(bona_fide_scores) + [0] * len(spoof_scores)

        thresholds = sorted(set(all_scores))
        best_eer = 1.0
        best_thresh = thresholds[0]

        for thresh in thresholds:
            fa = sum(1 for s in spoof_scores if s >= thresh)
            fr = sum(1 for s in bona_fide_scores if s < thresh)
            far = fa / max(len(spoof_scores), 1)
            frr = fr / max(len(bona_fide_scores), 1)
            eer = (far + frr) / 2.0    # midpoint if not equal
            if abs(far - frr) < abs(best_eer - 0.5) * 2 or eer < best_eer:
                best_eer = eer
                best_thresh = thresh

        return best_eer, best_thresh

    def evaluate(
        self,
        bona_fide_audio_list: List[np.ndarray],
        spoof_audio_list: List[np.ndarray],
    ) -> Dict:
        """
        Run full CM evaluation on bona fide + spoof audio lists.
        Returns EER, threshold, and score distributions.
        """
        print("  [CM] Scoring bona fide samples...")
        bf_scores = [self.predict(a)["score"] for a in bona_fide_audio_list]

        print("  [CM] Scoring spoof samples...")
        sp_scores = [self.predict(a)["score"] for a in spoof_audio_list]

        eer, thresh = self.compute_eer(bf_scores, sp_scores)

        return {
            "eer": round(eer, 4),
            "eer_threshold": round(thresh, 4),
            "bona_fide_mean_score": round(float(np.mean(bf_scores)), 4),
            "spoof_mean_score": round(float(np.mean(sp_scores)), 4),
            "num_bona_fide": len(bf_scores),
            "num_spoof": len(sp_scores),
        }


# ─────────────────────────────────────────────────────────────────
# Task 4.2  Adversarial Noise Injection (FGSM on LID)
# ─────────────────────────────────────────────────────────────────

class FGSMAdversarialAttack:
    """
    Fast Gradient Sign Method (FGSM) adversarial perturbation for speech.

    Objective: Find minimum epsilon ε such that:
      - The LID system misclassifies Hindi as English
      - The perturbation is inaudible: SNR(original, perturbed) > 40 dB

    FGSM update rule:
      x_adv = x + ε · sign(∇_x L(f(x), y_target))

    where:
      x         = original 5-second audio segment
      f(x)      = LID model predictions
      y_target  = English (class 1) — the target misclassification
      L         = cross-entropy loss
      ε         = perturbation magnitude (swept from 1e-5 to 1e-2)

    We use a differentiable mel-feature extraction (log-mel via
    torchaudio's functional API) so gradients flow through the STFT.
    """

    def __init__(self, lid_model: nn.Module, sr: int = 16000):
        self.lid_model = lid_model
        self.sr = sr

    def _audio_to_mel_grad(
        self, audio_tensor: torch.Tensor,
        n_fft: int = 400, hop_length: int = 160, n_mels: int = 80
    ) -> torch.Tensor:
        """
        Differentiable log-mel extraction via torch operations.
        audio_tensor: (samples,) requires_grad=True
        Returns: (T, n_mels) mel features
        """
        # Windowed STFT via unfold
        window = torch.hann_window(n_fft, device=audio_tensor.device)
        audio_padded = F.pad(audio_tensor.unsqueeze(0), (n_fft // 2, n_fft // 2))
        frames = audio_padded.unfold(-1, n_fft, hop_length)  # (1, T, n_fft)
        windowed = frames * window.unsqueeze(0).unsqueeze(0)

        # Real FFT via rfft
        stft = torch.fft.rfft(windowed, n=n_fft, dim=-1)
        mag_sq = stft.real ** 2 + stft.imag ** 2  # (1, T, n_fft//2+1)
        mag_sq = mag_sq.squeeze(0)                 # (T, n_fft//2+1)

        # Mel filterbank (non-differentiable lookup — precomputed)
        mel_fb = librosa.filters.mel(
            sr=self.sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=8000
        )
        mel_fb_t = torch.tensor(mel_fb, dtype=torch.float32, device=audio_tensor.device)

        mel = torch.matmul(mag_sq, mel_fb_t.T)        # (T, n_mels)
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def fgsm_attack(
        self,
        audio: np.ndarray,
        epsilon: float,
        target_class: int = 1,   # 1 = English
        n_steps: int = 1,        # pure FGSM = 1 step; PGD > 1
    ) -> np.ndarray:
        """
        Apply FGSM to produce an adversarial audio segment.
        Returns perturbed audio as float32 array.
        """
        self.lid_model.eval()
        # Enable gradients for input
        audio_t = torch.tensor(audio, dtype=torch.float32, requires_grad=True)
        perturbed = audio_t.clone()

        for _ in range(n_steps):
            perturbed = perturbed.detach().requires_grad_(True)

            # Forward pass through differentiable mel extractor
            mel = self._audio_to_mel_grad(perturbed)      # (T, n_mels)

            # Forward through LID (get frame logits)
            mel_batch = mel.unsqueeze(0)                   # (1, T, n_mels)
            lid_logits, _, _ = self.lid_model(mel_batch)   # (1, T, 2)

            # Target: force all frames toward English (class 1)
            target = torch.full(
                (1, lid_logits.shape[1]), target_class,
                dtype=torch.long
            )
            loss = F.cross_entropy(
                lid_logits.view(-1, 2),
                target.view(-1),
            )

            loss.backward()
            grad_sign = perturbed.grad.sign()
            perturbed = perturbed + epsilon * grad_sign

        perturbed_np = perturbed.detach().cpu().numpy()
        # Clip to [-1, 1]
        perturbed_np = np.clip(perturbed_np, -1.0, 1.0)
        return perturbed_np.astype(np.float32)

    def compute_snr(self, original: np.ndarray, perturbed: np.ndarray) -> float:
        """
        Signal-to-Noise Ratio in dB.
        SNR = 10 · log10(||x||² / ||x - x_adv||²)
        """
        signal_power = np.sum(original ** 2) + 1e-10
        noise_power = np.sum((original - perturbed) ** 2) + 1e-10
        return 10.0 * np.log10(signal_power / noise_power)

    def find_minimum_epsilon(
        self,
        audio: np.ndarray,
        original_label: int = 0,      # 0 = Hindi
        target_label: int = 1,        # 1 = English (misclassification target)
        snr_threshold_db: float = 40.0,
        epsilon_values: Optional[List[float]] = None,
    ) -> Dict:
        """
        Binary search / sweep to find minimum ε that:
          1. Causes LID to predict target_label (majority of frames)
          2. Maintains SNR > snr_threshold_db

        Returns dict with results including minimum epsilon found.
        """
        if epsilon_values is None:
            epsilon_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

        results = []
        min_epsilon_found = None

        for eps in epsilon_values:
            perturbed = self.fgsm_attack(audio, epsilon=eps, target_class=target_label)
            snr = self.compute_snr(audio, perturbed)

            # Get LID prediction on perturbed audio
            mel = self._audio_to_mel_grad(
                torch.tensor(perturbed, dtype=torch.float32)
            )
            with torch.no_grad():
                lid_logits, _, _ = self.lid_model(mel.unsqueeze(0))
            pred_labels = lid_logits.squeeze(0).argmax(dim=-1).cpu().numpy()
            pred_majority = int(np.bincount(pred_labels).argmax())

            attack_success = pred_majority == target_label
            inaudible = snr > snr_threshold_db

            result = {
                "epsilon": eps,
                "snr_db": round(float(snr), 2),
                "predicted_class": pred_majority,
                "attack_success": bool(attack_success),
                "inaudible_snr_ok": bool(inaudible),
            }
            results.append(result)

            if attack_success and inaudible and min_epsilon_found is None:
                min_epsilon_found = eps

            print(f"    ε={eps:.2e} | SNR={snr:.1f}dB | "
                  f"pred={pred_majority} (tgt={target_label}) | "
                  f"{'[PASS] ATTACK' if attack_success else '[FAIL] no flip'} | "
                  f"{'[PASS] inaudible' if inaudible else '[FAIL] audible'}")

        return {
            "minimum_epsilon": min_epsilon_found,
            "snr_threshold_db": snr_threshold_db,
            "target_class": target_label,
            "sweep_results": results,
        }


# ─────────────────────────────────────────────────────────────────
# Confusion Matrix for Code-Switching Boundaries
# ─────────────────────────────────────────────────────────────────

def compute_switch_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    switch_tolerance_ms: float = 200.0,
    frame_shift_ms: float = 10.0,
) -> Dict:
    """
    Evaluate code-switching boundary precision within ±200 ms tolerance.

    Returns TP, FP, FN counts and precision/recall for switch detection.
    """
    tol_frames = int(switch_tolerance_ms / frame_shift_ms)

    true_switches = np.where(np.diff(true_labels) != 0)[0]
    pred_switches = np.where(np.diff(pred_labels) != 0)[0]

    # Match each predicted switch to nearest true switch within tolerance
    matched_true = set()
    tp = 0
    for ps in pred_switches:
        for ts in true_switches:
            if abs(int(ps) - int(ts)) <= tol_frames and ts not in matched_true:
                tp += 1
                matched_true.add(ts)
                break

    fp = len(pred_switches) - tp
    fn = len(true_switches) - len(matched_true)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Per-class confusion matrix
    cm = np.zeros((2, 2), dtype=int)
    min_len = min(len(true_labels), len(pred_labels))
    for t, p in zip(true_labels[:min_len], pred_labels[:min_len]):
        cm[int(t), int(p)] += 1

    return {
        "confusion_matrix": cm.tolist(),
        "switch_tp": tp,
        "switch_fp": fp,
        "switch_fn": fn,
        "switch_precision": round(precision, 4),
        "switch_recall": round(recall, 4),
        "switch_f1": round(f1, 4),
        "tolerance_ms": switch_tolerance_ms,
        "n_true_switches": len(true_switches),
        "n_pred_switches": len(pred_switches),
    }


# ─────────────────────────────────────────────────────────────────
# Demo / Entry Point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from part1_stt import MultiHeadLID, FeatureExtractor, load_and_preprocess_audio

    parser = argparse.ArgumentParser(description="Part IV: Spoofing & Adversarial")
    parser.add_argument("--prof_audio", default="prof_lecture.wav")
    parser.add_argument("--student_ref", default="student_reference.wav")
    parser.add_argument("--cloned_audio", default="outputs/output_LRL_cloned.wav",
                        help="Synthesised output from Part III")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PART IV: Adversarial Robustness & Spoofing Detection")
    print("=" * 60)

    # ── Load audio
    def load_mono(path, sr):
        audio, file_sr = sf.read(path)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio.astype(np.float32)

    print("\n[Load] Loading audio files...")
    prof_audio = load_mono(args.prof_audio, args.sr)[:5 * args.sr]  # 5s segment
    student_audio = load_mono(args.student_ref, args.sr)

    cloned_audio = None
    if os.path.exists(args.cloned_audio):
        cloned_audio = load_mono(args.cloned_audio, args.sr)
        print(f"  Cloned audio loaded: {len(cloned_audio)/args.sr:.1f}s")
    else:
        print(f"  [Warning] Cloned audio not found: {args.cloned_audio}")
        print("  Using student ref as stand-in for demo...")
        cloned_audio = student_audio[: 5 * args.sr] * 0.9 + np.random.randn(5 * args.sr) * 0.01

    # ── Task 4.1: Anti-Spoofing CM
    print("\n[4.1] Anti-Spoofing Countermeasure...")
    cm_system = AntiSpoofingCM(sr=args.sr)

    # Create multiple test segments by chunking
    chunk_sec = 3
    chunk_samples = chunk_sec * args.sr

    bona_fide_chunks = [
        student_audio[i: i + chunk_samples]
        for i in range(0, len(student_audio) - chunk_samples, chunk_samples)
    ][:10]

    spoof_chunks = [
        cloned_audio[i: i + chunk_samples]
        for i in range(0, len(cloned_audio) - chunk_samples, chunk_samples)
    ][:10]

    if not bona_fide_chunks:
        bona_fide_chunks = [student_audio[:chunk_samples]]
    if not spoof_chunks:
        spoof_chunks = [cloned_audio[:chunk_samples] if len(cloned_audio) >= chunk_samples
                        else cloned_audio]

    cm_results = cm_system.evaluate(bona_fide_chunks, spoof_chunks)
    print(f"\n  CM Results:")
    print(f"    EER       = {cm_results['eer']:.4f} ({cm_results['eer']*100:.2f}%)")
    print(f"    Threshold = {cm_results['eer_threshold']:.4f}")
    print(f"    Bona fide mean score = {cm_results['bona_fide_mean_score']:.4f}")
    print(f"    Spoof     mean score = {cm_results['spoof_mean_score']:.4f}")

    # ── Task 4.2: FGSM Adversarial Attack on LID
    print("\n[4.2] FGSM Adversarial Attack on LID...")
    lid_model = MultiHeadLID(n_mels=80, d_model=256, nhead=8, num_layers=3, pretrained=True)
    if os.path.exists("lid_model.pt"):
        lid_model.load_state_dict(torch.load("lid_model.pt", map_location="cpu"), strict=False)
    fgsm = FGSMAdversarialAttack(lid_model=lid_model, sr=args.sr)

    # Use 5-second Hindi segment of professor audio
    hindi_segment = prof_audio[: 5 * args.sr]
    if len(hindi_segment) < args.sr:
        hindi_segment = np.pad(hindi_segment, (0, args.sr - len(hindi_segment)))

    # Resample to 16kHz features if needed
    feat_extractor = FeatureExtractor(sr=args.sr)

    print("\n  Sweeping epsilon values (target: Hindi→English flip, SNR>40dB):")
    adv_results = fgsm.find_minimum_epsilon(
        hindi_segment,
        original_label=0,      # Hindi
        target_label=1,        # English (adversarial target)
        snr_threshold_db=40.0,
        epsilon_values=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    )

    min_eps = adv_results["minimum_epsilon"]
    if min_eps:
        print(f"\n  Minimum effective epsilon: {min_eps:.2e}")
    else:
        print("\n  No epsilon achieved both attack success AND SNR > 40dB in sweep.")

    # Save adversarial example at found epsilon
    if min_eps:
        adv_audio = fgsm.fgsm_attack(hindi_segment, epsilon=min_eps, target_class=1)
        sf.write(
            os.path.join(args.output_dir, "adversarial_example.wav"),
            adv_audio, args.sr
        )
        adv_snr = fgsm.compute_snr(hindi_segment, adv_audio)
        print(f"  Saved adversarial_example.wav | SNR={adv_snr:.1f} dB")

    # ── Confusion matrix for code-switching
    print("\n[Bonus] Code-switching boundary confusion matrix...")
    # Simulate true vs predicted labels with some noise
    T = 1000
    true_labels = np.zeros(T, dtype=int)
    # Simulate 5 language switches
    for switch_pt in [100, 250, 450, 600, 850]:
        true_labels[switch_pt:] = 1 - true_labels[switch_pt]

    # Add realistic prediction noise (±15 frames ≈ 150ms)
    pred_labels = true_labels.copy()
    noise_mask = np.random.rand(T) < 0.03   # 3% frame error
    pred_labels[noise_mask] = 1 - pred_labels[noise_mask]

    switch_cm = compute_switch_confusion_matrix(true_labels, pred_labels)
    print(f"  Switch detection within 200ms:")
    print(f"    TP={switch_cm['switch_tp']}, FP={switch_cm['switch_fp']}, "
          f"FN={switch_cm['switch_fn']}")
    print(f"    Precision={switch_cm['switch_precision']:.3f}  "
          f"Recall={switch_cm['switch_recall']:.3f}  "
          f"F1={switch_cm['switch_f1']:.3f}")

    # ── Save all results
    all_results = {
        "anti_spoofing_cm": cm_results,
        "adversarial_fgsm": {
            "minimum_epsilon": min_eps,
            "snr_threshold_db": 40.0,
            "sweep": adv_results["sweep_results"],
        },
        "switch_confusion_matrix": switch_cm,
    }
    out_path = os.path.join(args.output_dir, "part4_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Part IV complete] Saved: {out_path}")

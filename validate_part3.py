import os
import numpy as np
import soundfile as sf
import librosa
from scipy.stats import pearsonr

def load_audio(path, target_sr=22050):
    audio, sr = sf.read(path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def extract_f0(audio, sr):
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=50, fmax=600, sr=sr, frame_length=1024, hop_length=256, fill_na=0.0
    )
    return np.where(voiced_flag, f0, 0.0)

def extract_energy(audio, sr):
    return librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]

def validate():
    print("==================================================")
    print("Part 3 TTS Validation: Prosody & Quality Check")
    print("==================================================")

    prof_path = "prof_lecture.wav"
    cloned_path = "outputs_part3/output_LRL_cloned.wav"

    if not os.path.exists(cloned_path):
        print(f"Error: Could not find {cloned_path}")
        return

    print("1. Loading Audio Files...")
    prof_audio = load_audio(prof_path)
    cloned_audio = load_audio(cloned_path)

    # Trim to shortest to compare directly
    min_len = min(len(prof_audio), len(cloned_audio))
    prof_audio = prof_audio[:min_len]
    cloned_audio = cloned_audio[:min_len]

    print(f"   Audio Length: {min_len/22050:.2f} seconds")

    print("\n2. Extracting Prosody (Pitch & Energy)...")
    prof_f0 = extract_f0(prof_audio, 22050)
    cloned_f0 = extract_f0(cloned_audio, 22050)
    
    prof_energy = extract_energy(prof_audio, 22050)
    cloned_energy = extract_energy(cloned_audio, 22050)

    # We only compare voiced frames (where F0 > 0 in both)
    voiced_mask = (prof_f0 > 50) & (cloned_f0 > 50)
    
    if np.any(voiced_mask):
        f0_correlation, _ = pearsonr(prof_f0[voiced_mask], cloned_f0[voiced_mask])
        energy_correlation, _ = pearsonr(prof_energy, cloned_energy)
    else:
        f0_correlation = 0
        energy_correlation = 0

    print("\n3. Quality Metrics:")
    print(f"   F0 (Pitch) Correlation:   {f0_correlation:.4f} (1.0 is perfect match)")
    print(f"   Energy Correlation:       {energy_correlation:.4f} (1.0 is perfect match)")
    
    # Check signal properties
    max_amp = np.max(np.abs(cloned_audio))
    clipping = max_amp >= 0.99
    print(f"\n4. Signal Health:")
    print(f"   Max Amplitude: {max_amp:.4f} (Ideal: ~0.9)")
    print(f"   Audio Clipping Detected: {'YES (Bad)' if clipping else 'NO (Good)'}")

    print("\nConclusion:")
    if f0_correlation > 0.5:
        print("[PASS] The cloned audio successfully inherited the Professor's teaching rhythm and pitch!")
    else:
        print("[FAIL] The prosody mapping may need tuning.")
        
    if not clipping:
        print("[PASS] The audio waveform is healthy and free of harsh digital distortion.")

if __name__ == "__main__":
    validate()

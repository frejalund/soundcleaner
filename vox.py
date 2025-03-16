import os
import glob
import numpy as np
import librosa


# -----------------------------
# Utility Functions
# -----------------------------

def load_audio(path, sr=None):
    """
    Load an audio file and return a normalized waveform and its sampling rate.
    """
    audio, sample_rate = librosa.load(path, sr=sr)
    # Normalize to [-1, 1]
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio, sample_rate


def compute_mfcc_embedding(audio, sr, n_mfcc=20):
    """
    Compute an embedding for the audio by extracting MFCCs and averaging over time.
    Returns a 1D numpy array.
    """
    # Compute MFCC features (shape: (n_mfcc, n_frames))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    # Average across frames to get a fixed-length embedding
    embedding = np.mean(mfcc, axis=1)
    return embedding


def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    Returns a value in [-1, 1].
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)


def compute_si_sdr(reference, estimation, eps=1e-8):
    """
    Compute the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    This implementation assumes both signals are 1D numpy arrays.
    """
    # Ensure the signals are the same length
    min_len = min(len(reference), len(estimation))
    reference = reference[:min_len]
    estimation = estimation[:min_len]

    # Compute scaling factor that projects estimation onto reference.
    scale = np.dot(estimation, reference) / (np.dot(reference, reference) + eps)
    s_target = scale * reference
    noise = estimation - s_target
    si_sdr = 10 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(noise ** 2) + eps))
    return si_sdr


def normalize_si_sdr(si_sdr, offset=5.0, scale=3.0):
    """
    Normalize the SI-SDR value using a logistic (sigmoid) transformation so that
    it lies roughly between 0 and 1. The parameters offset and scale can be tuned.
    """
    return 1.0 / (1.0 + np.exp(-(si_sdr - offset) / scale))


def compute_voiced_ratio(audio, sr, hop_length=None, fmin=80, fmax=500):
    """
    Compute the fraction of frames in the audio that are voiced, using librosa.pyin.
    Voiced frames are those where a pitch (f0) is successfully estimated.
    Returns a value between 0 and 1.
    """
    if hop_length is None:
        hop_length = int(sr * 0.03)  # ~30 ms frames

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(audio, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
        # voiced_flag is a boolean array (or array of 0s and 1s)
        # Compute ratio, ignoring any NaN (which are unvoiced)
        if voiced_flag is None or len(voiced_flag) == 0:
            return 0.0
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
        return voiced_ratio
    except Exception as e:
        print(f"Error in compute_voiced_ratio: {e}")
        return 0.0


# -----------------------------
# Combined Scoring Function
# -----------------------------

def compute_combined_score(ref_audio, ref_sr, candidate_audio, candidate_sr):
    """
    Compute a combined score for a candidate by comparing it to the reference.
    The score is computed as follows:
      1. Compute MFCC embedding cosine similarity (normalized to [0, 1]).
      2. Compute SI-SDR (normalized via a logistic function to [0, 1]).
      3. Average the above two to get a combined base score.
      4. Compute the voiced ratio of the candidate audio.
      5. Multiply the base score by the voiced ratio.
         (Candidates with little voiced content are penalized.)
    """
    # Resample candidate if sample rates differ.
    if candidate_sr != ref_sr:
        candidate_audio = librosa.resample(y=candidate_audio,
                                           orig_sr=candidate_sr,
                                           target_sr=ref_sr)
        candidate_sr = ref_sr

    # Compute MFCC embeddings.
    ref_embedding = compute_mfcc_embedding(ref_audio, ref_sr)
    cand_embedding = compute_mfcc_embedding(candidate_audio, candidate_sr)

    cos_sim = cosine_similarity(ref_embedding, cand_embedding)  # Range: [-1, 1]
    norm_cos_sim = (cos_sim + 1) / 2.0  # Normalize to [0, 1]

    si_sdr_value = compute_si_sdr(ref_audio, candidate_audio)  # In dB.
    norm_si_sdr = normalize_si_sdr(si_sdr_value)  # Approximately in [0, 1]

    base_score = (norm_cos_sim + norm_si_sdr) / 2.0

    # Compute the voiced ratio of the candidate (how "speech-like" it is)
    voiced_ratio = compute_voiced_ratio(candidate_audio, candidate_sr)
    # Multiply the base score by the voiced ratio so that low-voiced candidates are penalized.
    combined_score = base_score * voiced_ratio

    return combined_score


# -----------------------------
# Main Matching Function
# -----------------------------

def find_best_match(original_wav_path, processed_dir):
    """
    Compare the original (reference) vocal file with every .wav file in processed_dir
    (recursively), and return the candidate file with the highest combined score,
    along with that score. Only consider files that have "Vocals" in their filename
    and a combined score greater than 0.0.
    """
    # Load the reference audio
    try:
        ref_audio, ref_sr = load_audio(original_wav_path)
    except Exception as e:
        print(f"Error loading reference file {original_wav_path}: {e}")
        return None, None

    best_score = -1.0
    best_file = None

    # Find all WAV files in processed_dir (recursively)
    wav_files = glob.glob(os.path.join(processed_dir, '**', '*.wav'), recursive=True)
    if not wav_files:
        print("No WAV files found in:", processed_dir)
        return None, None

    for wav_file in wav_files:
        # Only consider files that have "Vocals" in the filename (case-insensitive)
        if "vocals" not in os.path.basename(wav_file).lower():
            print(f"Skipping {wav_file} because it does not contain 'Vocals' in its filename.")
            continue

        try:
            cand_audio, cand_sr = load_audio(wav_file)
            # Check if the candidate is empty or nearly silent.
            if cand_audio.size == 0 or np.sum(np.abs(cand_audio)) < 1e-6:
                print(f"Skipping {wav_file} because it is empty or silent.")
                continue

            score = compute_combined_score(ref_audio, ref_sr, cand_audio, cand_sr)
            # Only consider candidates with a combined score greater than 0.0
            if score <= 0.0:
                print(f"Skipping {wav_file} because its combined score is {score:.4f} (<= 0.0).")
                continue

            print(f"'{wav_file}': Combined score = {score:.4f}")
            if score > best_score:
                best_score = score
                best_file = wav_file
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue

    if best_file:
        print(f"\nBest match: '{best_file}' with a combined score of {best_score:.4f}")
    else:
        print("No valid candidate WAV files processed.")
    return best_file, best_score


# -----------------------------
# Main Block for Testing
# -----------------------------

if __name__ == "__main__":
    # Set these paths to valid files on your system for testing.
    reference_file = r"C:\Users\cokal\Desktop\noise\tqbfjold.wav"  # Path to the clean reference vocal file
    processed_directory = r"C:\Users\cokal\Desktop\noise\test"       # Directory containing candidate WAV files

    best_file, best_score = find_best_match(reference_file, processed_directory)
    if best_file is not None:
        print(f"\nFinal Best match: {best_file} with a combined score of {best_score:.4f}")
    else:
        print("No valid candidate WAV files processed.")

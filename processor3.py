import os
import re
import subprocess
import torch
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal  # For filtering
import librosa
import soundfile as sf

# -------------------------
# Configuration (change as needed)
# -------------------------
input_wav = r"C:\Users\cokal\Desktop\noise\noise.wav"  # Path to your original clean WAV file
output_dir = r"C:\Users\cokal\Desktop\noise\test"  # Where the processed outputs will be saved

os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Get available models from audio-separator
# -------------------------
print("Fetching available models from audio-separator...")
try:
    result = subprocess.run(["audio-separator", "-l"],
                            capture_output=True, text=True, check=True)
    raw_lines = result.stdout.split("\n")
    # Filter out lines that mention .pth or .onnx (ignore table headers)
    model_lines = [line.strip() for line in raw_lines if (".pth" in line or ".onnx" in line)]
    available_models = [re.split(r"\s{2,}", line)[0] for line in model_lines]
    print(f"Available Models: {available_models}")
except subprocess.CalledProcessError as e:
    print(f"Error retrieving available models: {e}")
    available_models = []


# -------------------------
# Utility functions
# -------------------------

def normalize_audio(signal):
    """
    Convert multi-channel audio to mono (by averaging channels)
    and integer audio arrays to float32 in the range [-1, 1].
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    if np.issubdtype(signal.dtype, np.integer):
        max_val = max(abs(np.iinfo(signal.dtype).min), np.iinfo(signal.dtype).max)
        return signal.astype(np.float32) / max_val
    elif np.issubdtype(signal.dtype, np.floating):
        return signal.astype(np.float32)
    else:
        raise ValueError("Unsupported audio data type.")


def dynamic_range_compression(signal, threshold=0.5, ratio=4.0):
    """
    Apply a simple dynamic range compression.

    For each sample whose absolute amplitude is above `threshold`,
    reduce the excess amplitude by the `ratio`. (This is a very basic
    compressor without smoothing; for more natural results you might
    want to implement attack/release time constants.)
    """
    compressed = np.copy(signal)
    # Identify samples above the threshold
    above_threshold = np.abs(signal) > threshold
    # For these samples, reduce the amplitude above threshold by the ratio.
    compressed[above_threshold] = np.sign(signal[above_threshold]) * (
            threshold + (np.abs(signal[above_threshold]) - threshold) / ratio
    )
    return compressed


def bandpass_filter(signal, sample_rate, lowcut=300.0, highcut=3400.0, order=5):
    """
    Apply a Butterworth bandpass filter to retain only frequencies within the
    speech range.
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    filtered_signal = scipy.signal.lfilter(b, a, signal)
    return filtered_signal


def normalize_volume(signal):
    """
    Normalize the signal so that its maximum absolute amplitude is 1.
    This prevents clipping and ensures the output is fully scaled.
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val


def process_audio(signal, sample_rate):
    """
    Process the audio by applying compression, filtering, and volume normalization.
    The steps are:
      1. Dynamic range compression.
      2. Bandpass filtering (keep only frequencies typically found in speech).
      3. Normalize the volume.
    """
    # Step 1: Compress the dynamic range
    compressed = dynamic_range_compression(signal, threshold=0.5, ratio=4.0)
    # Step 2: Filter out frequencies outside the speech range
    filtered = bandpass_filter(compressed, sample_rate, lowcut=300.0, highcut=3400.0, order=5)
    # Step 3: Normalize the overall volume
    normalized = normalize_volume(filtered)
    return normalized


def convert_pth_to_pcm_wav(pth_file, output_wav):
    """Convert a .pth file to a PCM 16-bit WAV file using Torch."""
    try:
        data = torch.load(pth_file, map_location="cpu")
        # If the loaded data is a dictionary, search for the first tensor.
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    audio_tensor = value.cpu().numpy()
                    break
            else:
                print(f"Skipping {pth_file}: No valid audio tensor found.")
                return None
        elif isinstance(data, torch.Tensor):
            audio_tensor = data.cpu().numpy()
        else:
            print(f"Skipping {pth_file}: Unsupported format.")
            return None

        # Convert to mono if multi-channel (by averaging)
        if audio_tensor.ndim > 1:
            audio_tensor = np.mean(audio_tensor, axis=0)

        # Normalize to float32 in [-1, 1]
        audio_tensor = normalize_audio(audio_tensor)

        sample_rate = 44100  # Set sample rate to 44100 Hz

        # --- Process the audio ---
        processed_audio = process_audio(audio_tensor, sample_rate)

        # Scale to int16 PCM range
        pcm_audio = np.int16(processed_audio * 32767)
        wav.write(output_wav, sample_rate, pcm_audio)
        print(f"Converted {pth_file} → {output_wav} (PCM 16-bit WAV)")
        return output_wav

    except Exception as e:
        print(f"Error converting {pth_file}: {e}")
        return None


def convert_flac_to_pcm_wav(flac_file, output_wav):
    """Convert a .flac file to a PCM 16-bit WAV file using soundfile."""
    try:
        # Read the FLAC file using soundfile (returns data and sample rate)
        data, sr = sf.read(flac_file)
        # Normalize (if multi-channel, average to mono)
        data = normalize_audio(data)

        # --- Process the audio ---
        processed_audio = process_audio(data, sr)

        # Scale to int16 PCM range
        pcm_audio = np.int16(processed_audio * 32767)
        wav.write(output_wav, sr, pcm_audio)
        print(f"Converted {flac_file} → {output_wav} (PCM 16-bit WAV)")
        return output_wav
    except Exception as e:
        print(f"Error converting {flac_file}: {e}")
        return None


# -------------------------
# Main processing loop for each available model
# -------------------------
for model_name in available_models:
    # Sanitize the model name for a safe Windows folder name.
    safe_model_name = re.sub(r'[<>:"/\\|?*]', '_', model_name)[:100]
    model_output_dir = os.path.join(output_dir, safe_model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Build the audio-separator command.
    cmd = [
        "audio-separator",
        "-m", model_name,
        "--output_dir", model_output_dir,
        "--use_autocast",  # Enables mixed precision GPU acceleration
        "--mdx_enable_denoise",  # Optional flag for improved separation quality
        "--log_level", "DEBUG",  # Debug mode to verify GPU usage
        input_wav
    ]

    print(f"Processing with {model_name}...")
    try:
        subprocess.run(cmd, check=True)
        print(f"Saved outputs in {model_output_dir}")

        # Now, search the output directory for .pth and .flac files.
        # Only process files that are instrumental tracks (their filenames include "instrumental").
        for file in os.listdir(model_output_dir):
            file_path = os.path.join(model_output_dir, file)

            # Check if this file is a instrumental track.
            if "instrumental" not in file.lower():
                print(f"Skipping non-instrumental track: {file}")
                os.remove(file_path)
                continue

            if file.endswith(".pth"):
                wav_path = file_path.replace(".pth", ".wav")
                converted = convert_pth_to_pcm_wav(file_path, wav_path)
                if converted:
                    os.remove(file_path)  # Remove the .pth file after successful conversion.
            elif file.endswith(".flac"):
                wav_path = file_path.replace(".flac", ".wav")
                converted = convert_flac_to_pcm_wav(file_path, wav_path)
                if converted:
                    os.remove(file_path)  # Remove the original FLAC file.
    except subprocess.CalledProcessError as e:
        print(f"Error processing {model_name}: {e}")

print("Processing complete. All outputs have been converted to PCM WAV format (instrumental tracks only).")

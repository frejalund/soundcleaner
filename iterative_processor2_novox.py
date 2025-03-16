import os
import shutil
import subprocess
import sys

from novox import find_best_match  # Import the function from your vox.py module


# -------------------------------
# Configuration
# -------------------------------
# These paths remain fixed for every iteration.
INPUT_WAV = r"C:\Users\cokal\Desktop\noise\noise.wav"  # Input for processor3.py
OUTPUT_DIR = r"C:\Users\cokal\Desktop\noise\test"  # Where processor3.py writes its outputs
ORIGINAL_WAV_PATH = r"C:\Users\cokal\Desktop\noise\engine.wav"  # Reference file for vox comparison

# Log file (you can change its location/name as desired)
LOG_FILE = r"C:\Users\cokal\Desktop\noise\best_log.txt"


def clear_directory(directory):
    """
    Remove all files and subdirectories from the given directory.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(directory, exist_ok=True)


def log_best_choice(iteration, best_file, best_score):
    """
    Append the best file path and its score for this iteration to a log file.
    """
    try:
        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"Iteration {iteration}: Best file: {best_file}, Score: {best_score:.4f}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")




def main():
    # Initialize the "best" score (distance) to a very low number.
    best_distance = -1.0
    iteration = 0

    while True:
        iteration += 1
        if iteration==6:
            break
        print(f"\n=== Iteration {iteration} ===")

        # ------------------------------------------------------------------
        # Step 1: Clean the output directory so that noise\test is empty.
        # ------------------------------------------------------------------
        print("Cleaning output directory...")
        clear_directory(OUTPUT_DIR)

        # ------------------------------------------------------------------
        # Step 2: Execute processor3.py.
        #
        # The processor3.py script uses INPUT_WAV as its input and populates
        # the OUTPUT_DIR with subfolders/files.
        # ------------------------------------------------------------------
        print("Running processor3.py...")
        try:
            # Using sys.executable ensures that the same Python interpreter is used.
            subprocess.run([sys.executable, "processor3.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing processor3.py: {e}")
            break

        # ------------------------------------------------------------------
        # Step 3: Run the vox analysis.
        #
        # The find_best_match() function (from vox.py) compares ORIGINAL_WAV_PATH
        # with all .wav files under OUTPUT_DIR and returns the best matching file
        # and its associated score.
        # ------------------------------------------------------------------
        print("Running vox analysis...")
        best_file, best_score = find_best_match(ORIGINAL_WAV_PATH, OUTPUT_DIR)
        if best_file is None:
            print("No valid best file found by vox. Exiting loop.")
            break

        print(f"Iteration {iteration}: Best file: {best_file}, Score: {best_score:.4f}")

        # ------------------------------------------------------------------
        # Step 4: Log the best file and score to a text file.
        # ------------------------------------------------------------------
        log_best_choice(iteration, best_file, best_score)

        # ------------------------------------------------------------------
        # Step 5: Replace the current input file with the best output,
        # and also save a history copy.
        #
        # The best processed file is copied to be the new INPUT_WAV (overwriting
        # the previous file) and also copied to a history file with a name like
        # h1.wav, h2.wav, h3.wav, etc.
        # ------------------------------------------------------------------
        try:
            # Define a history directory in the same folder as INPUT_WAV.
            history_dir = os.path.join(os.path.dirname(INPUT_WAV), "history")
            os.makedirs(history_dir, exist_ok=True)

            # Determine the next history file name.
            history_files = [
                f for f in os.listdir(history_dir)
                if f.lower().endswith(".wav") and f.startswith("h")
            ]
            numbers = []
            for f in history_files:
                num_str = f[1:-4]  # Remove the leading 'h' and trailing '.wav'
                try:
                    numbers.append(int(num_str))
                except:
                    pass
            next_number = max(numbers) + 1 if numbers else 1
            history_filename = os.path.join(history_dir, f"h{next_number}.wav")

            # Copy the best_file to both INPUT_WAV and the history file.
            shutil.copy(best_file, INPUT_WAV)
            shutil.copy(best_file, history_filename)
            print(f"Updated input file: {best_file} → {INPUT_WAV}")
            print(f"Saved history copy as: {history_filename}")
        except Exception as e:
            print(f"Error copying best file to input or history: {e}")
            break

        # ------------------------------------------------------------------
        # Step 6: Decide whether to continue the loop.
        #
        # If the new best score is greater than the previously recorded best_distance,
        # update best_distance and continue iterating. Otherwise, stop.
        # ------------------------------------------------------------------
        print(f"Score improved from {best_distance:.4f} to {best_score:.4f}. Continuing loop.")
        best_distance = best_score

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()

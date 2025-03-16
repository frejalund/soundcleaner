# Audio Sample Cleaning Program

This program was created entirely by me to separate the noise from the vocals in an iterative manner using the latest AI models.
While surfing online forums, I realized that people who were victims of the sonic weapon attack during the peaceful protest in Belgrade on March 15th, 2025, are traumatized and need to hear the sound emitted by the sonic weapon.

Here is the final result after 2 iterations (the program may over-do, so you have to monitor each output):

Iteration 1:




Most of the protesters were young adults and children, and there were 1 million of people or more during a peaceful protest that could have ended in a stampede. Here is the original video of the incident:


https://github.com/user-attachments/assets/a110b899-da2a-4665-ad6a-b7a481f96b22


## Overview

The process involves providing two audio samples:

1. **Sample to Clean:** The audio sample containing unwanted noise.
2. **Reference Sample:** An audio sample that represents the desired clean output.

The program compares the 'Sample to Clean' with the 'Reference Sample' and applies iterative processing to reduce noise, aiming to produce an output that closely resembles the reference.

## Reference Audio

For the cleaning process, a jet engine sound sample of less than 1 second is used for data comparison. The sample is sourced from Pixabay:

- **Source:** [Pixabay Sound Effects - Jet Engine](https://pixabay.com/sound-effects/search/jet-engine/)
- **File:** `starting-of-jet-engine-286108.mp3`
- **Timestamp:** Use the segment from 1:07 to 1:08 for the clean jet engine sound (excluding the powering down phase).

## Iterative Processing

The program employs a series of models to iteratively clean the audio sample. Below are the iterations performed:

```plaintext
**Iteration 1:**
- Best File: `C:\Users\cokal\Desktop\noise\test\MGM_HIGHEND_v4.pth\noise_(Instrumental)_MGM_HIGHEND_v4.wav`
- Score: 0.0274

**Iteration 2:**
- Best File: `C:\Users\cokal\Desktop\noise\test\3_HP-Vocal-UVR.pth\noise_(Instrumental)_3_HP-Vocal-UVR.wav`
- Score: 0.0429



Use GPT to figure out what the program really does and how it works.  You will need to hardcode the paths to wave files and folders, since I was focusing in the end result and not the best coding practices.


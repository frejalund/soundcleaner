# soundcleaner
This program was created entirely by me to separate the noise from the vocals in an iterative manner using the latest AI models.
While surfing online forums, I realized that people who were victims of the sonic weapon attack during the peaceful protest in Belgrade on March 15th, 2025, are traumatized and need to hear the sound emitted by the sonic weapon. Most of the protesters were young adults and children, and there were 1 million of people or more during a peaceful protest that could have ended in a stampede. Here is the original video of the incident:


https://github.com/user-attachments/assets/a110b899-da2a-4665-ad6a-b7a481f96b22



This works by providing the samlpe you want to clean, and the representative sample that shows what the result should sound like.

The jet engine sound sample of less than 1 second is used for data comparison:
https://pixabay.com/sound-effects/search/jet-engine/
file: starting-of-jet-engine-286108.mp3
Location: 1:07 to 1:08 for the clean sound of a jet engine (powering down does not matter)


The result which I have obtained is (sound only):


The models that have been used are (in this exact sequence, you do not need more than 5 iterations while cleaning the samples):

Iteration 1: Best file: C:\Users\cokal\Desktop\noise\test\MGM_HIGHEND_v4.pth\noise_(Instrumental)_MGM_HIGHEND_v4.wav, Score: 0.0274

Iteration 2: Best file: C:\Users\cokal\Desktop\noise\test\3_HP-Vocal-UVR.pth\noise_(Instrumental)_3_HP-Vocal-UVR.wav, Score: 0.0429



Use GPT to figure out what the program really does and how it works.  You will need to hardcode the paths to wave files and folders, since I was focusing in the end result and not the best coding practices.


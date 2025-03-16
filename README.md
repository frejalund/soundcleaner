# soundcleaner
This program was created entirely by me to separate the noise from the vocals in an iterative manner using the latest AI models.
While surfing online forums, I realized that people who were victims of the sonic weapon attack during the peaceful protest in Belgrade on March 15th, 2025, are traumatized and need to hear the sound emitted by the sonic weapon. Here is the original video of the incident:


https://github.com/user-attachments/assets/a110b899-da2a-4665-ad6a-b7a481f96b22



This works by providing the samlpe you want to clean, and the representative sample that shows what the result should sound like.

The jet engine sound sample of less than 1 second is used for data comparison:
https://pixabay.com/sound-effects/search/jet-engine/
file: starting-of-jet-engine-286108.mp3
Location: 1:07 to 1:08 for the clean sound of a jet engine (powering down does not matter)


The result which I have obtained is (sound only):


The models that have been used are (in this exact sequence):


Use GPT to figure out what the program really does and how it works.  You will need to hardcode the paths to wave files and folders, since I was focusing in the end result and not the best coding practices.


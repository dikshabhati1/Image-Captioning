# Image-Captioning

# Problem
Image caption Generator is a research area of Artificial Intelligence that deals with image understanding and a language description for that image. Generating well-formed sentences requires both syntactic and semantic understanding of the language. Being able to describe the content of an image using accurately formed sentences is a very challenging task, but it could also have a great impact, by **helping visually impaired people better understand the content of images**. 

# Approach
This problem uses an Encoder-Decoder model. Here encoder model will combine both the encoded form of the image and the encoded form of the text caption and feed to the decoder.
Our model will treat CNN as the ‘image model’ and the RNN/LSTM as the ‘language model’ to encode the text sequences of varying length. The vectors resulting from both the encodings are then merged and processed by a Dense layer to make a final prediction.

Create a merge architecture in order to keep the image out of the RNN/LSTM and thus be able to train the part of the neural network that handles images and the part that handles language separately, using images and sentences from separate training sets. 

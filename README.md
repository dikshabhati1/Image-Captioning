# Image-Captioning

# Problem
Image caption Generator is a research area of Artificial Intelligence that deals with image understanding and a language description for that image. Generating well-formed sentences requires both syntactic and semantic understanding of the language. Being able to describe the content of an image using accurately formed sentences is a very challenging task, but it could also have a great impact, by **helping visually impaired people better understand the content of images**. 

# Flask Demo
I have made a flask web app in which I have used google translate API to convert into the captions into different languages and Flask gtts to convert the captions into audio
<br>
<br>

**Note**: The below results are gif and images in which you are not able to hear the audio.If you want to hear audio results click on this ![video link](https://user-images.githubusercontent.com/70757239/124862115-6ecd3300-dfd2-11eb-8d87-87a630ad6fa8.mp4)<br>

![](https://github.com/dikshabhati1/Image-Captioning/blob/main/results/ezgif.com-gif-maker%20(1).gif)<br>

![](results/result1.JPG)<br>


![](results/result2.JPG)<br>


# Approach
This problem uses an Encoder-Decoder model. Here encoder model will combine both the encoded form of the image and the encoded form of the text caption and feed to the decoder.
Our model will treat CNN as the ‘image model’ and the RNN/LSTM as the ‘language model’ to encode the text sequences of varying length. The vectors resulting from both the encodings are then merged and processed by a Dense layer to make a final prediction.Create a merge architecture in order to keep the image out of the RNN/LSTM and thus be able to train the part of the neural network that handles images and the part that handles language separately, using images and sentences from separate training sets. 
Below is the architecture of our model

![](model-architecture.png)

# Results
Due to memeory issues the model is trained on 1300 images with 150 epochs and got a accuracy around 89%

![](results/accuracy.png)

# Reference
This [github repository](https://github.com/zhjohnchan/awesome-image-captioning) contains comprehensive collection of deep learning research papers from all premier conferences of image captioning and related area.

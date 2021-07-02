from flask import Flask, render_template, request
import cv2
from flask_gtts import gtts
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np


vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


embedding_size = 128
vocab_size = len(vocab)+1
max_len = 37

image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')


incept_model = ResNet50(include_top=True)
# take the 2nd last layer 
last = incept_model.layers[-2].output
cnn_model = Model(inputs = incept_model.input,outputs = last)


app = Flask(__name__)
gtts(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    
    global model, vocab, inv_vocab
    
    if request.method == 'POST':


        img = request.files['file']
        lang = request.form['Language']
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(img.filename))
        img.save(file_path)

   
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        image = np.reshape(image, (1,224,224,3))
  
    
        incept = cnn_model.predict(image).reshape(1,2048)

        text_in = ['startofseq']
        result = ''

        count = 0
        while tqdm(count < 20):

            count += 1

            encoded = []
            for i in text_in:
                encoded.append(vocab[i])

            padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)
            sampled_index = np.argmax(model.predict([incept, padded]))
            sampled_word = inv_vocab[sampled_index]

            if sampled_word != 'endofseq':
                result = result + ' ' + sampled_word

            text_in.append(sampled_word)
        
        hi = 0
        en = 0
        fr = 0
        el = 0
        de = 0
        it = 0
        ja = 0
        ur = 0
        es = 0
        ru = 0



        if lang == 'hi':
            hi = 1
            en = 0
            fr = 0
            el = 0
            de = 0
            it = 0
            ja = 0
            ur = 0
            es = 0
            ru = 0
        elif lang=='en':
            hi = 0
            en = 1
            fr = 0
            el = 0
            de = 0
            it = 0
            ja = 0
            ur = 0
            es = 0
            ru = 0  
        elif lang=='fr':
            hi = 0
            en = 0
            fr = 1
            el = 0
            de = 0
            it = 0
            ja = 0
            ur = 0
            es = 0
            ru = 0   
        elif lang=='el':
            hi = 0
            en = 0
            fr = 0
            el = 1
            de = 0
            it = 0
            ja = 0
            ur = 0
            es = 0
            ru = 0 
        elif lang=='de':
            hi = 0
            en = 0
            fr = 0
            el = 0
            de = 1
            it = 0
            ja = 0
            ur = 0
            es = 0
            ru = 0 
        elif lang=='it':
            hi = 0
            en = 0
            fr = 0
            el = 0
            de = 0
            it = 1
            ja = 0
            ur = 0
            es = 0
            ru = 0 
        elif lang=='ja':
            hi = 0
            en = 0
            fr = 0
            el = 0
            de = 0
            it = 0
            ja = 1
            ur = 0
            es = 0
            ru = 0 
        elif lang=='ur':
            hi = 0
            en = 0
            fr = 0
            el = 0
            de = 0
            it = 0
            ja = 0
            ur = 1
            es = 0
            ru = 0 
        elif lang=='es':
            hi = 0
            en = 0
            fr = 0
            el = 0
            de = 0
            it = 0
            ja = 0
            ur = 0
            es = 1
            ru = 0      
        else:
            hi = 0
            en = 0
            fr = 0
            el = 0
            de = 0
            it = 0
            ja = 0
            ur = 0
            es = 0
            ru = 1   



        
        translater = google_translator()
        result = translater.translate(result,lang_tgt=lang)

                    
        #language='en'
        #speech = gTTS(text = result, lang = language, slow = False)
        #speech.save('uploads/audio.mp3')
        #sound_file = 'uploads/audio.mp3'

        return result
    
    return None


if __name__=="__main__":
    app.run(debug=True)

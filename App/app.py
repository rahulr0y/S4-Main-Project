import streamlit as st
from cv2 import imread
from pydub import AudioSegment
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from skimage.color import rgb2gray


def to_spec():
    f = 'sample.mp3'
    aud, sr = librosa.load(f)
    mel_spectrogram = librosa.feature.melspectrogram(aud, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    librosa.display.specshow(log_mel_spectrogram, sr=sr)
    fig.savefig('temp.png', transparent=True)


def inp():
    img = imread('t.png')
    # img = Image.open('temp.png')
    img = np.array(img)
    img = rgb2gray(img)
    img = np.array(img)
    img = img.reshape(1, 540, 1080, 1)

    model = tf.keras.models.load_model('model.h5')
    arr = model.predict(img)
    print(arr)
    k = np.argmax(arr)
    return k


def main():
    class_names = ['flute', 'viola', 'cello', 'oboe', 'trumpet', 'saxophone']

    st.title('Music Instruments Classification')

    aud_file = st.file_uploader('Choose an audio file', type='mp3')

    if aud_file:
        aud = AudioSegment.from_mp3(aud_file)
        aud.export('sample.mp3', format='mp3')

        aud_file = open("sample.mp3", "rb")
        st.audio(aud_file)

        to_spec()
        st.write("Mel-Spectrogram")

        img = Image.open("temp.png")
        st.image(img)

        image = Image.open('temp.png')
        sunset_resized = image.resize((1080, 540))
        sunset_resized.save('t.png')

        k = inp()
        # class_names[k]

        st.write(f"The Instrument in the audio is {class_names[k]}")


main()

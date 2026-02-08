import os
import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model

IMG_SIZE = 224

def read_image(path, img_size=IMG_SIZE):
    img = load_img(path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    return img

def load_captions(caption_path):
    df = pd.read_csv(caption_path)
    df.columns = ["image","caption"]
    return df

def text_preprocessing(df):
    df['caption'] = df['caption'].str.lower()
    df['caption'] = df['caption'].str.replace('[^a-z ]','', regex=True)
    df['caption'] = df['caption'].str.replace('\s+',' ', regex=True)

    df['caption'] = "startseq " + df['caption'] + " endseq"
    return df

def create_tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer

def get_feature_extractor():
    base_model = DenseNet201()
    fe = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return fe

def extract_features(image_dir, images, fe):
    features = {}
    for image in images:
        path = os.path.join(image_dir, image)
        img = read_image(path)
        img = np.expand_dims(img, axis=0)
        feature = fe.predict(img, verbose=0)
        features[image] = feature
    return features

def save_tokenizer(tokenizer, path="tokenizer.pkl"):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path="tokenizer.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

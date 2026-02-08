import os
import numpy as np
import pickle
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Reshape, concatenate, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import *

# Paths
PROJECT_PATH = "data"
IMAGE_PATH = os.path.join(PROJECT_PATH, "Images")
CAPTION_PATH = os.path.join(PROJECT_PATH, "captions.txt")

# Load captions
df = load_captions(CAPTION_PATH)

# Clean text
df = text_preprocessing(df)

captions = df['caption'].tolist()

# Tokenizer
tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1

# Max length
max_length = max(len(c.split()) for c in captions)

# Save tokenizer and max_length
save_tokenizer(tokenizer)
pickle.dump(max_length, open("max_length.pkl","wb"))

print("Vocab Size:", vocab_size)
print("Max Length:", max_length)

# Unique images
images = df['image'].unique().tolist()[:1000]

# Feature extraction
print("Extracting Image Features...")
fe = get_feature_extractor()
features = extract_features(IMAGE_PATH, images, fe)
fe.save("feature_extractor.keras")

# Data Generator
def data_generator(df, features, tokenizer, max_length, vocab_size):

    while True:
        for _, row in df.iterrows():
            image = row['image']
            caption = row['caption']

            feature = features[image][0]
            seq = tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):
                in_seq = seq[:i]
                out_seq = seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                yield (np.array([feature]), np.array([in_seq])), np.array([out_seq])

# MODEL ARCHITECTURE
input1 = Input(shape=(1920,))
input2 = Input(shape=(max_length,))

img_features = Dense(256, activation='relu')(input1)
img_features_reshaped = Reshape((1,256))(img_features)

sentence_features = Embedding(vocab_size,256)(input2)
merged = concatenate([img_features_reshaped, sentence_features], axis=1)

sentence_features = LSTM(256)(merged)
x = Dropout(0.5)(sentence_features)
x = add([x,img_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model([input1,input2], output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

caption_model.summary()

# TRAIN MODEL
steps = len(df)
generator = data_generator(df, features, tokenizer, max_length, vocab_size)

print("Training Started...")
caption_model.fit(generator, epochs=20, steps_per_epoch=steps)

# Save trained model
caption_model.save("model.keras")

print("Training Complete and Model Saved!")

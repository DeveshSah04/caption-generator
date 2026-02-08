import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import *
import pickle

def remove_repeated_words(text):
    words = text.split()
    new_words = []
    for w in words:
        if len(new_words) == 0 or w != new_words[-1]:
            new_words.append(w)
    return " ".join(new_words)

def generate_caption_beam_search(feature, model, tokenizer, max_length, beam_index=3):

    start = [tokenizer.word_index['startseq']]
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            seq = pad_sequences([s[0]], maxlen=max_length)
            preds = model.predict([feature, seq], verbose=0)[0]

            word_preds = np.argsort(preds)[-beam_index:]

            for w in word_preds:
                next_seq = s[0] + [w]
                prob = s[1] + preds[w]
                temp.append([next_seq, prob])

        start_word = sorted(temp, key=lambda l: l[1])
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]

    caption_words = []
    for idx in start_word:
        word = tokenizer.index_word.get(idx)
        if word == "endseq":
            break
        if word != "startseq":
            caption_words.append(word)

    return " ".join(caption_words)

def generate_caption(image_path):

    model = load_model("model.keras", compile=False)
    fe = load_model("feature_extractor.keras", compile=False)

    tokenizer = load_tokenizer()
    max_length = pickle.load(open("max_length.pkl","rb"))

    img = read_image(image_path)
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)

    caption_text = generate_caption_beam_search(feature, model, tokenizer, max_length)

    final_caption = caption_text.strip()
    final_caption = remove_repeated_words(final_caption)
    final_caption = final_caption.capitalize()

    if not final_caption.endswith("."):
        final_caption += "."

    return final_caption


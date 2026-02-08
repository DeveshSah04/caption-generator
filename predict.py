import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import *
import pickle


def generate_caption(image_path):

    model = load_model("model.keras", compile=False)
    fe = load_model("feature_extractor.keras", compile=False)

    tokenizer = load_tokenizer()
    max_length = pickle.load(open("max_length.pkl","rb"))

    img = read_image(image_path)
    img = np.expand_dims(img, axis=0)
    feature = fe.predict(img, verbose=0)

    in_text = "startseq"

    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)

        yhat = model.predict([feature, seq], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break

        in_text += " " + word
        if word == "endseq":
            break

    final_caption = in_text.replace("startseq","").replace("endseq","").strip()

# Capitalize first letter
    final_caption = final_caption.capitalize()

    # Add period at end if missing
    if not final_caption.endswith("."):
        final_caption += "."

    return final_caption



if __name__ == "__main__":
    print(generate_caption("data/Images/667626_18933d713e.jpg"))

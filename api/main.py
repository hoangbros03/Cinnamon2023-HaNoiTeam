import time
import unicodedata
from typing import Optional

import torch
from fastapi import FastAPI
from models.transformers.model import Transformer
from pydantic import BaseModel

from utils.vocab_word import Vocab

# CONSTANT VARIABLES
TGT_VOCAB_PATH = "../utils/vocab/tokenize_tone.txt"
SRC_VOCAB_PATH = "../utils/vocab/tokenize_notone.txt"
CHECKPOINT_PATH = "../checkpoints/model_best.pt"


def load_model():
    """Load model transformer"""
    # Load the vocab
    tgt_vocab = Vocab(TGT_VOCAB_PATH)
    src_vocab = Vocab(SRC_VOCAB_PATH)

    # Model config
    d_model = 512
    n_head = 8
    n_hidden = 2048
    n_layer = 6
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    checkpoint_path = CHECKPOINT_PATH
    n_vocab_src = len(src_vocab)
    n_vocab_tgt = len(tgt_vocab)

    model = Transformer(
        n_vocab_src,
        n_vocab_tgt,
        d_model,
        n_hidden,
        n_head,
        n_layer,
        src_vocab.pad_id,
        device,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])
    model.to(device)
    model.eval()

    return model, tgt_vocab, src_vocab, device


class Text(BaseModel):
    """
    class of POST request
    """

    string: str
    preprocessed: Optional[bool] = True


app = FastAPI()


@app.get("/")
async def root():
    """
    Get request
    """
    return {"Message": "Welcome to Vietnamese diacritic restoration application!"}


@app.post("/predict")
async def restoration(text: Text):
    """
    Post request to restore the string
    """
    start_time = time.time()
    text_dict = text.model_dump()
    print(text_dict)
    model, tgt_vocab, src_vocab, device = load_model()
    time_cp1 = time.time()
    # if not text_dict.preprocessed:
    #     pass # Handle later if we have time
    sentence = text.string
    print(sentence)
    src_tokens = src_vocab.encode(sentence.split()).unsqueeze(0).to(device)
    tgt_tokens_tensor = torch.tensor([tgt_vocab.sos_id]).unsqueeze(0).to(device)
    predicted_sentence = []
    ori_sentence = sentence.split()

    try:
        for i in range(src_tokens.shape[1]):
            with torch.no_grad():
                output = model(src_tokens, tgt_tokens_tensor)
                next_token = output.argmax(dim=-1)[:, -1]
            if next_token.item() == tgt_vocab.eos_id:  # End token
                break

            tgt_tokens_tensor = torch.cat(
                (tgt_tokens_tensor, next_token.unsqueeze(1)), dim=-1
            )
            predicted_token = tgt_vocab.decode(
                [next_token.squeeze(0).cpu().numpy().tolist()]
            )
            if predicted_token == "<unk>":
                ori_token = ori_sentence[i]
                predicted_sentence.append(ori_token)
            else:
                predicted_sentence.append(predicted_token)
        text_return = " ".join(predicted_sentence)
    except IndexError:
        text_return = "An error occured"

    text_dict.update({"text_return": text_return})
    time_cp2 = time.time()
    print(f"Cp1: {time_cp1-start_time} cp2: {time_cp2-start_time}")
    return text_dict


@app.post("/remove")
async def removeDiacritic(text: Text):
    """
    Post request to remove the diacritics of the string
    """
    text_dict = text.model_dump()
    input = text.string
    normalized_input = unicodedata.normalize("NFD", input)
    stripped_input = "".join(
        c for c in normalized_input if not unicodedata.combining(c)
    )
    text_dict.update({"text_return": stripped_input})
    return text_dict

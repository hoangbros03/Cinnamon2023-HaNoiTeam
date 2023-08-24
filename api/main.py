import logging
import sys
import time
import unicodedata
from typing import Optional

import torch
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from file_download import UTIL_FOLDER_NAME
from pydantic import BaseModel

sys.path.append("../")  # noqa

from utils.vocab_word import Vocab

from models.transformers.model import Transformer  # isort: skip

# CONSTANT VARIABLES
TGT_VOCAB_PATH = f"{UTIL_FOLDER_NAME}/vn_words_tone.txt"
SRC_VOCAB_PATH = f"{UTIL_FOLDER_NAME}/vn_words_notone.txt"
CHECKPOINT_PATH = f"{UTIL_FOLDER_NAME}/model_best.pt"
WHITELIST = ["http://localhost:8501", "127.0.0.1"]

# Log configuration
logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("input_logger")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("input_log.log")
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# VARIABLES
loaded = False


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


@app.middleware("http")
async def validate(request: Request, next):
    """
    Middleware to check valid IP
    Parameters
    ----------
    request: The request given from frontend
    next: Next function
    Returns
    -------
    Message if not allowed, and nothing otherwise
    """
    # Client IP
    ip = str(request.client.host)

    # Check if it is allowed
    if ip not in WHITELIST:
        message = f"The IP {ip} trying to connect is not allowed."
        data = {"Message": message}
        logger.warning(f"Validate failed with message: {message}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=data)
    return await next(request)


model, tgt_vocab, src_vocab, device = load_model()


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

    time_cp1 = time.time()
    # if not text_dict.preprocessed:
    #     pass # Handle later if we have time
    sentence = text.string
    logger.info(f"Post request name: Predict. Input: {sentence}")
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
    logger.info(f"Post request name: Predict. Output: {text_return}")
    return text_dict


@app.post("/remove")
async def removeDiacritic(text: Text):
    """
    Post request to remove the diacritics of the string
    """
    text_dict = text.model_dump()
    input = text.string
    logger.info(f"Post request name: Remove. Input: {input}")
    normalized_input = unicodedata.normalize("NFD", input)
    stripped_input = "".join(
        c for c in normalized_input if not unicodedata.combining(c)
    )
    text_dict.update({"text_return": stripped_input})
    logger.info(f"Post request name: Remove. Output: {stripped_input}")
    return text_dict

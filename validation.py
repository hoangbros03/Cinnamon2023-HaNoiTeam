import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.data_loader import NMT_Dataset
from models.transformers.model import Transformer
from utils.vocab_word import Vocab


def seed_everything(SEED):
    """Seed everything"""
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    """Collate dataset function for each batch"""
    source = [i[0] for i in batch[0]]
    target = [i[1] for i in batch[0]]
    src_len = torch.as_tensor([i[2] for i in batch[0]])
    source = nn.utils.rnn.pad_sequence(
        source, batch_first=True, padding_value=src_vocab.pad_id
    )
    target = nn.utils.rnn.pad_sequence(
        target, batch_first=True, padding_value=tgt_vocab.pad_id
    )
    target_encoding = target[:, :-1]
    target_decoding = target[:, 1:]
    return source.long(), target_encoding.long(), target_decoding.long(), src_len.long()


def validate(model, dataloader, loss_fn, device):
    """Validating the model
    Returns: Loss value
    """
    model.eval()
    total_loss = 0
    for _, batch in enumerate(tqdm(dataloader)):
        x_encode, x_decode, y_train = (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
        )
        with torch.no_grad():
            out = model(x_encode, x_decode)
            out = out.reshape(-1, n_class)
            y_train = y_train.reshape(-1)
            loss = loss_fn(out, y_train)
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    seed_everything(69)

    # Load vocab
    tgt_vocab = Vocab("utils/vocab/vn_words_tone.txt")
    src_vocab = Vocab("utils/vocab/vn_words_notone.txt")

    # Model config
    d_model = 512
    n_head = 8
    n_hidden = 2048
    n_layer = 6
    dropout = 0.1
    n_epoch = 50
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lr = 0.001
    tokens_in_batch = 6000  # batch size in target language tokens

    data_eval = NMT_Dataset(
        "data/X_eval.txt", "data/Y_eval.txt", src_vocab, tgt_vocab, tokens_in_batch
    )

    dataloader_eval = DataLoader(
        data_eval, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    checkpoint = "checkpoints/model_best.pt"
    n_vocab_src = len(src_vocab)
    n_class = n_vocab_tgt = len(tgt_vocab)

    print("Loading data done!")
    print("Length of dataloader eval: ", len(dataloader_eval))

    model = Transformer(
        n_vocab_src,
        n_vocab_tgt,
        d_model,
        n_hidden,
        n_head,
        n_layer,
        src_vocab.pad_id,
        device,
        dropout,
    )
    model.to(device)
    validate(model, dataloader_eval, device, checkpoint)

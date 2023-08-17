import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from dataloader.data_loader import NMT_Dataset
from torch.utils.data import DataLoader
from models.transformers import Transformer
from models.transformers.schedul_optim import ScheduleOptimize
from utils.vocab_word import Vocab


def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
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


def accuracy(model, dataloader, device):
    model.eval()
    total_count = 0
    diff_count = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        x_encode, x_decode, y_train = batch[0].to(device), batch[1].to(device), batch[2]
        with torch.no_grad():
            out = model(x_encode, x_decode)
            out = out.reshape(-1, n_class)
            out = torch.argmax(out, axis=-1).cpu().numpy()
            y_train = y_train.reshape(-1).numpy()
            out = out - y_train
            diff = len(np.nonzero(out)[0])
            diff_count += diff
            total_count += len(y_train)
    return 1 - diff_count / total_count


def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    for idx, batch in enumerate(tqdm(dataloader)):
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


def train(
    model, dataloader_train, dataloader_eval, opt, loss_fn, device, checkpoint_path=None
):
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optim"])

    checkpoint_folder = "checkpoints"
    if os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder, exist_ok=True)

    print("Start training")
    count = 0
    best_eval_loss = 10000
    for epoch in range(n_epoch):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader_train)
        for _, batch in enumerate(pbar):
            count += 1
            x_encode, x_decode, y_train = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            # print(x_encode.shape, x_decode.shape)
            out = model(x_encode, x_decode)
            out = out.reshape(-1, n_class)
            y_train = y_train.reshape(-1)
            loss = loss_fn(out, y_train)
            loss.backward()
            if count % batches_per_step == 0:
                opt.update_and_step()
                opt.zero_grad()
            total_loss += loss.item()
            writer.add_scalar("batch_loss", loss.item(), count)
            pbar.set_postfix(
                {
                    "train loss": total_loss / len(dataloader_train),
                }
            )
        total_loss = total_loss / len(dataloader_train)
        eval_loss = validate(model, dataloader_eval, loss_fn, device)
        train_acc = accuracy(model, dataloader_train, device)
        val_acc = accuracy(model, dataloader_eval, device)
        writer.add_scalar("epoch", total_loss, epoch)
        writer.add_scalar("epoch", eval_loss, epoch)
        writer.add_scalar("epoch", train_acc, epoch)
        writer.add_scalar("epoch", val_acc, epoch)
        print(
            "Epoch: %s ---- Train loss: %f ---- Train accuracy: %f ---- Eval loss: %f ---- \
                Eval accuracy: %f"
            % (epoch, total_loss, train_acc, eval_loss, val_acc)
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "optim": opt.save_state_dict(),
            "loss": total_loss,
        }

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_checkpoint_path = os.path.join(checkpoint_folder, "model_best.pt")
            print(
                f"Best capture of model found at epoch {epoch}. Saving model_best.pt!"
            )
            torch.save(checkpoint, best_checkpoint_path)

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_folder, f"checkpoint_epoch_{epoch}.pt"
            )
            torch.save(checkpoint, checkpoint_path)

        if epoch == n_epoch - 1:
            last_checkpoint_path = os.path.join(checkpoint_folder, "model_last.pt")
            torch.save(checkpoint, last_checkpoint_path)

            print("Finished training. Saving the last checkpoint!")


if __name__ == "__main__":
    seed_everything(69)

    # Load vocab
    tgt_vocab = Vocab("utils/vocab/tgt_word_vocab.txt")
    src_vocab = Vocab("utils/vocab/src_word_vocab.txt")

    # Model config
    d_model = 512
    n_head = 8
    n_hidden = 2048
    n_layer = 6
    dropout = 0.1
    n_epoch = 50
    device = torch.device("cuda")
    lr = 0.001
    tokens_in_batch = 8000  # batch size in target language tokens
    batches_per_step = (
        40000 // tokens_in_batch
    )  # perform a training step, i.e. update parameters, once every so many batches

    # Visualize
    writer = SummaryWriter("runs/training-transformer")

    # Dataset
    data_train = NMT_Dataset(
        "data/X_train.txt",
        "data/Y_train.txt",
        src_vocab,
        tgt_vocab,
        tokens_in_batch,
    )
    data_eval = NMT_Dataset(
        "data/X_eval.txt", "data/Y_eval.txt", src_vocab, tgt_vocab, tokens_in_batch
    )
    dataloader_train = DataLoader(
        data_train, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    dataloader_eval = DataLoader(
        data_eval, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    checkpoint = None  # args.checkpoint
    n_vocab_src = len(src_vocab)
    n_class = n_vocab_tgt = len(tgt_vocab)
    # data = NMT_Dataset(vocab_vi=vocab_vi, vocab_ja=vocab_ja, vi_data_path=tgt_data_path, \
    # ja_data_path=src_data_path)

    print("Loading data done!")
    print("Length of dataloader train: ", len(dataloader_train))
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
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = LabelSmooth()
    opt = optim.Adam(model.parameters(), 1, betas=(0.9, 0.98), eps=1e-09)
    opt = ScheduleOptimize(opt, 0.2, d_model, 4000)
    train(model, dataloader_train, dataloader_eval, opt, loss_fn, device, checkpoint)

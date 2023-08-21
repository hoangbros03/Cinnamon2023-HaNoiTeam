import torch

from models.transformers import Transformer
from utils.vocab_word import Vocab


def predict(model, sentence, src_vocab, tgt_vocab, device):
    """Predict a input sentence
    Returns:
        predicted sentence with tone
    """
    model.eval()
    src_tokens = src_vocab.encode(sentence.split()).unsqueeze(0).to(device)
    tgt_tokens_tensor = torch.tensor([tgt_vocab.sos_id]).unsqueeze(0).to(device)

    predicted_sentence = []
    ori_sentence = sentence.split()

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

    translated_sentence = " ".join(predicted_sentence)
    return translated_sentence


def main():
    """Main function"""
    # Load vocab
    tgt_vocab = Vocab("utils/vocab/tgt_word_vocab.txt")
    src_vocab = Vocab("utils/vocab/src_word_vocab.txt")

    # Model config
    d_model = 512
    n_head = 8
    n_hidden = 2048
    n_layer = 6
    device = "cuda:0"

    checkpoint_path = "checkpoints/model_best.pt"
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

    sentence = "thu tuong ra quyet dinh tu chuc"
    translated_sentence = predict(model, sentence, src_vocab, tgt_vocab, device)
    print("Translated Sentence:", translated_sentence)


if __name__ == "__main__":
    main()

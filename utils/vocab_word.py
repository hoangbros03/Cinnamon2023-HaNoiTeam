import torch


class Vocab:
    def __init__(self, vocab_file):
        self.vocab_to_id = dict()
        self.id_to_vocab = dict()
        with open(vocab_file, "r", encoding="utf8") as rf:
            lines = rf.read().splitlines()
            # print(f"Number of vocab lines: {len(lines)}")
            for idx, line in enumerate(lines):
                self.vocab_to_id[line] = idx
                self.id_to_vocab[idx] = line
        self.sos_id = self["<sos>"]
        self.eos_id = self["<eos>"]
        self.pad_id = self["<pad>"]

    def __getitem__(self, token):
        return self.vocab_to_id[token]

    def __len__(self):
        return len(self.vocab_to_id)

    def get_all_vocab(self):
        return list(self.vocab_to_id.keys())

    def encode(self, words):
        """
        string convert to ids
        """
        ids = []
        for i in words:
            if i in self.vocab_to_id:
                ids.append(self.vocab_to_id[i])
            else:
                ids.append(self.vocab_to_id["<unk>"])

        return torch.as_tensor(
            [self.vocab_to_id["<sos>"]] + ids + [self.vocab_to_id["<eos>"]]
        )

    def decode(self, ids):
        """
        ids: list of id
        """
        words = []
        for i in ids:
            words.append(self.id_to_vocab[i])
        return " ".join(words)


if __name__ == "__main__":
    vocab = Vocab("utils/vocab/vocab.tgt")
    encode = vocab.encode("xin (c#hao) em iu".split())
    print(encode[0])
    decode = vocab.decode(encode.tolist())
    print(encode, "\n", decode, "\n")

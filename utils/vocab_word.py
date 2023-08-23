import torch


class Vocab:
    """Vocabulary class"""

    def __init__(self, vocab_file):
        """Initialize special tokens

        Args:
            vocab_file (txt file): txt file of Vietnamese tokenized vocab
        """
        self.vocab_to_id = {}
        self.id_to_vocab = {}
        with open(vocab_file, "r", encoding="utf8") as rf:
            lines = rf.read().splitlines()
            for idx, line in enumerate(lines):
                self.vocab_to_id[line] = idx
                self.id_to_vocab[idx] = line
        self.sos_id = self["<sos>"]
        self.eos_id = self["<eos>"]
        self.pad_id = self["<pad>"]
        self.unk_id = self["<unk>"]

    def __getitem__(self, token):
        """Get encoded token
        Returns: index of input token
        """
        return self.vocab_to_id[token]

    def __len__(self):
        """Get length of vocabulary"""
        return len(self.vocab_to_id)

    def get_all_vocab(self):
        """Get all vocabulary tokens"""
        return list(self.vocab_to_id.keys())

    def encode(self, words):
        """
        Convert token to index
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
        Convert index to token
        """
        words = []
        for i in ids:
            words.append(self.id_to_vocab[i])
        return " ".join(words)


if __name__ == "__main__":
    vocab = Vocab(
        "/home/khanh/workspace/Cinnamon2023-HaNoiTeam/utils/vocab/tokenize_tone.txt"
    )
    encode = vocab.encode("Toi di hoc  toi di hoc  toi di hoc ".split())
    decode = vocab.decode(encode.tolist())
    print(encode, "\n", decode, "\n")

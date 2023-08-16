import sys
from torch.utils.data import Dataset
from tqdm import tqdm
import itertools
from utils.vocab_word import Vocab

sys.path.append("..")


class NMT_Dataset(Dataset):
    def __init__(
        self,
        src_data_file,
        tgt_data_file,
        src_vocab,
        tgt_vocab,
        nums_token_in_batch=1500,
    ):
        super(NMT_Dataset, self).__init__()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.nums_token_in_batch = nums_token_in_batch
        src_data, tgt_data = self._get_data(src_data_file, tgt_data_file)
        assert len(src_data) == len(tgt_data), "lengths of src and tgt not equal"
        src_lengths = [len(x) for x in src_data]
        tgt_lengths = [len(x) for x in tgt_data]
        data = zip(src_data, tgt_data, src_lengths, tgt_lengths)
        self.data = sorted(data, key=lambda x: x[3])
        self.create_batch()

    def create_batch(self):
        chunks = [
            list(group) for _, group in itertools.groupby(self.data, key=lambda x: x[3])
        ]
        self.all_batches = []
        for chunk in chunks:
            chunk = list(chunk)
            chunk = sorted(chunk, key=lambda x: x[2])
            nums_seq = self.nums_token_in_batch // chunk[0][3]
            self.all_batches.extend(
                [chunk[i : i + nums_seq] for i in range(0, len(chunk), nums_seq)]
            )

    def __getitem__(self, idx):
        # return self.src_data[idx], self.tgt_data[idx]
        return self.all_batches[idx]

    def __len__(self):
        # return len(self.src_data)
        return len(self.all_batches)

    def __iter__(self):
        return

    def _get_data(self, src_file, tgt_file):
        src_data = []
        tgt_data = []
        with open(src_file, "r", encoding="utf8") as rf:
            lines = rf.readlines()
            for line in tqdm(lines):
                inp_encode = self.src_vocab.encode(line.strip().lower().split())
                src_data.append(inp_encode)
        with open(tgt_file, "r", encoding="utf8") as rf:
            lines = rf.readlines()
            for line in tqdm(lines):
                inp_decode = self.tgt_vocab.encode(line.strip().lower().split())
                tgt_data.append(inp_decode)
        for i in range(len(src_data)):
            assert src_data[i].shape == tgt_data[i].shape
        return src_data, tgt_data


if __name__ == "__main__":
    tgt_vocab = Vocab("utils/vocab/tgt_word_vocab.txt")
    src_vocab = Vocab("utils/vocab/src_word_vocab.txt")
    data_train = NMT_Dataset(
        "data/X_train_200k.txt", "data/Y_train_200k.txt", src_vocab, tgt_vocab
    )
    print(data_train[1])
    # print(tgt_vocab.decode(data_train[1][1]))

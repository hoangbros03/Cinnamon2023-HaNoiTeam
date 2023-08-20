import itertools

from torch.utils.data import Dataset
from tqdm import tqdm


class NMT_Dataset(Dataset):
    """Create dataset for training and evaluation"""

    def __init__(
        self,
        src_data_file,
        tgt_data_file,
        src_vocab,
        tgt_vocab,
        nums_token_in_batch=1500,
    ):
        """Initialize essential attributes"""
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
        """Create batch for loading tokens"""
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
        """Get each item token"""
        return self.all_batches[idx]

    def __len__(self):
        """Get length of dataset"""
        return len(self.all_batches)

    def _get_data(self, src_file, tgt_file):
        """Get data by convert token to index"""
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

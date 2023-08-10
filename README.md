#  Dataset Description

### Dataset Source and Preprocessing
The dataset was downloaded from [wikipedia](https://dumps.wikimedia.org/viwiki/) and was extracted using [wikiextractor](https://github.com/attardi/wikiextractor).

After extracting only text from the original dataset, some preprocessing steps were carried out:
- Split the text into sentences, each line has one sentence
- Remove numbers, special characters and punctuations
- Convert the sentences to lowercase
- Retain sentences with more than 10 words and less than 200 words
- No-tone sentences were generated by removing tone from the text.

### Dataset Split
After preprocessing, the dataset consists of a total of 4,315,334 sentences and is about 1.1 GB. The dataset is shuffled and divived into train/val/test with below config.
- Train dataset: 4295334 sentences
- Val dataset: 10000 sentences
- Test dataset: 10000 sentences

### How to use
The dataset zip file is stored on [google drive](https://drive.google.com/file/d/1cbtZO8T7RYEq0n58Jt-Be77Dj4XVFNqI/view?usp=drive_link). You can use below command for faster downloading.
```bash
gdown --fuzzy https://drive.google.com/file/d/1cbtZO8T7RYEq0n58Jt-Be77Dj4XVFNqI/view?usp=drive_link
```
Unzip the file, you will see the data with its ground truth files, for example with train set:
- train.tone: sentences with tone
- train.notone: sentences with no-tone

punctuations = [".", ",", "?", ";", "!", ":", "'", "(", ")", "[", "]", "\"", "...", "-", "~", "/", "@", "{", "}", "*"]
upperIndexes = []
puncIndexes = []
spaceIndexes = []
def preprocess(utf8_str):
    add = 0
    l = len(utf8_str)
    for i in range(l):
        if utf8_str[i].isupper():
            upperIndexes.append(i - add)
        elif utf8_str[i] in punctuations:
            puncIndexes.append((i - add, utf8_str[i]))
    for i in range(len(puncIndexes) - 1, -1, -1):
        utf8_str = utf8_str[:puncIndexes[i][0]] + utf8_str[puncIndexes[i][0] + 1:]
    for i in range(len(utf8_str)):
        if i > 0 and utf8_str[i] == " " and utf8_str[i - 1] == " ":
            spaceIndexes.append(i)
    for i in range(len(spaceIndexes) - 1, -1, -1):
        utf8_str = utf8_str[:spaceIndexes[i]] + utf8_str[spaceIndexes[i] + 1:]
    utf8_str = utf8_str.lower()
    return utf8_str
        
def postprocess(utf8_str):
    while spaceIndexes:
        ind = spaceIndexes.pop(0)
        utf8_str = " ".join((utf8_str[:ind], utf8_str[ind:]))
    while puncIndexes:
        ind, punc = puncIndexes.pop(0)
        utf8_str = punc.join((utf8_str[:ind], utf8_str[ind:]))
    while upperIndexes:
        ind = upperIndexes.pop(0)
        utf8_str = utf8_str[:ind] + utf8_str[ind].upper() + utf8_str[ind + 1:]
    for i in range(len(utf8_str) - 2, 0, -1):
        if utf8_str[i] == " " and utf8_str[i - 1] == " ":
            utf8_str = utf8_str[:i] + utf8_str[i + 1:]
    return utf8_str
    
if __name__ == "__main__":    
    s = "Chiều 20/8, tại Hội nghị gặp gỡ nhà đầu tư, CTCP Hoàng Anh    Gia Lai - HAGL (HAG) đã có những chia sẻ về mảng kinh doanh heo của tập đoàn. Chủ tịch Đoàn Nguyên Đức (bầu Đức)"
    s = preprocess(s)
    print(spaceIndexes)
    print(s)
    print(postprocess(s))

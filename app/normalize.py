import re
from collections import defaultdict
# punctuations excluding ending punctuation marks
punctuations = [",", ";", ":", "'", "(", ")", "[", "]", "\"", "-", "~", "/", "@", "{", "}", "*"] 
# ending punctuation marks
endPunctuations = [".", "?", "!", "..."]
upperIndexes = defaultdict(list)
puncIndexes = defaultdict(list)
endPuncList = []
spaceIndexes = defaultdict(list)
def preprocess(utf8_str):
    sentences = re.split('(?<=[.!?]) +', utf8_str)
    processed_sentences = []
    num = len(sentences)
    for j in range(num):
        sent = sentences[j]
        l = len(sent)
        for i in range(l):
            if sent[i].isupper():
                upperIndexes[j].append(i)
            elif sent[i] in punctuations:
                puncIndexes[j].append((i, sent[i]))
        for i in range(len(puncIndexes[j]) - 1, -1, -1):
            sent = sent[:puncIndexes[j][i][0]] + sent[puncIndexes[j][i][0] + 1:]
        for i in range(len(sent)):
            if i > 0 and sent[i] == " " and sent[i - 1] == " ":
                spaceIndexes[j].append(i)
        for i in range(len(spaceIndexes[j]) - 1, -1, -1):
            sent = sent[:spaceIndexes[j][i]] + sent[spaceIndexes[j][i] + 1:]
        sent = sent.lower()
        if sent[-1] in endPunctuations:
            endPuncList.append(sent[-1])
            sent = sent[:-1]
        else:
            endPuncList.append("")
        if sent[-1] == " ":
            sent = sent[:-1]
        processed_sentences.append(sent)
    return processed_sentences
        
def postprocess(sentences):
    num = len(sentences)
    output = ""
    for j in range(num):
        sent = sentences[j]
        while spaceIndexes[j]:
            ind = spaceIndexes[j].pop(0)
            sent = " ".join((sent[:ind], sent[ind:]))
        while puncIndexes[j]:
            ind, punc = puncIndexes[j].pop(0)
            sent = punc.join((sent[:ind], sent[ind:]))
        while upperIndexes[j]:
            ind = upperIndexes[j].pop(0)
            sent = sent[:ind] + sent[ind].upper() + sent[ind + 1:]
        for i in range(len(sent) - 1, 0, -1):
            if sent[i] == " " and sent[i - 1] == " ":
                sent = sent[:i-1] + sent[i:]
        if sent[-1] == " ":
            sent = sent[:-1]
        sent += endPuncList.pop(0)
        output += " " + sent
    return output
    
if __name__ == "__main__":    
    # s = "Chiều 20/8, tại Hội nghị gặp gỡ nhà đầu tư, CTCP Hoàng Anh    Gia Lai - HAGL (HAG) đã có những chia sẻ về mảng kinh doanh heo của tập đoàn . Chủ tịch Đoàn Nguyên Đức (bầu Đức)"
    s = "Việc Ukraine không thể có một   thành công   ! quyết định trên chiến trường đang làm dấy lên lo ngại cuộc xung đột đi vào ngõ cụt và hỗ trợ quốc tế dành cho Kiev có thể bị xói mòn  .  Vì đã quá mệt mỏi bởi xung đột, người dân Ukraine đang khao khát một chiến thắng vang dội và tại Washington, những lời kêu gọi cắt giảm viện trợ được cho là sẽ ngày càng nhiều lên khi cuộc bầu cử tổng thống Mỹ năm 2024 đến gần."
    s = preprocess(s)
    print(s)
    print(postprocess(s))
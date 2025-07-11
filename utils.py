# src/utils.py

import re
import pickle
from typing import List

def smart_tokenize(line: str) -> List[str]:
    """
    Tách token nâng cao:
      - Số nguyên hoặc số thập phân (vd. 37.5, 20) thành một token.
      - Chuỗi chữ/số liên tục (ví dụ Jiracek20) thành một token.
      - Dấu câu riêng.
    """
    pattern = r"\d+(?:\.\d+)?|\w+|[^\w\s]"
    return [t for t in re.findall(pattern, line, flags=re.UNICODE) if t.strip()]

def read_data(filepath: str, lowercase: bool = False):
    """
    Đọc file CoNLL (mỗi dòng 'token tag', câu cách nhau bởi dòng trống).
    Những dòng không có đủ token và tag sẽ bị bỏ qua (không lỗi).
    Trả về:
      - sentences: List[List[str]]
      - labels:    List[List[str]]
    """
    sentences, labels = [], []
    sent, labs = [], []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # nếu dòng trống: kết thúc một câu
            if not line:
                if sent:
                    sentences.append(sent)
                    labels.append(labs)
                    sent, labs = [], []
                continue

            parts = line.split()
            # Nếu không đủ 2 phần (token + tag), bỏ qua
            if len(parts) < 2:
                continue

            token = parts[0]
            tag = parts[-1]
            if lowercase:
                token = token.lower()

            sent.append(token)
            labs.append(tag)

    # Xử lý câu cuối nếu file không kết thúc bằng dòng trống
    if sent:
        sentences.append(sent)
        labels.append(labs)

    return sentences, labels

def build_maps(sentences, labels, min_freq: int = 1):
    """
    Tạo hai từ điển:
      - word2idx: ánh xạ token -> số nguyên (PAD=0, UNK=1).
      - label2idx: ánh xạ nhãn -> số nguyên (đảm bảo 'O' = 0).
    """
    from collections import Counter
    word_cnt = Counter(w for s in sentences for w in s)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w, cnt in word_cnt.items():
        if cnt >= min_freq:
            word2idx[w] = len(word2idx)

    label_set = sorted({l for labs in labels for l in labs})
    label2idx = {l: i for i, l in enumerate(label_set)}
    # đảm bảo 'O' có chỉ số 0
    if 'O' in label2idx and label2idx['O'] != 0:
        other = [k for k, v in label2idx.items() if v == 0][0]
        label2idx[other], label2idx['O'] = label2idx['O'], label2idx[other]

    return word2idx, label2idx

def save_pickle(obj, path: str):
    """Lưu object vào file .pkl."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    """Đọc object từ file .pkl."""
    with open(path, "rb") as f:
        return pickle.load(f)

def tokenize_ocr(filepath: str, lowercase: bool = False):
    """
    Đọc file văn bản OCR, tách từ bằng smart_tokenize().
    Trả về list các câu (mỗi câu là list token).
    """
    sentences = []
    with open(filepath, encoding="utf-8") as f:
        for ln in f:
            line = ln.strip()
            if not line:
                continue
            toks = smart_tokenize(line)
            if lowercase:
                toks = [t.lower() for t in toks]
            if toks:
                sentences.append(toks)
    return sentences

def extract_entities(tokens: List[str], labels: List[str]):
    """
    Gom thực thể theo định dạng BIO:
      Trả về dict: {entity_type: [entity_text,...]}
    """
    entities = {}
    cur, curt = [], None

    for tok, lab in zip(tokens, labels):
        if lab == 'O':
            if cur:
                entities.setdefault(curt, []).append(" ".join(cur))
                cur, curt = [], None
        else:
            tag, etype = lab.split('-', 1)
            if tag == 'B' or etype != curt:
                if cur:
                    entities.setdefault(curt, []).append(" ".join(cur))
                cur, curt = [tok], etype
            else:  # I-*
                cur.append(tok)

    if cur:
        entities.setdefault(curt, []).append(" ".join(cur))
    return entities

def load_ner_data(filepath):
    sentences = []
    labels = []
    with open(filepath, encoding="utf-8") as f:
        words = []
        labs = []
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append(words)
                    labels.append(labs)
                    words = []
                    labs = []
                continue
            splits = line.split()
            if len(splits) != 2:
                print(f"Lỗi dòng: {line}")
                continue
            word, lab = splits
            words.append(word)
            labs.append(lab)
        if words:
            sentences.append(words)
            labels.append(labs)
    return sentences, labels

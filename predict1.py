# src/predict1.py

import os
import sys
import torch
from model import BiLSTM_CRF
from utils import load_pickle, tokenize_ocr, extract_entities

THIS_DIR    = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
sys.path.append(THIS_DIR)

# 1. Load từ điển và mô hình
word2idx  = load_pickle(os.path.join(PROJECT_DIR, 'word2idx.pkl'))
label2idx = load_pickle(os.path.join(PROJECT_DIR, 'label2idx.pkl'))
idx2label = {v: k for k, v in label2idx.items()}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM_CRF(
    vocab_size=len(word2idx),
    tagset_size=len(label2idx),
    padding_idx=word2idx['<PAD>']
).to(device)

# Load checkpoint
model_path = os.path.join(PROJECT_DIR, 'bilstm_crf_best.pt')
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# 2. Hàm encode token
def encode(tokens):
    ids = [word2idx.get(t, word2idx['<UNK>']) for t in tokens]
    mask = [1] * len(ids)
    return (
        torch.LongTensor([ids]).to(device),
        torch.BoolTensor([mask]).to(device)
    )

# 3. Đọc và tokenize file OCR
ocr_file = os.path.join(DATA_DIR, 'Mau5.txt')
sentences = tokenize_ocr(ocr_file, lowercase=True)

# 4. Dự đoán và gom thuốc + liều lượng
found_drugs = set()
results = []
for tokens in sentences:
    X, M = encode(tokens)
    with torch.no_grad():
        pred_ids = model(X, mask=M)[0][:len(tokens)]
    labels = [idx2label[i] for i in pred_ids]
    entities = extract_entities(tokens, labels)

    drugs   = entities.get('DRUG', [])
    dosages = entities.get('DOSAGE', [])

    for i, drug in enumerate(drugs):
        drug_lower = drug.lower()
        if drug_lower in found_drugs:
            continue  # Bỏ qua nếu đã xuất hiện
        dosage = dosages[i] if i < len(dosages) else ""
        results.append((drug, dosage))
        found_drugs.add(drug_lower)

# 5. In kết quả
print("Thuốc và liều lượng phát hiện:")
for drug, dosage in results:
    if dosage:
        print(f"{drug} - {dosage}")
    else:
        print(f"{drug}")

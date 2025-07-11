import os
import torch
from utils import load_ner_data, tokenize_ocr
from predict1 import encode, extract_entities, model, idx2label


def predict(ocr_path: str):
    """
    Dự đoán và in các cặp DRUG - DOSAGE trên file OCR đã có sẵn.
    """
    sentences = tokenize_ocr(ocr_path, lowercase=True)
    seen = set()
    results = []
    for tokens in sentences:
        X, M = encode(tokens)
        pred_ids = model(X, mask=M)[0][:len(tokens)]
        pred_labels = [idx2label[i] for i in pred_ids]
        ents = extract_entities(tokens, pred_labels)

        drugs = ents.get('DRUG', [])
        dosages = ents.get('DOSAGE', [])
        unique_drugs = []
        for d in drugs:
            if d not in unique_drugs:
                unique_drugs.append(d)

        for i, drug in enumerate(unique_drugs):
            if drug in seen:
                continue
            dosage = dosages[i] if i < len(dosages) else None
            results.append((drug, dosage))
            seen.add(drug)

    # In kết quả
    for drug, dosage in results:
        if dosage:
            print(f"{drug} - {dosage}")
        else:
            print(drug)



def get_entities(labels: list) -> list:
    """
    Chuyển nhãn BIO thành danh sách entity spans: (start, end, type)
    """
    entities = []
    start, ent_type = None, None
    for i, lab in enumerate(labels):
        if lab.startswith('B-'):
            if start is not None:
                entities.append((start, i - 1, ent_type))
            start = i
            ent_type = lab.split('-', 1)[1]
        elif lab.startswith('I-'):
            continue
        else:
            if start is not None:
                entities.append((start, i - 1, ent_type))
                start, ent_type = None, None
    if start is not None:
        entities.append((start, len(labels) - 1, ent_type))
    return entities


def evaluate(dev_path: str):
    """
    Đánh giá model trên tập dữ liệu đã gán nhãn BIO.
    Tính Precision, Recall, F1-score, Accuracy.
    """
    sentences, true_labels_list = load_ner_data(dev_path)
    TP = FP = FN = 0
    total_tokens = correct_tags = 0

    for tokens, true_labels in zip(sentences, true_labels_list):
        X, M = encode(tokens)
        pred_ids = model(X, mask=M)[0][:len(tokens)]
        pred_labels = [idx2label[i] for i in pred_ids]

        true_ents = set(get_entities(true_labels))
        pred_ents = set(get_entities(pred_labels))

        TP += len(true_ents & pred_ents)
        FP += len(pred_ents - true_ents)
        FN += len(true_ents - pred_ents)

        total_tokens += len(true_labels)
        correct_tags += sum(1 for t, p in zip(true_labels, pred_labels) if t == p)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = correct_tags / total_tokens if total_tokens else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1_score:.4f}")
    print(f"Accuracy : {accuracy:.4f}")


if __name__ == "__main__":
    # Đường dẫn file cần thay tùy vào cấu trúc project của bạn
    ocr_file = r"C:/Ner_project/data/Mau2.txt"
    dev_file = r"C:/Ner_project/data/dev.txt"

    print("--- DỰ ĐOÁN TRÊN FILE OCR ---")
    predict(ocr_file)
    print("\n--- ĐÁNH GIÁ TRÊN FILE DEV ---")
    evaluate(dev_file)

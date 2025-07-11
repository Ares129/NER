# src/train.py

import os, sys, torch
from torch.utils.data import Dataset, DataLoader
from model import BiLSTM_CRF
from utils import read_data, build_maps, save_pickle

THIS_DIR    = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
sys.path.append(THIS_DIR)

# 1. Đọc dữ liệu (với nhãn DRUG + DOSAGE)
train_sents, train_labels = read_data(os.path.join(DATA_DIR, 'train5.txt'), lowercase=True)
valid_sents, valid_labels = read_data(os.path.join(DATA_DIR, 'valid5.txt'), lowercase=True)

# 2. Xây map và lưu lại
word2idx, label2idx = build_maps(train_sents + valid_sents, train_labels + valid_labels)
save_pickle(word2idx, os.path.join(PROJECT_DIR, 'word2idx.pkl'))
save_pickle(label2idx, os.path.join(PROJECT_DIR, 'label2idx.pkl'))

# 3. Dataset
class DrugDataset(Dataset):
    def __init__(self, s, l): self.sents, self.labels = s, l
    def __len__(self): return len(self.sents)
    def __getitem__(self, i): return self.sents[i], self.labels[i]

# 4. Collate: pad, mask, lengths
def collate_fn(batch):
    sents, labs = zip(*batch)
    lengths = [len(s) for s in sents]
    max_len = max(lengths)
    X,Y,M = [],[],[]
    for sent, lab in zip(sents, labs):
        ids  = [word2idx.get(w,1) for w in sent]
        tags = [label2idx[t]   for t in lab]
        mask = [1]*len(ids)
        pad  = max_len-len(ids)
        ids  += [word2idx['<PAD>']]*pad
        tags += [label2idx['O']]*pad
        mask += [0]*pad
        X.append(ids); Y.append(tags); M.append(mask)
    return (torch.LongTensor(X), torch.LongTensor(Y),
            torch.BoolTensor(M), torch.LongTensor(lengths))

train_loader = DataLoader(DrugDataset(train_sents, train_labels),
                          batch_size=16, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(DrugDataset(valid_sents, valid_labels),
                          batch_size=16, shuffle=False, collate_fn=collate_fn)

# 5. Khởi tạo model, optimizer, scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM_CRF(len(word2idx), len(label2idx), padding_idx=word2idx['<PAD>']).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

# 6. Huấn luyện
best_val = float('inf')
for epoch in range(1, 101):
    model.train(); total=0
    for Xb, Yb, Mb, Lb in train_loader:
        Xb, Yb, Mb, Lb = Xb.to(device), Yb.to(device), Mb.to(device), Lb.to(device)
        loss = model(Xb, tags=Yb, mask=Mb, lengths=Lb)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    avg_train = total/len(train_loader)

    model.eval(); vtotal=0
    with torch.no_grad():
        for Xb, Yb, Mb, Lb in valid_loader:
            Xb, Yb, Mb, Lb = Xb.to(device), Yb.to(device), Mb.to(device), Lb.to(device)
            vtotal += model(Xb, tags=Yb, mask=Mb, lengths=Lb).item()
    avg_val = vtotal/len(valid_loader)
    sched.step(avg_val)

    print(f"Epoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), os.path.join(PROJECT_DIR, 'bilstm_crf_best.pt'))
        




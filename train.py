import gc, os
import psutil
import joblib
import random
from tqdm import tqdm
import numpy  as np
import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn. model_selection import KFold, StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings, math
import matplotlib.pyplot as plt
from termcolor import colored
os.system("color")
warnings.filterwarnings("ignore")

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_dir", type = str,help="directory where weight files are saved/ to save files with / at last")
    parser.add_argument('-d',"--data_dir", type = str,help="where the competition data is there with / at last")
    parser.add_argument("--type", type = int,help="there are 6 types of network R E [1,7] - {4}")
    parser.add_argument("--pseudo", type = str,help="whether we need to train with pseudo label if so give the sub file else dont give this argument", default = None)
    parser.add_argument("--epochs", type = int,help="No of epoch", default = 100)
    parser.add_argument("--batch_size", type = int,help="Batch Size", default = 256)
    parser.add_argument("--adam", help="to use adam", action="store_true")
    return parser.parse_args()

args = arg()

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(colored('Perfectly Seeded ', 'green'))

seed_torch()
os.makedirs(args.weight_dir,exist_ok = True)

FOLD = 1
MAX_SEQ = 100
MIN_SAMPLES = 1
EMBED_DIM = 128
DROPOUT_RATE = 0.2
if args.type >=3:
    N_LAYER = 1
else:
    N_LAYER = 3
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 2e-3
if args.type != 1 and args.type < 4:
    NUM_STUDENTS = 5000
else:
    NUM_STUDENTS = 6000
EPOCHS = args.epochs
VERBOSE = 10
N_FEATURE = 2
TRAIN_BATCH_SIZE = args.batch_size
skf = KFold(random_state = 42)
PSEUDO_TRAIN = args.pseudo 
CV_METHOD = 0

if PSEUDO_TRAIN:
    pseudo_label = pd.read_csv(PSEUDO_TRAIN)
    pseudo_label['to_use'] = pseudo_label['correct'].apply(lambda x: 1 if (x > 0.8 or x < 0.2) else 0)
    pseudo_label['correct'] = (pseudo_label['correct'] > 0.5).astype('int32')

train_df = pd.read_csv(f'{args.data_dir}train.csv').sort_values(by = 'order_id')
test = pd.read_csv(f'{args.data_dir}test.csv')
sub = pd.read_csv(f'{args.data_dir}sample_output.csv')
print(colored('CSV UPLOADED SUCESSFULLY', 'green'))

if PSEUDO_TRAIN:
    pseudo_test = test.drop(['correct'], axis = 1).merge(pseudo_label, on  = 'row_id').sort_values(by = 'order_id')
    pseudo_test = pseudo_test[pseudo_test['to_use'] == 1].drop(['to_use'], axis = 1 )
    train_df = pd.concat([train_df, pseudo_test], axis = 0).sort_values(by = 'order_id')

train_df['feature_2']/=(3600*24)
train_df['feature_1']/=(3600*24)
test['feature_2']/=(3600*24)
test['feature_1']/=(3600*24)

skills = train_df[train_df.columns[4]].unique()
joblib.dump(skills, "skills.pkl.zip")
n_skill = len(skills) + 1
n_skill = 8092
print("number skills", len(skills))

group = train_df.groupby('student_id').apply(lambda r: (
                r['question_id'].values,
                r['bundle_id'].values,
                r['feature_3'].values,
                r['feature_4'].values,
                r['feature_5'].values,
                r[['feature_1', 'feature_2',]].values,
                r['correct'].values))
print(colored(' GROUPING FINISHED ', 'cyan'))
joblib.dump(group, f"{args.weight_dir}group.pkl.zip")
del train_df
gc.collect()

TRAIN_INDEX = list(group.index[:NUM_STUDENTS])
VALID_INDEX =  list(group.index[NUM_STUDENTS:])
train_group = group[group.index.isin(TRAIN_INDEX)]
valid_group = group[group.index.isin(VALID_INDEX)]

class SAINTDataset(Dataset):
    def __init__(self, group, n_skill, min_samples=1, max_seq=128):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            q,qb, qf3, qf4, qf5, qf, qa = group[user_id]
            if len(q) < min_samples:
                continue
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= min_samples:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], qa[:initial], qf[:initial], qb[:initial], qf3[:initial], qf4[:initial], qf5[:initial],)
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + (seq * self.max_seq)
                    end = start + self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], qa[start:end],qf[start:end], qb[start:end], qf3[start:end], qf4[start:end], qf5[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, qa, qf ,qb, qf3, qf4, qf5)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, qf_,qb_, qf3_, qf4_, qf5_ = self.samples[user_id]
        
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qf = np.zeros((self.max_seq,N_FEATURE), dtype=float)
        qb, qf3, qf4, qf5 = np.zeros(self.max_seq, dtype=int), np.zeros(self.max_seq, dtype=int), np.zeros(self.max_seq, dtype=int), np.zeros(self.max_seq, dtype=int)
        if seq_len == self.max_seq:
            q[:] = q_
            qa[:] = qa_
            qf[:] = qf_
            qb[:] = qb_
            qf3[:] = qf3_
            qf4[:] = qf4_
            qf5[:] = qf5_

        else:
            q[-seq_len:] = q_
            qa[-seq_len:] = qa_
            qf[-seq_len:] = qf_
            qb[-seq_len:] = qb_
            qf3[-seq_len:] = qf3_
            qf4[-seq_len:] = qf4_
            qf5[-seq_len:] = qf5_

        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_skill

        return x,qf[1:], q[1:], qa[1:], qb[1:], qf3[1:], qf4[1:], qf5[1:]

train_dataset = SAINTDataset(train_group, n_skill, min_samples=MIN_SAMPLES, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
valid_dataset = SAINTDataset(valid_group, n_skill, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)
print(colored(' Dataset and Dataloader Loaded Sucessfully ', 'green'))

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)



class SAINTModel(nn.Module):
    def __init__(self, n_skill, max_seq=128, embed_dim=128, dropout_rate=0.2, with_flstm = False):
        super(SAINTModel, self).__init__()
        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(2*n_skill+1, embed_dim)
        self.f3_embedding = nn.Embedding(11, embed_dim)
        self.f4_embedding = nn.Embedding(84, embed_dim)
        self.f5_embedding = nn.Embedding(4, embed_dim)
        
        self.pos_enc =  nn.Embedding(embed_dim, embed_dim)
        self.e_embedding = nn.Embedding(n_skill+1, embed_dim)
        self.b_embedding = nn.Embedding(5312, embed_dim)
        self.cat = nn.Sequential(
            nn.Linear(embed_dim*5, embed_dim),
            nn.LayerNorm(embed_dim),
        ) 
        self.cat1 = nn.Sequential(
            nn.Linear(embed_dim*4, embed_dim),
            nn.LayerNorm(embed_dim),
        ) 
        
        self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= N_LAYER, num_decoder_layers= N_LAYER, dropout = dropout_rate)
        
        self.gru = nn.LSTM(embed_dim, embed_dim, num_layers=N_LAYER, batch_first=False)
        if args.type >= 6:
            self.f_lstm = nn.LSTM(embed_dim, embed_dim, num_layers=N_LAYER, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_normal = nn.LayerNorm(embed_dim) 

        self.ffn = FFN(embed_dim)
        
        self.mlp1 = nn.Linear(N_FEATURE, 1024)
        self.mlp2 = nn.ReLU()
        self.mlp3 = nn.Linear(1024, 512)
        self.mlp4 = nn.ReLU()
        self.mlp5 = nn.Linear(512, embed_dim)
        if args.type >=6:
            self.pred = nn.Linear(embed_dim*2, 1)
        else:
            self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, x,feature, question_ids, bundle_ids, feature3, feature4, feature5):
        device = x.device 
        
        x1 = self.mlp1(feature)
        x1 = self.mlp2(x1)
        x1 = self.mlp3(x1)
        x1 = self.mlp4(x1)
        x1 = self.mlp5(x1)
        if args.type == 7:
            x1 = self.f_lstm(x1)[0]
        x2, x3,x4, x5 = self.f3_embedding(feature3) , self.f4_embedding(feature4) , self.f5_embedding(feature5) , self.b_embedding(bundle_ids)

        x = self.embedding(x)
        x = torch.cat([x,x1,x2,x3,x5], axis=2)
        x = self.cat(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).to(device)
        pos_x = self.pos_enc(pos_id)
        x = x + pos_x
        
        e = self.e_embedding(question_ids)
        e = torch.cat([e, x1,x2,x4], axis=2)
        e = self.cat1(e)
        e = e + pos_x
        
        x = x.permute(1, 0, 2)
        e = e.permute(1, 0, 2)
        
        att_mask = future_mask(x.size(0)).to(device)
        att_output = self.transformer( x,e, src_mask=att_mask, tgt_mask=att_mask, memory_mask = att_mask)
        if args.type != 3:
            att_output = self.gru(att_output)[0]
        att_output = self.layer_normal(att_output+e)
        att_output = att_output.permute(1, 0, 2)
        x = self.ffn(att_output)
        if args.type >= 6 or args.type <= 3:
            x = self.layer_normal(x + att_output)
        else:
            x = self.layer_normal(x + att_output+x1)
        if args.type >=6:
            x = torch.cat([x,x1], axis = -1)
        x = self.pred(x)
        return x.squeeze(-1)

def train_fn(model, dataloader, optimizer, scheduler, criterion, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        x = item[0].to(device).long()
        feature = item[1].to(device).float()
        target_id = item[2].to(device).long()
        label = item[3].to(device).float()
        bundle_id = item[4].to(device).long()
        feature3 = item[5].to(device).long()
        feature4 = item[6].to(device).long()
        feature5 = item[7].to(device).long()
        target_mask = (target_id != 0)

        optimizer.zero_grad()
        output= model(x,feature, target_id, bundle_id, feature3, feature4, feature5)
        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / (num_total+1e-6)
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc

def valid_fn(model, dataloader, criterion, device="cpu"):
    model.eval()

    valid_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        x = item[0].to(device).long()
        feature = item[1].to(device).float()
        target_id = item[2].to(device).long()
        label = item[3].to(device).float()
        bundle_id = item[4].to(device).long()
        feature3 = item[5].to(device).long()
        feature4 = item[6].to(device).long()
        feature5 = item[7].to(device).long()
        target_mask = (target_id != 0)

        output = model(x,feature, target_id, bundle_id, feature3, feature4, feature5)
        loss = criterion(output, label)
        valid_loss.append(loss.item())

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / (num_total+1e-6)
    auc = roc_auc_score(labels, outs)
    loss = np.mean(valid_loss)

    return loss, acc, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAINTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)
if args.adam:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)

best_auc = -np.inf
max_steps = 9999
step = 0
for epoch in range(EPOCHS):
    loss, acc, auc = train_fn(model, train_dataloader, optimizer,scheduler, criterion, device)
    print(colored("epoch - {}/{}  train: - {:.3f} acc - {:.3f} auc - {:.6f}".format(epoch+1, EPOCHS, loss, acc, auc),'magenta'))
    loss, acc, auc = valid_fn(model, valid_dataloader, criterion, device)
    print(colored("epoch - {}/{}  valid: - {:.3f} acc - {:.3f} auc - {:.6f}".format(epoch+1, EPOCHS, loss, acc, auc),'yellow'))
    if auc > best_auc:
        print(colored(f'\t - AUC Increased: from {best_auc} --> {auc}\n', 'green', attrs=['bold']))
        best_auc = auc
        step = 0
        torch.save(model.state_dict(), f"{args.weight_dir}sakt_model.pt")
    else:
        print(colored(f'\t - AUC Decreased: from {best_auc} <-- {auc}\n', 'red', attrs=['bold']))
        step += 1
        if step >= max_steps:
            break
    if epoch % VERBOSE == 0:
#         torch.save(model.state_dict(), f"{args.weight_dir}sakt_model_{epoch}.pt")
        pass
            
torch.save(model.state_dict(), f"{args.weight_dir}sakt_model_last.pt")

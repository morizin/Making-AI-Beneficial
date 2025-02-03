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
    parser.add_argument("--weight", type = str,help="direct path to weight file")
    parser.add_argument('-d',"--data_dir", type = str,help="where the competition data is there with / at last")
    parser.add_argument('-o',"--out_file", type = str,help="output file name with .csv at last")
    parser.add_argument("--type", type = int,help="there are 6 types of network R E [1,7] - {4}")
    parser.add_argument("--batch_size", type = int,help="Batch Size", default = 256)
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
VERBOSE = 10
N_FEATURE = 2
TRAIN_BATCH_SIZE = args.batch_size
skf = KFold(random_state = 42)
CV_METHOD = 0

train_df = pd.read_csv(f'{args.data_dir}train.csv').sort_values(by = 'order_id')
test = pd.read_csv(f'{args.data_dir}test.csv')
sub = pd.read_csv(f'{args.data_dir}sample_output.csv')
print(colored('CSV UPLOADED SUCESSFULLY', 'green'))

train_df['feature_2']/=(3600*24)
train_df['feature_1']/=(3600*24)
test['feature_2']/=(3600*24)
test['feature_1']/=(3600*24)

skills = train_df[train_df.columns[4]].unique()
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
del train_df
gc.collect()

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
        if with_flstm:
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAINTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)

model.to(device)
try:
    model.load_state_dict(torch.load(f'{args.weight}'), strict = True)
except:
    model = SAINTModel(n_skill, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE, with_flstm = True)
    model.to(device)
    model.load_state_dict(torch.load(f'{args.weight}'), strict = True)
    
model.eval()
prev_test_df = test.copy()

class TestDataset(Dataset):
    def __init__(self, samples, test_df, skills, max_seq=MAX_SEQ):
        super(TestDataset, self).__init__()
        self.samples = samples
        self.user_ids = [x for x in test_df["student_id"].unique()]
        self.test_df = test_df
        self.skills = skills
        self.n_skill = len(skills)
        self.max_seq = max_seq

    def __len__(self):
        return self.test_df.shape[0]

    def __getitem__(self, index):
        test_info = self.test_df.iloc[index]

        user_id = test_info["student_id"]
        target_id = test_info["question_id"]
        bundle_id = test_info["bundle_id"]
        feature = test_info[["feature_1", "feature_2"]].values  
        feature_3 = test_info["feature_3"]
        feature_4 = test_info["feature_4"]
        feature_5 = test_info["feature_5"]
        
        q = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        qf = np.zeros((self.max_seq,N_FEATURE), dtype=float)
        qb, qf3, qf4, qf5 = np.zeros(self.max_seq, dtype=int), np.zeros(self.max_seq, dtype=int), np.zeros(self.max_seq, dtype=int), np.zeros(self.max_seq, dtype=int)

        if user_id in self.samples.index:
            q_,qb_,qf3_,qf4_,qf5_,qf_,qa_ = self.samples[user_id]
            seq_len = len(q_)

            if seq_len >= self.max_seq:
                q  = q_[-self.max_seq:]
                qa = qa_[-self.max_seq:]
                qf = qf_[-self.max_seq:]
                qb = qb_[-self.max_seq:]
                qf3 = qf3_[-self.max_seq:]
                qf4 = qf4_[-self.max_seq:]
                qf5 = qf5_[-self.max_seq:]
            else:
                q[-seq_len:] = q_
                qa[-seq_len:] = qa_ 
                qf[-seq_len:] = qf_ 
                qb[-seq_len:] = qb_
                qf3[-seq_len:] = qf3_
                qf4[-seq_len:] = qf4_
                qf5[-seq_len:] = qf5_
        
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[1:].copy()
        x += (qa[1:] == 1) * self.n_skill
        
        questions = np.append(q[2:], [target_id])
        qf = np.append(qf[2:], [feature]).reshape(-1,N_FEATURE)
        qb = np.append(qb[2:], [bundle_id])
        qf3 = np.append(qf3[2:], [feature_3])
        qf4 = np.append(qf4[2:], [feature_4])
        qf5 = np.append(qf5[2:], [feature_5])
        
        return (x,qf,questions, qb, qf3, qf4, qf5)

prev_group = prev_test_df.groupby('student_id').apply(lambda r: (
                r['question_id'].values,
                r['bundle_id'].values,
                r['feature_3'].values,
                r['feature_4'].values,
                r['feature_5'].values,
    #             r[['feature_1', 'feature_2', "answered_correctly_student", 'sum', "answered_correctly_order", "answered_correctly_bundle", "answered_correctly_question"]].values,
                r[['feature_1', 'feature_2']].values,
                r['correct'].values))

# d = 0
for prev_user_id in tqdm(prev_group.index):
    if prev_user_id in group.index:
        group[prev_user_id] = (
            np.append(group[prev_user_id][0], prev_group[prev_user_id][0])[-MAX_SEQ:], 
            np.append(group[prev_user_id][1], prev_group[prev_user_id][1])[-MAX_SEQ:], 
            np.append(group[prev_user_id][2], prev_group[prev_user_id][2])[-MAX_SEQ:], 
            np.append(group[prev_user_id][3], prev_group[prev_user_id][3])[-MAX_SEQ:], 
            np.append(group[prev_user_id][4], prev_group[prev_user_id][4])[-MAX_SEQ:], 
            np.append(group[prev_user_id][5], prev_group[prev_user_id][5]).reshape(-1,N_FEATURE),
            np.append(group[prev_user_id][6], prev_group[prev_user_id][6])[-MAX_SEQ:]
        )

    else:
#         d += 1
        group[prev_user_id] = (
            prev_group[prev_user_id][0], 
            prev_group[prev_user_id][1],
            prev_group[prev_user_id][2],
            prev_group[prev_user_id][3],
            prev_group[prev_user_id][4],
            prev_group[prev_user_id][5],
            prev_group[prev_user_id][6],
        )

prev_test_df = test.copy()

test_dataset = TestDataset(group, test, skills)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

outs = []

for item in tqdm(test_dataloader):
    x = item[0].to(device).long()
    feature = item[1].to(device).float()
    target_id = item[2].to(device).long()
    bundle_id = item[3].to(device).long()
    feature3 = item[4].to(device).long()
    feature4 = item[5].to(device).long()
    feature5 = item[6].to(device).long()

    with torch.no_grad():
        output = model(x,feature, target_id, bundle_id, feature3, feature4, feature5)
    outs.extend(torch.sigmoid(output)[:, -1].view(-1).data.cpu().numpy())

sub['correct'] = outs
sub.to_csv(f'{args.out_file}', index = False)

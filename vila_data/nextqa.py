import os
import json
import pandas as pd
import numpy as np

def save_json(content, save_path):
    with open(save_path, 'w') as f:
        f.write(json.dumps(content, cls=NpEncoder))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


raw_train_csv = '/scratch_xijun/data/NExTVideo/nextqa/train.csv'
raw_val_csv = '/scratch_xijun/data/NExTVideo/nextqa/val.csv'
raw_train = pd.read_csv(raw_train_csv, delimiter=',')
raw_val = pd.read_csv(raw_val_csv, delimiter=',')
train = []
val = []
key = ['video', 'question', 'a0', 'a1', 'a2', 'a3', 'a4', 'answer', 'qid', 'type']
for i in range(len(raw_train)):
    data = {}
    for k in key:
        data[k] = raw_train.iloc[i][k]
    train.append(data)

for i in range(len(raw_val)):
    data = {}
    for k in key:
        data[k] = raw_val.iloc[i][k]
    val.append(data)

vid_map = json.load(open('/scratch_xijun/data/NExTVideo/map_vid_vidorID.json'))


new_train = []
new_val = []
for qa in train:
    qa_dict = {}
    qa_dict['video'] = vid_map[str(qa['video'])]
    qa_dict['num_option'] = 5
    qa_dict['qid'] = '_'.join([qa['type'], str(qa['video']), str(qa['qid'])])
    for i in range(5):
        qa_dict['a{}'.format(str(i))] = qa['a{}'.format(str(i))]+'.'
    qa_dict['answer'] = qa['answer']
    qa_dict['question'] = qa['question']+'?'
    new_train.append(qa_dict)

for qa in val:
    qa_dict = {}
    qa_dict['video'] = vid_map[str(qa['video'])]
    qa_dict['num_option'] = 5
    qa_dict['qid'] = '_'.join([qa['type'], str(qa['video']), str(qa['qid'])])
    for i in range(5):
        qa_dict['a{}'.format(str(i))] = qa['a{}'.format(str(i))]+'.'
    qa_dict['answer'] = qa['answer']
    qa_dict['question'] = qa['question']+'?'
    new_val.append(qa_dict)



save_json(new_train, '/scratch_xijun/data/NExTVideo/final/train.json')
save_json(new_val, '/scratch_xijun/data/NExTVideo/final/val.json')
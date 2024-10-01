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

train_path = '/scratch_xijun/data/QVH/highlight_train_release.jsonl'
val_path = '/scratch_xijun/data/QVH/highlight_val_release.jsonl'
test_path = '/scratch_xijun/data/QVH/highlight_test_release.jsonl'

train = load_jsonl(train_path)
val = load_jsonl(val_path)
test = load_jsonl(test_path)

new_train = []
new_val = []
new_test = []
for i, qa in enumerate(train):
    qa_dict = {}
    qa_dict['video'] = qa['vid']
    qa_dict['qid'] = 'QVHighlight_' + str(qa['qid'])
    qa_dict['query'] = qa['query']
    qa_dict['duration'] = qa['duration']
    qa_dict['relevant_windows'] = qa['relevant_windows']
    new_train.append(qa_dict)

for i, qa in enumerate(val):
    qa_dict = {}
    qa_dict['video'] = qa['vid']
    qa_dict['qid'] = 'QVHighlight_' + str(qa['qid'])
    qa_dict['query'] = qa['query']
    qa_dict['duration'] = qa['duration']
    qa_dict['relevant_windows'] = qa['relevant_windows']
    new_val.append(qa_dict)

for i, qa in enumerate(test):
    qa_dict = {}
    qa_dict['video'] = qa['vid']
    qa_dict['qid'] = 'QVHighlight_' + str(qa['qid'])
    qa_dict['query'] = qa['query']
    qa_dict['duration'] = qa['duration']
    new_test.append(qa_dict)


save_json(new_train, '/scratch_xijun/data/QVH/qvh/train.json')
save_json(new_val, '/scratch_xijun/data/QVH/qvh/val.json')
save_json(new_test, '/scratch_xijun/data/QVH/qvh/test.json')
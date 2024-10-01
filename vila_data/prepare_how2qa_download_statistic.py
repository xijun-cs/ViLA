import codecs
import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json(content, save_path):
    with open(save_path, 'w') as f:
        f.write(json.dumps(content, cls=NpEncoder))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]

how2qa_videos = []
train_path = '/scratch_xijun/data/How2QA/ann/how2qa/how2qa_train_release.jsonl'
val_path = '/scratch_xijun/data/How2QA/ann/how2qa/how2qa_val_release.jsonl'

train = load_jsonl(train_path)
val = load_jsonl(val_path)

print(len(train) + len(val))


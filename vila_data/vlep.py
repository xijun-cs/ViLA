import os
import json
import pandas as pd

def save_json(content, save_path):
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


train_path = '/scratch_xijun/data/VLEP/vlep_train_release.jsonl'
val_path = '/scratch_xijun/data/VLEP/vlep_dev_release.jsonl'

train = load_jsonl(train_path)
val = load_jsonl(val_path)

new_train = []
new_val = []

for i, qa in enumerate(train):
    qa_dict = {}
    qa_dict['video'] = qa['vid_name']
    qa_dict['num_option'] = 2
    qa_dict['qid'] = 'VLEP_' + str(qa['example_id'])

    for j in range(2):
        qa_dict['a{}'.format(str(j))] = qa['events'][j]
    qa_dict['answer'] = qa['answer']
    # qa_dict['question'] = qa['q']
    qa_dict['start'] = qa['ts'][0]
    qa_dict['end'] = qa['ts'][1]

    new_train.append(qa_dict)

for i, qa in enumerate(val):
    qa_dict = {}
    qa_dict['video'] = qa['vid_name']
    qa_dict['num_option'] = 2
    qa_dict['qid'] = 'VLEP_' + str(qa['example_id'])

    for j in range(2):
        qa_dict['a{}'.format(str(j))] = qa['events'][j]
    qa_dict['answer'] = qa['answer']
    # qa_dict['question'] = qa['q']
    qa_dict['start'] = qa['ts'][0]
    qa_dict['end'] = qa['ts'][1]

    new_val.append(qa_dict)

save_json(new_train, '/scratch_xijun/data/VLEP/train.json')
save_json(new_val, '/scratch_xijun/data/VLEP/val.json')
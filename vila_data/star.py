import os
import json
import pandas as pd

def save_json(content, save_path):
    with open(save_path, 'w') as f:
        f.write(json.dumps(content))
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


train_path = '/scratch_xijun/data/STAR/STAR_train.json'
val_path = '/scratch_xijun/data/STAR/STAR_val.json'

train = json.load(open(train_path))
val = json.load(open(val_path))

new_train = []
new_val = []
for qa in train:
    qa_dict = {}
    qa_dict['video'] = qa['video_id']
    qa_dict['num_option'] = 4
    qa_dict['qid'] = qa['question_id']
    for i, choice in enumerate(qa['choices']):
        qa_dict['a{}'.format(str(i))] = choice['choice']
        if choice['choice'] == qa['answer']:
            answer = i
    qa_dict['answer'] = answer
    qa_dict['question'] = qa['question']
    qa_dict['start'] = qa['start']
    qa_dict['end'] = qa['end']
    new_train.append(qa_dict)

for qa in val:
    qa_dict = {}
    qa_dict['video'] = qa['video_id']
    qa_dict['num_option'] = 4
    qa_dict['qid'] = qa['question_id']
    for i, choice in enumerate(qa['choices']):
        qa_dict['a{}'.format(str(i))] = choice['choice']
        if choice['choice'] == qa['answer']:
            answer = i
    qa_dict['answer'] = answer
    qa_dict['question'] = qa['question']
    qa_dict['start'] = qa['start']
    qa_dict['end'] = qa['end']
    new_val.append(qa_dict)


save_json(new_train, '/scratch_xijun/data/STAR/train.json')
save_json(new_val, '/scratch_xijun/data/STAR/val.json')
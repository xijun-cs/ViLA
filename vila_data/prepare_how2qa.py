import codecs
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
with open("how2_vid_mapping.json", "r") as f:
    mapping = json.load(f)

bad = []
for i, qa in enumerate(train):
    try:
        id =  qa['vid_name']
        name = mapping[id].split('_')[0]
        how2qa_videos.append(name)
    except:
        bad.append(qa)


for i, qa in enumerate(val):
    try:
        id = qa['vid_name']
        name = mapping[id].split('_')[0]
        how2qa_videos.append(name)
    except:
        bad.append(qa)

video_list_file = '/scratch_xijun/data/How2QA/download_instructions/howto100m_videos.txt'
videos = []
names = []
for video_url in codecs.open(video_list_file, "r", "utf-8"):
    video_url = video_url.strip()
    if not video_url:
        continue
    videos.append(video_url)
    name = video_url.split('/')[-1].split('.')[0]
    names.append(name)

# print(how2qa_videos[:10])
# print(len(how2qa_videos))
# print(len(bad))

how2qa_videos_url_list = open('/scratch_xijun/data/How2QA/howtoqa_videos.txt', 'w')
for name in how2qa_videos:
    try:
        idx = names.index(name)
        how2qa_videos_url_list.write(videos[idx]+'\n')
    except:
        how2qa_videos_url_list.write('http://howto100m.inria.fr/dataset/'+name+'.mp4' + '\n')
        bad.append(name)


how2qa_videos_url_list.close()
print(len(bad))





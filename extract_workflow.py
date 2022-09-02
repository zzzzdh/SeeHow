import os
import json
import bbox
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import pymysql
import difflib
import re

playlist_list = ['2', '5']
data_dir = '/home/cheer/Project/Workflow/data'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
model.to('cuda')
S_LEN = 128

label_dict = {'0':'others', '6':'enter_text', '7':'enter_text_popup_a', '8':'enter_text_popup_d', '9':'delete', '10':'popup', '11':'select', '12':'deselect', '13':'scroll', '14':'switch', '15':'enter'}

config = {
  'host': '172.17.0.2',
  'port': 3306,
  'user': 'root',
  'passwd':'123',
  'db': 'workflow',
  'charset': 'utf8',
  'cursorclass': pymysql.cursors.DictCursor
}

def compute_action_overlap(y1, y2, y3, y4):
  if y3 > y2 or y1 > y4:
    return 0
  else:
    y_list = np.array([y1, y2, y3, y4])
    y_list = np.sort(y_list)
    return (y_list[2] - y_list[1]) / (y_list[3] - y_list[0])

def find_clip(action_overlaps):
  clips = []
  if len(action_overlaps):
    for i in range(len(action_overlaps)):
      if action_overlaps[i]['overlap'] < 0.1:
        start = action_overlaps[i]['id']
        end = action_overlaps[i]['id']
      if i < len(action_overlaps) - 1 and action_overlaps[i+1]['overlap'] < 0.1:
        end = action_overlaps[i]['id']
        action = action_overlaps[i]['action']
        code = action_overlaps[i]['code']
        clips.append({'clip': [start, end], 'action': action, 'code': code})
    end = action_overlaps[i]['id']
    action = action_overlaps[i]['action']
    code = action_overlaps[i]['code']
    clips.append({'clip': [start, end], 'action': action, 'code': code})
  return clips  

def rectify(i, captions):
  if i > int(captions[-1].split()[0]):
    return int(captions[-1].split()[0]), int(captions[-1].split()[1])
  elif i < int(captions[0].split()[1]):
    return int(captions[0].split()[0]), int(captions[0].split()[1])
  for caption in captions:
    if i in range(int(caption.split()[0]), int(caption.split()[1])):
      return int(caption.split()[0]), int(caption.split()[1])
#  dist_i = []
#  for caption in captions:
#    dist_i.append(min([abs(int(caption.split()[0]) - i), abs(int(caption.split()[1]) - i)]))
#  dist_min = np.argmin(dist_i)
#  return int(captions[dist_min].split()[0]), int(captions[dist_min].split()[1])

def merge_two(clip1, clip2):
  clip = {}
  clip['frame'] = [clip1['frame'][0], clip2['frame'][1]]
  clip['line'] = [clip1['line'][0], clip2['line'][1]]
  clip['caption'] = clip1['caption'] + ' ' + clip2['caption']
  return clip

def split_sentence(s):
  s_list = []
  for i in range(int(len(s.split()) / S_LEN) + 1):
    s_list.append(' '.join(s.split()[i * S_LEN:(i+1) * S_LEN]))
  return s_list

def compute_next_sentence(s1, s2):
  s1_list = split_sentence(s1)
  s2_list = split_sentence(s2)
  scores = [] 
  for s1_clip in s1_list:
    for s2_clip in s2_list:
      #print (len(s1_clip.split()), len(s2_clip.split()), len(s1.split()), len(s2.split()))
      token_list = tokenizer.encode(s1_clip + ',' + s2_clip, add_special_tokens = True)
      input_ids = torch.tensor(token_list).unsqueeze(0).to('cuda')
      segments_ids = [0] * (token_list.index(1010) + 1) + [1] * (len(token_list) - token_list.index(1010) - 1)
      segments_tensors = torch.tensor([segments_ids]).to('cuda')
      outputs = model(input_ids, token_type_ids=segments_tensors)
      seq_relationship_scores = outputs[0]
      scores.append(seq_relationship_scores[0][0])
  scores = np.array(scores, dtype = float)
  return np.mean(scores)

def group_clip(clips):
  for i in range(len(clips)):
    if clips[i]['flag'] == 0:
      if i == 0:
        clips[1].update(merge_two(clips[0], clips[1]))
      elif i < len(clips) - 1:
        next_1 = compute_next_sentence(clips[i-1]['caption'], clips[i]['caption'])
        next_2 = compute_next_sentence(clips[i]['caption'], clips[i+1]['caption'])
        if next_1 > next_2:
          clips[i-1].update(merge_two(clips[i-1], clips[i]))
        else:
          clips[i+1].update(merge_two(clips[i], clips[i+1]))
      else:
          clips[-2].update(merge_two(clips[-2], clips[-1]))
  new_clips = []
  for clip in clips:
    if clip['flag']:
      new_clips.append(clip)
  for i in range(len(new_clips)):
    if new_clips[i]['action'] == 'select' or new_clips[i]['action'] == 'deselect':
      if len(new_clips[i]['code']):
        for j in range(i):
          code_sim = difflib.SequenceMatcher(None, new_clips[i]['code'], new_clips[j]['code']).quick_ratio()
          if code_sim > 0.9:
            new_clips[i]['parent'] = str(j + 1)
            #print (i, new_clips[i]['frame'], new_clips[i]['code'], j, new_clips[j]['frame'], new_clips[j]['code']) 
  return new_clips
        
def merge_caption(clips, captions):
  clip_list = []
  for clip in clips:
    start_index = end_index = len(captions) - 1
    for i in range(len(captions)):
      if clip['clip'][0] in range(int(captions[i].split()[0]), int(captions[i].split()[1])):
        start_index = i
      if clip['clip'][1] in range(int(captions[i].split()[0]), int(captions[i].split()[1])):
        end_index = i
    text_1 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[start_index:end_index + 1])))
    if len(clip_list) == 0 and start_index > 0:
      text_0 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[0:start_index])))
      clip_list.append({'frame': [0, clip['clip'][0]], 'line': [0, start_index], 'caption': text_0, 'flag': 0})
    elif len(clip_list) == 0 and start_index == 0:
      clip_list.append({'frame': clip['clip'], 'key_frame': clip['clip'], 'line': [start_index, end_index + 1], 'caption': text_1, 'flag': 1, 'action': clip['action'], 'code': clip['code'], 'parent': None})
    else:
      text_0 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[clip_list[-1]['line'][1]:start_index])))
      clip_list.append({'frame': [clip_list[-1]['frame'][1], clip['clip'][0]], 'line': [clip_list[-1]['line'][1], start_index], 'caption': text_0, 'flag': 0})
    clip_list.append({'frame': clip['clip'], 'key_frame': clip['clip'], 'line': [start_index, end_index + 1], 'caption': text_1, 'flag': 1, 'action': clip['action'], 'code': clip['code'], 'parent': None})
  if len(clip_list) == 0:
    return clip_list
  elif clip_list[-1]['line'][1] < len(captions) - 1:
    text_0 = ' '.join(list(map(lambda x: ' '.join(x.split()[2:]), captions[clip_list[-1]['line'][1]:len(captions)])))
    clip_list.append({'frame': [clip_list[-1]['frame'][1], int(captions[-1].split()[1])], 'line': [clip_list[-1]['line'][1], len(captions)], 'caption': text_0, 'flag': 0})
  return clip_list

def compute_clip(folder, actions, annotations, captions):
  action_overlaps = []
  old_ocr_box = bbox.BBox2D([0, 0, 0, 0], mode = bbox.XYXY)
  for i in range(len(actions)):
    action_box = bbox.BBox2D([int(annotations[i].split()[1]), int(annotations[i].split()[2]), int(annotations[i].split()[3]), int(annotations[i].split()[4])], mode = bbox.XYXY)
    s, e = rectify(i, captions)
    with open(os.path.join(data_dir, 'OCR', folder, actions[e-1].split()[0] + '.json'), 'r') as json_file:
      json_data = json.load(json_file)
    ocr_boxes = []
    ious = []
    codelines = []
    for line in json_data['lines']:
      ocr_boxes.append(bbox.BBox2D([line['vertice']['x_min'], line['vertice']['y_min'], line['vertice']['x_max'], line['vertice']['y_max']], mode = bbox.XYXY))
      codelines.append(line['text'])
    for ocr_box in ocr_boxes:
      ious.append(bbox.metrics.jaccard_index_2d(action_box, ocr_box))
    ious = np.array(ious)
    max_iou_index = np.argmax(ious)
    if ious[max_iou_index] > 0.05:
      if  actions[i].split()[1] == '6' or actions[i].split()[1] == '9' or actions[i].split()[1] == '11' or actions[i].split()[1] == '12':
        action_overlap = compute_action_overlap(int(ocr_boxes[max_iou_index].y1), int(ocr_boxes[max_iou_index].y2), int(old_ocr_box.y1), int(old_ocr_box.y2))
        old_ocr_box = ocr_boxes[max_iou_index]
        action_overlaps.append({'overlap': action_overlap, 'id': i, 'action': label_dict[actions[i].split()[1]], 'code': codelines[max_iou_index]})
  clips = find_clip(action_overlaps)
  return clips

def main():
  connect = pymysql.connect(**config)
  cursor = connect.cursor()
  #cursor.execute('DROP TABLE IF EXISTS video')
  #cursor.execute('CREATE TABLE video (Id int primary key auto_increment, Name varchar(20), Step int, Frame varchar(100), Fragment varchar(100), Code varchar(200), Action varchar(20), Parent int, Caption text, Summary text)')
  folder_list = []
  folder_list_all = os.listdir(os.path.join(data_dir, 'OCR'))
  for folder in folder_list_all:
    for playlist in playlist_list:
      if re.search(r'^' + playlist + '_\d+', folder):
        folder_list.append(folder)
  for folder in tqdm(folder_list):
    print (folder)
    with open(os.path.join(data_dir, 'Actions', folder + '.txt'), 'r') as action_file:
      actions = action_file.readlines()
    with open(os.path.join(data_dir, 'Annotations', folder + '.txt'), 'r') as annotation_file:
      annotations = annotation_file.readlines()
    with open(os.path.join(data_dir, 'Captions', folder + '.txt'), 'r') as caption_file:
      captions = caption_file.readlines()
    clips = compute_clip(folder, actions, annotations, captions)
    clips = merge_caption(clips, captions)
    clips = group_clip(clips)

    for i in range(len(clips)):
      fragment = ','.join(list(map(lambda x: str(x), clips[i]['frame'])))
      key_frame = '{:05},{:05}'.format(clips[i]['key_frame'][0] + 1, clips[i]['key_frame'][1] + 1)
      cursor.execute('INSERT INTO video (Name, Caption, Step, Frame, Fragment, Code, Action, Parent) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)', (folder, clips[i]['caption'], i+1, key_frame, fragment, clips[i]['code'], clips[i]['action'], clips[i]['parent']))

  print (cursor.execute('SELECT * FROM video'))
  connect.commit()
  cursor.close()
  connect.close()

if __name__ == '__main__':
  main()

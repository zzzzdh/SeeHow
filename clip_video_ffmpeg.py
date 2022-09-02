import glob
import os
import re
import codecs
import webvtt
from tqdm import tqdm

video_path = '/home/cheer/Project/Workflow/data/videos'
output_path = '/home/cheer/Project/Workflow/data/Images'
playlist_list = ['1', '2', '3', '4', '5', '6']
exts = ['.mp4', '.mkv', '.webm']

def rectify(file_name):
  file_name = file_name.replace(os.path.splitext(file_name)[1], '.en.vtt')
  with open(file_name, 'r') as vtt_file:
    subtitles = vtt_file.readlines()
  i = 0
  for subtitle in subtitles:
    if re.search('position:0%', subtitles[i]) and len(subtitles[i+1].strip()) == 0 and len(subtitles[i+2].strip()) > 0:
      del subtitles[i+1]
    i += 1
    if i-2 > len(subtitles):
      break
  with open(os.path.join(video_path, 'tmp.en.vtt'), 'w') as vtt_file:
    vtt_file.writelines(subtitles)

def parse_vtt(save_dir):
  file_name = os.path.join(video_path, 'tmp.en.vtt')
  caption_list = []
  merge_list = []
  for caption in webvtt.read(file_name):
    start = int(caption.start.split(':')[0])*3600 + int(caption.start.split(':')[1])*60 + int(float(caption.start.split(':')[2]))
    end = int(caption.end.split(':')[0])*3600 + int(caption.end.split(':')[1])*60 + int(float(caption.end.split(':')[2]))
    if len(caption.text.split('\n')) > 1:
      text = caption.text.split('\n')[1]
    else:
      text = caption.text.split('\n')[0]
    if end > start:
      caption_list.append('{} {} {}\n'.format(start, end, text))
  i = 1
  for caption in caption_list:
    start = caption.split()[0]
    end = caption.split()[1]
    text = ' '.join(caption.split()[2:])
    if i % 2 == 0:
      text_two = text_two + ' ' + text
      merge_list.append('{} {} {}\n'.format(start_old, end, text_two))
    else:
      start_old = start
      text_two = text
    i += 1       
  with open(save_dir.replace('Images', 'Captions') + '.txt', 'w') as caption_file:
    caption_file.writelines(merge_list)

def make_dir(playlist, video_number):
  folder_name = str(playlist) + '_' + str(video_number)
  if not os.path.exists(os.path.join(output_path, folder_name)):
    print ('create folder {}'.format(folder_name))
    os.makedirs(os.path.join(output_path, folder_name))
  else:
    print ('folder {} exist'.format(folder_name))
  return os.path.join(output_path, folder_name)

def clip(playlist, file_name, video_number):
  #if os.path.exists(file_name.replace(os.path.splitext(file_name)[1], '.en.vtt')):
  #  print (file_name.replace(os.path.splitext(file_name)[1], '.en.vtt'))
    save_dir = make_dir(playlist, video_number)
  #  rectify(file_name)
  #  parse_vtt(save_dir)
    cmd = "ffmpeg -i '" + file_name + "' -q:v 1 -r 1 " + save_dir + "/%05d.jpg"
    os.system(cmd)

def start():
  for playlist in playlist_list:
    video_list = []
    file_list = os.listdir(os.path.join(video_path, str(playlist)))
    for file_name in file_list:
      if os.path.splitext(file_name)[1] in exts:
        video_list.append(os.path.join(video_path, str(playlist), file_name))
    video_number = 0
    for video_name in tqdm(video_list):
      clip(playlist, video_name, video_number)
      video_number += 1

if __name__ == '__main__':
  start()

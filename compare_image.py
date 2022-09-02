from skimage.measure import compare_ssim
import imutils
import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing

image_path = '/home/cheer/Project/Workflow/data/Images'
output_path = '/home/cheer/Project/Workflow/data/Annotations'

def compare_frame(frameA, frameB):
  grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)

  score, diff = compare_ssim(grayA, grayB, full=True)
  diff = (diff * 255).astype("uint8")
  #print("SSIM: {}".format(score))

  thresh = cv2.threshold(diff, 180, 255, cv2.THRESH_BINARY_INV)[1]
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if imutils.is_cv2() else cnts[1]

  return diff, thresh, cnts, score

def convert_box(cnts):
  box = []
  for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 6 and h > 15: 
      box.append([x, y, x+w, y+h])
  box = np.array(box)
  return box

def find_max(boxes_nms):
  if len(boxes_nms) == 0:
    box_max = np.zeros((7), dtype = np.int32)
    return box_max
  boxes = []
  for box_nms in boxes_nms:
    box_nms = np.append(box_nms, (box_nms[2]-box_nms[0])*(box_nms[3]-box_nms[1]))
    boxes.append(box_nms)
  boxes = np.array(boxes)
  idx = np.argsort(boxes[:,4])
  x_center = boxes[idx[-1]][0] + (boxes[idx[-1]][2] - boxes[idx[-1]][0]) / 2
  y_center = boxes[idx[-1]][1] + (boxes[idx[-1]][3] - boxes[idx[-1]][1]) / 2
  box_max = np.append(boxes[idx[-1]], [x_center, y_center])
  box_max = np.array(box_max, dtype = np.int32)
  return box_max

def non_max_suppression(boxes, overlapThresh):
  if len(boxes) == 0:
    return []
  if boxes.dtype.kind == "i":
    boxes = boxes.astype("float")

  pick = []
  x1 = boxes[:,0]
  y1 = boxes[:,1]
  x2 = boxes[:,2]
  y2 = boxes[:,3]
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  idxs = np.argsort(y2)

  while len(idxs) > 0:
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    xx1 = np.maximum(x1[i], x1[idxs[:last]])
    yy1 = np.maximum(y1[i], y1[idxs[:last]])
    xx2 = np.minimum(x2[i], x2[idxs[:last]])
    yy2 = np.minimum(y2[i], y2[idxs[:last]])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    overlap = (w * h) / area[idxs[:last]] 
    idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

  return boxes[pick].astype("int")

def start():
  video = q.get()
  print (video)
  box_list = []
  image_list = os.listdir(os.path.join(image_path, video))
  image_list.sort()
  image_old = cv2.imread(os.path.join(image_path, video, image_list[0]))
  for image_file in image_list:
    image = cv2.imread(os.path.join(image_path, video, image_file))
    diff, thresh, cnts, score= compare_frame(image_old, image)
    boxes = convert_box(cnts)
    boxes_nms = non_max_suppression(boxes, 0.3)
    max_box = find_max(boxes_nms)
    image_old = image.copy()
    box_list.append('{} {} {} {} {} {} {}\n'.format(image_file.split('.')[0], max_box[0], max_box[1], max_box[2], max_box[3], max_box[5], max_box[6]))
  with open(os.path.join(output_path, video + '.txt'), 'w') as annotation_file:
     annotation_file.writelines(box_list) 
      
if __name__ == '__main__':
  folder_list = os.listdir(image_path)
  annotation_list = os.listdir(output_path)
  for annotation in annotation_list:
    if os.path.splitext(annotation)[0] in folder_list:
      folder_list.remove(os.path.splitext(annotation)[0])
      print ('remove ' + os.path.splitext(annotation)[0])
  q = multiprocessing.Manager().Queue()
  pool = multiprocessing.Pool(5)
  for folder in folder_list:
    q.put(folder)
  qsize = q.qsize()
  for _ in range(qsize):
    pool.apply_async(start, ())
  pool.close()
  pool.join()


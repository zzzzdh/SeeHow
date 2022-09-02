import cv2
import numpy as np
import xml.etree.ElementTree as ET

def connect(boxes):
  main_boxes = []
  conn_boxes = [boxes[0]]
  com_boxes = boxes
  i = 0
  while len(com_boxes):
    current_box = conn_boxes[i]
    find_connect = 0
    if current_box in com_boxes:
      com_boxes.remove(current_box)
    for com_box in com_boxes:
      if abs(current_box[3] + current_box[1] - com_box[3] - com_box[1]) < 10 and min(abs(current_box[0] - com_box[2]), abs(current_box[2] - com_box[0])) < current_box[3] - current_box[1]:
        line_box = [min(current_box[0], com_box[0]), min(current_box[1], com_box[1]), max(current_box[2], com_box[2]), max(current_box[3], com_box[3])]
        find_connect = 1
        com_boxes.remove(com_box)
        if len(conn_boxes):
          conn_boxes.remove(conn_boxes[i])
        conn_boxes.insert(i, line_box)
        break
    if find_connect == 0:
      i += 1
      conn_boxes.append(com_box)
     
  #main_boxes = list(set(main_boxes))
  #main_boxes = np.array(list(map(lambda x: list(map(lambda y: int(y), x.split(','))), main_boxes)))
  return conn_boxes

def draw_box():
  tree = ET.parse(os.path.join(data_dir, 'tmp', 'output.hocr'))
  root = tree.getroot()
  ocr_boxes = []
  for body in root.iter('{http://www.w3.org/1999/xhtml}body'):
    for div in body.iter('{http://www.w3.org/1999/xhtml}div'):
      for p in div.iter('{http://www.w3.org/1999/xhtml}p'):
        for span in p.iter('{http://www.w3.org/1999/xhtml}span'):
          if span.get('class') == 'ocr_line':
            box = span.get('title').split(';')[0].split()[1:5]
            box = list(map(lambda x: int(x), box))
            ocr_boxes.append(box)
  image = cv2.imread(os.path.join(data_dir, 'tmp', '00008.jpg'))
  for box in ocr_boxes:
    image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
  cv2.imwrite(os.path.join(data_dir, 'tmp', 'ocr.jpg'), image)

  image = cv2.imread(os.path.join(data_dir, 'tmp', '00008.jpg'))
  with open(os.path.join(data_dir, 'tmp', '00008.json'), 'r') as json_file:
    json_data = json.load(json_file)
  main_boxes = []
  for line in json_data['lines']:
    line_box = [line['vertice']['x_min'], line['vertice']['y_min'], line['vertice']['x_max'], line['vertice']['y_max']]
    word_boxes = line['boxes']
    word_boxes = np.array(word_boxes)
    #image = cv2.rectangle(image, (line_box[0], line_box[1]), (line_box[2], line_box[3]), (0, 255, 0), 4)
    for w_box in word_boxes:
      #if 500 < w_box[0][0] < 1900 and 120 < w_box[0][1] < 910:
      main_boxes.append([min(w_box[:,0]), min(w_box[:, 1]), max(w_box[:,0]), max(w_box[:, 1])])
      w_box = w_box.reshape((-1, 1, 2))
      image = cv2.polylines(image, [w_box], 1, (0, 0, 255), 2)
  main_boxes = connect(main_boxes)
  for m_box in main_boxes:
    image = cv2.rectangle(image, (m_box[0], m_box[1]), (m_box[2], m_box[3]), (0, 255, 0), 2)
  cv2.imwrite(os.path.join(data_dir, 'tmp', 'gocr.jpg'), image)

if __name__ == '__main__':
  draw_box()

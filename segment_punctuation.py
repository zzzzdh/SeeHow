from deepsegment import DeepSegment
import pymysql
import re
from tqdm import tqdm

playlist_list = ['2', '5']

config = {
  'host': '172.17.0.2',
  'port': 3306,
  'user': 'root',
  'passwd':'123',
  'db': 'workflow',
  'charset': 'utf8',
  'cursorclass': pymysql.cursors.DictCursor
}

def rectify(new_results):
  for i in range(len(new_results) - 1):
    if new_results[i]['name'] == new_results[i + 1]['name']:
      if len(new_results[i]['caption']) > len(new_results[i + 1]['caption']) and len(new_results[i]['caption'].split(',')) > 1:
        new_results[i + 1]['caption'] = new_results[i]['caption'].split(',')[-1] + ' ' + new_results[i + 1]['caption']
        new_results[i]['caption'] = ','.join(new_results[i]['caption'].split(',')[:-1])
      elif len(new_results[i]['caption']) < len(new_results[i + 1]['caption']) and len(new_results[i + 1]['caption'].split(',')) > 1:
        new_results[i]['caption'] = new_results[i]['caption'] + ' ' + new_results[i + 1]['caption'].split(',')[0]
        new_results[i + 1]['caption'] = ','.join(new_results[i + 1]['caption'].split(',')[1:])
  return new_results

def main():
  segmenter = DeepSegment('en')
  connect = pymysql.connect(**config)
  cursor = connect.cursor()
  cursor.execute('SELECT Id, Name, Caption FROM video')
  results = cursor.fetchall()
  new_results = []
  for result in tqdm(results):
    for playlist in playlist_list:
      if re.search(r'^' + playlist + '_\d+', result['Name']):
        new_result = {}
        new_result['caption'] = ', '.join(segmenter.segment_long(result['Caption']))
        new_result['name'] = result['Name']
        new_result['id'] = result['Id']
        new_results.append(new_result)
  new_results = rectify(new_results)
  for new_result in tqdm(new_results):
    cursor.execute('UPDATE video SET Caption=%s WHERE Id=%s', (new_result['caption'], new_result['id']))
  print (cursor.execute('SELECT * FROM video'))
  connect.commit()
  cursor.close()
  connect.close()

if __name__ == '__main__':
  main()


from transformers import pipeline
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

def main():
  summarizer = pipeline('summarization', device=0)
  connect = pymysql.connect(**config)
  cursor = connect.cursor()
  cursor.execute('SELECT Id, Name, Caption FROM video')
  results = cursor.fetchall()
  for result in tqdm(results):
    for playlist in playlist_list:
      if re.search(r'^' + playlist + '_\d+', result['Name']):
        summary_text = summarizer(result['Caption'], min_length=10, max_length=100)[0]['summary_text']
        cursor.execute('UPDATE video SET Summary=%s WHERE Id=%s', (summary_text, result['Id']))
  print (cursor.execute('SELECT * FROM video'))
  connect.commit()
  cursor.close()
  connect.close()

if __name__ == '__main__':
  main()

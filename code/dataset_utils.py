import pickle as pickle
import pandas as pd
import ast


def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  out_dataset = pd.DataFrame({'id':dataset['id'], 
                              'sentence':dataset['sentence'],
                              'subject_word':dataset['subject_word'],
                              'object_word':dataset['object_word'],
                              'label':dataset['label'],})
  return out_dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_word'], dataset['object_word']):
    temp = ''
    temp = e01 + '[SEP]' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

def tokenized_dataset_xlm(dataset, tokenizer):
  """dataset을 주어진 tokenizer로 tokenize하는 함수(xlm only)

  Args:
      dataset (DataFrame): 변곧내(변수이름이 곧 내용)
      tokenizer (Any): 사용하고자 하는 tokenizer

  Returns:
      Any: tokenized된 문장들
  """
  concat_entity = []
  for e01, e02 in zip(dataset['subject_word'], dataset['object_word']):
    temp = ''
    temp = e01 + '</s></s>' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=256,
      add_special_tokens=True,
      )
  return tokenized_sentences

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def label_to_num(label):
  num_label = []
  with open('./pickle/dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('./pickle/dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def label_to_num_binary(label):
  num_label = []
  for v in label:
    if v == 'no_relation':
      num_label.append(0)
    else:
      num_label.append(1)
  
  return num_label

def num_to_label_binary(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  for v in label:
    if v == 0:
      origin_label.append('no_relation')
    else:
      origin_label.append('relation')
  
  return origin_label
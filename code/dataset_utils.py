import pickle as pickle
import pandas as pd
import ast

from config.config import call_config
from entity_token_adder import add_typed_entity_marker_punct
from preprocessing import preprocessing

conf = call_config()

def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label'],})
  return out_dataset

def tokenized_dataset(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  
  if conf.use_entity_marker:
    concat_entity = []
    for e01, e01_type, e02, e02_type in zip(dataset['subject_word'], dataset['subject_type'], dataset['object_word'], dataset['object_type']):
      temp = ''
      temp = e01 + '[SEP]' + e02
      temp = f'@ * {e01_type} * {e01} @ [SEP] & ^ {e02_type} ^ {e02} &'
      concat_entity.append(temp)
    
    # use entity marker
    dataset = add_typed_entity_marker_punct(dataset)
  else:
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
      max_length=128,
      add_special_tokens=True,
      return_token_type_ids=True
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
    temp = e01 + '</s>' + e02
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128,
      add_special_tokens=True,
      )
  return tokenized_sentences

def load_data(dataset_dir, train=False):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd_dataset = pd.read_csv(dataset_dir)
  
  dataset = preprocessing(pd_dataset, train=train)
  
  return dataset

def load_test_dataset(dataset_dir, tokenizer, MODEL_NAME=None):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  if MODEL_NAME and MODEL_NAME[:10] == "xlm-roberta":
    tokenized_test = tokenized_dataset_xlm(test_dataset, tokenizer)
  else:
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

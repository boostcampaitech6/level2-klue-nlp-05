import pickle as pickle
import os
import pandas as pd
import torch


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_dataset(dataset):
  """load_data.load_data에서 정보를 preprocessing하는 함수. 

  Args:
      dataset (DataFrame): csv파일을 DataFrame으로 변환한 상태

  Returns:
      DataFrame: dataset에서 원하는 형태의 preprocessing을 거진 DataFrame
  """
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]
    j = j[1:-1].split(',')[0].split(':')[1]

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 
                              'sentence':dataset['sentence'],
                              'subject_entity':subject_entity,
                              'object_entity':object_entity,
                              'label':dataset['label'],})
  return out_dataset

def load_data(dataset_dir):
  """csv파일을 불러와 원하는 형태로 preprocessing해주고 DataFrame을 반환하는 함수

  Args:
      dataset_dir (str): 불러올 csv파일의 경로

  Returns:
      DataFrame: preprocessing까지 완료된 DataFrame
  """
  pd_dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(pd_dataset)
  
  return dataset

def tokenized_dataset(dataset, tokenizer):
  """dataset을 주어진 tokenizer로 tokenize하는 함수

  Args:
      dataset (DataFrame): 변곧내(변수이름이 곧 내용)
      tokenizer (Any): 사용하고자 하는 tokenizer

  Returns:
      Any: tokenized된 문장들
  """
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
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
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
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
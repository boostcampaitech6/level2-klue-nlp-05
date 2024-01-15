from preprocessing import train_data_preprocessing
from tqdm.auto import tqdm
import pickle as pickle
import pandas as pd
import ast


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
  concat_entity = []
  for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
    temp = ''
    temp = e01 + "와 " + e02 + "의 관계"
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=195,
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

def tokenized_dataset_pretrain(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  data = []
  for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):
    subj = item["subject_entity"]
    obj = item["object_entity"]

    concat_entity = tokenizer.sep_token.join([subj, obj])
    # roberta 모델은 token_type_ids 레이어를 사용하지 않습니다.
    output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
    data.append(output)
  return data

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  pd.set_option('mode.chained_assignment',  None)
  pd_dataset = pd.read_csv(dataset_dir)
  cleaned_dataset = train_data_preprocessing(pd_dataset)
  dataset = preprocessing_dataset(cleaned_dataset)
  
  return dataset

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  pd_dataset = pd.read_csv(dataset_dir)
  test_dataset = preprocessing_dataset(pd_dataset)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def load_pretrain_data(dataset_dir1, dataset_dir2):
  train_dataset = pd.read_csv(dataset_dir1)
  test_dataset = pd.read_csv(dataset_dir2)

  # remove duplicated data
  pd.set_option('mode.chained_assignment',  None)
  train_dataset = train_data_preprocessing(train_dataset)

  # train, test data concat
  pretrain_dataset = pd.concat([train_dataset, test_dataset])
  dataset = preprocessing_dataset(pretrain_dataset)
  
  return dataset

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

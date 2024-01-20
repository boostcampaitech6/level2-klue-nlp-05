from tqdm.auto import tqdm
import pickle as pickle
import pandas as pd
import torch
import ast

def tokenized_dataset_entity(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  sentences = []
  ss, os = [], []

  for idx, row in dataset.iterrows():
    sentence = row['sentence']
    subject_word, object_word = row['subject_word'], row['object_word']
    subject_start_idx, subject_end_idx = row['subject_start_idx'], row['subject_end_idx']
    object_start_idx, object_end_idx = row['object_start_idx'], row['object_end_idx']
    subject_type, object_type = row['subject_type'], row['object_type']

    new_sentence = ''
    new_sentence += subject_word + '-' + object_word + '[SEP]'

    if subject_start_idx < object_start_idx:
      new_sentence += sentence[:subject_start_idx]
      new_sentence += f'<S:{subject_type}>'
      new_sentence += sentence[subject_start_idx:subject_end_idx+1]
      new_sentence += f'</S:{subject_type}>'
      new_sentence += sentence[subject_end_idx+1:object_start_idx]
      new_sentence += f'<O:{object_type}>'
      new_sentence += sentence[object_start_idx:object_end_idx+1]
      new_sentence += f'</O:{object_type}>'
      new_sentence += sentence[object_end_idx+1:]
    else:
      new_sentence += sentence[:object_start_idx]
      new_sentence += f'<O:{object_type}>'
      new_sentence += sentence[object_start_idx:object_end_idx+1]
      new_sentence += f'</O:{object_type}>'
      new_sentence += sentence[object_end_idx+1:subject_start_idx]
      new_sentence += f'<S:{subject_type}>'
      new_sentence += sentence[subject_start_idx:subject_end_idx+1]
      new_sentence += f'</S:{subject_type}>'
      new_sentence += sentence[subject_end_idx+1:]

    sentences.append(new_sentence)

    encoded_inputs = tokenizer(
        new_sentence,
        truncation=True,
        max_length=180,
        add_special_tokens=True,
        )
    input_ids = encoded_inputs['input_ids']
    subject_token_id, object_token_id = tokenizer.convert_tokens_to_ids([f'<S:{subject_type}>', f'<O:{object_type}>'])
    
    if subject_token_id in input_ids:
      ss.append(input_ids.index(subject_token_id))
    else:
      ss.append(len(input_ids)-1)

    if object_token_id in input_ids:
      os.append(input_ids.index(object_token_id))
    else:
      os.append(len(input_ids)-1)

  tokenized_sentences = tokenizer(
      sentences,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=180,
      add_special_tokens=True,
      )
  
  tokenized_sentences['ss'] = torch.tensor(ss, dtype=torch.long)
  tokenized_sentences['os'] = torch.tensor(os, dtype=torch.long)

  return tokenized_sentences


def tokenized_dataset_prompt(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  concat_entity = []
  for e01, e02 in zip(dataset['subject_word'], dataset['object_word']):
    temp = ''
    temp = e01 + "와 " + e02 + "의 관계"
    concat_entity.append(temp)
  tokenized_sentences = tokenizer(
      concat_entity,
      list(dataset['sentence']),
      return_tensors="pt",
      max_length=195,
      padding=True,
      truncation=True,
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


def tokenized_dataset_pretrain(dataset, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  data = []
  for _, item in tqdm(dataset.iterrows(), desc="tokenizing", total=len(dataset)):
    subj = item["subject_word"]
    obj = item["object_word"]
    concat_entity = subj + "와 " + obj + "의 관계"
    output = tokenizer(concat_entity, item["sentence"], padding=True, truncation=True, max_length=256, add_special_tokens=True)
    data.append(output)
  return data


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    dataset = pd.read_csv(dataset_dir)  
    return dataset


def load_test_dataset(dataset_dir, tokenizer, conf):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  if conf.model.mode_name[:10] == "xlm-roberta":
    tokenized_test = tokenized_dataset_xlm(test_dataset, tokenizer)
  else:
    tokenized_test = tokenized_dataset_prompt(test_dataset, tokenizer)

  if conf.utils.isCustom:
    tokenized_test = tokenized_dataset_entity(test_dataset, tokenizer)

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
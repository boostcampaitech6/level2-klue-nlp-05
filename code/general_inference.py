from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from dataset_utils import load_test_dataset, num_to_label, num_to_label_binary
from omegaconf import OmegaConf
from general_classifier_train import set_seed
from custom_datasets import RE_Dataset
from tqdm import tqdm
from custom_model import CustomModel3

import torch.nn.functional as F
import pandas as pd
import torch

import numpy as np
import argparse


def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device), 
          ss = data['ss'].to(device),
          os = data['os'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")
  print(args)
  
  set_seed(42)
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  # 스페셜 토큰 추가
  special_tokens = ['<S:ORG>','<S:PER>','<S:POH>','<S:LOC>','<S:DAT>','<S:NOH>','</S:ORG>','</S:PER>','</S:POH>','</S:LOC>','</S:DAT>','</S:NOH>','<O:ORG>','<O:PER>','<O:POH>','<O:LOC>','<O:DAT>','<O:NOH>','</O:ORG>','</O:PER>','</O:POH>','</O:LOC>','</O:DAT>','</O:NOH>']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

  ## load my model
  BINARY_MODEL_NAME = './best_model/Binary'
  GENERAL_MODEL_NAME = './best_model/General'
  
  binary_model = CustomModel3.load_pretrained(BINARY_MODEL_NAME, conf, 
                                             config=AutoConfig.from_pretrained(conf.model.model_name))
  general_model = CustomModel3.load_pretrained(GENERAL_MODEL_NAME, conf, 
                                             config=AutoConfig.from_pretrained(conf.model.model_name))

  binary_model.encoder.resize_token_embeddings(len(tokenizer))
  general_model.encoder.resize_token_embeddings(len(tokenizer))
  
  binary_model.parameters
  general_model.parameters
  binary_model.to(device)
  general_model.to(device)

  ## load binary test datset
  test_dataset_dir = conf.path.test_path
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(binary_model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label_binary(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

  print("---- binary inference successed!! ----")
  
  ## Output after binary inference
  binary_output = pd.DataFrame({'id':test_id,
                                'pred_label':pred_answer,
                                'probs':output_prob,})
  
  # 'relation' label의 id 추출
  grouped = binary_output.groupby('pred_label')

  general_classify_ids = []
  no_relation_rows = []

  # if row['probs'][0] > "학습시 정확도":
  for idx, row in binary_output.iterrows():
    if row['probs'][0] > 0.765:
      false_prob = row['probs'][0]
      true_prob = row['probs'][1]
      a = [true_prob/29 for _ in range(30)]
      a[0] = false_prob

      new_row = row.to_frame().T
      new_row.at[idx, 'probs'] = a
      no_relation_rows.append(new_row)
    else:
      general_classify_ids.append(row['id'])
    
  final_output = pd.concat(no_relation_rows, ignore_index=True)
      
  origin_df = pd.read_csv(conf.path.test_path)
  filtered_df = origin_df[origin_df['id'].isin(general_classify_ids)]
  filtered_df.to_csv('../dataset/test/general_test.csv')

  ## load general test datset
  test_dataset_dir = '../dataset/test/general_test.csv'
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference(general_model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.

  print("---- general inference successed!! ----")

  ## Output after general inference
  general_output = pd.DataFrame({'id':test_id,
                                'pred_label':pred_answer,
                                'probs':output_prob,})
  
  final_output = pd.concat([final_output, general_output])

  final_output = final_output.sort_values(by='id', ascending=True)

  final_output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
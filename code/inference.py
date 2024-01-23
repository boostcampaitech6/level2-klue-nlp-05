from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from dataset_utils import load_test_dataset, num_to_label
from custom_datasets import RE_Dataset
from base_train import set_seed
from tqdm import tqdm

import torch.nn.functional as F
import pandas as pd
import torch
import numpy as np
import argparse
import os

from config.config import call_config
from custom_model import CustomModel, CustomModel2, CustomModel3

def custom_inference(model, tokenized_sent, device):
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

def base_inference(model, tokenized_sent, device):
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
          token_type_ids=data['token_type_ids'].to(device)
          )

    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)

  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")
  parser.add_argument('--model_dir', "-m", type=str, default="./best_model")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")
  print(args)
  
  set_seed(conf.utils.seed)
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  if conf.utils.TOKEN_TYPE == 1:
    # 스페셜 토큰 (기호형)
    special_tokens = ['<S:ORG>','<S:PER>','<S:POH>','<S:LOC>','<S:DAT>','<S:NOH>','</S:ORG>','</S:PER>','</S:POH>','</S:LOC>','</S:DAT>','</S:NOH>','<O:ORG>','<O:PER>','<O:POH>','<O:LOC>','<O:DAT>','<O:NOH>','</O:ORG>','</O:PER>','</O:POH>','</O:LOC>','</O:DAT>','</O:NOH>']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
  elif conf.utils.TOKEN_TYPE == 2:
    # 일반 토큰 (신근이형)
    new_tokens = ['[SE]','[/SE]','[OE]','[/OE]']
    tokenizer.add_tokens(new_tokens)

  ## load my model
  MODEL_NAME = conf.model.model_name
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  if conf.custom_model==1:
    model_config = AutoConfig.from_pretrained(args.model_dir)
    model = CustomModel(conf.model.model_name, config=model_config)
    model.load_state_dict(torch.load(os.path.join(MODEL_NAME, 'pytorch_model.bin')))
  elif conf.custom_model==2:
    model_config = AutoConfig.from_pretrained(args.model_dir)
    model = CustomModel2(conf.model.model_name, config=model_config)
    model.load_state_dict(torch.load(os.path.join(MODEL_NAME, 'pytorch_model.bin')))
  elif conf.custom_model==3:
    model = CustomModel3(conf, config=model_config)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(args.model_dir))
  else:
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=model_config)
  
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = conf.path.test_path
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, MODEL_NAME)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  if conf.custom_model > 0:
    pred_answer, output_prob = custom_inference(model, Re_test_dataset, device) # model에서 class 추론
  else:
    pred_answer, output_prob = base_inference(model, Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
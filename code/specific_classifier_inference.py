from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from dataset_utils import load_test_dataset, num_to_label
from omegaconf import OmegaConf
from train import set_seed
from datasets import RE_Dataset
from tqdm import tqdm

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
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def pair_separater(file_path):
    """csv에 ['entity pair']column을 기준으로 분리해서 
    1. 각 csv파일로 저장해주는 함수

    Args:
        file_path (str): entity pair가 마구잡이로 섞인 csv파일의 경로

    Returns:
        list: pair의 리스트
    """
    df = pd.read_csv(file_path)
    grouped = df.groupby('entity_pair')
    pair_list = []
    for name, group in grouped:
       file_name = f'../dataset/test/{name}_test.csv'
       group.to_csv(file_name, index=False)
       pair_list.append(name)
    return pair_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="base_config")

    args, _ = parser.parse_known_args()
    conf = OmegaConf.load(f"./config/{args.config}.yaml")
    
    set_seed(conf.utils.seed)
    """
        주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = conf.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    pair_list = pair_separater(conf.path.test_path)

    test_id_final = pd.Series()
    pred_answer_final = []
    output_prob_final = []

    # entity pair별로 추론진행
    for pair in pair_list:
        ## load my model
        MODEL_NAME = f'./best_model/{pair}' # model dir.
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.parameters
        model.to(device)

        ## load test datset
        test_id, test_dataset, test_label = load_test_dataset(f'../dataset/test/{pair}_test.csv', 
                                                              tokenizer)
        Re_test_dataset = RE_Dataset(test_dataset, 
                                     test_label)

        ## predict answer
        pred_answer, output_prob = inference(model, Re_test_dataset, device) # model에서 class 추론
        pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
        
        test_id_final = pd.concat([test_id_final, test_id], axis=0, ignore_index=True)
        pred_answer_final += pred_answer
        output_prob_final += output_prob


    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id':test_id_final,
                           'pred_label':pred_answer_final,
                           'probs':output_prob_final})
    output = output.sort_values(by='id', ascending=True)

    output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')
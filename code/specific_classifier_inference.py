from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from dataset_utils import load_test_dataset, num_to_label, num_to_label_binary
from omegaconf import OmegaConf
from general_classifier_train import set_seed
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

    ## load my model
    BINARY_MODEL_NAME = './best_model/Binary'
    binary_model = AutoModelForSequenceClassification.from_pretrained(BINARY_MODEL_NAME)
    binary_model.parameters
    binary_model.to(device)

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

    specific_classify_ids = []
    no_relation_rows = []

    # if row['probs'][0] > "학습시 정확도":
    for idx, row in binary_output.iterrows():
        if row['probs'][0] > 0.75:
            false_prob = row['probs'][0]
            true_prob = row['probs'][1]
            a = [true_prob/29 for _ in range(30)]
            a[0] = false_prob

            new_row = row.to_frame().T
            new_row.at[idx, 'probs'] = a
            no_relation_rows.append(new_row)
        else:
            specific_classify_ids.append(row['id'])
        
    final_output = pd.concat(no_relation_rows, ignore_index=True)
        
    origin_df = pd.read_csv(conf.path.test_path)
    filtered_df = origin_df[origin_df['id'].isin(specific_classify_ids)]
    filtered_df.to_csv('../dataset/test/general_test.csv')

    pair_list_train = ['PER-DAT', 'ORG-PER', 'PER-ORG', 
                       'PER-POH']#, 'ORG-ORG', 'PER-PER']  
    pair_list_test = pair_separater('../dataset/test/general_test.csv') 

    exception_pair = set(pair_list_test) - set(pair_list_train)
    exception_pair = list(exception_pair)

    for pair in pair_list_train:
        ## load specific test datset
        test_dataset_dir = f'../dataset/test/{pair}_test.csv'
        test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
        Re_test_dataset = RE_Dataset(test_dataset ,test_label)

        curr_pair_probs = []

        for fold in range(conf.utils.stratifiedKFold):
            SPECIFIC_MODEL_NAME = f'./best_model/{pair}/{fold}_true_labels'
            specific_model = AutoModelForSequenceClassification.from_pretrained(SPECIFIC_MODEL_NAME)
            specific_model.parameters
            specific_model.to(device)

            ## predict answer
            _, output_prob = inference(specific_model, Re_test_dataset, device) # model에서 class 추론
            curr_pair_probs.append(output_prob)

        pair_probs = []
        pair_answer = []

        for idx in range( len(curr_pair_probs[0]) ):
           avg_list = [curr_pair_probs[0][idx],
                       curr_pair_probs[1][idx],
                       curr_pair_probs[2][idx],
                       curr_pair_probs[3][idx],
                       curr_pair_probs[4][idx]]
           avg_prob = [sum(x)/len(x) for x in zip(*avg_list)]
           pair_probs.append(avg_prob)

        for prob in pair_probs:
           result = np.argmax(prob, axis=-1)
           pair_answer.append(result)

        pair_answer = num_to_label(pair_answer)
        
        ## Output after specific pair inference
        pair_output = pd.DataFrame({'id':test_id,
                                    'pred_label':pair_answer,
                                    'probs':pair_probs,})
    
        final_output = pd.concat([final_output, pair_output])

    print("---- specific inference successed!! ----")

    for pair in exception_pair:
        GENERAL_MODEL_NAME = f'./best_model/General'
        general_model = AutoModelForSequenceClassification.from_pretrained(GENERAL_MODEL_NAME)
        general_model.parameters
        general_model.to(device)

        test_id, test_dataset, test_label = load_test_dataset(f'../dataset/test/{pair}_test.csv', 
                                                              tokenizer)
        Re_test_dataset = RE_Dataset(test_dataset,
                                     test_label)
        
        pred_answer, output_prob = inference(general_model, Re_test_dataset, device)
        pred_answer = num_to_label(pred_answer)

        general_output = pd.DataFrame({'id':test_id,
                                       'pred_label':pred_answer,
                                       'probs':output_prob})
        final_output = pd.concat([final_output, general_output])

    final_output = final_output.sort_values(by='id', ascending=True)

    final_output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print('---- Finish! ----')
import pandas as pd
import torch
import numpy as np
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F

from load_data import RE_Dataset, load_test_dataset, num_to_label
from model import CustomModel
from train import set_seed


def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
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
  parser.add_argument("--config", "-c", type=str, default="best_config")
  parser.add_argument('--model_dir', "-m", type=str, default="./best_model/model.pt")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")
  print(args)
  
  set_seed(conf.utils.seed)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # load tokenizer
  Tokenizer_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
  # add special tokens
  special_tokens = ['<S:ORG>','</S:ORG>', 
                    '<S:PER>', '</S:PER>', 
                    '<S:POH>', '</S:POH>', 
                    '<S:LOC>', '</S:LOC>',
                    '<S:DAT>', '</S:DAT>',
                    '<S:NOH>', '</S:NOH>', 
                    '<O:ORG>', '</O:ORG>', 
                    '<O:PER>', '</O:PER>', 
                    '<O:POH>', '</O:POH>', 
                    '<O:LOC>', '</O:LOC>',
                    '<O:DAT>', '</O:DAT>', 
                    '<O:NOH>', '</O:NOH>']
  tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

  # load my model
  MODEL_NAME = conf.model.model_name
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  model = CustomModel(conf, config=model_config)
  # resize token embeddings
  model.encoder.resize_token_embeddings(len(tokenizer))
  model.load_state_dict(torch.load(args.model_dir))
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = conf.path.test_path
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  # predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device)
  pred_answer = num_to_label(pred_answer)
  
  # make csv file with predicted answer
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  print('---- Finish! ----')
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoModelForCausalLM
from omegaconf import OmegaConf
from dataset_utils import load_data, label_to_num, tokenized_dataset, tokenized_dataset_xlm
from datasets import RE_Dataset
from metrics import compute_metrics

import numpy as np
import argparse
import random
import torch
import wandb
import os


def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")

  set_seed(42)

  wandb.login()
  wandb.init(project=conf.wandb.project_name)

  # load model and tokenizer
  MODEL_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("../dataset/train/train.csv")

  if MODEL_NAME=="beomi/llama-2-ko-7b":
    question_template = "### Human: 다음 두 문장의 관계를 entailment, neutral, contradiction 중 하나로 분류해줘. "
    train_instructions = [f'{question_template}\npremise: {x}\nhypothesis: {y}\n\n### Assistant: {label_to_num[z]}' for x,y,z in zip(train_dataset['premise'],train_dataset['hypothesis'],train_dataset['label'])]
  train_label = label_to_num(train_dataset['label'].values)

  # tokenizing dataset
  if MODEL_NAME[:10] == "xlm-roberta":
    tokenized_train = tokenized_dataset_xlm(train_dataset, tokenizer)
  else:
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  
  # Split train dataset into train, valid.
  RE_train_dataset, RE_dev_dataset = torch.utils.data.random_split(RE_train_dataset, [int(len(RE_train_dataset)*0.8), len(RE_train_dataset)-int(len(RE_train_dataset)*0.8)])

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  if MODEL_NAME == "beomi/llama-2-ko-7b":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    device_map = {"": 0}
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrainedAutoModelForCausalLM.from_pretrained(
                                                                                      MODEL_NAME,
                                                                                      quantization_config=quantization_config,
                                                                                      device_map=device_map,
                                                                                      trust_remote_code=True,
                                                                                      torch_dtype=torch_dtype,
                                                                                      use_auth_token=False,
                                                                                  )    
  else:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results',          # output directory
    save_total_limit=conf.utils.top_k,  # number of total save model.
    save_steps=conf.train.save_steps,                 # model saving step.
    num_train_epochs=conf.train.epochs,              # total number of training epochs
    learning_rate=conf.train.learning_rate,               # learning_rate
    per_device_train_batch_size=conf.train.batch_size,  # batch size per device during training
    per_device_eval_batch_size=conf.train.batch_size,   # batch size for evaluation
    warmup_steps=conf.train.warmup_steps,                # number of warmup steps for learning rate scheduler
    logging_dir='./logs',            # directory for storing logs
    logging_steps=conf.train.logging_steps,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps=conf.train.eval_steps,            # evaluation step.
    load_best_model_at_end = True,
    metric_for_best_model="micro f1 score",
    report_to="wandb",
    adam_beta1=conf.train.adam_beta1,
    adam_beta2=conf.train.adam_beta2,
    adam_epsilon=conf.train.adam_epsilon,
    lr_scheduler_type=conf.train.lr_scheduler_type,
    fp16=conf.train.fp16
  )
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=RE_train_dataset,
    eval_dataset=RE_dev_dataset,
    compute_metrics=compute_metrics
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model')
  
  wandb.finish()

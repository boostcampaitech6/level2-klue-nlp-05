from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from omegaconf import OmegaConf
from dataset_utils import load_data, label_to_num, tokenized_dataset_prompt, tokenized_dataset_xlm
from custom_trainer import CustomTrainer
from custom_datasets import RE_Dataset
from metrics import compute_metrics

import torch.optim as optim
import numpy as np
import transformers
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

def optuna_hp_space(trial):
  return {
    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.4),
    "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000)
  }


if __name__ == '__main__':
  """
    custom모델이 아닌 xlm-roberta, klue/roberta 모델들을 train합니다.
  
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", "-c", type=str, default="base_config")
  parser.add_argument("--train_type", "-type", type=str, default="train")

  args, _ = parser.parse_known_args()
  conf = OmegaConf.load(f"./config/{args.config}.yaml")

  set_seed(conf.utils.seed)

  wandb.login()
  wandb.init(project=conf.wandb.project_name, entity='level2-klue-nlp-05')

  # load model and tokenizer
  MODEL_NAME = conf.model.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  # load dataset
  train_dataset = load_data(conf.path.train_path)
  dev_dataset = load_data(conf.path.dev_path)

  train_label = label_to_num(train_dataset['label'].values)
  dev_label = label_to_num(dev_dataset['label'].values)

  # tokenizing dataset
  if MODEL_NAME.split('-')[0] == 'xlm':
    tokenized_train = tokenized_dataset_xlm(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset_xlm(dev_dataset, tokenizer)
  else:
    tokenized_train = tokenized_dataset_prompt(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset_prompt(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  print(device)
  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 30

  if conf.model.use_tapt_model:
    MODEL_NAME = conf.path.tapt_model_path

  def model_init():
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  print(model.config)
  model.parameters
  model.to(device)


  training_args = TrainingArguments(
    output_dir='./results',
    save_total_limit=conf.utils.top_k,
    save_steps=conf.train.save_steps,
    num_train_epochs=conf.train.epochs,
    learning_rate=conf.train.learning_rate,
    per_device_train_batch_size=conf.train.batch_size,
    per_device_eval_batch_size=conf.train.batch_size,
    weight_decay=conf.train.weight_decay,
    warmup_steps=conf.train.warmup_steps,
    logging_dir='./logs',
    logging_steps=conf.train.logging_steps,
    evaluation_strategy='steps',
    eval_steps=conf.train.eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model="micro f1 score",
    lr_scheduler_type="cosine",
    report_to="wandb",
    fp16=conf.train.fp16,
    gradient_accumulation_steps=conf.train.gradient_accumulation_step
  )

  if args.train_type == "train":
    trainer = CustomTrainer(
      model=model,
      args=training_args,
      train_dataset=RE_train_dataset,
      eval_dataset=RE_dev_dataset,
      data_collator=data_collator,
      compute_metrics=compute_metrics
    )
    trainer.train()
    model.save_pretrained('./best_model')
  elif args.train_type == "hp_search":
    trainer = CustomTrainer(
	    model=None,
	    args=training_args,
	    train_dataset=RE_train_dataset,
	    eval_dataset=RE_dev_dataset,
      data_collator=data_collator,
	    compute_metrics=compute_metrics,
	    model_init=model_init
	  )
    best_trials = trainer.hyperparameter_search(n_trials=10,
                                               direction="maximize",
                                               backend="optuna",
                                               hp_space=optuna_hp_space)
    print(best_trials)

  wandb.finish()
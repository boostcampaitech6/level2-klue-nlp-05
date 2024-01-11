from datasets import load_dataset, Dataset, DatasetDict
from dataclasses import dataclass, field
from typing import Optional
import torch
from peft import LoraConfig
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, pipeline
from trl import SFTTrainer

from dataset_utils import load_data, label_to_num, tokenized_dataset, tokenized_dataset_xlm

tqdm.pandas()

train_data = load_data("../dataset/train/train.csv")

question_template = "### Human: 다음 문장 내 두 단어를 no_relation, org:member_of, org:top_members/employees, org:alternate_names, per:date_of_birth, org:place_of_headquarters, per:employee_of, per:origin, per:title, org:members, per:schools_attended, per:colleagues, per:alternate_names, per:spouse, org:founded_by, org:political/religious_affiliation, per:children, org:founded, org:number_of_employees/members, per:place_of_birth, org:dissolved, per:parents, per:religion, per:date_of_death, per:place_of_residence, per:other_family, org:product, per:siblings, per:product, per:place_of_death 중 하나로 분류해줘. "
train_label = label_to_num(train_data['label'].values)

train_instructions = [f'{question_template}\nsentence: {sentence}\n{x}:{y}\n\n### Assistant: {z}' for sentence,x,y,z in zip(train_data['sentence'],train_data['subject_entity'],train_data['object_entity'],train_label)]
ds_train = Dataset.from_dict({"text": train_instructions})
instructions_ds_dict = DatasetDict({"train": ds_train, "validation": ds_train})


model_name = "beomi/llama-2-ko-7b"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default=model_name, metadata={"help": "the model name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})


script_args = ScriptArguments()


if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    device_map = {"": 0}
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    cache_dir='/data/ephemeral/home/tmp'
)


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# tokenized_ds = tokenizer(ds_train['text'], padding=True, truncation=True, max_length=script_args.seq_length)

# dataset = tokenized_ds

# train_df, val_df = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    disable_tqdm=False,
)

if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=instructions_ds_dict['train'],
    eval_dataset=instructions_ds_dict['validation'],
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
)


trainer.train()

trainer.save_model(training_args.output_dir)


model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map={'':0},
)


query = instructions_ds_dict['eval']['text'][1].split('### Assistant: ')[0] + '### Assistant:'
queries = [instructions_ds_dict['eval']['text'][i].split('### Assistant: ')[0] + '### Assistant:' for i in range(len(instructions_ds_dict['eval']))]
sequences = pipeline(
    queries,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=3,
    early_stopping=True,
    # do_sample=True,
)


results = []

for seq in sequences:
  result = seq[0]['generated_text'].split('### Assistant:')[1]
  results.append(result)

labels = []

for label in instructions_ds_dict['eval']['text']:
  result = label.split('### Assistant:')[1]
  labels.append(result)

print("Accuracy: ", (len([1 for x, y in zip(results, labels) if y in x]) / len(labels)))
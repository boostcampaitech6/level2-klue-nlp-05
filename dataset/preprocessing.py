import pandas as pd
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def extract_columns(df):
    subject_word, subject_start_idx, subject_end_idx, subject_type = [], [], [], []
    for data in df['subject_entity']:
        data = ast.literal_eval(data)
        subject_word.append(data['word'])
        subject_start_idx.append(data['start_idx'])
        subject_end_idx.append(data['end_idx'])
        subject_type.append(data['type'])
    df['subject_word'], df['subject_start_idx'], df['subject_end_idx'], df['subject_type'] = subject_word, subject_start_idx, subject_end_idx, subject_type

    object_word, object_start_idx, object_end_idx, object_type = [], [], [], []
    for data in df['object_entity']:
        data = ast.literal_eval(data)
        object_word.append(data['word'])
        object_start_idx.append(data['start_idx'])
        object_end_idx.append(data['end_idx'])
        object_type.append(data['type'])
    df['object_word'], df['object_start_idx'], df['object_end_idx'], df['object_type'] = object_word, object_start_idx, object_end_idx, object_type

    df.drop(columns=['subject_entity', 'object_entity'], inplace=True)

    return df

def data_cleaning(df):
    df.drop_duplicates(subset=['sentence', 'subject_word','object_word','label'], inplace=True)
    duplicates = df[df.duplicated(subset=['sentence', 'subject_word','object_word'], keep=False)]
    df.drop(duplicates[duplicates['label'] == 'no_relation'].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index

    return df

# def data_augmentation(df):
#     augmented_data = []
#     tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
#     model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')

#     print("***** Running data augmentation *****")
#     for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
#         sentence = row['sentence']
#         subject_start_idx, object_start_idx = row['subject_start_idx'], row['object_start_idx']
#         subject_end_idx, object_end_idx = row['subject_end_idx'], row['object_end_idx']
        
#         if subject_start_idx < object_start_idx:
#             masked_sentence = sentence[:subject_start_idx] + '[MASK]' + sentence[subject_end_idx+1:object_start_idx] + '[MASK]' + sentence[object_end_idx+1:]
#         else:
#             masked_sentence = sentence[:object_start_idx] + '[MASK]' + sentence[object_end_idx+1:subject_start_idx] + '[MASK]' + sentence[subject_end_idx+1:]

#         input_ids = tokenizer.encode(masked_sentence, return_tensors="pt")

#         with torch.no_grad():
#             outputs = model(input_ids)
#             predictions = outputs[0]

#         mask_indices = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
#         predicted_tokens = []

#         for mask_index in mask_indices:
#             predicted_index = torch.argmax(predictions[0, mask_index]).item()
#             predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
#             predicted_tokens.append(predicted_token)

#         new_sentence = masked_sentence.replace('[MASK]', predicted_tokens[0], 1).replace('[MASK]', predicted_tokens[1], 1)
        
#         if ('[PAD]' in new_sentence) or ('[UNK]' in new_sentence):
#             continue
#         else:
#             if subject_start_idx < object_start_idx:
#                 row['subject_word'], row['object_word'] = predicted_tokens[0], predicted_tokens[1]
#                 row['subject_start_idx'] = new_sentence.find(predicted_tokens[0])
#                 row['subject_end_idx'] = row['subject_start_idx'] + len(predicted_tokens[0]) - 1
#                 row['object_start_idx'] = new_sentence.find(predicted_tokens[1], row['subject_end_idx']+1)
#                 row['object_end_idx'] = row['object_start_idx'] + len(predicted_tokens[1]) - 1
#                 row['sentence'] = new_sentence
#                 augmented_data.append(row)
#             else:
#                 row['object_word'], row['subject_word'] = predicted_tokens[0], predicted_tokens[1]
#                 row['object_start_idx'] = new_sentence.find(predicted_tokens[0])
#                 row['object_end_idx'] = row['object_start_idx'] + len(predicted_tokens[0]) - 1
#                 row['subject_start_idx'] = new_sentence.find(predicted_tokens[1], row['object_end_idx']+1)
#                 row['subject_end_idx'] = row['subject_start_idx'] + len(predicted_tokens[1]) - 1
#                 row['sentence'] = new_sentence
#                 augmented_data.append(row)

#     df = pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)
#     df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#     df['id'] = df.index
#     print("Complete!")

#     return df
    
train, validation, test = pd.read_csv("./train/train.csv"), pd.read_csv("./validation/validation.csv"), pd.read_csv("./test/test.csv")
train, validation, test = extract_columns(train), extract_columns(validation), extract_columns(test)
train = data_cleaning(train)
# train = data_augmentation(train)
train.to_csv("./train/train_final.csv", index=False)
validation.to_csv('./validation/validation_final.csv', index=False)
test.to_csv("./test/test_final.csv", index=False)
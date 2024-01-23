import pandas as pd
import ast

def extract_columns(df):
    # subject_entity
    subject_word, subject_start_idx, subject_end_idx, subject_type = [], [], [], []
    for data in df['subject_entity']:
        data = ast.literal_eval(data)
        subject_word.append(data['word'])
        subject_start_idx.append(data['start_idx'])
        subject_end_idx.append(data['end_idx'])
        subject_type.append(data['type'])
    df['subject_word'], df['subject_start_idx'], df['subject_end_idx'], df['subject_type'] = subject_word, subject_start_idx, subject_end_idx, subject_type
    # object_entity
    object_word, object_start_idx, object_end_idx, object_type = [], [], [], []
    for data in df['object_entity']:
        data = ast.literal_eval(data)
        object_word.append(data['word'])
        object_start_idx.append(data['start_idx'])
        object_end_idx.append(data['end_idx'])
        object_type.append(data['type'])
    df['object_word'], df['object_start_idx'], df['object_end_idx'], df['object_type'] = object_word, object_start_idx, object_end_idx, object_type
    # drop subject_entity, object_entity
    df.drop(columns=['subject_entity', 'object_entity'], inplace=True)

    return df

def data_cleaning(df):
    # remove duplicates
    df.drop_duplicates(subset=['sentence', 'subject_word','object_word','label'], inplace=True)
    duplicates = df[df.duplicated(subset=['sentence', 'subject_word','object_word'], keep=False)]
    df.drop(duplicates[duplicates['label'] == 'no_relation'].index, inplace=True)

    return df
    
train, validation, test = pd.read_csv("./train/train.csv"), pd.read_csv("./validation/validation.csv"), pd.read_csv("./test/test.csv")
train, validation, test = extract_columns(train), extract_columns(validation), extract_columns(test)
train = data_cleaning(train)
train.to_csv("./train/train_final.csv", index=False)
validation.to_csv('./validation/validation_final.csv', index=False)
test.to_csv("./test/test_final.csv", index=False)
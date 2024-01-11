import pandas as pd
import ast
from config.config import call_config

conf = call_config()

def preprocessing(df, train=False):
    # seperate subject_entity
    subject_word, subject_start_idx, subject_end_idx, subject_type = [], [], [], []
    for data in df['subject_entity']:
        data = ast.literal_eval(data)
        subject_word.append(data['word'])
        subject_start_idx.append(data['start_idx'])
        subject_end_idx.append(data['end_idx'])
        subject_type.append(data['type'])
    df['subject_word'], df['subject_start_idx'], df['subject_end_idx'], df['subject_type'] = subject_word, subject_start_idx, subject_end_idx, subject_type
    # seperate object_entity
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
    # add entity_pair
    df['entity_pair'] = df['subject_type'] + "-" + df['object_type']
    
    
      # use duplicated sentence preprocessing
    if conf.dup_preprocessing & train:
        outlier_sentence=[18458,6749,8364,22258,10202,277,10320,25094]
        df = df.drop(outlier_sentence)
        df.drop_duplicates(['sentence', 'subject_word', 'object_word'], keep='first', inplace=True)
        df = df.reset_index(drop=True)
        
    if conf.type_pair_preprocessing & train:
        for label in df['label'].unique():
            label_df = df[df['label'] == label]
            
            type_counts = label_df['object_type'].value_counts()

            types_to_remove = type_counts[type_counts <= 30].index
            
            df = df[~((df['label'] == label) & (df['object_type'].isin(types_to_remove)))]


    return df


if __name__ == '__main__':
    train, test = pd.read_csv("../dataset/train/train.csv"), pd.read_csv("../dataset/test/test.csv")
    train, test = preprocessing(train), preprocessing(test)
    train.to_csv("../dataset/train/train_final.csv")
    test.to_csv("../dataset/test/test_final.csv")
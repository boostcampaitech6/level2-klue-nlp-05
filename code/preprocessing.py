import pandas as pd
import ast

def train_data_preprocessing(df):
    '''
    train data에서 중복 데이터가 존재해 이들을 삭제
    '''

    # sentence, subject_entity, object_entity, label 모두 중복인 데이터 삭제
    new_df = df.drop_duplicates(['sentence', 'subject_entity','object_entity','label'])

    # sentence, sub_entity, obj_entity가 중복되는 data에 대해 label이 no_relation인 데이터 삭제
    idx_list = [12829, 32299, 11511, 3296, 25094]   # 중복 데이터의 id

    for i in range(len(idx_list)):
        idx = new_df[new_df["id"] == idx_list[i]].index
        new_df.drop(idx, inplace=True)

    return new_df

def preprocessing(df):
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

    return df

train, test = pd.read_csv("../dataset/train/train.csv"), pd.read_csv("../dataset/test/test.csv")
train = train_data_preprocessing(train)
train, test = preprocessing(train), preprocessing(test)
train.to_csv("../dataset/train/train_final.csv")
test.to_csv("../dataset/test/test_final.csv")
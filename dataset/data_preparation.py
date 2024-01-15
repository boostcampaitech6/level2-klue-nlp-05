import pandas as pd
import ast

# def train_dev_split(df):
#     class_0_sample = df[df['label']=='no_relation'].sample(2315, random_state=42)
#     class_1_sample = df[df['label']=='org:dissolved'].sample(5, random_state=42)
#     class_2_sample = df[df['label']=='org:founded'].sample(10, random_state=42)
#     class_3_sample = df[df['label']=='org:place_of_headquarters'].sample(97, random_state=42)
#     class_4_sample = df[df['label']=='org:alternate_names'].sample(39, random_state=42)
#     class_5_sample = df[df['label']=='org:member_of'].sample(52, random_state=42)
#     class_6_sample = df[df['label']=='org:members'].sample(61, random_state=42)
#     class_7_sample = df[df['label']=='org:political/religious_affiliation'].sample(6, random_state=42)
#     class_8_sample = df[df['label']=='org:product'].sample(117, random_state=42)
#     class_9_sample = df[df['label']=='org:founded_by'].sample(5, random_state=42)
#     class_10_sample = df[df['label']=='org:top_members/employees'].sample(256, random_state=42)
#     class_11_sample = df[df['label']=='org:number_of_employees/members'].sample(8, random_state=42)
#     class_12_sample = df[df['label']=='per:date_of_birth'].sample(6, random_state=42)
#     class_13_sample = df[df['label']=='per:date_of_death'].sample(6, random_state=42)
#     class_14_sample = df[df['label']=='per:place_of_birth'].sample(5, random_state=42)
#     class_15_sample = df[df['label']=='per:place_of_death'].sample(5, random_state=42)
#     class_16_sample = df[df['label']=='per:place_of_residence'].sample(62, random_state=42)
#     class_17_sample = df[df['label']=='per:origin'].sample(59, random_state=42)
#     class_18_sample = df[df['label']=='per:employee_of'].sample(121, random_state=42)
#     class_19_sample = df[df['label']=='per:schools_attended'].sample(5, random_state=42)
#     class_20_sample = df[df['label']=='per:alternate_names'].sample(52, random_state=42)
#     class_21_sample = df[df['label']=='per:parents'].sample(13, random_state=42)
#     class_22_sample = df[df['label']=='per:children'].sample(13, random_state=42)
#     class_23_sample = df[df['label']=='per:siblings'].sample(12, random_state=42)
#     class_24_sample = df[df['label']=='per:spouse'].sample(20, random_state=42)
#     class_25_sample = df[df['label']=='per:other_family'].sample(17, random_state=42)
#     class_26_sample = df[df['label']=='per:colleagues'].sample(110, random_state=42)
#     class_27_sample = df[df['label']=='per:product'].sample(33, random_state=42)
#     class_28_sample = df[df['label']=='per:religion'].sample(6, random_state=42)
#     class_29_sample = df[df['label']=='per:title'].sample(359, random_state=42)
    
#     dev = pd.concat([class_0_sample, 
#                      class_1_sample, 
#                      class_2_sample, 
#                      class_3_sample,
#                      class_4_sample, 
#                      class_5_sample, 
#                      class_6_sample, 
#                      class_7_sample,
#                      class_8_sample, 
#                      class_9_sample, 
#                      class_10_sample,
#                      class_11_sample, 
#                      class_12_sample, 
#                      class_13_sample, 
#                      class_14_sample,
#                      class_15_sample, 
#                      class_16_sample, 
#                      class_17_sample, 
#                      class_18_sample,
#                      class_19_sample, 
#                      class_20_sample, 
#                      class_21_sample, 
#                      class_22_sample,
#                      class_23_sample, 
#                      class_24_sample, 
#                      class_25_sample, 
#                      class_26_sample, 
#                      class_27_sample, 
#                      class_28_sample, 
#                      class_29_sample
#                     ])
    
#     train = df.drop(dev.index)
#     dev = dev.sample(frac=1, random_state=42)

#     return train, dev

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

    return df

# def data_augmentation(df):
#     relation_sample = df[df['label']!='no_relation']
#     entity_pair_sample = relation_sample.sample(frac=1, random_state=42)[['subject_word','subject_type','object_word','object_type']].sample(num_samples, random_state=42).reset_index(drop=True)
#     relation_sample['id'] = range(len(train), len(train) + len(relation_sample))

#     for idx in range(len(relation_sample)):
#         # sentence
#         relation_sample['sentence'][idx] = relation_sample['sentence'][idx].replace(relation_sample['subject_word'][idx], entity_pair_sample['subject_word'][idx])
#         relation_sample['sentence'][idx] = relation_sample['sentence'][idx].replace(relation_sample['object_word'][idx], entity_pair_sample['object_word'][idx])
#         # label
#         relation_sample['label'][idx] = 'no_relation'
#         # subject_word
#         relation_sample['subject_word'][idx] = entity_pair_sample['subject_word'][idx]
#         # subject_start_idx
#         relation_sample['subject_start_idx'][idx] = relation_sample['sentence'][idx].find(entity_pair_sample['subject_word'][idx])
#         # subject_end_idx
#         relation_sample['subject_end_idx'][idx] = relation_sample['subject_start_idx'][idx] + len(entity_pair_sample['subject_word'][idx])
#         # subject_type
#         relation_sample['subject_type'][idx] = entity_pair_sample['subject_type'][idx]
#         # object_word
#         relation_sample['object_word'][idx] = entity_pair_sample['object_word'][idx]
#         # object_start_idx
#         relation_sample['object_start_idx'][idx] = relation_sample['sentence'][idx].find(entity_pair_sample['object_word'][idx])
#         # object_end_idx
#         relation_sample['object_end_idx'][idx] = relation_sample['object_start_idx'][idx] + len(entity_pair_sample['object_word'][idx])
#         # object_type
#         relation_sample['object_type'][idx] = entity_pair_sample['object_type'][idx]
    
#     df = pd.concat([df, relation_sample])
#     df = df.sample(frac=1, random_state=42)

#     return df

train, dev, test = pd.read_csv("./train/train.csv"), pd.read_csv("./train/dev.csv"), pd.read_csv("./test/test.csv")
# train, dev = train_dev_split(df)
# dev.to_csv("./train/dev.csv", index=False)
train, dev, test = preprocessing(train), preprocessing(dev), preprocessing(test)
# train = data_augmentation(train)
train.to_csv("./train/train_final.csv", index=False)
dev.to_csv('./train/dev_final.csv', index=False)
test.to_csv("./test/test_final.csv", index=False)
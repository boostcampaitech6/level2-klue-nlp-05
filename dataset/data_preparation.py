import pandas as pd
import ast

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

def data_cleaning(df):
    # 중복 데이터 제거
    # sentence, subject_word, object_word, label이 중복인 데이터 제거
    df.drop_duplicates(subset=['sentence', 'subject_word','object_word','label'], inplace=True)
    # sentence, subject_word, object_word가 중복이지만 레이블이 다른 데이터 -> no_relation 데이터 제거
    duplicates = df[df.duplicated(subset=['sentence', 'subject_word','object_word'], keep=False)]
    df.drop(duplicates[duplicates['label'] == 'no_relation'].index, inplace=True)
    # label별 entity 타입 수정
    # 1. org:top_members/employees
    df['subject_type'][df[(df['label']=='org:top_members/employees') & (df['subject_type']=='PER')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:top_members/employees') & (df['object_type']=='ORG')].index] = 'PER'
    df['object_type'][df[(df['label']=='org:top_members/employees') & (df['object_type']=='ORG')].index] = 'PER'
    df['object_type'][df[(df['label']=='org:top_members/employees') & (df['object_type']=='NOH')].index] = 'PER'
    # 2. org:members
    df['subject_type'][df[(df['label']=='org:members') & (df['subject_type']=='PER')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:members') & (df['object_type']=='PER')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:members') & (df['object_type']=='NOH')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:members') & (df['object_type']=='DAT')].index] = 'ORG'
    # 3. org:product
    df['object_type'][df[df['label']=='org:product'].index] = 'POH'
    # 4. per:title
    df['object_type'][df[df['label']=='per:title'].index] = 'POH'
    # 5. org:alternate_names
    df['object_type'][df[(df['label']=='org:alternate_names') & (df['object_type']=='PER')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:alternate_names') & (df['object_type']=='DAT')].index] = 'ORG'
    # 6. per:employee_of
    df['object_type'][df[(df['label']=='per:employee_of') & (df['object_type']=='DAT')].index] = 'ORG'
    df['object_type'][df[(df['label']=='per:employee_of') & (df['object_type']=='NOH')].index] = 'ORG'
    # 7. org:place_of_headquarters
    df['subject_type'][df[(df['label']=='org:place_of_headquarters') & (df['subject_type']=='PER')].index] = 'ORG'
    df['object_type'][df[df['label']=='org:place_of_headquarters'].index] = 'LOC'
    # 8. per:product
    df['object_type'][df[df['label']=='per:product'].index] = 'POH'
    # 9. org:number_of_employees/members
    # 10. per:children
    df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='LOC')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='NOH')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='DAT')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='ORG')].index] = 'PER'
    # 11. per:place_of_residence
    df['object_type'][df[df['label']=='per:place_of_residence'].index] = 'LOC'
    # 12. per:alternate_names
    df['object_type'][df[(df['label']=='per:alternate_names') & (df['object_type']=='ORG')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:alternate_names') & (df['object_type']=='LOC')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:alternate_names') & (df['object_type']=='ORG')].index] = 'PER'
    # 13. per:other_family
    df['object_type'][df[(df['label']=='per:other_family') & (df['object_type']=='LOC')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:other_family') & (df['object_type']=='ORG')].index] = 'PER'
    # 14. per:colleagues
    df['object_type'][df[(df['label']=='per:colleagues') & (df['object_type']=='ORG')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:colleagues') & (df['object_type']=='DAT')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:colleagues') & (df['object_type']=='LOC')].index] = 'PER'
    # 15. per:origin
    df['object_type'][df[(df['label']=='per:origin') & (df['object_type']=='DAT')].index] = 'POH'
    df['object_type'][df[(df['label']=='per:origin') & (df['object_type']=='PER')].index] = 'POH'
    df['object_type'][df[(df['label']=='per:origin') & (df['object_type']=='NOH')].index] = 'POH'
    # 16. per:siblings
    # 17. per:spouse
    df['object_type'][df[(df['label']=='per:spouse') & (df['object_type']=='LOC')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:spouse') & (df['object_type']=='ORG')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:spouse') & (df['object_type']=='DAT')].index] = 'PER'
    # 18. org:founded
    # 19. org:political/religious_affiliation
    df['object_type'][df[(df['label']=='org:political/religious_affiliation') & (df['object_type']=='LOC')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:political/religious_affiliation') & (df['object_type']=='PER')].index] = 'ORG'
    df['object_type'][df[(df['label']=='org:political/religious_affiliation') & (df['object_type']=='DAT')].index] = 'ORG'
    # 20. org:member_of
    df['object_type'][df[(df['label']==' org:member_of') & (df['object_type']=='NOH')].index] = 'ORG'
    df['object_type'][df[(df['label']==' org:member_of') & (df['object_type']=='DAT')].index] = 'ORG'
    df['object_type'][df[(df['label']==' org:member_of') & (df['object_type']=='PER')].index] = 'ORG'
    # 21. per:parents
    df['object_type'][df[(df['label']=='per:parents') & (df['object_type']=='LOC')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:parents') & (df['object_type']=='DAT')].index] = 'PER'
    df['object_type'][df[(df['label']=='per:parents') & (df['object_type']=='NOH')].index] = 'PER'
    # 22. org:dissolved
    # 23. per:schools_attended
    df['object_type'][df[df['label']=='per:schools_attended'].index] = 'ORG'
    # 24. per:date_of_death
    df['object_type'][df[df['label']=='per:date_of_death'].index] = 'DAT'
    # 25. per:date_of_birth
    df['object_type'][df[df['label']=='per:date_of_birth'].index] = 'DAT'
    # 26. per:place_of_birth
    df['object_type'][df[df['label']=='per:place_of_birth'].index] = 'LOC'
    # 27. per:place_of_death
    df['object_type'][df[df['label']=='per:place_of_death'].index] = 'LOC'
    # 28. org:founded_by
    # 29. per:religion
    df['object_type'][df[(df['label']=='per:religion') & (df['object_type']=='LOC')].index] = 'ORG'

    return df

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

def data_augmentation(df):
    no_relation_df, relation_df = df[df['label']=='no_relation'], df[df['label']!='no_relation']
    no_relation_df['id'], relation_df = range(len(df), len(df)+len(no_relation_df)), range(len(df)+len(no_relation_df),len(df)+len(no_relation_df)+len(relation_df))
    
    for idx, row in no_relation_df.iterrows():
        sentence = row['sentence']
        subject_word, object_word = row['subject_word'], row['object_word']
        subject_start_idx, object_start_idx = row['subject_start_idx'], row['object_start_idx']
        
        if subject_start_idx < object_start_idx:
            if len(subject_word) < len('MASK'):
                object_start_idx += len('MASK') - len(subject_word)
            else:
                object_start_idx -= len(subject_word) - len('MASK')
            no_relation_df['subject_end_idx'][idx] = subject_start_idx + len('MASK')
            no_relation_df['object_start_idx'][idx] = object_start_idx
            no_relation_df['object_end_idx'][idx] = object_start_idx + len('MASK')
            no_relation_df['subject_word'][idx] = 'MASK'
            no_relation_df['object_word'][idx] = 'MASK'
            no_relation_df['sentence'][idx] = sentence.replace(subject_word, 'MASK')
            no_relation_df['sentence'][idx] = no_relation_df['sentence'][idx].replace(object_word, 'MASK')
        else:
            if len(object_word) < len('MASK'):
                subject_start_idx += len('MASK') - len(object_word)
            else:
                subject_start_idx -= len(object_word) - len('MASK')
            no_relation_df['object_end_idx'][idx] = object_start_idx + len('MASK')
            no_relation_df['subject_start_idx'][idx] = subject_start_idx
            no_relation_df['subject_end_idx'][idx] = subject_start_idx + len('MASK')
            no_relation_df['subject_word'][idx] = 'MASK'
            no_relation_df['object_word'][idx] = 'MASK'
            no_relation_df['sentence'][idx] = sentence.replace(object_word, 'MASK')
            no_relation_df['sentence'][idx] = no_relation_df['sentence'][idx].replace(subject_word, 'MASK')
    no_relation_df['subject_type'] = 'MASK'
    no_relation_df['object_type'] = 'MASK'

    for idx, row in relation_df.iterrows():
        sentence = row['sentence']
        subject_word, object_word = row['subject_word'], row['object_word']
        subject_start_idx, object_start_idx = row['subject_start_idx'], row['object_start_idx']
        
        if subject_start_idx < object_start_idx:
            if len(subject_word) < len('MASK'):
                object_start_idx += len('MASK') - len(subject_word)
            else:
                object_start_idx -= len(subject_word) - len('MASK')
            relation_df['subject_end_idx'][idx] = subject_start_idx + len('MASK')
            relation_df['object_start_idx'][idx] = object_start_idx
            relation_df['object_end_idx'][idx] = object_start_idx + len('MASK')
            relation_df['subject_word'][idx] = 'MASK'
            relation_df['object_word'][idx] = 'MASK'
            relation_df['sentence'][idx] = sentence.replace(subject_word, 'MASK')
            relation_df['sentence'][idx] = relation_df['sentence'][idx].replace(object_word, 'MASK')
        else:
            if len(object_word) < len('MASK'):
                subject_start_idx += len('MASK') - len(object_word)
            else:
                subject_start_idx -= len(object_word) - len('MASK')
            relation_df['object_end_idx'][idx] = object_start_idx + len('MASK')
            relation_df['subject_start_idx'][idx] = subject_start_idx
            relation_df['subject_end_idx'][idx] = subject_start_idx + len('MASK')
            relation_df['subject_word'][idx] = 'MASK'
            relation_df['object_word'][idx] = 'MASK'
            relation_df['sentence'][idx] = sentence.replace(object_word, 'MASK')
            relation_df['sentence'][idx] = relation_df['sentence'][idx].replace(subject_word, 'MASK')
    relation_df['subject_type'] = 'MASK'
    relation_df['object_type'] = 'MASK'
    
    df = pd.concat([df, no_relation_df, relation_df])
    df = df.sample(frac=1, random_state=42)
    
    return df


# download dev set
train, dev, test = pd.read_csv("./train/train.csv"), pd.read_csv("./train/dev.csv"), pd.read_csv("./test/test.csv")
train, dev, test = preprocessing(train), preprocessing(dev), preprocessing(test)
train = data_cleaning(train)
train = data_augmentation(train)
train.to_csv("./train/train_final.csv", index=False)
dev.to_csv('./train/dev_final.csv', index=False)
test.to_csv("./test/test_final.csv", index=False)

# split train, dev set
# df, test = pd.read_csv("./train/train.csv"), pd.read_csv("./test/test.csv")
# df, test = preprocessing(df), preprocessing(test)
# df = data_cleaning(df)
# train, dev = train_dev_split(df)
# train.to_csv("./train/train_final.csv", index=False)
# dev.to_csv('./train/dev_final.csv', index=False)
# test.to_csv("./test/test_final.csv", index=False)

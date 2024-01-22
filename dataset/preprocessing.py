import pandas as pd
import ast

def data_preparation(df):
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

# def data_cleaning(df):
#     # 중복 데이터 제거
#     # sentence, subject_word, object_word, label이 중복인 데이터 제거
#     df.drop_duplicates(subset=['sentence', 'subject_word','object_word','label'], inplace=True)
#     # sentence, subject_word, object_word가 중복이지만 레이블이 다른 데이터 -> no_relation 데이터 제거
#     duplicates = df[df.duplicated(subset=['sentence', 'subject_word','object_word'], keep=False)]
#     df.drop(duplicates[duplicates['label'] == 'no_relation'].index, inplace=True)
#     # label별 entity 타입 수정
#     # 1. org:top_members/employees
#     df['subject_type'][df[(df['label']=='org:top_members/employees') & (df['subject_type']=='PER')].index] = 'ORG'
#     df['object_type'][df[(df['label']=='org:top_members/employees') & (df['object_type']=='ORG')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:top_members/employees') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:top_members/employees') & (df['object_type']=='NOH')].index] = 'POH'
#     # 2. org:members
#     df['subject_type'][df[(df['label']=='org:members') & (df['subject_type']=='PER')].index] = 'ORG'
#     df['object_type'][df[(df['label']=='org:members') & (df['object_type']=='PER')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:members') & (df['object_type']=='NOH')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:members') & (df['object_type']=='DAT')].index] = 'POH'
#     # 3. org:product
#     df['object_type'][df[df['label']=='org:product'].index] = 'POH'
#     # 4. per:title
#     df['object_type'][df[df['label']=='per:title'].index] = 'POH'
#     # 5. org:alternate_names
#     df['object_type'][df[(df['label']=='org:alternate_names') & (df['object_type']=='PER')].index] = 'ORG'
#     df['object_type'][df[(df['label']=='org:alternate_names') & (df['object_type']=='DAT')].index] = 'POH'
#     # 6. per:employee_of
#     df['object_type'][df[(df['label']=='per:employee_of') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:employee_of') & (df['object_type']=='NOH')].index] = 'POH'
#     # 7. org:place_of_headquarters
#     df['subject_type'][df[(df['label']=='org:place_of_headquarters') & (df['subject_type']=='PER')].index] = 'ORG'
#     df['object_type'][df[(df['label']=='org:place_of_headquarters') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:place_of_headquarters') & (df['object_type']=='NOH')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:place_of_headquarters') & (df['object_type']=='PER')].index] = 'POH'
#     # 8. per:product
#     df['object_type'][df[df['label']=='per:product'].index] = 'POH'
#     # 9. org:number_of_employees/members
#     # 10. per:children
#     df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='NOH')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:children') & (df['object_type']=='ORG')].index] = 'POH'
#     # 11. per:place_of_residence
#     df['object_type'][df[(df['label']=='per:place_of_residence') & (df['object_type']=='ORG')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:place_of_residence') & (df['object_type']=='DAT')].index] = 'POH'
#     # 12. per:alternate_names
#     df['object_type'][df[(df['label']=='per:alternate_names') & (df['object_type']=='ORG')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:alternate_names') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:alternate_names') & (df['object_type']=='ORG')].index] = 'POH'
#     # 13. per:other_family
#     df['object_type'][df[(df['label']=='per:other_family') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:other_family') & (df['object_type']=='ORG')].index] = 'POH'
#     # 14. per:colleagues
#     df['object_type'][df[(df['label']=='per:colleagues') & (df['object_type']=='ORG')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:colleagues') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:colleagues') & (df['object_type']=='LOC')].index] = 'POH'
#     # 15. per:origin
#     df['object_type'][df[(df['label']=='per:origin') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:origin') & (df['object_type']=='PER')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:origin') & (df['object_type']=='NOH')].index] = 'POH'
#     # 16. per:siblings
#     # 17. per:spouse
#     df['object_type'][df[(df['label']=='per:spouse') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:spouse') & (df['object_type']=='ORG')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:spouse') & (df['object_type']=='DAT')].index] = 'POH'
#     # 18. org:founded
#     # 19. org:political/religious_affiliation
#     df['object_type'][df[(df['label']=='org:political/religious_affiliation') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:political/religious_affiliation') & (df['object_type']=='PER')].index] = 'POH'
#     df['object_type'][df[(df['label']=='org:political/religious_affiliation') & (df['object_type']=='DAT')].index] = 'POH'
#     # 20. org:member_of
#     df['object_type'][df[(df['label']==' org:member_of') & (df['object_type']=='NOH')].index] = 'POH'
#     df['object_type'][df[(df['label']==' org:member_of') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']==' org:member_of') & (df['object_type']=='PER')].index] = 'POH'
#     # 21. per:parents
#     df['object_type'][df[(df['label']=='per:parents') & (df['object_type']=='LOC')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:parents') & (df['object_type']=='DAT')].index] = 'POH'
#     df['object_type'][df[(df['label']=='per:parents') & (df['object_type']=='NOH')].index] = 'POH'
#     # 22. org:dissolved
#     # 23. per:schools_attended
#     df['object_type'][df[(df['label']=='per:schools_attended') & (df['object_type']=='LOC')].index] = 'ORG'
#     # 24. per:date_of_death
#     df['object_type'][df[df['label']=='per:date_of_death'].index] = 'DAT'
#     # 25. per:date_of_birth
#     df['object_type'][df[df['label']=='per:date_of_birth'].index] = 'DAT'
#     # 26. per:place_of_birth
#     df['object_type'][df[df['label']=='per:place_of_birth'].index] = 'LOC'
#     # 27. per:place_of_death
#     df['object_type'][df[df['label']=='per:place_of_death'].index] = 'LOC'
#     # 28. org:founded_by
#     # 29. per:religion
#     df['object_type'][df[(df['label']=='per:religion') & (df['object_type']=='LOC')].index] = 'POH'

#     return df

train, validation, test = pd.read_csv("./train/train.csv"), pd.read_csv("./train/validation.csv"), pd.read_csv("./test/test.csv")
train, validation, test = data_preparation(train), data_preparation(validation), data_preparation(test)
# train = data_cleaning(train)
train.to_csv("./train/train_final.csv", index=False)
validation.to_csv('./train/validation_final.csv', index=False)
test.to_csv("./test/test_final.csv", index=False)
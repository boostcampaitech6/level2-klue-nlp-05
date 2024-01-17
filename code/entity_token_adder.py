import pandas as pd
import ast

def add_typed_entity_marker_original(file_path):
    """기존각 엔티티 토큰을 스페셜 토큰에 추가하는 방식

    Args:
        file_path (str): csv파일경로

    Returns:
        list: 스페셜 토큰에 추가할 엔티티 토큰의 list
    """
    entity_tokens = set()
    df = pd.read_csv(file_path)

    result = []

    for idx, row in df.iterrows():
        sentence = row['sentence']
        subject_entity = ast.literal_eval(row['subject_entity'])
        object_entity = ast.literal_eval(row['object_entity'])

        new_sentence = ''

        curr_entity_tokens = [f'<S:{subject_entity["type"]}>',
                              f'</S:{subject_entity["type"]}>',
                              f'<O:{object_entity["type"]}>',
                              f'</O:{object_entity["type"]}>']
        entity_tokens.update(curr_entity_tokens)

        if subject_entity['start_idx'] < object_entity['start_idx']:
            new_sentence += sentence[ :subject_entity['start_idx'] ]
            new_sentence += curr_entity_tokens[0]
            new_sentence += sentence[ subject_entity['start_idx']:subject_entity['end_idx']+1 ]
            new_sentence += curr_entity_tokens[1]
            new_sentence += sentence[ subject_entity['end_idx']+1:object_entity['start_idx'] ]
            new_sentence += curr_entity_tokens[2]
            new_sentence += sentence[ object_entity['start_idx']:object_entity['end_idx']+1 ]
            new_sentence += curr_entity_tokens[3]
            new_sentence += sentence[ object_entity['end_idx']+1: ]
        else:
            new_sentence += sentence[ :object_entity['start_idx'] ]
            new_sentence += curr_entity_tokens[2]
            new_sentence += sentence[ object_entity['start_idx']:object_entity['end_idx']+1 ]
            new_sentence += curr_entity_tokens[3]
            new_sentence += sentence[ object_entity['end_idx']+1:subject_entity['start_idx'] ]
            new_sentence += curr_entity_tokens[0]
            new_sentence += sentence[ subject_entity['start_idx']:subject_entity['end_idx']+1 ]
            new_sentence += curr_entity_tokens[1]
            new_sentence += sentence[ subject_entity['end_idx']+1: ]

        result.append(new_sentence)
    
    df['sentence'] = result
    df.to_csv('typed_entity_marker_original_train.csv')

    return list(entity_tokens) 


def add_typed_entity_marker_punct(dataset):
    """각 엔티티 토큰은 @, *, &, ^로 고정. 다만 논문과 다르게 우리가 사용할 한글 토크나이저에
       1. 엔티티 타입이 없을 수 있어 [UNK]으로 되는것을 방지하고자 엔티티 타입 자체를 스페셜 토큰으로 추가
       2. wordpiece 토크나이저를 사용할 수 있으므로 object entity의 토큰을 논문의 '#'대신 '&'로 대체사용 
       3. dataset은 prprocessing한 dataset 입니다.
   
    Args:
        file_path (str): csv파일경로

    Returns:
        list: 스페셜 토큰에 추가할 엔티티 타입의 list
    """
    
    entity_list = {'ORG':'조직', 'PER':'사람', 'POH':'명사', 'DAT':'날짜', 'LOC':'위치', 'NOH':'숫자'}
    
    entity_tokens = set()
    df = dataset.copy()

    result = []

    for idx, row in df.iterrows():
        sentence = row['sentence']
        
        subject_type, objecet_type = row['subject_type'], row['object_type']
        subject_start_idx, subject_end_idx = row['subject_start_idx'], row['subject_end_idx']
        object_start_idx, object_end_idx = row['object_start_idx'], row['object_end_idx']

        new_sentence = ''

        curr_entity_tokens = [entity_list[subject_type],
                              entity_list[objecet_type]]
        entity_tokens.update(curr_entity_tokens)

        if subject_start_idx < object_start_idx:
            new_sentence += sentence[ :subject_start_idx ]
            new_sentence += f"@ * {curr_entity_tokens[0]} * {sentence[subject_start_idx:subject_end_idx+1]} @"
            new_sentence += sentence[subject_end_idx+1:object_start_idx]
            new_sentence += f"& ^ {curr_entity_tokens[1]} ^ {sentence[object_start_idx:object_end_idx+1]} &"
            new_sentence += sentence[object_end_idx+1:]
        else:
            new_sentence += sentence[ :object_start_idx ]
            new_sentence += f"& ^ {curr_entity_tokens[1]} ^ {sentence[object_start_idx:object_end_idx+1]} &"
            new_sentence += sentence[object_end_idx+1:subject_start_idx]
            new_sentence += f"@ * {curr_entity_tokens[0]} * {sentence[subject_start_idx:subject_end_idx+1]} @"
            new_sentence += sentence[subject_end_idx+1:]
        result.append(new_sentence)
    
    df['sentence'] = result
    # df.to_csv('typed_entity_marker_punct_train.csv')
    
    concat_entity = []
    for e01, e01_type, e02, e02_type in zip(dataset['subject_word'],dataset['subject_type'], dataset['object_word'], dataset['object_type']):
      temp = ''
      temp = f'@ * {entity_list[e01_type]} * {e01} @ [SEP] & ^ {entity_list[e02_type]} ^ {e02} &'
      concat_entity.append(temp)

    return concat_entity, df

# add_typed_entity_marker_original("../dataset/train/train.csv")
# add_typed_entity_marker_punct("../dataset/train/train.csv") 
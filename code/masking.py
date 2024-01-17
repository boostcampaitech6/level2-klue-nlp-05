import ast
import random
import re
import pandas as pd

# 문장을 단어로 분리하면서 각 단어의 시작 및 끝 인덱스를 반환하는 함수
def split_sentence_into_words_with_indices(sentence):
    words_with_indices = []
    for match in re.finditer(r'\S+', sentence):
        start, end = match.span()
        word = match.group()
        words_with_indices.append((word, start, end - 1)) 
    return words_with_indices

def masking(df):
    # 각 문장에서 엔티티가 아닌 단어를 하나 랜덤하게 선택하여 [MASK]로 마스킹하고, 엔티티 인덱스 업데이트
    masked_sentences = []
    updated_subject_entities = []
    updated_object_entities = []
    for _, row in df.iterrows():
        sentence = row['sentence']
        words_with_indices = split_sentence_into_words_with_indices(sentence)
        subject_entity = ast.literal_eval(row['subject_entity'])
        object_entity = ast.literal_eval(row['object_entity'])

        # 엔티티가 포함된 단어 범위 추출
        entity_word_indices = []
        for entity in [subject_entity, object_entity]:
            start_pos, end_pos = entity['start_idx'], entity['end_idx']
            entity_word_indices.extend([idx for idx, (word, start, end) in enumerate(words_with_indices) if start <= start_pos and end >= end_pos])

        # 엔티티가 아닌 단어 찾기
        non_entity_word_positions = [idx for idx, _ in enumerate(words_with_indices) if idx not in entity_word_indices]

        # 랜덤하게 마스킹
        if non_entity_word_positions:
            mask_word_idx = random.choice(non_entity_word_positions)
            masked_word, masked_word_start, masked_word_end = words_with_indices[mask_word_idx]
            words_with_indices[mask_word_idx] = ("[MASK]", masked_word_start, masked_word_end)

            # 마스킹된 단어가 엔티티보다 앞에 위치하는 경우, 엔티티 인덱스 조정
            mask_length_diff = len("[MASK]") - len(masked_word)
            for entity in [subject_entity, object_entity]:
                if entity['start_idx'] > masked_word_end:
                    entity['start_idx'] += mask_length_diff
                    entity['end_idx'] += mask_length_diff

        # 마스킹된 문장
        masked_sentence = ' '.join([word for word, _, _ in words_with_indices])
        masked_sentences.append(masked_sentence)
        updated_subject_entities.append(str(subject_entity))
        updated_object_entities.append(str(object_entity))
        
    return masked_sentences, updated_subject_entities, updated_object_entities

def object_entity_masking(df):
    masked_sentences = []
    updated_subject_entities = []
    updated_object_entities = []
    for _, row in df.iterrows():
        obj_entity = ast.literal_eval(row['object_entity'])
        word = obj_entity['word']
        start_idx = obj_entity['start_idx']
        end_idx = obj_entity['end_idx']

        # 엔티티 단어 마스킹
        masked_sentence = row['sentence'][:start_idx] + '[MASK]' + row['sentence'][end_idx + 1:]

        masked_sentences.append(masked_sentence)

        # object entity 업데이트
        new_end_idx = start_idx + len('[MASK]') - 1
        updated_object_entities.append(str({'word': '[MASK]', 'start_idx': start_idx, 'end_idx': new_end_idx, 'type': obj_entity['type']}))
        
    return masked_sentences, updated_object_entities

def subject_entity_masking(df):
    masked_sentences = []
    updated_subject_entities = []
    updated_object_entities = []
    for _, row in df.iterrows():
        subj_entity = ast.literal_eval(row['subject_entity'])
        word = subj_entity['word']
        start_idx = subj_entity['start_idx']
        end_idx = subj_entity['end_idx']

        # 엔티티 단어 마스킹
        masked_sentence = row['sentence'][:start_idx] + '[MASK]' + row['sentence'][end_idx + 1:]

        masked_sentences.append(masked_sentence)

        # subject entity 업데이트
        new_end_idx = start_idx + len('[MASK]') - 1
        updated_subject_entities.append(str({'word': '[MASK]', 'start_idx': start_idx, 'end_idx': new_end_idx, 'type': subj_entity['type']}))
        
    return masked_sentences, updated_subject_entities


def Mask(df):
    df_temp = pd.DataFrame()
    
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        df_random = df_label.sample(frac=0.1)
        df_random['sentence'], df_random['subject_entity'], df_random['object_entity'] = masking(df_random)
        df_temp = pd.concat([df_temp, df_random])
        df_temp = df_temp.reset_index(drop=True)
        
    return df_temp


def subject_entity_Mask(df):
    df_temp = pd.DataFrame()
    
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        df_random = df_label.sample(frac=0.05)
        # object entity masking
        df_random['sentence'], df_random['subject_entity'] = subject_entity_masking(df_random)
        df_temp = pd.concat([df_temp, df_random])
        df_temp = df_temp.reset_index(drop=True)
        
    return df_temp

def object_entity_Mask(df):
    df_temp = pd.DataFrame()
    
    for label in df['label'].unique():
        df_label = df[df['label'] == label]
        df_random = df_label.sample(frac=0.05)
        # object entity masking
        df_random['sentence'], df_random['object_entity'] = object_entity_masking(df_random)
        df_temp = pd.concat([df_temp, df_random])
        df_temp = df_temp.reset_index(drop=True)
        
    return df_temp
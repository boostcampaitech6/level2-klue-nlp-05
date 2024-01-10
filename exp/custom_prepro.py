from tqdm import tqdm
import pandas as pd


class Processor:
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.new_tokens = ['[PER]', '[ORG]', '[DAT]', '[LOC]', '[POH]', '[NOH]']
        self.tokenizer.add_tokens(self.new_tokens)
        self.LABEL_TO_ID = {'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, \
                'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, \
                'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, \
                'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, \
                'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
        

    def tokenize(self, sentence, subj_type, obj_type, ss, se, os, oe):
        sents = []
        subj_type , obj_type = f"[{subj_type}]", f"[{obj_type}]"
        tokens = self.tokenizer.tokenize(sentence)
        for i_t, token in enumerate(tokens):
            if i_t == ss:
                new_ss = len(sents)
                sents.extend(['@'] + ['*'] + subj_type + ['*'] + [token])
            if i_t == se:
                sents.extend([token] + ['@'])
            if i_t == os:
                new_os = len(sents)
                sents.extend(["#"] + ['^'] + obj_type + ['^'] + [token])
            if i_t == oe:
                sents.extend([token] + ['#'])
        sents = sents[:self.args.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        
        return input_ids, new_ss + 1, new_os + 1

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = pd.read_csv(fh)

        for d in tqdm(data):
            ss, se = d['subject_start_idx'], d['subject_end_idx']
            os, oe = d['object_start_idx'], d['object_end_idx']

            sentence = d['sentence']

            input_ids, new_ss, new_os = self.tokenize(sentence, d['subject_type'], d['object_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['label']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }
            features.append(feature)

        return features
    
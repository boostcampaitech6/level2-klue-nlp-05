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
        
    def word_idx_extract(self, words, ns, ne):
        
        word_indices = []
        start_index = 0

        for word in words:
            end_index = start_index + len(word) - 1
            word_indices.append((start_index, end_index))
            start_index = end_index + 2

        word_idx=[]
        for i, (start, end) in enumerate(word_indices):
            if ns in range(start, end + 1) or ne in range(start, end + 1):
                word_idx.append(i)
                
        return word_idx[0] , word_idx[-1]


    def token_location(self, list1, list2):
        for i in range(len(list1) - len(list2) + 1):
            if list1[i:i + len(list2)] == list2:
                index = i
                return i, i+len(list2)-1

    def tokenize(self, sentence, subject_word, object_word, subj_type, obj_type, ss, se, os, oe):
        
        words = sentence.split()

        sws, swe = self.word_idx_extract(words, ss,se)
        ows, owe = self.word_idx_extract(words, os,oe)

        subj_tokens= self.tokenizer.tokenize(subject_word)
        obj_tokens= self.tokenizer.tokenize(object_word)

        sents =[]
        subj_type , obj_type = f"[{subj_type}]", f"[{obj_type}]"
        subj_token_collect = []
        obj_token_collect = []

        for idx, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            if idx not in range(sws,swe+1) and idx not in range(ows,owe+1):
                sents.extend(tokens)

            else:
                if sws <= idx and idx <= swe:
                    subj_token_collect.extend(tokens)
                    if idx == swe:
                        ts, te = self.token_location(subj_token_collect, subj_tokens)
                        tokens = subj_token_collect[:ts] + ['@'] + ['*'] + [subj_type] + ['*'] + subj_tokens + ['@'] + subj_token_collect[te+1:]
                        new_ss = len(sents) + len(subj_token_collect[:ts])
                        sents.extend(tokens)

                if ows <= idx and idx <= owe:
                    obj_token_collect.extend(tokens)
                    if idx == owe:
                        ts, te = self.token_location(obj_token_collect, obj_tokens)
                        tokens = obj_token_collect[:ts] + ["#"] + ['^'] + [obj_type] + ['^'] + obj_tokens + ["#"] + obj_token_collect[te+1:]
                        new_os = len(sents) + len(obj_token_collect[:ts])
                        sents.extend(tokens)

        sents = sents[:self.args.model.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        
        return input_ids, new_ss + 1, new_os + 1

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = pd.read_csv(fh)

        for _, d in tqdm(data.iterrows()):
            ss, se = int(d['subject_start_idx']), int(d['subject_end_idx'])
            os, oe = int(d['object_start_idx']), int(d['object_end_idx'])
            input_ids, new_ss, new_os = self.tokenize(d['sentence'],d['subject_word'],d['object_word'], d['subject_type'], d['object_type'], ss, se, os, oe)
            rel = self.LABEL_TO_ID[d['label']]

            feature = {
                'input_ids': input_ids,
                'labels': rel,
                'ss': new_ss,
                'os': new_os,
            }
            features.append(feature)

        return features
    

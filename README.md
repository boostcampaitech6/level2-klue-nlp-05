# 문장 내 개체간 관계 추출
관계 추출은 문장의 단어에 대한 속성과 관계를 예측하는 문제입니다. 이는 지식 그래프 구축을 위한 핵심 구성 요소로, 여러 자연어처리 응용 프로그램에서 중요합니다. 문장, 단어에 대한 정보를 통해 문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 

# 평가 지표
### Micro f1
micro-precision과 micro-recall의 조화 평균이며, 각 샘플에 동일한 importance를 부여해, 샘플이 많은 클래스에 더 많은 가중치를 부여합니다. 데이터 분포상 많은 부분을 차지하고 있는 no_relation class는 제외하고 F1 score가 계산 됩니다.

### AUPRC
x축은 Recall, y축은 Precision이며, 모든 class에 대한 평균적인 AUPRC로 계산해 score를 측정 합니다. imbalance한 데이터에 유용한 metric 입니다.

# Data
아래와 같은 구조를 가지고 있습니다.

- id : sentence에 대한 고유 id
- sentence : entity 쌍이 있는 sentence
- subject_entity : word, start_idx, end_idx, type으로 구성
- object_entity : word, start_idx, end_idx, type으로 구성
- label : 총 30개의 label로 구성
- source : wikipedia / wikitree / policy_briedfing으로 구성

# 팀 구성 및 역할
| 이름 | 역할 |
| :--- | :--- |
| 김기호 | • EDA <br/> • Data Preparation <br/> • Fine-tuning <br/> • Hyperparameter Tuning &emsp; |
| 박상기 | • EDA <br/> • Data Preparation &emsp; |
| 방신근 | • 개발환경구축 <br/> • Paper research <br/> • Custom model 구현 <br/> • Hyperparameter Tuning &emsp; |
| 심재혁 | • Entity token adder 구현 <br/> • Entity restriction <br/> • Binary classifier &emsp; |
| 황순열 | • Model Selection <br/> • Ensemble 기능 구현 <br/> • Hyperparameter Tuning <br/> • LLAMA-7b를 활용한 추론 &emsp; |
| 김건우 | • TAPT <br/> • dynamic padding <br/> • input prompt <br/> • gradient accumulation <br/> • data cleaning &emsp; |
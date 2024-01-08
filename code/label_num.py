import pickle

def label_to_num(label):
    """문자열로 되어있던 class를 숫자 label로 변환

    Args:
        label (str): 문자열 label

    Returns:
        int: 숫자 label
    """
    num_label = []
    with open('dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
  
    return num_label


def num_to_label(label):
    """숫자로 되어있던 class를 원본 문자열 label로 변환

    Args:
        label (int): 숫자 label

    Returns:
        str: 문자열 label
    """
    origin_label = []
    with open('dict_num_to_label.pkl', 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])
  
    return origin_label
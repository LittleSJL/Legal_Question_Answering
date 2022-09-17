import json
from sklearn.model_selection import train_test_split


def generate_data_id(data_selection, human_selection=True, meaning_selection=False, human_only=False):
    """
    generate question id set for different purposes

        data_selection: {train, test}
            train: generate question id used for training
            test：generate question id used for testing
        human_selection:
            if adding human questions into training set
        meaning_selection:
            if adding meaningless QG question into training set
        human_only:
            if only need human questions in the training set
    """
    with open('data/PGT/data_token.json') as f:
        data = json.load(f)

    human_id = []  # 326
    qg_meaningful_id = []  # 211
    qg_meaningless_id = []  # 411
    for section in data:
        for paragraph in section['paragraphs']:
            for qas in paragraph['qas']:
                for qa in qas:
                    if qa['type']['original_type'] == 'Human' and qa['type']['reformulation_type'] == 'Original':
                        human_id.append(qa['question_id'])
                    if qa['type']['original_type'] == 'QG' and qa['type']['meaning_type'] == 'Meaningful' and \
                            qa['type']['reformulation_type'] == 'Original':
                        qg_meaningful_id.append(qa['question_id'])
                    if qa['type']['original_type'] == 'QG' and qa['type']['meaning_type'] == 'Meaningless' and \
                            qa['type']['reformulation_type'] == 'Original':
                        qg_meaningless_id.append(qa['question_id'])

    # test set is always the same, only sample from human questions
    test, left_for_train = train_test_split(human_id, test_size=0.3, random_state=2022)  # 228, 98

    # generate different training sets
    train = None
    if meaning_selection and human_selection:  # QG + Human
        train = left_for_train + qg_meaningful_id + qg_meaningless_id  # 309 + 411 = 720
    if meaning_selection and not human_selection:  # QG
        train = qg_meaningful_id + qg_meaningless_id  # 211 + 411 = 622
    if not meaning_selection and human_selection:  # QG-meaningful + Human
        train = left_for_train + qg_meaningful_id  # 98 + 211 = 309
    if not meaning_selection and not human_selection:  # QG-meaningful
        train = qg_meaningful_id  # 211
    if human_only:  # only human
        train = left_for_train

    if data_selection == 'test':
        return test
    else:
        return train


def generate_test_retriever(refor_selection=False):
    """
    generate test set for retriever: only need question_id, question, paragraph_id

    refor_selection:
        False: only original questions in the test set
        True: adding reformulation questions into test set
    """
    test_id = generate_data_id('test')
    questions = []
    paragraph_ids = []
    questions_id = []
    with open('data/PGT/data_token.json') as f:
        data = json.load(f)
        for section in data:
            for paragraph in section['paragraphs']:
                paragraph_id = paragraph['paragraph_id']
                for qas in paragraph['qas']:
                    if refor_selection:  # choose to add reformulation questions
                        if qas[0]['question_id'] in test_id:
                            # if original question is sampled, then all its reformulation questions are sampled as well
                            for qa in qas:
                                questions.append(qa['question'])
                                paragraph_ids.append(paragraph_id)
                                questions_id.append(qa['question_id'])
                    else:  # choose not to add reformulation questions
                        for qa in qas:
                            if qa['question_id'] in test_id:  # only original questions are sampled
                                questions.append(qa['question'])
                                paragraph_ids.append(paragraph_id)
                                questions_id.append(qa['question_id'])
    return questions, paragraph_ids, questions_id
import json


def writeQidTextAnswer():
    with open('data/PGT/data_token.json') as f:
        data = json.load(f)

    qid2_text_answer = {}
    for section in data:
        for paragraph in section['paragraphs']:
            for qas in paragraph['qas']:
                for qa in qas:
                    qid2_text_answer[qa['question_id']] = {'question_text': qa['question'],
                                                           'answer_text': qa['answer_text']}

    path = 'model/reader/qid_2_text_answer.json'
    with open(path, 'w') as f:
        json.dump(qid2_text_answer, f)
    print('Saving qid_2_text_answer file to', path)


def writeId2Para():
    with open('data/PGT/data_token.json') as f:
        data = json.load(f)

    id2para = {}
    for section in data:
        for paragraph in section['paragraphs']:
            para_id = paragraph.get('paragraph_id')
            para_context = paragraph.get('context')
            id2para[para_id] = para_context

    path = 'model/reader/id2para.json'
    with open(path, 'w') as f:
        json.dump(id2para, f)
    print('Saving id2para file to', path)


writeQidTextAnswer()
writeId2Para()

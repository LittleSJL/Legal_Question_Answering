import json
import os
import sys

path_cwd = os.getcwd()
path_tools = os.path.join(path_cwd, 'pipeline')
sys.path.append(path_tools)

from retriever import retriever
from retriever.retrieve_data import generate_test_retriever


class Retriever:
    def __init__(self, model_path, para_dic=None):
        self.ranker = retriever.get_class('tfidf')(tfidf_path=model_path)
        self.para_dic = para_dic

    def retrieve(self, question, k=1):
        doc_ids, doc_scores = self.ranker.closest_docs(question, k)
        return doc_ids, doc_scores

    def predict_list(self, questions, k=1):
        predict = []
        for question in questions:
            doc_ids, _ = self.retrieve(question, k)
            if not doc_ids:
                print('retrieve nothing!')
            predict.append(doc_ids)
        return predict

    def fetch_text(self, para_id):
        return self.para_dic.get(para_id)


def get_para(data):
    para_dic = {}
    for sec in data:
        for para in sec['paragraphs']:
            para_dic[para['paragraph_id']] = para['context']
    return para_dic


def evaluate_recall(true, predict, questions_id, refor_selection):
    """
    true: [id, id, ...]
    predict: [[id, id, ...], [id, id, ...], ...] - retrieved id may be a list (top-5 or top-10)

    Recall for a sample is a binary value that 1 means getting the correct paragraph in the top-n retrieved list otherwise 0.
    """
    correct = 0
    total = 0
    for true_id, predict_id, qa_id in zip(true, predict, questions_id):
        if refor_selection == qa_id.split('-')[-1]:
            total += 1
            if true_id in predict_id:
                correct += 1
    recall = correct / total
    print('Recall:', recall)
    return recall


def evaluate_MRR(true, predict, questions_id, refor_selection):
    """
    true: [id, id, ...]
    predict: [[id, id, ...], [id, id, ...], ...] - retrieved id may be a list (top-5 or top-10)

    Mean Reciprocal Rank: calculates the reciprocal of the rank of the correct paragraph
    """
    MRR = 0
    total = 0
    for true_id, predict_id, qa_id in zip(true, predict, questions_id):
        if refor_selection == qa_id.split('-')[-1]:
            total += 1
            if true_id in predict_id:
                index = predict_id.index(true_id)
                MRR += 1 / (index + 1)
    MRR = MRR / total
    print('MRR:', MRR)
    return MRR


# generate test set
print('---Loading test set---')
questions, paragraph_ids, questions_id = generate_test_retriever(refor_selection=True)

# load retriever
print('---Loading retriever---')
model_path = 'model/retriever/pgt_handbook_paragraph-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
paragraph_retriever = Retriever(model_path)

# run prediction and evaluate (Recall, MRR)
print('---Running prediction---')
predict = paragraph_retriever.predict_list(questions, k=5)
print('----------------------Evaluation results----------------------')
evaluate_recall(paragraph_ids, predict, questions_id, refor_selection='Original')
evaluate_MRR(paragraph_ids, predict, questions_id, refor_selection='Original')
print('----------------------Evaluation results----------------------')

# write the retriever prediction into json file for later BERT reader
id_2_paragraph = {}
for qa_id, predict_paragraph_id in zip(questions_id, predict):
    id_2_paragraph[qa_id] = predict_paragraph_id
path = 'model/retriever/retriever_prediction.json'
with open(path, "w") as f:
    json.dump(id_2_paragraph, f)
print('Writing prediction of retriever to', path)

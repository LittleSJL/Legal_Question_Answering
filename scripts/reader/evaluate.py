from __future__ import print_function
from collections import Counter
import string
import re
import json
import sys
from scipy import stats
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='Prediction file to be evaluated')
# choose which version of question to be evaluated: Original, Paraphrase, Noisy
parser.add_argument('--refor', type=str, help='Reformulation version to be evaluated', default='Original')
args = parser.parse_args()



def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calculate f1 score for a single prediction-true pair
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Calculate exact match for a single prediction-true pair
    """
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, predictions, true_answer):
    """
    Since there are top-3 predictions, take the maximum scores over all of the predictions for a given question
    """
    scores_for_ground_truths = []
    for prediction in predictions:
        score = metric_fn(prediction, true_answer)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_single(true_answer, prediction):
    """
    Calculate the f1 score and exact math for a single predictions-true pair
    """
    prediction = [item['text'] for item in prediction]  # [text, text]
    exact_match = metric_max_over_ground_truths(exact_match_score, prediction, true_answer)
    f1 = metric_max_over_ground_truths(f1_score, prediction, true_answer)
    return exact_match, f1


def sort_prediction(prediction):
    """
    [{text, score} {text, score}, ...] - order the prediction by probability score, get the top-3 prediction
    """

    def getScore(ele):
        return ele['probability']

    prediction.sort(key=getScore, reverse=True)
    return prediction[:3]


def evaluate(predictions, refor_selection='Original'):
    """
    predictions: {id: [{text, score} {text, score}, ...], ...}
        qid to multiple {answer_text, probability}

    refor_selection:
        choose which version to be evaluated: Original, Paraphrase, Noisy
    """
    with open('model/reader/qid_2_text_answer.json') as f:
        qid_2_text_answer = json.load(f)

    f1 = exact_match = total = 0

    for items in predictions.items():
        qa_id = items[0]
        refor_type = qa_id.split('-')[-1]
        if refor_type == refor_selection:
            prediction = sort_prediction(items[1])
            true_answer = qid_2_text_answer.get(qa_id)['answer_text']
            total += 1
            exact_match_item, f1_item = evaluate_single(true_answer, prediction)
            exact_match += exact_match_item
            f1 += f1_item

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    # print(exact_match, f1)
    return {'exact_match': exact_match, 'f1': f1}


# choose the result file you want to evaluate
prediction = args.file
# prediction = 'output/reader/prediction_upper.json'

with open(prediction) as prediction_file:
    predictions_1 = json.load(prediction_file)

# choose the version of question you want to evaluate: {Original, Paraphrase, Noisy}
results = evaluate(predictions_1, refor_selection=args.refor)
print('-------------Evaluation results on ' + args.refor + ' questions-------------')
print('Exact match:', results.get('exact_match'))
print('F1 score:', results.get('f1'))

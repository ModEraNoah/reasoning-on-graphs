import argparse
import glob
import json
import os
import re
import string
from sklearn.metrics import precision_score
import ast

def remove_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

def match_equal(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 == s1

def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)

def eval_hit(prediction, answer):
    if len(prediction) < 1:
        return 0

    prediction = prediction[0]
    for a in answer:
        if match_equal(prediction, a):
            return 1
    return 0

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    for a in answer:
        for p in prediction:
            if match_equal(p, a):
                matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)

    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall

def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]

def preprocess_prediction(prediction):
    # Case 1: prediction is already a list (e.g., ["Eastern Time Zone"])
    if isinstance(prediction, list):
        pass

    # Case 2: prediction is a string representation of a list (e.g., "['54']")
    elif isinstance(prediction, str):
        try:
            # Safely evaluate the string to convert it into a list
            parsed_list = ast.literal_eval(prediction)

            # Ensure the parsed result is a list (e.g., ['54'] becomes ['54'])
            if isinstance(parsed_list, list):
                prediction = parsed_list
            else:
                # Handle cases where the parsed result is a single string (e.g., "54" becomes ["54"])
                prediction = [parsed_list]
        except (ValueError, SyntaxError):
            if prediction[0] == "[" and prediction[-1] == "]":
                prediction = prediction[1:-1].split(",")
            # Fallback: Split by newline or other delimiters if parsing fails
            else:
                prediction = prediction.split("\n")
        
        # convert prediction to string
        prediction = list(map(str, prediction))

        # filter out empty elements
        prediction = list(filter(bool, prediction))

    # Case 3: prediction is neither a list nor a string (unlikely, but handle it)
    else:
        prediction = [str(prediction)]  # Convert to a single-item list
        prediction = list(map(str, prediction))

    return prediction

def eval_result(predict_file, cal_f1=True, topk = -1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = f"detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data['id']
            prediction = data['prediction']
            answer = data['ground_truth']
            prediction = remove_think(prediction)

            if cal_f1:
                prediction = preprocess_prediction(prediction)
                prediction = extract_topk_prediction(prediction, topk)

                if len(prediction) < 1:
                    f1_score = precision_score = recall_score = acc = hit = 0
                    f1_list.append(f1_score)
                    precission_list.append(precision_score)
                    recall_list.append(recall_score)
                    acc_list.append(acc)
                    hit_list.append(hit)
                    f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}) + '\n')

                    continue

                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_score = min(f1_score, 1)
                precision_score = min(precision_score, 1)
                recall_score = min(recall_score, 1)

                prediction_str = ' '.join(prediction)
                acc = eval_acc(prediction_str, answer)
                acc = min(acc, 1)

                hit = eval_hit(prediction, answer)

                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                acc_list.append(acc)
                hit_list.append(hit)

                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'f1': f1_score, 'precission': precision_score, 'recall': recall_score}) + '\n')
            else:
                acc = eval_acc(prediction, answer)
                acc = min(acc, 1)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps({'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit}) + '\n')
    
    if len(f1_list) > 0:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(sum(recall_list) * 100 / len(recall_list))
    else:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(sum(hit_list) * 100 / len(hit_list))
    print(result_str)
    result_name = f"eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', type=str, default='results/KGQA/csqa/alpaca_default/test')
    argparser.add_argument('--cal_f1', action="store_true")
    argparser.add_argument('--top_k', type=int, default=-1)
    args = argparser.parse_args()
    
    eval_result(args.d, args.cal_f1, args.top_k)
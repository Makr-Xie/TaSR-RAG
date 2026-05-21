import argparse
from metrics import metric_max_over_ground_truths, exact_match_score, match, f1, extract_answer, qa_recall_score
import json
from tqdm import tqdm
# from rouge import Rouge

def load_file(file_path):
    """
    Load data from a JSON or JSONL file.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        list: List of dictionaries loaded from the file.

    Raises:
        ValueError: If the file format is not supported.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, dict):
                    return [data]
                elif isinstance(data, list):
                    return data
                else:
                    raise ValueError("Unsupported JSON structure. Expecting list or dict.")
            elif file_path.endswith('.jsonl'):
                data = [json.loads(line.strip()) for line in f if line.strip()]
                return data
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, help="File containing results with both generated responses and ground truth answers")
    parser.add_argument("--metric", type=str, default="em", choices=["em", "accuracy", "match", "rouge", "f1"], help="Metric to use for evaluation")
    return parser.parse_args()

def main():
    args = get_args()

    results = load_file(args.results_file)
    
    scores = []
    recall_scores = []
    cnt=0
    cnt_insufficient = 0
    updated_results = []
    # rouge_score = Rouge()
    for item in tqdm(results):
        # response = item['response']
        response = item['pred']

        question = item['question']

        if 'asqa' in args.results_file:
            answers=  []
            for ans in item['qa_pairs']:
                answers.extend(ans['short_answers'])
        else:
            answers = item['answer']
        if not answers:
            print(f"Warning: No answers provided for ID {item.get('id', 'unknown')}")
            item['correctness'] = False
            updated_results.append(item)
            continue
        
        
        if args.metric == "em":
            # Extract the content inside <answer>...</answer> before EM
            response_for_em = extract_answer(response)
            metric_result = metric_max_over_ground_truths(
                exact_match_score, response_for_em, answers
            )
            # print("response for em:", response_for_em, "answers:", answers, "EM:", metric_result)
            if "facts are insufficient" in response_for_em:
                cnt_insufficient += 1
            else:
                recall_scores.append(metric_max_over_ground_truths(qa_recall_score, response_for_em, answers))
        elif args.metric == "accuracy":
            response =  response.replace('\n','').strip()
            answer = answers[0][0]
            if response==answer:
                metric_result = 1.0
            else:
                metric_result = 0.0
        elif args.metric == "match":
            metric_result = match(question, response, answers)
        # elif args.metric == "rouge":
            # metric_result = Rouge(rouge_score, response, answers)
        elif args.metric == "f1":
            metric_result = f1(response, answers)
        else:
            raise NotImplementedError(f"Metric {args.metric} is not implemented.")
        
        # Add correctness field
        correctness = bool(metric_result == 1.0 if args.metric in ["em", "accuracy", "match"] else metric_result >= 0.5)
        # Reorder to put correctness as second field
        ordered_item = {"question": item["question"], "correctness": correctness}
        for k, v in item.items():
            if k not in ["question", "correctness"]:
                ordered_item[k] = v
        updated_results.append(ordered_item)
        scores.append(metric_result)
    
    # Write updated results back to file
    with open(args.results_file, 'w', encoding='utf-8') as f:
        for item in updated_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    if scores:
        overall = sum(scores) / len(scores)
        print(f'Overall result: {overall}')
        if args.metric == "em" and recall_scores:
            overall_recall = sum(recall_scores) / len(recall_scores)
            if overall + overall_recall == 0:
                overall_f1 = 0.0
            else:
                overall_f1 = 2 * overall * overall_recall / (overall + overall_recall)
            print(f'Overall recall: {overall_recall}')
            print(f'Overall F1: {overall_f1}')
            print(f'Recall samples (excluding insufficient): {len(recall_scores)}')
    else:
        print("No scores were calculated. Please check your input file.")

    print(f"Number of insufficient responses: {cnt_insufficient} out of {len(results)}")
    print(f"Updated results with correctness field saved to: {args.results_file}")

if __name__ == "__main__":
    main()

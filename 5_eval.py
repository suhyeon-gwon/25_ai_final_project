from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json, argparse

def main(args):
    with open(args.input_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)[:10]  # 앞 10개만

    references = []
    candidates_corr = []
    candidates_not_corr = []

    for data in dataset:
        reference = [data["gold_answer"].strip().split()]  # [[...]]
        candidate_corr = data["corr_inf_answer"].strip().split()
        candidate_not_corr = data["not_corr_inf_answer"].strip().split()
        
        references.append(reference)
        candidates_corr.append(candidate_corr)
        candidates_not_corr.append(candidate_not_corr)

    # BLEU 계산
    smoothie = SmoothingFunction().method4
    bleu_corr = corpus_bleu(references, candidates_corr, smoothing_function=smoothie)
    bleu_not_corr = corpus_bleu(references, candidates_not_corr, smoothing_function=smoothie)

    print(f"Corpus BLEU (corr_inf_answer):      {bleu_corr:.4f}")
    print(f"Corpus BLEU (not_corr_inf_answer): {bleu_not_corr:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='data/inferred/translated_result.json')
    args = parser.parse_args()

    main(args)

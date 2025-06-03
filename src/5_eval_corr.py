from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json, argparse
import numpy as np
from tqdm import tqdm

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def semantic_sim(gold, ko):
    embeddings = model.encode([gold, ko], convert_to_tensor=True)
    semantic_similarity = float(util.cos_sim(embeddings[0], embeddings[1]))

    return semantic_similarity

def main(args):
    results = []
    with open(args.input_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
        
    references = []
    candidates_corr = []
    
    for data in tqdm(dataset):
        gold = data["gold_corrected"]
        ko = data["ko_corrected"]
        
        # 1. sentence bert
        result = semantic_sim(gold, ko)
        results.append(result)
        
        # 2. bleu
        reference = [data["gold_corrected"].strip().split()]
        candidate_corr = data["ko_corrected"].strip().split()
        
        references.append(reference)
        candidates_corr.append(candidate_corr)

    # BLEU 계산
    smoothie = SmoothingFunction().method4
    bleu_corr = corpus_bleu(references, candidates_corr, smoothing_function=smoothie)
    print("corr")
    print(f"Average Semantic Similarity:    {np.mean(results)}")
    print(f"Corpus BLEU (corr_inf_answer):      {bleu_corr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/inferred/correct_result.json")
    args = parser.parse_args()

    main(args)

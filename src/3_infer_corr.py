from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, json, argparse, os, random
from tqdm import tqdm

def randomize_input(input_file, rate): # randomly choose error or cor text
    with open(args.input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    indices = list(range(len(data)))
    random.shuffle(indices)

    ko_error_indices = indices[:int(len(indices)*rate)]
    ko_cor_indices = indices[int(len(indices)*rate):]

    ko_error = [{
        'ko_error': data[i]["ko_error"],
        'ko_cor': data[i]['ko_cor'],
        'en_cor': data[i]['en_cor'],
        'num_errors': data[i]['num_errors'],
        'input': data[i]['ko_error'], 
        'tag': 'error', 
        } for i in ko_error_indices]
    ko_cor = [{
        'ko_error': data[i]["ko_error"],
        'ko_cor': data[i]['ko_cor'],
        'en_cor': data[i]['en_cor'],
        'num_errors': data[i]['num_errors'],
        'input': data[i]['ko_cor'], 
        'tag': 'cor', 
        } for i in ko_cor_indices]

    ko_error.extend(ko_cor)
    random.shuffle(ko_error)

    return ko_error 

## --- noise correction ---
def load_model(model_path):
    tokenizer_corr = AutoTokenizer.from_pretrained(model_path)
    model_corr = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model_corr.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model_corr.to(device)
    return tokenizer_corr, model_corr, device

def correct_text(input_text, tokenizer_corr, model_corr, device, max_length):
    inputs_ids = tokenizer_corr(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model_corr.generate(
        inputs_ids["input_ids"],
        attention_mask=inputs_ids["attention_mask"],
        max_length=max_length,
        temperature=0.0,
    )
    return tokenizer_corr.decode(outputs[0], skip_special_tokens=True)
    
def correct_infer(input_path, output_dir, tokenizer_corr, model_corr, device, max_length, rate):
    input_data = randomize_input(input_path, rate)
    corrected_results = []
    for d in tqdm(input_data):
        input_sentence = "<s>"+d["input"]+"</s>"
        corrected = correct_text(input_sentence, tokenizer_corr, model_corr, device, max_length)
        print(input_sentence, "-->", corrected)
        
        corrected_results.append({
            'ko_error': d["ko_error"],
            'ko_cor': d['ko_cor'],
            'en_cor': d['en_cor'],
            'num_errors': d['num_errors'],
            'input': d['input'], 
            'tag': d['tag'], 
            'gold_corrected': d['ko_cor'], # answer
            'ko_corrected': corrected,
        })

    output_file = os.path.join(output_dir, "correct_result.json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(corrected_results, file, ensure_ascii=False, indent=4)

    print(f"Saved {output_file} !!!")
    return corrected_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/best")
    parser.add_argument("--input_file", type=str, default="data/final/error_corr_kor_en_test.json")
    parser.add_argument("--output_dir", type=str, default="data/inferred")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--rate", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_path)
    correct_infer(args.input_file, args.output_dir, tokenizer, model, device, args.max_length, args.rate)

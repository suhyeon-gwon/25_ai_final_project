from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, json, argparse, os, random
from tqdm import tqdm
random.seed(43)

## --- translate ---
def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as file:
        input_data = json.load(file)[:10]
        
    translated_results = []

    # load translation model
    tokenizer_trans = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    model_trans = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model_trans = model_trans.to(device)

    for i in range(len(input_data)):
        # corrected
        inputs_corr = tokenizer_trans(text=input_data[i]['ko_corrected'], max_length=args.max_length, truncation=True, return_tensors='pt').to(device)
        outputs_corr = model_trans.generate(inputs_corr['input_ids'], temperature=0.0)
        decoded_outputs_corr = tokenizer_trans.decode(outputs_corr[0], skip_special_tokens=True)
        print("~~>", decoded_outputs_corr)
        
        # not corrected
        inputs_no = tokenizer_trans(text=input_data[i]['input'], max_length=args.max_length, truncation=True, return_tensors='pt').to(device)
        outputs_no = model_trans.generate(inputs_no['input_ids'], temperature=0.0)
        decoded_outputs_no = tokenizer_trans.decode(outputs_no[0], skip_special_tokens=True)
        print("~~>", decoded_outputs_no)
        
        
        block = {
            'corr_input': input_data[i]['ko_corrected'],
            'not_corr_input': input_data[i]['input'],
            'gold_answer': input_data[i]['en_cor'],
            'corr_inf_answer': decoded_outputs_corr,
            'not_corr_inf_answer': decoded_outputs_no,
            'tag': input_data[i]['tag'],
        }
        translated_results.append(block)

    output_file = os.path.join(args.output_dir, f"translated_result.json")
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(translated_results, file, ensure_ascii=False, indent=4)

    print(f"Saved {output_file} !!!")
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/best")
    parser.add_argument("--input_file", type=str, default="data/inferred/correct_result.json")
    parser.add_argument("--output_dir", type=str, default="data/inferred")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    main(args)

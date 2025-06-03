import json, os, random, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

random.seed(43)

# organize data and translate
def make_data(list_dir, save_dir, train_or_test, max_length):
    # load translation model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(train_or_test)
    data_final = []
    for file_path in os.listdir(list_dir):
        if file_path.endswith('json'):
            print(file_path.split('.json')[0])
            with open(os.path.join(list_dir, file_path), 'r', encoding='utf-8') as file:
                data_raw = json.load(file)
                
                # limit train to 5000, test to 500
                if train_or_test=="train":
                    threshold = 5000 if (len(data_raw['data']) > 5000) else len(data_raw['data'])
                else:
                    threshold = 500 if (len(data_raw['data']) > 500) else len(data_raw['data'])
                    
                for data in tqdm(data_raw['data'][:threshold]):
                    error = data['annotation']['err_sentence']
                    corr = data['annotation']['cor_sentence']
                    num_errors = len(data['annotation']['errors']) # how many errors

                    # translate
                    inputs = tokenizer(text=corr, max_length=max_length, truncation=True, return_tensors='pt').to(device)
                    outputs = model.generate(inputs['input_ids'], temperature=0.0) # deterministic
                    decoded_outputs = tokenizer.decode(outputs[0]).split("<pad>")[-1].split("</s>")[0].strip()
            
                    block = {
                            'ko_error': error,
                            'ko_cor':  corr,
                            'en_cor': decoded_outputs,
                            'num_errors': num_errors,
                        }
                    data_final.append(block)
        
    random.shuffle(data_final) # shuffle
    save_path = f'{save_dir}/error_corr_kor_en_{train_or_test}.json'
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(data_final, file, ensure_ascii=False, indent=4)
    print(f"Saved {save_path} !!! len: {len(data_final)}")
    
def main(args):
    make_data(args.list_dir_train, args.save_dir, 'train', args.max_length) # train
    make_data(args.list_dir_test, args.save_dir, 'test', args.max_length) # test
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_dir_train", type=str, default="data/raw/train")
    parser.add_argument("--list_dir_test", type=str, default="data/raw/test")
    parser.add_argument("--save_dir", type=str, default="data/final")
    parser.add_argument("--max_length", type=int, default=512)

    args = parser.parse_args()
    main(args)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import Dataset
import torch, json, argparse, os
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

def to_dataset(train_data_path):
    with open(train_data_path, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
    return Dataset.from_list(data_list)

def tokenize_fn(batch, tokenizer, max_length):
    ko_error_texts = ["<s>"+str(x)+"</s>" for x in batch["ko_error"]]
    ko_cor_texts = ["<s>"+str(x)+"</s>" for x in batch["ko_cor"]]

    model_inputs = tokenizer(
        ko_error_texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False
    )

    labels = tokenizer(
        ko_cor_texts,
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )["input_ids"]

    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label_seq]
        for label_seq in labels
    ]

    model_inputs["labels"] = labels
    return model_inputs


def train(args):
    os.makedirs(f"{args.new_model_path}/best", exist_ok=True)
    os.makedirs(f"{args.new_model_path}/final", exist_ok=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")

    dataset = to_dataset(args.train_data_path)
    dataset = dataset.train_test_split(test_size=0.1, seed=43)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names  # 이걸 넣어야 충돌 안 남
    )

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    val_dataloader = DataLoader(
        tokenized_dataset["test"],
        batch_size=args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6,
    )
    best_val_loss = float('inf')
    best_model_path = f'{args.new_model_path}/best'

    # train
    train_list = []
    val_list = []
    model.train()
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            scheduler.step() # scheduler
        print(f"Train Loss: {total_loss / len(train_dataloader):.4f}")
        train_list.append(total_loss / len(train_dataloader))
                          
        # eval
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        current_val_loss = val_loss / len(val_dataloader)
        val_list.append(current_val_loss)

        print(f"Val Loss: {current_val_loss:.4f}")

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
        model.train()
        
    # save model
    model.save_pretrained(f'{args.new_model_path}/final')
    tokenizer.save_pretrained(f'{args.new_model_path}/final')
    print(f"Model saved to {args.new_model_path}")
    
    pd.DataFrame({
        'epoch': list(range(1, args.epochs + 1)),
        'train_loss': train_list,
        'val_loss': val_list
    }).to_csv(f"{args.new_model_path}/loss_log.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="data/final/error_corr_kor_en_train.json")
    parser.add_argument("--new_model_path", type=str, default="model")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    args = parser.parse_args()
    train(args)

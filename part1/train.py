from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, Trainer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset as TorchDataset
import argparse

from trainer import Trainer,TrainerConfig

from tqdm import tqdm

import os 

import utils

block_size = 128

def get_finetune_dataset(tokenizer,finetune_corpus_path):

    train_path = finetune_corpus_path + '/birth_places_train.tsv'
    test_path = finetune_corpus_path + '/birth_places_dev.tsv'

    train_dataset, valid_dataset = load_dataset('text', data_files={'train': train_path, 'dev': test_path}, split=['train', 'dev'])


    def get_labels(examples):
        return tokenizer(examples['text'], add_special_tokens=True, truncation=True, max_length=1024)

    tokenized_train = train_dataset.map(get_labels, batched=True, num_proc=1, remove_columns=["text"])
    tokenized_valid = valid_dataset.map(get_labels, batched=True, num_proc=1, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        result["labels"] = result["labels"][1:]
        result["input_ids"] = result["input_ids"][:-1]
        result["attention_mask"] = result["attention_mask"][:-1]
        return result


    lm_train = tokenized_train.map(group_texts, batched=True, num_proc=4)
    lm_valid = tokenized_valid.map(group_texts, batched=True, num_proc=4)

    lm_train.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    lm_valid.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])

    return lm_train, lm_valid

def get_pretrain_dataset(tokenizer,split_ratio=0.1):
    dataset = load_dataset('text', data_files={'train': './dataset/pretrain/wiki.txt'}, split='train')
    # 4. Load and preprocess the dataset using datasets library
    

    # Split the dataset into training and validation sets
    split_dataset = dataset.train_test_split(test_size=split_ratio)
    train_dataset = split_dataset['train']
    valid_dataset = split_dataset['test']

    # 5. Tokenize the dataset

    def get_labels(examples):
        return tokenizer(examples['text'], add_special_tokens=True, truncation=True, max_length=1024)

    tokenized_train = train_dataset.map(get_labels, batched=True, num_proc=1, remove_columns=["text"])
    tokenized_valid = valid_dataset.map(get_labels, batched=True, num_proc=1, remove_columns=["text"])


    # 6. Group texts into blocks of block_size
    block_size = 128

    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated['input_ids'])
        # Ensure the total length is a multiple of block_size
        total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        result["labels"] = result["labels"][1:]
        result["input_ids"] = result["input_ids"][:-1]
        result["attention_mask"] = result["attention_mask"][:-1]
        return result

    lm_train = tokenized_train.map(group_texts, batched=True, num_proc=4)
    lm_valid = tokenized_valid.map(group_texts, batched=True, num_proc=4)

    # turn datasets into list of dicts
    # lm_train = lm_train.to_list()
    # lm_valid = lm_valid.to_list()

    # 7. Set the format to PyTorch tensors
    lm_train.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])
    lm_valid.set_format(type='torch', columns=['input_ids', 'attention_mask','labels'])

    return lm_train, lm_valid

def main():

    if os.getenv("DEBUG_TRAIN",'0') == '1':
        import debugpy
        debugpy.listen(('localhost', 5678))
        debugpy.wait_for_client()
        debugpy.breakpoint()

    # 1. Define GPT-2 Configuration
    config = GPT2Config(
        vocab_size=50257,          # Update based on your tokenizer
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        num_hidden_layers=12,
        num_attention_heads=12,
    )

    # 2. Initialize the GPT-2 model from scratch
    model = GPT2LMHeadModel(config)

    # 3. Load your custom tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("my_own_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure tokenizer is set up correctly
    model.resize_token_embeddings(len(tokenizer))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    if args.function == 'pretrain':
        assert args.outputs_path is not None
        # max_epochs=650
        # batch_size=128
        # learning_rate=args.pretrain_lr
        # lr_decay=True
        # warmup_tokens=512*20
        # final_tokens=650*len(pretrain_dataset)*block_size
        # num_workers=4
        # writer=writer
        lm_train, lm_valid = get_pretrain_dataset(tokenizer,split_ratio=0.1)
        config = TrainerConfig(
            max_epochs=650,
            batch_size=64,
            learning_rate=1e-4,
            lr_decay=True,
            warmup_tokens=512*20,
            final_tokens=650*len(lm_train)*block_size,
            num_workers=2,
            ckpt_path=f"{args.outputs_path}/pretrain.pt"
        )

        trainer = Trainer(
            config=config,
            model=model,
            train_dataset=lm_train,
            test_dataset=lm_valid,
            criterion=criterion,
        )

        # 10. Start Training
        trainer.train()

    elif args.function == 'finetune':
        assert args.outputs_path is not None

        lm_train, lm_valid = get_finetune_dataset(tokenizer,'./dataset/sft')
        # TODO [part c] [part f]:
        # - Given:
        #     1. A finetuning corpus specified in args.finetune_corpus_path
        #     2. A path args.reading_params_path containing pretrained model
        #         parameters, or None if finetuning without a pretrained model
        #     3. An output path args.writing_params_path for the model parameters
        # - Goals:
    #     1. If args.reading_params_path is specified, load these parameters
    #         into the model
    #     2. Finetune the model on this corpus
    #     3. Save the resulting model in args.writing_params_path
    # - Make sure to use the following hyperparameters:
    #     [part d] Hyperparameters for finetuning WITHOUT a pretrained model:
    #         max_epochs=75
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
    #     [part f] Hyperparameters for finetuning WITH a pretrained model:
    #         max_epochs=10
    #         batch_size=256
    #         learning_rate=args.finetune_lr
    #         lr_decay=True
    #         warmup_tokens=512*20
    #         final_tokens=200*len(pretrain_dataset)*block_size
    #         num_workers=4
    #         writer=writer
        #     You can use the args.reading_params_path flag to switch between the
        #     number of epochs for each case.
        if args.reading_params_path is not None:
            model.load_state_dict(torch.load(args.reading_params_path))
            
            config = TrainerConfig(
                max_epochs=10,
                batch_size=64,
                learning_rate=5e-5,
                lr_decay=True,
                warmup_tokens=512*20,
                final_tokens=200*len(lm_train)*block_size,
                num_workers=4,
                ckpt_path=f"{args.outputs_path}/finetune_with_pretrained.pt"
            )
        else:
            config = TrainerConfig(
                max_epochs=75,
                batch_size=64,
                learning_rate=1e-4,
                lr_decay=True,
                warmup_tokens=512*20,
                final_tokens=200*len(lm_train)*block_size,
                num_workers=4,
                ckpt_path=f"{args.outputs_path}/finetune_without_pretrained.pt"
            )
            
        trainer = Trainer(
            config=config,
            model=model,
            train_dataset=lm_train,
            test_dataset=lm_valid,
            criterion=criterion,
        )

        # 10. Start Training
        trainer.train()

    elif args.function == 'evaluate':
        assert args.outputs_path is not None
        assert args.reading_params_path is not None
        assert args.eval_corpus_path is not None
        model.load_state_dict(torch.load(args.reading_params_path))
        correct = 0
        total = 0
        with open(args.outputs_path, 'w', encoding='utf-8') as fout:
            predictions = []
            for line in tqdm(open(args.eval_corpus_path, encoding='utf-8')):
                x = line.split('\t')[0]
                x = x + '⁇'
                # x = torch.tensor([pretrain_dataset.stoi[s] for s in x],
                #                 dtype=torch.long)[None,...].to(device)
                x = tokenizer(x, add_special_tokens=True, truncation=True, max_length=1024, return_tensors='pt')
                pred = utils.sample(model, x, 32, sample=False)[0]
                # completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
                completion = tokenizer.decode(pred, skip_special_tokens=True)
                pred = completion.split('⁇')[1]
                predictions.append(pred)
                fout.write(pred + '\n')
            total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
        if total > 0:
            print(f'Correct: {correct} out of {total}: {correct/total*100}%')
        else:
            print(f'Predictions written to {args.outputs_path}; no targets provided')
        return None
    # 9. Initialize the Trainer




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='train', choices=['pretrain', 'finetune', 'evaluate'])
    parser.add_argument('--outputs_path', type=str, default=None)
    parser.add_argument('--reading_params_path', type=str, default=None)
    parser.add_argument('--eval_corpus_path', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main()
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, Trainer
from datasets import load_dataset
import torch
from torch.utils.data import Dataset as TorchDataset
import argparse
from tqdm import tqdm
import os

import utils
from models import GPTConfig, GPT
from trainer import Trainer,TrainerConfig
import dataset

def main(args):

    # Keep the block size 128
    # Why is the pretraining corpus always required (even if we're not pretraining?)
    # It's because we're using it as a hack to always have the same vocabulary
    # (that is, the same mapping from character to integer, and we build the 
    # vocab from the pretraining corpus.)
    block_size = 128
    text = open(args.pretrain_corpus_path).read()
    pretrain_dataset = dataset.CharCorruptionDataset(text, block_size)

    # We don't suggest you change these hyperparameters, as they're known to work.
    # use them for both the vanilla and the synthesizer models
    # TODO[Optional]: change gpt config for different model setups
    mconf = GPTConfig(pretrain_dataset.vocab_size, pretrain_dataset.block_size,
    n_layer=4, n_head=8, n_embd=256)
    # Ensure tokenizer is set up correctly
    model = GPT(mconf).to(args.device)

    if args.function == 'pretrain':
        assert args.pretrain_corpus_path is not None
        assert args.outputs_path is not None
        ### TODO: Implement pretraining, you can refer this setup in train.py
        ### TODO[Optional]: change hyperparameters for ablation study
        # max_epochs=650
        # batch_size=128
        # learning_rate=args.pretrain_lr
        # lr_decay=True
        # warmup_tokens=512*20
        # final_tokens=650*len(pretrain_dataset)*block_size
        # num_workers=4
        # writer=writer

    elif args.function == 'finetune':
        assert args.outputs_path is not None
        assert args.finetune_corpus_path is not None

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
        

    elif args.function == 'evaluate':
        assert args.outputs_path is not None
        assert args.reading_params_path is not None
        assert args.eval_corpus_path is not None
        model.load_state_dict(torch.load(args.reading_params_path))
        correct = 0
        total = 0
        with open(args.outputs_path, 'w') as fout:
            predictions = []
            for line in tqdm(open(args.eval_corpus_path)):
                x = line.split('\t')[0]
                x = x + '⁇'
                x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[None,...].to(args.device)
                pred = utils.sample(model, x, 32, sample=False)[0]
                completion = ''.join([pretrain_dataset.itos[int(i)] for i in pred])
                pred = completion.split('⁇')[1]
                predictions.append(pred)
                fout.write(pred + '\n')
            total, correct = utils.evaluate_places(args.eval_corpus_path, predictions)
        if total > 0:
            print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))
        else:
            print('Predictions written to {}; no targets provided'.format(args.outputs_path))
    # 9. Initialize the Trainer




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='train', choices=['pretrain', 'finetune', 'evaluate'])
    parser.add_argument('--pretrain_corpus_path', type=str, default=None)
    parser.add_argument('--finetune_corpus_path', type=str, default=None)
    parser.add_argument('--outputs_path', type=str, default=None)
    parser.add_argument('--reading_params_path', type=str, default=None)
    parser.add_argument('--eval_corpus_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    main(args)

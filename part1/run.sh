    # parser = argparse.ArgumentParser()
    # parser.add_argument('--function', type=str, default='train', choices=['pretrain', 'finetune', 'evaluate'])
    # parser.add_argument('--outputs_path', type=str, default=None)
    # parser.add_argument('--reading_params_path', type=str, default=None)
    # parser.add_argument('--eval_corpus_path', type=str, default=None)

python train.py \
    --function finetune \
    --outputs_path ./output \
    --reading_params_path ./output/finetune_without_pretrained.pt

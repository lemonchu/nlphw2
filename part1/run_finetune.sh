

# finetune with/without pretrained

python run.py \
    --function finetune \
    --outputs_path /root/autodl-tmp/output \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --finetune_corpus_path ./dataset/finetune/birth_places_train.tsv \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path /root/autodl-tmp/output/pretrain.pt
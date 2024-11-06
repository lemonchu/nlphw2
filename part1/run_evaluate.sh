python run.py \
    --function evaluate \
    --outputs_path /root/autodl-tmp/output/predictions.txt \
    --pretrain_corpus_path ./dataset/pretrain/wiki.txt \
    --eval_corpus_path ./dataset/finetune/birth_places_dev.tsv \
    --reading_params_path /root/autodl-tmp/output/finetune_with_pretrained.pt
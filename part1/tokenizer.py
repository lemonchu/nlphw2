# from tokenizers.models import BPE
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# from tokenizers.trainers import BpeTrainer
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# from tokenizers.pre_tokenizers import Whitespace
# tokenizer.pre_tokenizer = Whitespace()

# files = [f"./dataset/pretrain/wiki.txt"]
# tokenizer.train(files, trainer)
# tokenizer.save("tokenizer/tokenizer-wiki.json")

# tokenizer = Tokenizer.from_file("tokenizer/tokenizer-wiki.json")

from datasets import load_dataset
from part1.loading import load_training_corpus

from transformers import AutoTokenizer
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

pretrain_path = "./dataset/pretrain/wiki.txt"
training_corpus = load_training_corpus(pretrain_path, old_tokenizer)

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokenizer.save_pretrained("my_own_tokenizer")
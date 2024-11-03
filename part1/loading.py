from datasets import load_dataset
from transformers import AutoTokenizer

def load_sft_data(dataset_path, split='train'):
    """
    Load a dataset using Hugging Face's datasets library
    
    Args:
    dataset_path (str): Path to the local dataset
    split (str): Dataset split, default is 'train'
    
    Returns:
    dataset: Loaded dataset object (with Hugging Face's datasets format)
    """
    # load dataset by index
    dataset = load_dataset('csv', data_files={split: f'{dataset_path}/birth_places_{split}.tsv'}, delimiter='\t',column_names=['question','answer'],header=None)
    
    print(f"Successfully loaded {split} split from '{dataset_path}'")
    return dataset[split]
    

def load_training_corpus(dataset_path ,tokenizer):
    """
    Load a training corpus from a dataset
    """
    output = []
    with open(dataset_path, 'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            output.extend(tokenizer.tokenize(line))
    return output
    

if __name__ == "__main__":
    # Example usage, test the function
    dataset_path = "./dataset/sft"
    # train_data = load_sft_data(dataset_path, split='train')
    # dev_data = load_sft_data(dataset_path, split='dev')
    
    # # if train_data and dev_data:
    # try:
    #     print(f"Training set size: {len(train_data)}")
    #     assert len(train_data) == 2000, "Training set size should be 2000"
    #     print(f"Dev set size: {len(dev_data)}")
    #     assert len(dev_data) == 500, "Dev set size should be 500"
    #     print(f"Dataset columns: {train_data.column_names}")
    #     print(f"First data sample: {train_data[0]}")
    # except Exception as e:
    #     print(f"Error loading dataset: {e}")

    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    try:
        pretrain_path = "./dataset/pretrain/wiki.txt"
        training_corpus = load_training_corpus(pretrain_path, old_tokenizer)
        print(len(training_corpus))
        print(training_corpus[:100])
    except Exception as e:
        print(f"Error loading training corpus: {e}")

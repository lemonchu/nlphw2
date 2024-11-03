# Homework Assignment: Training Your Own Model Using Hugging Face

## Objective:
The goal of this assignment is to learn how to train your own model using the Hugging Face Transformers library. You will explore key steps in the pipeline, including loading datasets, training your own tokenizer, and training a model from scratch.

## Instructions:

### Part 1: Load the SFT Dataset
- **File:** `load_dataset.py`
- Your first task is to complete the `load_dataset.py` script. 
  
  **Steps:**
  1. Use the `datasets` library from Hugging Face to load  SFT dataset. (`./dataset/sft`)
  <!-- 2. Apply necessary preprocessing steps to clean the data. -->
  <!-- 3. Ensure the data is formatted correctly for training (i.e., tokenization, formatting, etc.). -->

### Part 2: Train Your Own Tokenizer
- **File:** `tokenizer.py`
- In this part, you will train your own tokenizer using SentencePiece on a pretraining dataset. Follow the instructions in the file and use the `dataset/pretrain/wiki.txt` corpus for tokenizer training.

  **Steps:**
  1. You may want to follow this [tutorial](https://huggingface.co/docs/tokenizers/en/quicktour) to train your own tokenizer.
  2. 

### Part 3: Fine-Tune a Pretrained Model
- **File:** `train_model.py`
- In this part, you will use your trained tokenizer to fine-tune a pretrained model on the dataset you loaded in Part 1. You will use the Hugging Face Transformers library for this task.

  **Steps:**
  1. Load a pretrained model using the `transformers` library. For example, you may use a model like BERT, GPT, or any other model that fits your task.
  2. Use your custom-trained tokenizer to tokenize the dataset.
  3. Configure a training pipeline, specifying parameters such as batch size, learning rate, and number of epochs.
  4. Fine-tune the model using Hugging Face's `Trainer` API or a custom training loop.
  5. Save the fine-tuned model for further evaluation.

  **Deliverable:** A fine-tuned model that can be used for evaluation or further training.

### Part 4: Evaluate and Test Your Model
- **File:** `evaluate_model.py`
- In this part, you will evaluate the performance of your fine-tuned model on a test dataset. You will generate metrics such as accuracy, F1 score, or any relevant metric for your task.

  **Steps:**
  1. Load the fine-tuned model from Part 3.
  2. Prepare a test dataset and tokenize it using your custom tokenizer.
  3. Evaluate the model on the test data using metrics such as accuracy, precision, recall, or F1 score.
  4. Print and interpret the results to assess the model's performance.
  5. Optionally, test the model with a few sample inputs to observe its behavior.

  **Deliverable:** A detailed evaluation report of your model's performance, including metrics and test case results.

## Submission:
- Submit your completed `load_dataset.py`, `tokenizer.py`, `train_model.py`, and `evaluate_model.py` scripts, along with any additional files required to run the code (e.g., `wiki.txt` dataset, fine-tuned model).
- Make sure all code is thoroughly commented and well-structured for readability.

### Notes:
- Ensure that your code is properly tested.
- Use Hugging Face documentation as needed for assistance.
- Reach out with any questions or concerns during the process.

Good luck and have fun!

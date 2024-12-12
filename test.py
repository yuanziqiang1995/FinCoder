from datasets import load_dataset
import os
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
dataset = load_dataset("yale-nlp/FinanceMath")

# print the first example on the validation set
print(dataset["validation"][0])

# print the first example on the test set
print(dataset["test"][0])
# FinCoder


## Installation
For details on the project's environment dependencies, please see the `requirements.txt` file.
## Data Preprocessing
Preprocessing data from a benchmark, exemplified by the FinQA dataset:

```shell script
cd data/finqa
python preprocess.py --input input_file_path --output output_file_path
```
This step is designed to organize the data format, ensuring it is well-prepared for subsequent use.

## Generating Code Annotation
### Step 1: Variable Definition Generation
```shell script
cd ../..
python run_finqa_stepwise1.py \
    --key api_key \
    --api api_url \
    --model model_name \
    --start 0 \
    --input input_file_path \
    --greedy True \
    --end -1 
```
All parameters, except those related to the API, are set with default values in file A. 
Among these, the 'greedy' parameter determines whether to employ self-consistency, 
while 'start' and 'end' specify the positions where data processing begins and ends, respectively.

The generated data is stored in the `output` folder in JSONL format.
 Given the potential for interruptions and failures during prolonged, heavy use of third-party APIs, 
 the 'start' and 'end' parameters can be utilized to control the starting point of data processing for multiple attempts. 
 There may be more than one JSONL file produced as a result. To facilitate further processing,
 it is necessary to organize these JSONL files.


First, copy all files from the `outputs` folder that were generated in this round of experiments and contain 'correct'
 in their filenames 
to the `data/finqa/jsonl` folder.

```shell script
cd data/finqa
python convert_finqa.py
```
The name of the output file can be modified in the last line of the main function in `convert_finqa.py`.
### Step 2: Code Calculation Process Generation
Using the results from step1 as input, running run_finqa_stepwise2.py will yield the final annotations. 
The default parameters for step2 are the same as those in the first step, except for the input filename.

```shell script
cd ../..
python run_finqa_stepwise2.py \
    --key api_key \
    --api api_url \
    --model model_name \
    --start 0 \
    --input input_file_path \
    --greedy True \
    --end -1 
```
 The method for organizing the output JSONL files is also the same as in step1.
## Fine-tuning
## Test
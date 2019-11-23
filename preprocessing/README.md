# Training Data Preprocessing
Scripts for preprocessing the ATM training data.
## Usage
The first step converts the 'raw' mlf files into prompt-response pairs and saves them (and related meta-data) into a directory:
### 1. Process the transcriptions and scripts and save the processed files as readable .txt files:
```
python magic_preprocess_raw.py /path/to/scripts.mlf /path/to/responses/transcription.mlf /path/to/destination/directory --fixed_sections A B --exclude_sections A B --multi_sections E --multi_sections_master SE0006 --speaker_grades_path /path/to/speaker/grades.lst --exclude_grades 2
```
Generates files:
`responses.txt` `prompts.txt` `speakers.txt` `conf.txt` `sections.txt` `prompt_ids.txt`
with the same number of lines where each line in each file corresponds to one another.


The second step depends on whether the dataset is going to be used for evaluation or for training.

If the dataset is going to be used for training:
### 2. (train) Generate the required tfrecords files from the processed .txt files
```
python magic_preprocess_to_tfrecords.py /directory/with/preprocessed/txt/files /path/to/word-list/file/input.wlist.index /destination/directory --valid_fraction 0.1 --preprocessing_type train
```

If the dataset is going to be used for evaluation:
### 2.a (eval) Create an evaluation set by shuffling prompts and responses to generate file with a mix of positive (on-topic) and negative (off-topic) examples.
```
python magic_preprocess_shuffle.py /directory/with/unshuffled/.txt/data /destination/directory --samples 10
```
### 2.b (eval) Generate the required tfrecords files from the processed and shuffled .txt files
```
python magic_preprocess_to_tfrecords.py /directory/with/shuffled/txt/data /path/to/word-list/file/input.wlist.index /destination/directory --preprocessing_type test --sorted_topics_path /path/to/sorted-topics.txt(optional)
```

### 

## Relevant Paths:
#### Scripts (prompts):
###### BULATS
`/home/alta/BLTSpeaking/convert-v2/4/lib/script.v7/scripts.mlf`
###### LINSK
`/home/alta/Linguaskills/convert/lib/script/scripts.mlf`

#### Responses:
###### BLXXXgrp24
`/home/alta/BLTSpeaking/exp-ar527/grammar/detection/data/word/mlf/BLXXXgrd00_grd01_grd02_uns00_uns10_uns11.mlf`

###### BLXXXeval{1,3}
`/home/alta/BLTSpeaking/exp-ar527/grammar/detection/data/word/mlf/BLXXX{dataset}.mlf`

###### LINSKuns03
`/home/alta/Linguaskills/exp-graphemic-kmk/D3/align-LINSKuns03.mlp.mpe/lib/wlabs/train.mlf`

###### LINSKevl03
`/home/alta/Linguaskills/exp-graphemic-kmk/D3/align-LINSKevl03.mlp.mpe/lib/wlabs/train.mlf`

#### Grades:
###### BLXXXeval{1,3} and BLXXXgrp24
  `/home/alta/BLTSpeaking/convert-v2/4/lib/grades-orig/ BLXXX{dataset}.lst`
  
###### LINSK
  `/home/alta/Linguaskills/convert/lib/grades-orig/LINSK{dataset}.lst`
  
  


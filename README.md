# HATM

## Dependencies

### To install
pip install --user tensorflow-gpu == 1.4.1

### To put in .bash_profile
export LD_LIBRARY_PATH="/home/mifs/am969/cuda-8.0/lib64:${LD_LIBRARY_PATH}" <br />
export PATH="/home/mifs/am969/cuda-8.0/bin:${PATH}" <br />
export PATH=$HOME/.local/bin:$PATH <br />
export PYTHONPATH=$PYTHONPATH:/home/alta/relevance/vr311/attention-topic-model/ <br />
export PYTHONPATH="$PYTHONPATH:~/.local/lib/" <br />

## Training

Train ATM and then train HATM using ATM as checkpoint <br />
Alternatively train SimGrid <br />

(To fill with example commands)

## Testing

The following sequence of instructions give an example of how to evaluate performance on a data-set of prompt-response pairs.
The trained model directory here is called `com1` but it could be any name (depending on the name given to the model during training).
Step 0 is optional as you may have already preprocessed the data-set into the desired format of a tfrecord.

### 0.a Process the transcriptions and scripts and save the processed files as readable .txt files

```
 ./path/to/HATM/preprocessing/magic_preprocess_raw.py /path/to/test/scripts.mlf /path/to/test/responses/transcription.mlf /path/to/destination/directory --fixed_sections A B --exclude_sections A B --multi_sections E --multi_sections_master SE0006 --speaker_grades_path /path/to/test/speaker/grades.lst --exclude_grades 2
```
Generates files:
`responses.txt` `prompts.txt` `speakers.txt` `conf.txt` `sections.txt` `prompt_ids.txt`
with the same number of lines where each line in each file corresponds to one another.

### 0.b Create an evaluation set by shuffling prompts and responses to generate file with a mix of positive (on-topic) and negative (off-topic) examples

```
./path/to/HATM/preprocessing/magic_preprocess_shuffle.py /directory/with/unshuffled/.txt/data /destination/directory --samples 1
```

### 0.c Generate the required tfrecords files from the processed and shuffled .txt files

```
./path/to/HATM/preprocessing/magic_preprocess_to_tfrecords.py /directory/with/shuffled/txt/data /path/to/word-list/file/input.wlist.index /destination/directory --preprocessing_type test
```

### 1. Navigate to directory containing trained model checkpoint

```
cd /path/to/com1/
```

The trained model directory should have a structure similar to the following:

    .
    ├── ...
    ├── atm                                        # Symbolic link to `/path/to/HATM/`
    ├── LOGs
    ├── data
    ├── model                                      # Contains checkpoints for trained model after each epoch
    │   ├── checkpoint          
    │   ├── net_arch.pickle
    │   ├── prompt_embeddings.txt
    │   ├── weights.ckpt-1.data-00000-of-00001
    │   ├── ...
    │   ├── weights.ckpt-3.data-00000-of-00001     # 3 epochs in this model
    │   ├── ...
    │   └── weights.ckpt.meta
    ├── ss.sh
    ├── submit_test.sh
    ├── submit_train.sh
    ├── train_cmd.sh
    ├── uu.sh
    └── ...

### 2. Run test script

```
./atm/hatm/run/step_test_hatm.py /path/to/directory/with/test/tfrecord/relevance.test.tfrecords eval_unseen --epoch 3
```

The above command is typically put into a bash script such as `uu.sh`. Then, the command is executed on the GPU using `./submit_test.sh uu.sh`

#### Typical contents of `submit_test.sh`

```
#!/bin/bash
#$ -S /bin/bash

export CUDAPATH="/home/mifs/am969/cuda-8.0/targets/x86_64-linux/lib/:/home/mifs/am969/cuda-8.0/lib64:/home/mifs/ar527/bin/OpenBLAS"

qsub -cwd -j y -o LOGs/LOG.test -l qp=cuda-low -l osrel='*' -l mem_grab=160G -l gpuclass='kepler' ${1}
```

#### Typical contents of `uu.sh`

```
#!/bin/bash
#$ -S /bin/bash

export LD_LIBRARY_PATH="/home/mifs/am969/cuda-8.0/lib64:${LD_LIBRARY_PATH}"

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE

./atm/hatm/run/step_test_hatm.py /home/alta/relevance/vr311/data/LINSKuns03evl03_ALL_naive/tfrecords/relevance.test.tfrecords eval_unseen --epoch 3

```

### 3. Viewing results

You should still be within `/path/to/com1/`. Now the directory structure should have a new directory created named `eval_unseen` which should contain the results of interest.

    .
    ├── ...
    ├── eval_unseen                                      
    │   ├── labels-probs.txt
    │   ├── labels.txt
    │   ├── predictions.txt
    │   └── results.txt
    └── ...
    
    
## Acknowledgements

The code for the ATM and the HATM models was originally forked from [attention-topic-model](https://github.com/KaosEngineer/attention-topic-model) (though there have been substantial changes since the original fork).

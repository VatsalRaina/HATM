# HATM

## Dependencies

### To install
pip install --user tensorflow-gpu == 1.4.1

### To put in .bash_profile
export LD_LIBRARY_PATH="/home/mifs/am969/cuda-8.0/lib64:${LD_LIBRARY_PATH}"
export PATH="/home/mifs/am969/cuda-8.0/bin:${PATH}"
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$PYTHONPATH:/home/alta/relevance/vr311/attention-topic-model/
export PYTHONPATH="$PYTHONPATH:~/.local/lib/"

## Training

Train ATM and then train HATM using ATM as checkpoint
Alternatively train SimGrid

(To fill with example commands)

## Testing

The following sequence of instructions give an example of how to evaluate performance on a data-set of prompt-response pairs.
The trained model directory here is called 'com1' but it could be any name (depending on the name given to the model during training).
Step 0 is optional as you may have already preprocessed the data-set into the desired format of a tfrecord.

### 0.a Process the transcriptions and scripts and save the processed files as readable .txt files



### 0.b Create an evaluation set by shuffling prompts and responses to generate file with a mix of positive (on-topic) and negative (off-topic) examples

### 0.c Generate the required tfrecords files from the processed and shuffled .txt files

### 1. Navigate to directory containing trained model checkpoint

### 2. Run test script


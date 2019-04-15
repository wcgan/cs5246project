This README file contains instructions to reproduce the results in our CS5246 Project: Performing Sentimenet Analysis With BERT

The dataset used for performance comparison between models are in the folder /data, with 3 files: rt-polarity.train, rt-polarity.dev, rt-polarity.test

1. To obtain results for the RNN model, 

First ensure that you are using a python3.6 environment with PyTorch 0.4.1 installed.

Train the model by running:

python rnn.py --train_file Data/rt-polarity/train.tsv \
  --val_file Data/rt-polarity/dev.tsv \
  --emb_file_txt [path to GloVe 300d] \
  --output_file RNN_Output/model_file \
  --epochs 10

Then, obtain the testing accuracy with:

python run_rnn.py Data/rt-polarity/test.tsv RNN_Output/model_file

2. To obtain results for the CNN model,


3. To obtain results for the bmLSTM model,


4. To obtain results for the original BERT model,

First ensure that you are using a python3.6 environment with PyTorch 0.4.1 and pytorch-pretrained-bert 0.3.0 installed.

Then, simply run

python run_classifier_new.py \
  --task_name SST-2 \
  --do_train \
  --do_test \
  --do_lower_case \
  --data_dir Data/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --output_dir Output/base/ \
  --load_dir Output/base/ \
  --save_dir Output/base/ \
  --gradient_accumulation_steps 32 \
  --eval_batch_size 1 \
  --model base \

5. To fine-tune BERT with the attention approach, ensure the dependencies in 4. are installed.

Then, simply run

python run_classifier_new.py \
  --task_name SST-2 \
  --do_train \
  --do_test \
  --do_lower_case \
  --data_dir Data/rt-polarity/ \
  --bert_model bert-base-uncased \
  --max_seq_length 128 \
  --train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 3.0 \
  --output_dir Output/attn/ \
  --load_dir Output/attn/ \
  --save_dir Output/attn/ \
  --gradient_accumulation_steps 32 \
  --eval_batch_size 1 \
  --model attention \
  --seed 5246 \

6. To fine-tune BERT with adversarial training,
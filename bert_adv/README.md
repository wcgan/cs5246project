# bert-adv

This repo combines bert: https://github.com/google-research/bert,
with text adversarial: https://github.com/tensorflow/models/tree/master/research/adversarial_text.

Follow the original Bert instruction for downloading pretrained uncased_L-12_H-768_A-12.
Then for classification, run the following:

```
export BERT_BASE_DIR=/path/to/project/bert/uncased_L-12_H-768_A-12
python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=59 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=at\
  --train_batch_size=8\
  --adv_training_method=rp
```
Data dir should contain the tab seperated files.
This repo supports rp and at for adv_training_method option.


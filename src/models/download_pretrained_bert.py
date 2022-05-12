from transformers import BertModel, BertTokenizer

PRE_TRAINED_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

tokenizer.save_pretrained('./bert_cache_dir/')
model.save_pretrained('./bert_cache_dir/')
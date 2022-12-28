import os
# Important: the following env vars is a hack to emulated distributed environment on a single GPU. Remove all of the env vars if you run
# with more than one GPU and the torch.distributed or the deepspeed launcher
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import torch
import datasets
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

# args
torch.manual_seed(42)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=15, per_device_eval_batch_size=15, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True)

# model
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
model.resize_token_embeddings(len(tokenizer))

# dataset
squad = datasets.load_dataset('squad')
tokenized = squad_dataset.map(lambda x: tokenizer(x['context']), batched=True)

# class NetflixDataset(Dataset):
#     def __init__(self, txt_list, tokenizer, max_length):
#         self.input_ids = []
#         self.attn_masks = []
#         self.labels = []
#         for txt in txt_list:
#             encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
#                                        max_length=max_length, padding="max_length")
#             self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
#             self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.attn_masks[idx]


# train
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})
trainer.train()

# save the trained model
trainer.save_model('gpt2-netflix')

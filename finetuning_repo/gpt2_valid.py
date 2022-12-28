import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy

# dataset
ds = load_dataset('squad')
all = []
maxlen = 0
for x in ds['train']['context']:
  all.append("CONTEXT: " + x.replace('\n', ' ') + "\n")
for i, x in enumerate(ds['train']['question']):
  all[i] += "QUESTION: " + x + "\n"
for i, x in enumerate(ds['train']['answers']):
  all[i] += "ANSWER: " + x['text'][0]
  if len(all[i]) > maxlen:
    maxlen = len(all[i])
  if len(x['text']) > 1:
    raise Exception("More than one answer")
print("Max length: ", maxlen)

tokenizer = AutoTokenizer.from_pretrained("gpt2", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

class TxtDataset(Dataset):
    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in all:
            encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                       max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

dataset = TxtDataset(all[:-100], tokenizer, 1024)

# training
training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                  per_device_train_batch_size=15, per_device_eval_batch_size=15, warmup_steps=100,
                                  weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./ds_config_gptj6b.json')
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
model.resize_token_embeddings(len(tokenizer))
train_size = int(0.9 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                              'attention_mask': torch.stack([f[1] for f in data]),
                                                              'labels': torch.stack([f[0] for f in data])})
trainer.train()

# save the trained model
trainer.save_model('gpt2-squad')

# sample outputs
for x in all[-1:]:
  x = x.split('\n')[:-1]
  x = "\n".join(x) + "\n"
  generated = tokenizer("<|startoftext|>" + x, return_tensors="pt").input_ids.cuda()
  attention_mask = torch.ones(generated.shape).cuda()
  sample_outputs = model.generate(generated, attention_mask=attention_mask, do_sample=True, top_k=50, max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=3)
  for i, sample_output in enumerate(sample_outputs):
      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

import pdb; pdb.set_trace()

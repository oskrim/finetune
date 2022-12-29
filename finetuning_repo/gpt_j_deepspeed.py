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
from datasets import load_dataset
from torch.utils.data import Dataset, random_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, IntervalStrategy
torch.manual_seed(42)

train = False
batch_size = 6
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

# dataset
max_len = 512
ds = load_dataset('squad')
all = []
for x in ds['train']['context']:
  all.append("CONTEXT: " + x.replace('\n', ' ') + "\n")
for i, x in enumerate(ds['train']['question']):
  all[i] += "QUESTION: " + x + "\n"
for i, x in enumerate(ds['train']['answers']):
  all[i] += "ANSWER: " + x['text'][0]
  if len(x['text']) > 1:
    raise Exception("More than one answer")
all = [x for x in all if len(x) < max_len][:500]
print("dataset len: ", len(all))

if train:
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=4.3, logging_steps=100, save_strategy=IntervalStrategy.NO,
                                      per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, warmup_steps=100,
                                      weight_decay=0.01, logging_dir='./logs', fp16=True, deepspeed='./ds_config_gptj6b.json')
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
    model.resize_token_embeddings(len(tokenizer))
 
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
    dataset = TxtDataset(all[:-10], tokenizer, max_len)

    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                                  'labels': torch.stack([f[0] for f in data])})
    trainer.save_model('gptj-savetest')
    trainer.train()
    trainer.save_model('gptj-squad')
else:
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
    # model = AutoModelForCausalLM.from_pretrained("./gptj-squad").cuda()
    model.resize_token_embeddings(len(tokenizer))

def model_outputs(x, n=1):
  generated = tokenizer("<|startoftext|>" + x, return_tensors="pt").input_ids.cuda()
  attention_mask = torch.ones(generated.shape).cuda()
  out = model.generate(generated, attention_mask=attention_mask, do_sample=True, top_k=50, max_length=300, top_p=0.95, temperature=1.9, num_return_sequences=n)
  return [tokenizer.decode(x, skip_special_tokens=True) for x in out]

import pdb; pdb.set_trace()

# sample outputs, SQUAD
for x in all[-10:]:
  x = x.split('\n')[:-1]
  x = "\n".join(x) + "\n"
  sample_outputs = model_outputs(x)
  for i, sample_output in enumerate(sample_outputs):
      print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

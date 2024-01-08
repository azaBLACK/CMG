from transformers import GPT2DoubleHeadsModel
from transformers.models.gpt2.modeling_gpt2 import GPT2DoubleHeadsModelOutput
from torch.nn import BCELoss, Sigmoid, CrossEntropyLoss
import torch
import numpy as np
from torchtext import data
import logging
import random
import argparse

from data.data import NormalField, ParallelDataset
from data.lazy_iterator import BucketIterator, Iterator
import time

from model.dl4nmt import train
from pathlib import Path
import json
import os
from torchtext.vocab import Vectors
class Multitask(GPT2DoubleHeadsModel):
    
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        # mc_token_ids=None,
        labels=None,
        meta_label=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):


            >>> import torch
            >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = GPT2DoubleHeadsModel.from_pretrained('gpt2')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.logits
            >>> mc_logits = outputs.mc_logits

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        cls_token_ids = (input_ids==102).nonzero()[:,1]
        mc_logits = self.multiple_choice_head(hidden_states, cls_token_ids).squeeze(-1)

        mc_loss = None
        if meta_label is not None:
            meta_label = meta_label.float()
            sigmoid = Sigmoid()
            loss_fct = BCELoss()
            probs = sigmoid(mc_logits)
            mc_loss = loss_fct(probs.view(-1), meta_label.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction='none')
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            lm_loss = (lm_loss.view(shift_logits.size(0), -1)*probs.view(-1,1)).mean()

        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss,) + output
            return ((lm_loss,) + output) if lm_loss is not None else output

        return GPT2DoubleHeadsModelOutput(
            loss=lm_loss,
            mc_loss=mc_loss,
            logits=lm_logits,
            mc_logits=mc_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    from transformers import (GPT2LMHeadModel,
                              GPT2DoubleHeadsModel,
                              BertTokenizer,
                              TrainingArguments,
                              Trainer)
    from models import Multitask
    import sys
    import datasets
    import pandas as pd
    from transformers.trainer_utils import get_last_checkpoint, is_main_process

    if __name__ == '__main__':
        model_name, cmc_path, clc_path, simile_path, max_length, epochs, save_path, simile_script_path, cache_dir, = sys.argv[
                                                                                                                     1:]
        max_length = int(max_length)
        epochs = int(epochs)
        last_checkpoint = get_last_checkpoint(save_path)

        # simile_ds = datasets.load_dataset(simile_script_path, data_dir=cmc_path, split='train')
        tokenizer = BertTokenizer.from_pretrained(model_name)

        def process_dataset(examples, col_name='sent'):
            encoding = tokenizer(examples[col_name], max_length=max_length, truncation=True, padding='max_length')
            encoding["labels"] = encoding["input_ids"].copy()
            return encoding

        def load_clc():
            ds = datasets.Dataset.from_text(clc_path, cache_dir=cache_dir)
            ds = ds.add_column('meta_label', [-100] * len(ds))
            ds = ds.map(process_dataset, fn_kwargs={'col_name': 'text'}, remove_columns=['text'],
                        cache_file_name=cache_dir + '/clc_tensor.cache')
            return ds

        def load_simile():
            ds = datasets.load_dataset(simile_script_path, data_dir=simile_path, split='train')
            ds.rename_column_('label', 'meta_label')
            ds = ds.map(process_dataset, fn_kwargs={'col_name': 'sent'}, remove_columns=['sent', 'tokens'])
            ds.remove_columns_(['tags'])
            return ds

        def load_cmc():
            df = pd.read_csv(cmc_path, sep='\t', names=['label', 'sent'])
            ds = datasets.Dataset.from_pandas(df)
            ds.rename_column_('label', 'meta_label')
            ds = ds.map(process_dataset, fn_kwargs={'col_name': 'sent'}, remove_columns=['sent'])
            return ds

        ds_simile = load_simile()
        ds_cmc = load_cmc()

        ds_clc = load_clc()
        ds = datasets.concatenate_datasets([ds_simile, ds_cmc, ds_clc])

        model = Multitask.from_pretrained(model_name)

        args = TrainingArguments(
            output_dir=save_path,
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=2,
            num_train_epochs=epochs,
            save_strategy='steps',
            logging_strategy='steps',
            save_steps=2000,
            logging_steps=500,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            tokenizer=tokenizer,
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
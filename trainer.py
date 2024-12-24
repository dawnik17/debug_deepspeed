import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any
import json
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from .base import CTrainer
from functools import wraps
import bz2
import pickle
import functools
import pandas as pd
from copy import deepcopy
import deepspeed
from typing import Literal, Dict, Union
from collections import defaultdict
import math

from loss import SigLipLoss


class SiglipTrainer(Trainer):
    def __init__(self, emb_type="bos_token", sentence_transformers_model=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.siglip = SigLipLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(10))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10)

        self.emb_type = emb_type
        self.sentence_transformers_model = sentence_transformers_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get Query Output
        query_inputs = inputs["query"]
        query_model_output = model(
            input_ids=query_inputs["input_ids"], 
            attention_mask=query_inputs["attention_mask"], 
            return_dict=True
        )

        # Get Product Output
        product_inputs = inputs["passage"]
        product_model_output = model(
            input_ids=product_inputs["input_ids"], 
            attention_mask=product_inputs["attention_mask"], 
            return_dict=True
        )

        # Get Embeddings
        if self.sentence_transformers_model:
            p_reps = product_model_output.sentence_embedding
            q_reps = query_model_output.sentence_embedding
        
        else:
            p_reps = self.get_embeddings(product_model_output, product_inputs["attention_mask"])
            q_reps = self.get_embeddings(query_model_output, query_inputs["attention_mask"])

        p_reps = F.normalize(p_reps, dim=-1).contiguous()
        q_reps = F.normalize(q_reps, dim=-1).contiguous()

        # Calculate Loss
        with torch.no_grad():
            self.logit_scale.clamp_(0, math.log(100))
        loss = self.siglip(p_reps, q_reps, self.logit_scale.exp(), self.logit_bias, False)
        return loss

    def get_embeddings(self, model_output, attention_mask):
        """
        model_output: [batch, seqlen, hiddim]
        attention_mask: [batch, seqlen]

        return: [batch, emb_dim]
        """
        if self.emb_type == "mistral-nv-scratch":
            return model_output["sentence_embeddings"]

        elif self.emb_type == "eos_token":
            last_hidden_states = model_output.last_hidden_state
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]

                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        elif self.emb_type == "bos_token":
            last_hidden_states = model_output.last_hidden_state
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]

            if left_padding:
                pad_sequence_lengths = (1 - attention_mask).sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]

                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    pad_sequence_lengths,
                ]

            else:
                return last_hidden_states[:, 0]
        else:
            raise

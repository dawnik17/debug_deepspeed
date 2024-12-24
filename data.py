import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple

import json
import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

import sys
from arguments import DataArguments


class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        if os.path.isdir(args.train_data):
            train_datasets = []

            for file in os.listdir(args.train_data):
                print(f"Loading {file}..")
                temp_dataset = datasets.load_dataset(
                    "json",
                    data_files=os.path.join(args.train_data, file),
                    split="train",
                )

                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(
                            list(range(len(temp_dataset))),
                            args.max_example_num_per_dataset,
                        )
                    )

                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset(
                "json", data_files=args.train_data, split="train"
            )

        self.args = args
        self.dataset = self.dataset.shuffle(seed=42)

        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)

        self.pid2text = json.load(
            open("path..", "r")
        )

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]["query"]

        retrieval_instruction = (
            self.dataset[item].get("query_instruction_for_retrieval")
            or self.args.query_instruction_for_retrieval
        )

        if retrieval_instruction:
            query = retrieval_instruction + query

        passages = []

        assert isinstance(self.dataset[item]["positive"], list)
        pos = random.choice(self.dataset[item]["positive"])
        passages.append(pos)

        if self.args.use_dataset_neg and "negative" in self.dataset[item]:
            if len(self.dataset[item]["negative"]) < self.args.train_group_size - 1:
                num = math.ceil(
                    (self.args.train_group_size - 1)
                    / len(self.dataset[item]["negative"])
                )
                negs = random.sample(
                    self.dataset[item]["negative"] * num, self.args.train_group_size - 1
                )
            else:
                negs = random.sample(
                    self.dataset[item]["negative"], self.args.train_group_size - 1
                )

            passages.extend(negs)

        passages = [self.pid2text[pid] for pid in passages]

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [
                self.args.passage_instruction_for_retrieval + p for p in passages
            ]

        return query, passages


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    query_max_len: int = 64
    passage_max_len: int = 194

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])

        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        return {"query": q_collated, "passage": d_collated}

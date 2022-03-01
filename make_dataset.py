# Script for downloading the suqad v2 dataset and creating a dataset for training

# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import urllib.request
import argparse
from transformers import BertTokenizerFast
import torch
import json
from pathlib import Path


class SquadDownloader:
    def __init__(self, save_path):
        self.save_path = save_path + '/squad'

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.download_urls = {
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json': 'train-v2.0.json',
            'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json': 'dev-v2.0.json',
            'https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/': 'evaluate-v2.0.py',
        }

    def download(self):
        for item in self.download_urls:
            url = item
            file = self.download_urls[item]

            print('Downloading:', url)
            if os.path.isfile(self.save_path + '/' + file):
                print('** Download file already exists, skipping download')
            else:
                response = urllib.request.urlopen(url)
                with open(self.save_path + '/' + file, "wb") as handle:
                    handle.write(response.read())


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            # When the gold label is off by one character
            answer['answer_end'] = end_idx - 1
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            # When the gold label is off by two characters
            answer['answer_end'] = end_idx - 2


def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(
            i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(
            i, answers[i]['answer_end'] - 1))
        # if None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length
    encodings.update({'start_positions': start_positions,
                     'end_positions': end_positions})


# run and get path to save using argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument(
        '--datadir', type=str,   default='./', help='data directory for training and validation data. Default is current dir')
    argparser.add_argument(
        '--onlydownload', type=bool,   default=True, help='Should it also preprocess and tokenize')
    argparser.add_argument(
        '--tokenizername', type=str,   default="bert-base-uncased", help='The fast tokenizer to use')
    args = argparser.parse_args()

    squad_downloader = SquadDownloader(args.datadir)
    squad_downloader.download()
    if args.onlydownload:
        print('Starting preprocessing!')
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizername)
        train_contexts, train_questions, train_answers = read_squad(
            args.datadir + 'squad/train-v2.0.json')
        val_contexts, val_questions, val_answers = read_squad(
            args.datadir + 'squad/dev-v2.0.json')
        add_end_idx(train_answers, train_contexts)
        add_end_idx(val_answers, val_contexts)
        train_encodings = tokenizer(
            train_contexts, train_questions, truncation=True, padding=True)
        val_encodings = tokenizer(
            val_contexts, val_questions, truncation=True, padding=True)

        add_token_positions(train_encodings, train_answers)
        add_token_positions(val_encodings, val_answers)

        # Saving preprocessed data
        print('Saving preprocessed data')
        torch.save(train_encodings, "./squad/train_encodings")
        torch.save(val_encodings, "./squad/val_encodings")
    print('Done!')

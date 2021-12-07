"""
ArgKP baseline implementation. Extracts key points from a given dataset of
comments. Implements Appendix B from https://arxiv.org/pdf/2010.05369.pdf

Created by: Michiel van der Meer <m.t.van.der.meer@liacs.leidenuniv.nl>
"""


import argparse
import json
import operator
from collections import defaultdict
from itertools import permutations, product

import numpy as np
import pandas as pd
import scipy
import pickle
import spacy
import torch
import transformers
from torch.utils.data import Dataset

nlp = spacy.load("en_core_web_sm")

DEFAULT_CANDIDATE_NUM_TOKENS_MAX = 15
DEFAULT_CANDIDATE_NUM_TOKENS_MIN = 1
DEFAULT_CANDIDATE_MIN_QUAL = 0.5

class KPDataset(torch.utils.data.Dataset):
    """
    Dataset used in forward passes through ArgKP model to find matches between
    comments and candidate key points.
    """
    def __init__(self, comments, kp_candidates):
        self.model_name = 'roberta-large'
        self.comments = comments.reset_index()
        self.kp_candidates = kp_candidates

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)

        self.data = list(product(range(len(self.kp_candidates)), range(len(self.comments))))


    def __getitem__(self, idx):
        kp_candidate, comment = self.data[idx]
        sentence = self.kp_candidates[kp_candidate] + self.tokenizer.sep_token + self.comments.iloc[comment]['english']
        encoding = self.tokenizer([sentence], padding="max_length", truncation=True)
        item = {key: torch.tensor(val).squeeze() for key, val in encoding.items()}

        return item

    def __len__(self):
        return len(self.data)


class KP2KPDataset(torch.utils.data.Dataset):
    """
    Dataset used in forward passes through ArgKP model to find merges between
    key point candidates.
    """
    def __init__(self, kp_candidates):
        self.model_name = 'roberta-large'
        self.kp_candidates = kp_candidates
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.data = list(permutations(range(len(self.kp_candidates)), 2))

    def __getitem__(self, idx):
        kp_candidate, kp_candidate_1 = self.data[idx]
        sentence = self.kp_candidates[kp_candidate] + self.tokenizer.sep_token + self.kp_candidates[kp_candidate_1]
        encoding = self.tokenizer([sentence], padding="max_length", truncation=True)
        item = {key: torch.tensor(val).squeeze() for key, val in encoding.items()}
        return item

    def __len__(self):
        return len(self.data)


def extract_candidates(comments, cand_max_tokens=DEFAULT_CANDIDATE_NUM_TOKENS_MAX, cand_min_quality=DEFAULT_CANDIDATE_MIN_QUAL):
    """
    Given a list of comments, extract a list of candidate key points.
    """
    candidates = []
    for _, candidate in comments.iterrows():
        doc = nlp(candidate['english'], disable=['ner'])
        if len(doc) > DEFAULT_CANDIDATE_NUM_TOKENS_MIN and len(doc) <= cand_max_tokens and all([t.is_ascii for t in doc]) and not any([x.lower_ == 'and' for x in doc]) and candidate['quality_scores'] > cand_min_quality:
            candidates.append(candidate['english'])

    print(f"Selected {len(candidates)} / {len(comments)} candidates")
    return candidates


def match_comments(texts, candidates, threshold, batch_size):
    """
    Match a list of `texts` to candidates given a matching threshold.
    """
    dataset = KPDataset(texts, candidates)
    training_args = transformers.TrainingArguments('argkp_baseline', report_to='none', per_device_eval_batch_size=batch_size)
    trainer = transformers.Trainer(
        args=training_args,
        model=transformers.AutoModelForSequenceClassification.from_pretrained('state/'),
    )
    predictions = trainer.predict(dataset)
    k2c = defaultdict(lambda: [])
    comment_scores = defaultdict(lambda: {})
    for i, pred in enumerate(predictions.predictions):
        kp_idx, comment_idx = dataset.data[i]
        soft_pred = scipy.special.softmax(pred)
        if np.argmax(soft_pred) == 1 and soft_pred[1] > threshold:
            comment_scores[comment_idx][kp_idx] = soft_pred[1]

    for comment_idx, kps_dict in comment_scores.items():
        kp_idx, kp_score = max(kps_dict.items(), key=operator.itemgetter(1))
        k2c[dataset.kp_candidates[kp_idx]].append((dataset.comments.iloc[comment_idx]['english'], kp_score))
    return k2c


def filter_candidates(k2c, threshold, batch_size):
    """
    Filter out candidates that are similar to each other by merging them.
    Re-match the comments that were matched to merged-away candidate key points.
    """
    x = list(zip(k2c.keys(), [len(k2c[key]) for key in k2c]))
    keys = sorted(x, reverse=True, key=lambda x: x[1])
    dataset = KP2KPDataset([x[0] for x in keys])
    training_args = transformers.TrainingArguments('argkp_baseline', report_to='none', per_device_eval_batch_size=batch_size)
    trainer = transformers.Trainer(
        args=training_args,
        model=transformers.AutoModelForSequenceClassification.from_pretrained('state/'),
    )
    predictions = trainer.predict(dataset)

    num_kps = len(keys)
    num_other_kps = len(keys) - 1
    preds = np.zeros((num_kps, num_kps, 2))

    for i, k in enumerate(keys):
        outputs = predictions.predictions[(i*num_other_kps):(i*num_other_kps)+num_other_kps, :]
        outputs = np.insert(outputs, i, (0,0), axis=0)
        preds[i,:]= outputs

    comment_pool = []
    merged = 0
    for i, key_point in reversed(list(enumerate(keys))):
        for j, _ in enumerate(keys):
            if i == j:
                continue
            avg_score = (preds[i,j] + preds[j, i]) / 2
            if np.argmax(avg_score) == 1 and avg_score[1] > threshold:
                comment_pool.extend([x for x, _ in k2c[key_point[0]]])
                comment_pool.append(key_point[0])
                k2c.pop(key_point[0])
                keys.pop(i)
                merged += 1
                break

    if len(comment_pool) > 0:
        comment_df = pd.DataFrame({'english': comment_pool})
        new_k2c = match_comments(comment_df, list(k2c.keys()), threshold, batch_size)
        for k in new_k2c:
            k2c[k].extend(new_k2c[k])
    print(f"Merged {merged} KP candidates")
    x = list(zip(k2c.keys(), [len(k2c[key]) for key in k2c]))
    sorted_keys = sorted(x, reverse=True, key=lambda x: x[1])
    print("Final list of KPs:")
    for k, count in sorted_keys:
        print(f"({count:3}) {k}")
    return k2c, sorted_keys


def get_sentences(texts, candidates, threshold=0.9, batch_size=16):
    """
    Given a list of comments and candidates, get a list of extracted comments
    as final key points.
    """
    k2c = match_comments(texts, candidates, threshold, batch_size)
    k2c, sorted_kps = filter_candidates(k2c, threshold, batch_size)
    return dict(k2c), sorted_kps


def get_args_per_topic(df, topic_id, num_comments=100, batch_size=16, cand_max_tokens=DEFAULT_CANDIDATE_NUM_TOKENS_MAX, cand_min_quality=DEFAULT_CANDIDATE_MIN_QUAL):
    """
    For a specific topic (using `topic_id`) get a list of pro and con key
    points from a given DataFrame `df`.
    Considers only up to `num_comments` comments.
    Use `batch_size` for the batch size of forward passes through the ArgKP model.
    """
    if num_comments < 0:
        text_pro = df[(df.project == topic_id) & (df.extracted_from == 'pro')]
        text_con = df[(df.project == topic_id) & (df.extracted_from == 'con')]
    else:
        text_pro = df[(df.project == topic_id) & (df.extracted_from == 'pro')][:num_comments]
        text_con = df[(df.project == topic_id) & (df.extracted_from == 'con')][:num_comments]
    pro_cands = extract_candidates(text_pro, cand_max_tokens=cand_max_tokens, cand_min_quality=cand_min_quality)
    con_cands = extract_candidates(text_con, cand_max_tokens=cand_max_tokens, cand_min_quality=cand_min_quality)
    pro_kps, _ = get_sentences(text_pro, pro_cands, batch_size=batch_size)
    con_kps, _ = get_sentences(text_con, con_cands, batch_size=batch_size)
    return pro_kps, con_kps


def main(config):
    df = pd.read_csv(config['infile'])
    comment2id = {}
    for i, row in df.iterrows():
        if row['english'] in comment2id:
            # if row['project'] in comment2id[row['english']]:
            #     print(row)
            comment2id[row['english']][row['project']] = i
        else:
            comment2id[row['english']] = {row['project']: i}

    summary_dict = defaultdict(lambda: {'pro':{}, 'con':{}})

    # Single topic for now to reduce computations
    for i in range(config['option'],config['option']+1):
        pro, con = get_args_per_topic(
            df, i,
            num_comments=config['num_comments'],
            batch_size=config['batch_size'],
            cand_max_tokens=config['candidate_max_tokens'],
            cand_min_quality=config['candidate_min_quality'],
            )
        summary_dict[i]['pro'] = pro
        summary_dict[i]['con'] = con

    # Dump
    with open(f"dump_baseline_option{config['option']}.pkl", 'wb') as f:
        pickle.dump({'summary_dict': dict(summary_dict), 'config': config, 'comment2id':comment2id}, f)

    # Print results
    results = []
    for polarity in summary_dict[config['option']]:
        for key_point, comments in summary_dict[config['option']][polarity].items():
            results.append({
                'key_point': key_point,
                'key_point_motivation_idx': comment2id[key_point][config['option']],
                'stance': polarity,
                'support': len(comments),
                'comments': [(comment, float(match_score), comment2id[comment][config['option']]) for comment, match_score in comments]
            })
            print(f"\"{key_point}\", {len(comments)}, {polarity}")

    with open(f"baseline_result{config['option']}.json", 'w') as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='input .csv')
    parser.add_argument('--threshold', default=0.9, type=float, help="Threshold on the argument matching confidence")
    parser.add_argument('--num_comments', default=100, type=int, help="How many comments to analyze")
    parser.add_argument('--batch_size', default=16, type=int, help="how many datapoints in a batch")
    parser.add_argument('--candidate_max_tokens', default=DEFAULT_CANDIDATE_NUM_TOKENS_MAX, type=int, help="how many tokens a comment can have max for it to be considered a candidate")
    parser.add_argument('--candidate_min_quality', default=DEFAULT_CANDIDATE_MIN_QUAL, type=float, help="minimum quality score for a comment to be considered a candidate")
    parser.add_argument('--option', default=0, type=int, help="which topic to run")
    args = parser.parse_args()

    config = vars(args)
    print("Parameters:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")
    main(config)

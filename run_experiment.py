# © Copyright IBM Corporation 2022.
#
# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0
import logging
import os
import random
import re
import string
import uuid

from argparse import ArgumentParser
from typing import Sequence

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from filelock import FileLock
from sib import SIB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


STOP_WORDS_FILE = './stop_words.txt'
OUT_DIR = './output'


def remove_stop_words_and_punctuation(texts):
    with open(STOP_WORDS_FILE, 'r') as f:
        stop_words = [line.strip() for line in f if line.strip()]
    escaped_stop_words = [re.escape(stop_word) for stop_word in stop_words]
    regex_pattern = r"\b(" + "|".join(escaped_stop_words) + r")\b"
    # remove stop words
    texts = [re.sub(r" +", r" ", re.sub(regex_pattern, "", str(text).lower())).strip() for text in texts]
    # remove punctuation
    texts = [t.translate(t.maketrans(string.punctuation, ' ' * len(string.punctuation))) for t in texts]
    return [' '.join(t.split()) for t in texts]


def stem(texts):
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=False)
    return [" ".join(stemmer.stem(word).lower().strip()
            for word in nltk.word_tokenize(text)) for text in texts]


def get_embeddings(texts):
    # apply text processing to prepare the texts
    texts = remove_stop_words_and_punctuation(texts)
    texts = stem(texts)

    # create the vectorizer and transform data to vectors
    vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, max_features=10000, stop_words=None, use_idf=False, norm=None)
    vectors = vectorizer.fit_transform(texts)
    return vectors


def get_cluster_labels(texts: Sequence[str], n_clusters: int) -> Sequence[int]:
    embedding_vectors = get_embeddings(texts)
    logging.info('Finished generating embedding vectors')
    algorithm = SIB(n_clusters=n_clusters, n_init=10, n_jobs=-1, max_iter=15, random_state=1024, tol=0.02)
    clustering_model = algorithm.fit(embedding_vectors)
    logging.info(f'Finished clustering embeddings for {len(texts)} texts into {n_clusters} clusters')
    cluster_labels = clustering_model.labels_.tolist()
    return cluster_labels


def record_list_of_results(res_file_name, agg_file_name, list_of_dicts):
    if len(list_of_dicts) == 0:
        return
    lock_path = os.path.abspath(os.path.join(res_file_name, os.pardir, 'result_csv_files.lock'))
    with FileLock(lock_path):
        logging.debug("Inside lock")
        if os.path.isfile(res_file_name):
            orig_df = pd.read_csv(res_file_name)
            df = pd.concat([orig_df, pd.DataFrame(list_of_dicts)])
            df_agg = df.groupby(by=['setting_name', 'eval_file', 'labeling_budget']).mean()\
                .sort_values(by=['eval_file', 'labeling_budget', 'setting_name'])
            df_agg.to_csv(agg_file_name)
        else:
            df = pd.DataFrame(list_of_dicts)
        df.to_csv(res_file_name, index=False)
        logging.debug("Releasing lock")


def train(texts, labels, keep_classification_layer=True, model_name_or_path="bert-base-uncased",
          batch_size=64, learning_rate=3e-5, num_epochs=10):
    model_id = str(uuid.uuid1())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_inputs = tokenizer(texts, add_special_tokens=True, max_length=128, padding=True, truncation=True)
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_inputs), labels))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-06)
    num_labels = len(set(labels))
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    model_path = os.path.join(OUT_DIR, model_id)
    os.makedirs(model_path)
    logging.info(f'training model {model_id} with {num_labels} output classes, '
                 f'starting from base model {model_name_or_path}')
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model.fit(x=train_dataset.shuffle(1000).batch(batch_size), validation_data=None, epochs=num_epochs)
    if not keep_classification_layer:
        model.classifier._name = "dummy"  # change classifier layer name so it will not be reused
    model.save_pretrained(model_path)
    model.config.__class__.from_pretrained(model_name_or_path).save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    return model_path


def infer(texts, model_name_or_path, num_labels, infer_batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    logging.info(f'running inference using model {model_name_or_path} on {len(texts)} texts')
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=num_labels)
    predictions = []
    for x in tqdm.tqdm(range(0, len(texts), infer_batch_size)):
        batch_texts = texts[x:x + infer_batch_size]
        batch_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="tf")
        batch_res = model(batch_input).logits.numpy().tolist()
        predictions.extend(batch_res)

    return np.argmax(predictions, axis=1)


def evaluate(eval_texts, eval_labels, model_path, label_encoder):
    eval_predictions = infer(eval_texts, model_path, num_labels=len(label_encoder.classes_))
    eval_predictions = label_encoder.inverse_transform(eval_predictions)
    accuracy = np.mean([gold_label == prediction for gold_label, prediction in zip(eval_labels, eval_predictions)])
    return accuracy


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--labeling_budget', type=int, required=True)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--inter_training_epochs', type=int, default=1)
    parser.add_argument('--finetuning_epochs', type=int, default=10)
    parser.add_argument('--num_clusters', type=int, default=50)

    args = parser.parse_args()
    logging.info(args)

    # set random seed
    random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # generate pseudo-labels for unlabeled texts
    unlabeled_texts = pd.read_csv(args.train_file)['text'].tolist()
    logging.info(f'Clustering {len(unlabeled_texts)} unlabeled texts into {args.num_clusters} clusters')
    clustering_pseudo_labels = get_cluster_labels(unlabeled_texts, args.num_clusters)

    # inter-train model on clustering pseudo-labels
    inter_trained_model_path = train(unlabeled_texts, clustering_pseudo_labels, keep_classification_layer=False,
                                     num_epochs=args.inter_training_epochs)

    # sample *labeling_budget* examples with their gold labels from the train file for fine-tuning
    labeled_data_sample = pd.read_csv(args.train_file).sample(n=args.labeling_budget, random_state=args.random_seed)
    label_encoder = LabelEncoder()
    sample_labels = label_encoder.fit_transform(labeled_data_sample['label'].tolist())
    sample_texts = labeled_data_sample['text'].tolist()

    # fine-tune over the pretrained model using the given sample
    model_finetuned_over_base_path = \
        train(sample_texts, sample_labels, keep_classification_layer=True, num_epochs=args.finetuning_epochs)

    # fine-tune over the inter-trained model using the given sample
    model_finetuned_over_intermediate_path = \
        train(sample_texts, sample_labels, keep_classification_layer=True, num_epochs=args.finetuning_epochs,
              model_name_or_path=inter_trained_model_path)

    # evaluate classification accuracy over the *eval_file*
    eval_df = pd.read_csv(args.eval_file)
    eval_texts = eval_df['text'].tolist()
    eval_labels = eval_df['label'].tolist()

    model_finetuned_over_base_accuracy = \
        evaluate(eval_texts, eval_labels, model_finetuned_over_base_path, label_encoder)
    model_finetuned_over_intermediate_accuracy = \
        evaluate(eval_texts, eval_labels, model_finetuned_over_intermediate_path, label_encoder)

    results = [{'eval_file': args.eval_file, 'setting_name': 'base', 'labeling_budget': args.labeling_budget,
                'accuracy': model_finetuned_over_base_accuracy},
               {'eval_file': args.eval_file, 'setting_name': 'intermediate', 'labeling_budget': args.labeling_budget,
                'accuracy': model_finetuned_over_intermediate_accuracy}]
    # save evaluation results to csv files
    record_list_of_results(res_file_name='output/results.csv', agg_file_name='output/aggregated_results.csv',
                           list_of_dicts=results)

    logging.info(f'Fine-tuned over base:\neval_file: {args.eval_file}, model: {model_finetuned_over_base_path}, '
                 f'accuracy: {model_finetuned_over_base_accuracy}')
    logging.info(f'Fine-tuned over intermediate:\neval_file: {args.eval_file}, '
                 f'model: {model_finetuned_over_intermediate_path}, '
                 f'accuracy: {model_finetuned_over_intermediate_accuracy}')

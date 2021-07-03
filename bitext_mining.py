"""
This scripts show how to mine parallel (translated) sentences from two list of monolingual sentences.
As input, you specific two text files that have sentences in every line. Then, the
LaBSE model is used to find parallel (translated) across these two files.
The result is written to disc.
A large source for monolingual sentences in different languages is:
http://data.statmt.org/cc-100/
This script requires that you have FAISS installed:
https://github.com/facebookresearch/faiss
todo: progress bar
"""
import os
import stanza
from sentence_transformers import SentenceTransformer, models
from bitext_mining_utils import *
import gzip
import tqdm
from sklearn.decomposition import PCA
import torch
from nltk import sent_tokenize
import pandas as pd
import csv
from collections import defaultdict
import langid
from tqdm import tqdm


# if use_pca:
#     # We use a smaller number of training sentences to learn the PCA
#     train_sent = []
#     num_train_sent = 20000
#
#     with file_open(source_file) as fSource, file_open(target_file) as fTarget:
#         for line_source, line_target in zip(fSource, fTarget):
#             if min_sent_len <= len(line_source.strip()) <= max_sent_len:
#                 sentence = line_source.strip()
#                 train_sent.append(sentence)
#
#             if min_sent_len <= len(line_target.strip()) <= max_sent_len:
#                 sentence = line_target.strip()
#                 train_sent.append(sentence)
#
#             if len(train_sent) >= num_train_sent:
#                 break
#
#     print("Encode training embeddings for PCA")
#     train_matrix = model.encode(train_sent, show_progress_bar=True, convert_to_numpy=True)
#     pca = PCA(n_components=pca_dimensions)
#     pca.fit(train_matrix)
#
#     dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=pca_dimensions, bias=False, activation_function=torch.nn.Identity())
#     dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
#     model.add_module('dense', dense)


def get_sentences(source_file, target_file, min_sent_len, max_sent_len, tokenized, en_tokenizer, sl_tokenizer):
    if tokenized:
        # print("Read source file")
        source_sentences = set()
        with file_open(source_file) as fIn:
            for line in tqdm.tqdm(fIn):
                line = line.strip()
                if len(line) >= min_sent_len and len(line) <= max_sent_len:
                    source_sentences.add(line)

        # print("Read target file")
        target_sentences = set()
        with file_open(target_file) as fIn:
            for line in tqdm.tqdm(fIn):
                line = line.strip()
                if len(line) >= min_sent_len and len(line) <= max_sent_len:
                    target_sentences.add(line)
    else:
        # print("Read source file")
        source_sentences = set()
        with file_open(source_file) as fIn:
            # text = sent_tokenize(fIn.read(), language='slovene')
            doc = sl_tokenizer(fIn.read())
            text = [sentence.text for sentence in doc.sentences]
            for line in text:
                if len(line) >= min_sent_len and len(line) <= max_sent_len:
                    source_sentences.add(line)

        # print("Read target file")
        target_sentences = set()
        with file_open(target_file) as fIn:
            # text = sent_tokenize(fIn.read(), language='english')
            doc = en_tokenizer(fIn.read())
            text = [sentence.text for sentence in doc.sentences]
            for line in text:
                if len(line) >= min_sent_len and len(line) <= max_sent_len:
                    target_sentences.add(line)

    # print("Source Sentences:", len(source_sentences))
    # print("Target Sentences:", len(target_sentences))

    return source_sentences, target_sentences


def encode_sentences(model, source_sentences, target_sentences):
    ### Encode source sentences
    source_sentences = list(source_sentences)

    # print("Encode source sentences")
    source_embeddings = model.encode(source_sentences, show_progress_bar=True, convert_to_numpy=True)

    ### Encode target sentences
    target_sentences = list(target_sentences)

    # print("Encode target sentences")
    target_embeddings = model.encode(target_sentences, show_progress_bar=True, convert_to_numpy=True)

    return source_sentences, source_embeddings, target_sentences, target_embeddings


def score(source_embeddings, target_embeddings, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe):
    # Normalize embeddings
    x = source_embeddings
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    y = target_embeddings
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Perform kNN in both directions
    x2y_sim, x2y_ind = kNN(x, y, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
    x2y_mean = x2y_sim.mean(axis=1)

    y2x_sim, y2x_ind = kNN(y, x, knn_neighbors, use_ann_search, ann_num_clusters, ann_num_cluster_probe)
    y2x_mean = y2x_sim.mean(axis=1)

    # Compute forward and backward scores
    margin = lambda a, b: a / b
    fwd_scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
    bwd_scores = score_candidates(y, x, y2x_ind, y2x_mean, x2y_mean, margin)
    fwd_best = x2y_ind[np.arange(x.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y.shape[0]), bwd_scores.argmax(axis=1)]

    indices = np.stack(
        [np.concatenate([np.arange(x.shape[0]), bwd_best]), np.concatenate([fwd_best, np.arange(y.shape[0])])], axis=1)
    scores = np.concatenate([fwd_scores.max(axis=1), bwd_scores.max(axis=1)])
    seen_src, seen_trg = set(), set()

    return indices, scores, seen_src, seen_trg


def write_sentences_to_disk(output_file, indices, scores, seen_src, seen_trg, source_sentences, target_sentences,
                    min_threshold):
    # Extact list of parallel sentences
    print("Write sentences to disc")
    sentences_written = 0
    with gzip.open(output_file, 'wt', encoding='utf8') as fOut:
        for i in np.argsort(-scores):
            src_ind, trg_ind = indices[i]
            src_ind = int(src_ind)
            trg_ind = int(trg_ind)

            if scores[i] < min_threshold:
                break

            if src_ind not in seen_src and trg_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(trg_ind)
                fOut.write("{:.4f}\t{}\t{}\n".format(scores[i], source_sentences[src_ind].replace("\t", " "),
                                                     target_sentences[trg_ind].replace("\t", " ")))
                sentences_written += 1

    print("Done. {} sentences written".format(sentences_written))


def write_best_sentences_to_csv(output_file, file_id, indices, scores, seen_src, seen_trg, source_sentences, target_sentences,
                    min_threshold):
    # Extact list of parallel sentences
    print("Write sentences to disc")
    sentences_written = 0
    output_file = f'output/best/{output_file}'
    with open(output_file, 'w', encoding='utf8') as fOut:
        header = ['file_id', 'score', 'src', 'tgt']
        writer = csv.writer(fOut)

        # write the header
        writer.writerow(header)

        for i in np.argsort(-scores):
            src_ind, tgt_ind = indices[i]
            src_ind = int(src_ind)
            tgt_ind = int(tgt_ind)

            if scores[i] < min_threshold:
                break

            if src_ind not in seen_src and tgt_ind not in seen_trg:
                seen_src.add(src_ind)
                seen_trg.add(tgt_ind)
                # fOut.write("{:.4f}\t{}\t{}\n".format(scores[i], source_sentences[src_ind].replace("\t", " "),
                #                                      target_sentences[tgt_ind].replace("\t", " ")))

                data = [file_id, scores[i], source_sentences[src_ind], target_sentences[tgt_ind]]
                print(data)

                # write the data
                writer.writerow(data)
                sentences_written += 1

        print("Done. {} sentences written".format(sentences_written))


def write_all_sentences_to_csv_one2one(output_file, file_id, indices, scores, source_sentences, target_sentences,
                                       min_threshold):
    # Extact list of parallel sentences
    print("Write sentences to disc")
    sentences_written = 0
    output_file = f'output/one2one/{output_file}'
    with open(output_file, 'w', encoding='utf8') as fOut:
        header = ['file_id', 'source_id', 'target_id', 'score', 'src', 'tgt']
        writer = csv.writer(fOut)

        # write the header
        writer.writerow(header)

        written_scores = set()
        for i in np.argsort(-scores):
            src_ind, tgt_ind = indices[i]
            src_ind = int(src_ind)
            tgt_ind = int(tgt_ind)
            score = scores[i]

            if score not in written_scores:
                written_scores.add(score)
                data = [file_id, src_ind, tgt_ind, score, source_sentences[src_ind], target_sentences[tgt_ind]]
                # print(data)

                # write the data
                writer.writerow(data)

        print("Done. {} sentences written".format(sentences_written))


def write_all_sentences_to_csv_merged(output_file, file_id, indices, scores, source_sentences, target_sentences,
                                       min_threshold):
    # merge directions
    tgt2src = defaultdict(list)
    src2tgt = defaultdict(list)
    written_scores = set()
    for i in np.argsort(-scores):
        src_ind, tgt_ind = indices[i]
        src_ind = int(src_ind)
        tgt_ind = int(tgt_ind)
        score = scores[i]

        # filter duplicates and sentence pairs below given threshold
        if score not in written_scores and score > min_threshold:
            written_scores.add(score)
            data = [file_id, src_ind, tgt_ind, score, source_sentences[src_ind], target_sentences[tgt_ind]]
            tgt2src[src_ind].append(data)
            src2tgt[tgt_ind].append(data)

    # simplify and merge target sentences
    tgt2src_merged = {}
    for ind, value in tgt2src.items():
        src_idx, src_sent = "", ""
        merged_tgt = []
        scores = []
        for file_id, src_ind, tgt_ind, score, source_sentence, target_sentence in value:
            src_idx, src_sent = src_ind, source_sentence
            merged_tgt.append(target_sentence)
            scores.append(score)
        merged_tgt = " ".join(merged_tgt)
        avg_score = np.mean(scores)
        if len(value) > 1:
            merged = True
        else:
            merged = False
        tgt2src_merged[src_idx] = [file_id, avg_score, src_sent, merged_tgt, merged]

    # simplify and merge source sentences
    src2tgt_merged = {}  # NOTE:
    for ind, value in src2tgt.items():
        tgt_idx, tgt_sent = "", ""
        merged_src = []
        scores = []
        for file_id, src_ind, tgt_ind, score, source_sentence, target_sentence in value:
            tgt_idx, tgt_sent = tgt_ind, target_sentence
            merged_src.append(source_sentence)
            scores.append(score)
        merged_src = " ".join(merged_src)
        avg_score = np.mean(scores)
        if len(value) > 1:
            merged = True
        else:
            merged = False
        src2tgt_merged[tgt_idx] = [file_id, avg_score, merged_src, tgt_sent, merged]  # debug: does this solve src and tgt language issues?

    if file_id == '8937837' or file_id == '9062549':
        print()

    # eliminate single sentences that were merged into complex sentences
    added_sentences = ""
    output = []
    merged_both_way = [pair for pair in tgt2src_merged.values()] + [pair for pair in src2tgt_merged.values()]
    merged_both_way.sort(key=lambda x: (x[4], x[1]), reverse=True)  # sort first by merged and by score
    for pair in merged_both_way:
        if pair[2] not in added_sentences or pair[3] not in added_sentences:
            added_sentences += pair[2] + pair[3]  # add source and target sentences to added sentences
            output.append(pair)
    output.sort(key=lambda x: x[1], reverse=True)  # sort by score

    # verify if all source is slo and all target is eng
    for file_id, avg_score, src, tgt, merged in output:
        src_lang = langid.classify(src)[0]
        tgt_lang = langid.classify(tgt)[0]
        if src_lang != 'sl' or tgt_lang != 'en':
            raise ValueError(f'In file {file_id}, src_lang is not Slovene or tgt_lang is not English!')

    # write to disk
    output_file = f'output/merged/{output_file}'
    with open(output_file, 'w', encoding='utf8') as fOut:
        header = ['file_id', 'avg_score', 'src', 'tgt', 'merged']
        writer = csv.writer(fOut)

        # write the header
        writer.writerow(header)

        for row in output:
            writer.writerow(row)


def main():
    # Only consider sentences that are between min_sent_len and max_sent_len characters long
    min_sent_len = 0
    max_sent_len = 200000

    # We base the scoring on k nearest neighbors for each element
    knn_neighbors = 4

    # Do we want to use exact search of approximate nearest neighbor search (ANN)
    # Exact search: Slower, but we don't miss any parallel sentences
    # ANN: Faster, but the recall will be lower
    use_ann_search = False

    # Number of clusters for ANN. Each cluster should have at least 10k entries
    ann_num_clusters = 32768

    # How many cluster to explorer for search. Higher number = better recall, slower
    ann_num_cluster_probe = 3

    # To save memory, we can use PCA to reduce the dimensionality from 768 to for example 128 dimensions
    # The encoded embeddings will hence require 6 times less memory. However, we observe a small drop in performance.
    use_pca = False
    pca_dimensions = 128

    # Model we want to use for bitext mining. LaBSE achieves state-of-the-art performance
    model_name = 'LaBSE'
    model = SentenceTransformer(model_name)

    # stanza tokenizers
    en_tokenizer =  stanza.Pipeline('en', use_gpu=True, processors='tokenize')
    sl_tokenizer =  stanza.Pipeline('sl', use_gpu=True, processors='tokenize')
    langid.set_languages(['sl', 'en'])

    # input folders
    source_folder = 'data/kas/slo_eng_abs/slo'
    # target_folder = 'data/kas/slo_eng_abs/eng'
    tokenized = False

    # Min score for text pairs. Note, score can be larger than 1
    min_threshold = 1.1

    for file in tqdm(os.scandir(source_folder)):
        # get file id
        file_id = file.name.split('-')[1]
        print('FILE ID:', file_id)

        # Input files. We interpret every line as sentence.
        source_file = f"data/kas/slo_eng_abs/slo/kas-{file_id}-abs-sl.txt"
        target_file = f"data/kas/slo_eng_abs/eng/kas-{file_id}-abs-en.txt"
        output_file = f"{file_id}.csv"

        # get sentences
        source_sentences, target_sentences = get_sentences(source_file,
                                                           target_file,
                                                           min_sent_len,
                                                           max_sent_len,
                                                           tokenized,
                                                           en_tokenizer,
                                                           sl_tokenizer)

        # encode sentences
        source_sentences, source_embeddings, target_sentences, target_embeddings = encode_sentences(model,
                                                                                                    source_sentences,
                                                                                                    target_sentences)

        # calculate scores
        indices, scores, seen_src, seen_trg = score(source_embeddings,
                                                    target_embeddings,
                                                    knn_neighbors,
                                                    use_ann_search,
                                                    ann_num_clusters,
                                                    ann_num_cluster_probe)

        # # write sentences to disk
        # write_sentences_to_disk(output_file,
        #                 indices,
        #                 scores,
        #                 seen_src,
        #                 seen_trg,
        #                 source_sentences,
        #                 target_sentences,
        #                 min_threshold)

        # write_best_sentences_to_csv(output_file,
        #                 file_id,
        #                 indices,
        #                 scores,
        #                 seen_src,
        #                 seen_trg,
        #                 source_sentences,
        #                 target_sentences,
        #                 min_threshold)

        write_all_sentences_to_csv_one2one(output_file,
                                           file_id,
                                           indices,
                                           scores,
                                           source_sentences,
                                           target_sentences,
                                           min_threshold)

        write_all_sentences_to_csv_merged(output_file,
                                           file_id,
                                           indices,
                                           scores,
                                           source_sentences,
                                           target_sentences,
                                           min_threshold)

if __name__ == '__main__':
    main()

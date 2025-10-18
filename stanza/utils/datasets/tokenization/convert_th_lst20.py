"""Processes the tokenization section of the LST20 Thai dataset

The dataset is available here:

https://aiforthai.in.th/corpus.php

The data should be installed under ${EXTERN_DATA}/thai/LST20_Corpus

python3 -m stanza.utils.datasets.tokenization.convert_th_lst20 extern_data data/tokenize

Unlike Orchid and BEST, LST20 has train/eval/test splits, which we relabel train/dev/test.

./scripts/run_tokenize.sh UD_Thai-lst20 --dropout 0.05 --unit_dropout 0.05
"""


import argparse
import glob
import os
import sys

from stanza.utils.datasets.tokenization.process_thai_tokenization import write_section, convert_processed_lines, reprocess_lines

def read_document(lines, spaces_after, split_clauses):
    document = []
    sentence = []

    replace_nbsp = "\xa0" in "".join(lines)  # Avoid per-token check unless needed
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if sentence:
                if spaces_after:
                    sentence[-1] = (sentence[-1][0], True)
                document.append(sentence)
                sentence = []
        else:
            # Fast path: only replace nbsp if necessary
            pieces = line_stripped.split("\t")
            if replace_nbsp:
                for i, p in enumerate(pieces):
                    if "\xa0" in p:  # Replace only on string with nbsp
                        pieces[i] = p.replace("\xa0", " ")

            token = pieces[0]
            # shortcut for 'O' check so we do not build pieces[3]
            # but can't avoid due to original logic (index required)
            if split_clauses and token == '_' and pieces[3] == 'O':
                if sentence:
                    sentence[-1] = (sentence[-1][0], True)
                    document.append(sentence)
                    sentence = []
            elif token == '_':
                sentence[-1] = (sentence[-1][0], True)
            else:
                sentence.append((token, False))

    if sentence:
        if spaces_after:
            sentence[-1] = (sentence[-1][0], True)
        document.append(sentence)
        sentence = []
    return [[document]]

def retokenize_document(lines):
    processed_lines = []
    sentence = []
    for line in lines:
        line = line.strip()
        if not line:
            if sentence:
                processed_lines.append(sentence)
                sentence = []
        else:
            pieces = line.split("\t")
            if pieces[0] == '_':
                sentence.append(' ')
            else:
                sentence.append(pieces[0])
    if sentence:
        processed_lines.append(sentence)

    processed_lines = reprocess_lines(processed_lines)
    paragraphs = convert_processed_lines(processed_lines)
    return paragraphs


def read_data(input_dir, section, resegment, spaces_after, split_clauses):
    glob_path = os.path.join(input_dir, section, "*.txt")
    filenames = glob.glob(glob_path)
    print("  Found {} files in {}".format(len(filenames), glob_path))
    if len(filenames) == 0:
        raise FileNotFoundError("Could not find any files for the {} section.  Is LST20 installed in {}?".format(section, input_dir))
    documents = []
    for filename in filenames:
        with open(filename) as fin:
            lines = fin.readlines()
        if resegment:
            document = retokenize_document(lines)
        else:
            document = read_document(lines, spaces_after, split_clauses)
        documents.extend(document)
    return documents

def add_lst20_args(parser):
    parser.add_argument('--no_lst20_resegment', action='store_false', dest="lst20_resegment", default=True, help='When processing th_lst20 tokenization, use pythainlp to resegment the text.  The other option is to keep the original sentence segmentation.  Currently our model is not good at that')
    parser.add_argument('--lst20_spaces_after', action='store_true', dest="lst20_spaces_after", default=False, help='When processing th_lst20 without pythainlp, put spaces after each sentence.  This better fits the language but gets lower scores for some reason')
    parser.add_argument('--split_clauses', action='store_true', dest="split_clauses", default=False, help='When processing th_lst20 without pythainlp, turn spaces which are labeled as between clauses into sentence splits')

def parse_lst20_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help="Directory to use when processing lst20")
    parser.add_argument('output_dir', help="Directory to use when saving lst20")
    add_lst20_args(parser)
    return parser.parse_args()



def convert(input_dir, output_dir, args):
    input_dir = os.path.join(input_dir, "thai", "LST20_Corpus")
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Could not find LST20 corpus in {}".format(input_dir))

    for (in_section, out_section) in (("train", "train"),
                                      ("eval", "dev"),
                                      ("test", "test")):
        print("Processing %s" % out_section)
        documents = read_data(input_dir, in_section, args.lst20_resegment, args.lst20_spaces_after, args.split_clauses)
        print("  Read in %d documents" % len(documents))
        write_section(output_dir, "lst20", out_section, documents)

def main():
    args = parse_lst20_args()
    convert(args.input_dir, args.output_dir, args)

if __name__ == '__main__':
    main()

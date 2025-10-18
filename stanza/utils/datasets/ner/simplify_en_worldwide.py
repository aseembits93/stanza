import argparse
import os
import tempfile

import stanza
from stanza.utils.default_paths import get_default_paths
from stanza.utils.datasets.ner.utils import read_tsv
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()

PUNCTUATION = """!"#%&'()*+, -./:;<=>?@[\\]^_`{|}~"""
MONEY_WORDS = {"million", "billion", "trillion", "millions", "billions", "trillions", "hundred", "hundreds",
               "lakh", "crore", # south asian english
               "tens", "of", "ten", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "couple"}

# Doesn't include Money but this case is handled explicitly for processing
LABEL_TRANSLATION = {
    "Date":         None,
    "Misc":         "MISC",
    "Product":      "MISC",
    "NORP":         "MISC",
    "Facility":     "LOC",
    "Location":     "LOC",
    "Person":       "PER",
    "Organization": "ORG",
}

def isfloat(num):
    """Optimized float check: skip ValueError path for typical non-numeric strings earlier."""
    # Fast path: Integers and common floats (positive, negative, with '.' or 'e')
    if isinstance(num, (int, float)):
        return True
    if not isinstance(num, str):
        return False
    s = num.strip()
    # Heuristic: Quickly filter out common cases that cannot possibly be a float (e.g. alphabetic tokens)
    # ASCII check (saves exceptions for money words, months, etc.)
    # Early out for empty string
    if not s:
        return False
    # Most tokens are not numbers: if first char isn't digit/./-/+, rule out early
    first = s[0]
    if not (first.isdigit() or first in {'-', '+', '.'}):
        return False
    # Let float() do full number work
    try:
        float(s)
        return True
    except ValueError:
        return False


def process_label(line, is_start=False):
    """
    Converts our stuff to conll labels

    event, product, work of art, norp -> MISC
    take out dates - can use Stanza to identify them as dates and eliminate them
    money requires some special care
    facility -> location  (there are examples of Bridge and Hospital in the data)
    the version of conll we used to train CoreNLP NER is here:

    Overall plan:
    Collapse Product, NORP, Money (extract only the symbols), into misc.
    Collapse Facilities into LOC
    Deletes Dates

    Rule for currency is that we take out labels for the numbers that return True for isfloat()
    Take out words that categorize money (Million, Billion, Trillion, Thousand, Hundred, Ten, Nine, Eight, Seven, Six, Five,
    Four, Three, Two, One)
    Take out punctuation characters

    If we remove the 'B' tag, then move it to the first remaining tag.

    Replace tags with 'O'
    is_start parameter signals whether or not this current line is the new start of a tag. Needed for when
    the previous line analyzed is the start of a MONEY tag but is removed because it is a non symbol- need to
    set the starting token that is a symbol to the B-MONEY tag when it might have previously been I-MONEY
    """
    if not line:
        return []
    token = line[0]
    biggest_label = line[1]
    position, label_name = biggest_label[:2], biggest_label[2:]

    # Fast-path checks for Money
    if label_name == "Money":
        lowertok = token.lower()
        # Inline condition rearrangement: MONEY_WORDS and PUNCTUATION are presumed to have fast lookup (set/str).
        # Check PUNCTUATION (str) first as it's usually shorter than MONEY_WORDS (set), then float.
        if token in PUNCTUATION or lowertok in MONEY_WORDS or isfloat(token):
            label_name = "O"
            is_start = True
            position = ""
        else:
            label_name = "MISC"
            if is_start:
                position = "B-"
                is_start = False

    elif not label_name or label_name == "O":
        pass
    elif label_name in LABEL_TRANSLATION:
        label_name = LABEL_TRANSLATION[label_name]
        if label_name is None:
            position = ""
            label_name = "O"
            is_start = False
    else:
        raise ValueError("Oops, missed a label: %s" % label_name)
    return [token, position + label_name, is_start]


def write_new_file(save_dir, input_path, old_file, simplify):
    starts_b = False
    with open(input_path, "r+", encoding="utf-8") as iob:
        new_filename = (os.path.splitext(old_file)[0] + ".4class.tsv") if simplify else old_file
        with open(os.path.join(save_dir, new_filename), 'w', encoding='utf-8') as fout:
            for i, line in enumerate(iob):
                if i == 0 or i == 1:  # skip over the URL and subsequent space line.
                    continue
                line = line.strip()
                if not line:
                    fout.write("\n")
                    continue
                label = line.split("\t")
                if simplify:
                    try:
                        edited = process_label(label, is_start=starts_b)  # processed label line labels
                    except ValueError as e:
                        raise ValueError("Error in %s at line %d" % (input_path, i)) from e
                    assert edited
                    starts_b = edited[-1]
                    fout.write("\t".join(edited[:-1]))
                    fout.write("\n")
                else:
                    fout.write("%s\t%s\n" % (label[0], label[1]))


def copy_and_simplify(base_path, simplify):
    with tempfile.TemporaryDirectory(dir=base_path) as tempdir:
        # Condense Labels
        input_dir = os.path.join(base_path, "en-worldwide-newswire")
        final_dir = os.path.join(base_path, "4class" if simplify else "9class")
        os.makedirs(tempdir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        for root, dirs, files in os.walk(input_dir):
            if root[-6:] == "REVIEW":
                batch_files = os.listdir(root)
                for filename in batch_files:
                    file_path = os.path.join(root, filename)
                    write_new_file(final_dir, file_path, filename, simplify)

def main(args=None):
    BASE_PATH = "C:\\Users\\SystemAdmin\\PycharmProjects\\General Code\\stanza source code"
    if not os.path.exists(BASE_PATH):
        paths = get_default_paths()
        BASE_PATH = os.path.join(paths["NERBASE"], "en_worldwide")

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default=BASE_PATH, help="Where to find the raw data")
    parser.add_argument('--simplify', default=False, action='store_true', help='Simplify to 4 classes... otherwise, keep all classes')
    parser.add_argument('--no_simplify', dest='simplify', action='store_false', help="Don't simplify to 4 classes")
    args = parser.parse_args(args=args)

    copy_and_simplify(args.base_path, args.simplify)

if __name__ == '__main__':
    main()




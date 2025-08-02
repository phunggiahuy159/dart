import os
import json


def make_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def read_text(path):
    """Read text file and return list of lines"""
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts


def save_text(texts, path):
    """Save list of texts to file"""
    with open(path, 'w', encoding='utf-8') as file:
        for text in texts:
            file.write(text.strip() + '\n')


def read_jsonlist(path):
    """Read JSONL file and return list of dictionaries"""
    data = list()
    with open(path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line))
    return data


def save_jsonlist(list_of_json_objects, path, sort_keys=True):
    """Save list of dictionaries to JSONL file"""
    with open(path, 'w', encoding='utf-8') as output_file:
        for obj in list_of_json_objects:
            output_file.write(json.dumps(obj, sort_keys=sort_keys) + '\n')


def split_text_word(texts):
    """Split texts by words"""
    texts = [text.split() for text in texts]
    return texts
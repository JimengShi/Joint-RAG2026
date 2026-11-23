import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

import collections
import copy
import re
import string



# Download NLTK data for tokenization
nltk.download('punkt')


def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())



def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def f1_score(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


def em_score(gold: str, predicted: str) -> float:
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()

    return 1.0 if gold_tokens == predicted_tokens else 0.0


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def bleu_score(pred, gold):
    """
    It shows word overlap between new answer and ground-truth. It penalizes rephrasing, even if the meaning is correct.
    """
    from nltk.translate.bleu_score import sentence_bleu

    score = sentence_bleu([pred.split()], gold.split())
    return score



def rouge_score(pred, gold):
    """
    ROUGE-L focuses on longest common subsequences, good for long sentences, as it captures structural similarity. But Less sensitive to semantic meaning.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(pred, gold)
    return scores['rougeL'].fmeasure



_semantic_score_model = None


def semantic_score(pred, gold):
    """
    Meaning similarity using contextual embeddings (e.g., from BERT or SentenceTransformers).
    Higher scores mean the new answer is semantically closer to the ground-truth, even if worded differently.
    Captures paraphrasing and meaning better than BLEU/ROUGE.
    """
    global _semantic_score_model
    from sentence_transformers import SentenceTransformer, util

    if _semantic_score_model is None:
        _semantic_score_model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = _semantic_score_model.encode([pred, gold])
    similarity = util.cos_sim(embeddings[0], embeddings[1])

    return similarity.item()




def precision_at_k(retrieved_docs, relevant_docs):
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(retrieved_docs)



def recall_at_k(retrieved_docs, relevant_docs):
    relevant_retrieved = [doc for doc in retrieved_docs if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0






# class Tokens(object):
#     """A class to represent a list of tokenized text."""
#     TEXT = 0
#     TEXT_WS = 1
#     SPAN = 2
#     POS = 3
#     LEMMA = 4
#     NER = 5

#     def __init__(self, data, annotators, opts=None):
#         self.data = data
#         self.annotators = annotators
#         self.opts = opts or {}

#     def __len__(self):
#         """The number of tokens."""
#         return len(self.data)

#     def slice(self, i=None, j=None):
#         """Return a view of the list of tokens from [i, j)."""
#         new_tokens = copy.copy(self)
#         new_tokens.data = self.data[i: j]
#         return new_tokens

#     def untokenize(self):
#         """Returns the original text (with whitespace reinserted)."""
#         return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

#     def words(self, uncased=False):
#         """Returns a list of the text of each token
#         Args:
#             uncased: lower cases text
#         """
#         if uncased:
#             return [t[self.TEXT].lower() for t in self.data]
#         else:
#             return [t[self.TEXT] for t in self.data]

#     def offsets(self):
#         """Returns a list of [start, end) character offsets of each token."""
#         return [t[self.SPAN] for t in self.data]

#     def pos(self):
#         """Returns a list of part-of-speech tags of each token.
#         Returns None if this annotation was not included.
#         """
#         if 'pos' not in self.annotators:
#             return None
#         return [t[self.POS] for t in self.data]

#     def lemmas(self):
#         """Returns a list of the lemmatized text of each token.
#         Returns None if this annotation was not included.
#         """
#         if 'lemma' not in self.annotators:
#             return None
#         return [t[self.LEMMA] for t in self.data]

#     def entities(self):
#         """Returns a list of named-entity-recognition tags of each token.
#         Returns None if this annotation was not included.
#         """
#         if 'ner' not in self.annotators:
#             return None
#         return [t[self.NER] for t in self.data]

#     def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
#         """Returns a list of all ngrams from length 1 to n.
#         Args:
#             n: upper limit of ngram length
#             uncased: lower cases text
#             filter_fn: user function that takes in an ngram list and returns
#             True or False to keep or not keep the ngram
#             as_string: return the ngram as a string vs list
#         """

#         def _skip(gram):
#             if not filter_fn:
#                 return False
#             return filter_fn(gram)

#         words = self.words(uncased)
#         ngrams = [(s, e + 1)
#                 for s in range(len(words))
#                 for e in range(s, min(s + n, len(words)))
#                 if not _skip(words[s:e + 1])]

#         # Concatenate into strings
#         if as_strings:
#             ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

#         return ngrams

#     def entity_groups(self):
#         """Group consecutive entity tokens with the same NER tag."""
#         entities = self.entities()
#         if not entities:
#             return None
#         non_ent = self.opts.get('non_ent', 'O')
#         groups = []
#         idx = 0
#         while idx < len(entities):
#             ner_tag = entities[idx]
#             # Check for entity tag
#             if ner_tag != non_ent:
#                 # Chomp the sequence
#                 start = idx
#                 while (idx < len(entities) and entities[idx] == ner_tag):
#                     idx += 1
#                 groups.append((self.slice(start, idx).untokenize(), ner_tag))
#             else:
#                 idx += 1
#         return groups


# class Tokenizer(object):
#     """Base tokenizer class.
#     Tokenizers implement tokenize, which should return a Tokens class.
#     """

#     def tokenize(self, text):
#         raise NotImplementedError

#     def shutdown(self):
#         pass

#     def __del__(self):
#         self.shutdown()


# class SimpleTokenizer(Tokenizer):
#     ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
#     NON_WS = r'[^\p{Z}\p{C}]'

#     def __init__(self, **kwargs):
#         """
#         Args:
#             annotators: None or empty set (only tokenizes).
#         """
#         self._regexp = regex.compile(
#             '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
#             flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
#         )
#         self.annotators = set()

#     def tokenize(self, text):
#         data = []
#         matches = [m for m in self._regexp.finditer(text)]
#         for i in range(len(matches)):
#             # Get text
#             token = matches[i].group()

#             # Get whitespace
#             span = matches[i].span()
#             start_ws = span[0]
#             if i + 1 < len(matches):
#                 end_ws = matches[i + 1].span()[0]
#             else:
#                 end_ws = span[1]

#             # Format data
#             data.append((
#                 token,
#                 text[start_ws: end_ws],
#                 span,
#             ))
#         return Tokens(data, self.annotators)

# tokenizer = SimpleTokenizer()

# def has_answer(answers, text, match_type="string"):
#     text = unicodedata.normalize('NFD', text)
#     if match_type == 'string':
#         text = tokenizer.tokenize(text).words(uncased=True)
#         for single_answer in answers:
#             single_answer = unicodedata.normalize('NFD', single_answer)
#             single_answer = tokenizer.tokenize(single_answer)
#             single_answer = single_answer.words(uncased=True)
#             for i in range(0, len(text) - len(single_answer) + 1):
#                 if single_answer == text[i: i + len(single_answer)]:
#                     return 1
#     return 0

# def _normalize_answer(s):
#     def remove_articles(text):
#         return re.sub(r"\b(a|an|the)\b", " ", text)

#     def white_space_fix(text):
#         return " ".join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))

# def EM_compute(answer_list, prediction):
#     return max([int(_normalize_answer(prediction) == _normalize_answer(ground_truth)) for ground_truth in answer_list])



# def f1_score_v2(answers, pred):
#     """
#     https://github.com/RUCAIBox/REAR/blob/master/rear/src/utils.py
#     """
#     def get_tokens(s):
#         if not s: return []
#         return _normalize_answer(s).split()

#     def compute_f1(a_gold, a_pred):
#         gold_toks = get_tokens(a_gold)
#         pred_toks = get_tokens(a_pred)
#         common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#         num_same = sum(common.values())
#         if len(gold_toks) == 0 or len(pred_toks) == 0:
#             return int(gold_toks == pred_toks)
#         if num_same == 0:
#             return 0
#         precision = 1.0 * num_same / len(pred_toks)
#         recall = 1.0 * num_same / len(gold_toks)
#         f1 = (2 * precision * recall) / (precision + recall)
#         return f1
#     return max([compute_f1(x, pred) for x in answers])



# def f1_score_v3(pred, gold):
#     pred_tokens = word_tokenize(pred.lower())
#     gold_tokens = word_tokenize(gold.lower())
#     common = set(pred_tokens) & set(gold_tokens)
#     if len(common) == 0:
#         return 0.0
#     precision = len(common) / len(pred_tokens)
#     recall = len(common) / len(gold_tokens)
#     return 2 * precision * recall / (precision + recall)

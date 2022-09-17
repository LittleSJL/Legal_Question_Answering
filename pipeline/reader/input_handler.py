import collections

import six
import json
import tensorflow as tf
import os
import sys
path_cwd = os.getcwd()
path_tools = os.path.join(path_cwd, 'pipeline')
sys.path.append(path_tools)


from reader import tokenization
from sklearn.model_selection import train_test_split


def generate_data_id(data_selection, human_selection=True, meaningless_selection=False, human_only=False):
    """
    generate question id set for different purposes

        data_selection: {train, test}
            train: generate question id used for training
            test：generate question id used for testing
        human_selection:
            if adding human questions into training set
        meaningless_selection:
            if adding meaningless QG question into training set
        human_only:
            if only need human questions in the training set
    """
    with open('data/PGT/data_token.json') as f:
        data = json.load(f)

    human_id = []  # 326
    qg_meaningful_id = []  # 211
    qg_meaningless_id = []  # 411
    for section in data:
        for paragraph in section['paragraphs']:
            for qas in paragraph['qas']:
                for qa in qas:
                    if qa['type']['original_type'] == 'Human' and qa['type']['reformulation_type'] == 'Original':
                        human_id.append(qa['question_id'])
                    if qa['type']['original_type'] == 'QG' and qa['type']['meaning_type'] == 'Meaningful' and \
                            qa['type']['reformulation_type'] == 'Original':
                        qg_meaningful_id.append(qa['question_id'])
                    if qa['type']['original_type'] == 'QG' and qa['type']['meaning_type'] == 'Meaningless' and \
                            qa['type']['reformulation_type'] == 'Original':
                        qg_meaningless_id.append(qa['question_id'])

    # test set is always the same, only sample from human questions
    test, left_for_train = train_test_split(human_id, test_size=0.3, random_state=2022)  # 228, 98

    # generate different training sets
    train = None
    if meaningless_selection and human_selection:  # QG + Human
        train = left_for_train + qg_meaningful_id + qg_meaningless_id  # 309 + 411 = 720
    if meaningless_selection and not human_selection:  # QG
        train = qg_meaningful_id + qg_meaningless_id  # 211 + 411 = 622
    if not meaningless_selection and human_selection:  # QG-meaningful + Human
        train = left_for_train + qg_meaningful_id  # 98 + 211 = 309
    if not meaningless_selection and not human_selection:  # QG-meaningful
        train = qg_meaningful_id  # 211
    if human_only:  # only human
        train = left_for_train

    if data_selection == 'test':
        return test
    else:
        return train


def read_test_pipeline(refor_selection=False):
    """
    generate test examples to test the whole pipeline: the reader will read the paragraphs retrieved by retriever
        need qa_id, question_text, doc_tokens, original_answer

    retriever_prediction:
        the retriever prediction {qa_id: [top-5 retrieved paragraphs]}
    id2para:
        mapping paragraph id to its corresponding context
    qid_2_text_answer:
        mapping question id to its question text and answer
    """
    with open('model/retriever/retriever_prediction.json') as f:
        retriever_prediction = json.load(f)  # {qa_id: [top-5 passage], ...}
    with open('model/reader/id2para.json') as f:
        id2para = json.load(f)  # {para_id: para_context, ...}
    with open('model/reader/qid_2_text_answer.json') as f:
        qid_2_text_answer = json.load(f)  # {qid: {question_text, answer}, ...}

    examples = []
    question_id = generate_data_id('test')  # question id for test set (only original question)

    if refor_selection:  # if choose to include reformulation questions, then we need to expand the qid list
        new_question_id = []
        for qid in question_id:
            new_question_id.append(qid)
            new_question_id.append(qid[:-8] + 'Paraphrase')  # add paraphrase version
            new_question_id.append(qid[:-8] + 'Noisy')  # add noisy version
        question_id = new_question_id

    for qa_id in question_id:
        retrieved_para_id_list = retriever_prediction.get(qa_id)  # for each q, get the retriever prediction
        for para_id in retrieved_para_id_list:  # 1 question - top-5 paragraphs - 5 test examples
            def is_whitespace(c):
                if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                    return True
                return False

            para_context = id2para.get(para_id)  # mapping from id to context
            doc_tokens = []  # pre-process it to doc_tokens
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in para_context:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            example = SquadExample(
                qas_id=qa_id,
                question_text=qid_2_text_answer.get(qa_id).get('question_text'),  # mapping qid to question text
                doc_tokens=doc_tokens,
                orig_answer_text=qid_2_text_answer.get(qa_id).get('answer_text'),  # mapping qid to answer text
                start_position=None,
                end_position=None)
            examples.append(example)

    return examples


class SquadExample(object):
    """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
        s += ", question_text: %s" % (
            tokenization.printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_data_reader(is_training, human_selection=True, meaningless_selection=False, refor_selection=True,
                     human_only=False):
    """
    generate data samples for reader training and testing

        is_training:
            True: generate training examples
            False: generate testing examples
        human_selection:
            if adding human questions into training set
        meaningless_selection:
            if adding meaningless QG question into training set
        refor_selection:
            if adding reformulation questions into data examples
        human_only:
            if only need human questions in the training set
    """
    if is_training:
        question_id = generate_data_id('train', human_selection, meaningless_selection, human_only=human_only)
    else:
        question_id = generate_data_id('test')

    if not is_training:
        path = 'data/PGT/data_token.json'
    else:
        if meaningless_selection:
            path = 'data/PGT/data_sentence.json'
        else:
            path = 'data/PGT/data_token.json'

    with tf.gfile.Open(path, "r") as reader:
        input_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            def create_example(qa):
                qas_id = qa["question_id"]
                question_text = qa["question"]

                orig_answer_text = qa['answer_text']
                answer_offset = qa["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset + answer_length -
                                                   1]
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position)
                return example

            for qas in paragraph["qas"]:
                if refor_selection:  # choose to add reformulation questions
                    # if original question is sampled, then all its reformulation questions are sampled as well
                    if qas[0]['question_id'] in question_id:
                        for qa in qas:
                            examples.append(create_example(qa))
                else:  # choose not to add reformulation questions
                    # only original questions are sampled
                    for qa in qas:
                        if qa['question_id'] in question_id:
                            examples.append(create_example(qa))

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 20:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (unique_id))
                tf.logging.info("example_index: %s" % (example_index))
                tf.logging.info("doc_span_index: %s" % (doc_span_index))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                tf.logging.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                ]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    tf.logging.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    tf.logging.info("start_position: %d" % (start_position))
                    tf.logging.info("end_position: %d" % (end_position))
                    tf.logging.info(
                        "answer: %s" % (tokenization.printable_text(answer_text)))

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

            # Run callback
            output_fn(feature)

            unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

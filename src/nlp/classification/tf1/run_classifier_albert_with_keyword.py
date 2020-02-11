# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-09-02
Description :
auther : wcy
"""
# import modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import collections
import csv
from albert_zh import modeling
import albert_zh.optimization_finetuning as optimization
from albert_zh import tokenization
import tensorflow as tf
from metrics import get_metrics_ops, get_metrics

curr_path = os.getcwd()
sys.path.append(curr_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

__all__ = []

# config
BERT_PATH = "/home/ran_wang_math/ccf_data/pretrain_model/albert_base_zh"
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data_dir", os.path.join(curr_path, "data", "ccf_data_zr_deal", "train_1.csv"), None)
flags.DEFINE_string("test_data_dir", os.path.join(curr_path, "data", "ccf_data_zr_deal", "dev_1.csv"), None)
flags.DEFINE_string("dev_data_dir", os.path.join(curr_path, "data", "ccf_data_zr_deal", "dev_1.csv"), None)
flags.DEFINE_string("model_dir", None, None)
flags.DEFINE_bool("eval_all_ckpt", True, None)

# data repeat
flags.DEFINE_integer("data_repeat", None, None)

# loss
flags.DEFINE_integer("label_num", 3, None)

flags.DEFINE_string("bert_config_file", os.path.join(BERT_PATH, "bert_config.json"), None)
flags.DEFINE_string("vocab_file", os.path.join(BERT_PATH, "vocab.txt"), None)

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint", "gs://ccf_cp/ccf/pretrain_model/ernie1.0/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 64, None)

flags.DEFINE_bool("do_train", None, None)
flags.DEFINE_bool("do_eval", None, None)
flags.DEFINE_bool("do_predict", None, None)
flags.DEFINE_string("predict_ckpt", None, None)
flags.DEFINE_string("predict_file_write_path", None, None)

flags.DEFINE_integer("train_batch_size", None, None)
flags.DEFINE_integer("eval_batch_size", 128, None)
flags.DEFINE_integer("predict_batch_size", 128, None)
flags.DEFINE_integer("save_checkpoints_steps", None, None)
flags.DEFINE_integer("iterations_per_loop", None, None)

flags.DEFINE_float("learning_rate", None, None)
flags.DEFINE_float("weight_decay_rate", 0.01, None)
flags.DEFINE_float("num_train_epochs", None, None)
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("use_tpu", True, None)
tf.flags.DEFINE_string("tpu_name", "a-wcy-train-bert", None)
tf.flags.DEFINE_string("tpu_zone", "us-central1-b", None)
tf.flags.DEFINE_string("gcp_project", None, None)
tf.flags.DEFINE_string("master", None, None)
flags.DEFINE_integer("num_tpu_cores", 8, None)


# define class
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class ccfKeyProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])

            # keyword
            text_c = tokenization.convert_to_unicode(line[4])

            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    text_c=text_c,
                    label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])

            text_c = tokenization.convert_to_unicode(line[4])

            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    text_c=text_c,
                    label=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])

            text_c = tokenization.convert_to_unicode(line[3])

            label = tokenization.convert_to_unicode("0")
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    text_c=text_c,
                    label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]


# define function
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    tokens_c = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if example.text_c != '':
        tx = example.text_c.split("_")

        tokens_c = []
        for t in tx:
            tokens_c.extend(tokenizer.tokenize(t))
            tokens_c.append('[SEP]')
        tokens_c = tokens_c[:-1]

    if tokens_b and tokens_c:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_three_seq(tokens_a, tokens_b, tokens_c, max_seq_length - 4)

    elif tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_c:
        for token in tokens_c:
            tokens.append(token)
            segment_ids.append(1)

        # todo 判断
        if tokens[-1] != "[SEP]":
            tokens.append("[SEP]")
            segment_ids.append(1)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
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

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(
                "Writing example %d of %d" %
                (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_b) > 0:
            tokens_b.pop()
        else:
            tokens_a.pop()


def _truncate_three_seq(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_b) > 0:
            tokens_b.pop()
        elif len(tokens_c) > 0:
            tokens_c.pop()
        else:
            tokens_a.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        ln_type = bert_config.ln_type
        if ln_type == 'preln':  # add by brightmart, 10-06. if it is preln, we need to an additonal layer: layer normalization as suggested in paper "ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE"
            print("ln_type is preln. add LN layer.")
            output_layer = layer_norm(output_layer)
        else:
            print("ln_type is postln or other,do nothing.")

        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # todo 08-29 try temp-loss
        ###############bi_tempered_logistic_loss############################################################################
        # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
        # t1=0.9 #t1=0.90
        # t2=1.05 #t2=1.05
        # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
        # tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
        ##############bi_tempered_logistic_loss#############################################################################

        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(
                "  name = %s, shape = %s" %
                (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(
                features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(
                    per_example_loss,
                    label_ids,
                    logits,
                    is_real_example):

                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

                conf_mat = get_metrics_ops(label_ids, predictions, 3, is_real_example)

                accuracy = tf.metrics.accuracy(
                    labels=label_ids,
                    predictions=predictions,
                    weights=is_real_example)

                loss = tf.metrics.mean(
                    values=per_example_loss,
                    weights=is_real_example)

                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                    "cf": conf_mat
                }

            eval_metrics = (
                metric_fn, [
                    per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with
        # TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info(
                "Writing example %d of %d" %
                (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = ccfKeyProcessor()

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.model_dir)

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.model_dir,
        keep_checkpoint_max=10000,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.train_data_dir)
        num_train_steps = int(
            len(train_examples) /
            FLAGS.train_batch_size *
            FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.dev_data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        steps_and_files = []
        filenames = tf.gfile.ListDirectory(FLAGS.model_dir)
        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = os.path.join(FLAGS.model_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

        result_list = list()
        for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)
            result_list.append([global_step, result])

        for step, result in result_list:
            tf.logging.info("\n\n------ step ------" + str(step))
            pre, rec, f1 = get_metrics(result["cf"], 3)
            tf.logging.info("eval_precision: {}".format(pre))
            tf.logging.info("eval_recall: {}".format(rec))
            tf.logging.info("eval_f1: {}".format(f1))
            tf.logging.info("eval_accuracy: {}".format(result["eval_accuracy"]))
            tf.logging.info("eval_loss: {}".format(result["eval_loss"]))
            tf.logging.info("-------------------------\n\n")

    if FLAGS.do_predict:

        """
        在进行predict 之前首先使用ckpt 对应的dev 数据进行eval, 防止模型使用错误
        """
        eval_examples = processor.get_dev_examples(FLAGS.dev_data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=FLAGS.predict_ckpt)
        pre, rec, f1 = get_metrics(result["cf"], FLAGS.label_num)
        tf.logging.info("eval_precision: {}".format(pre))
        tf.logging.info("eval_recall: {}".format(rec))
        tf.logging.info("eval_f1: {}".format(f1))
        tf.logging.info("eval_accuracy: {}".format(result["eval_accuracy"]))
        tf.logging.info("eval_loss: {}".format(result["eval_loss"]))
        tf.logging.info("-------------------------\n\n")

        # 临时添加
        # with tf.gfile.GFile(FLAGS.predict_file_write_path + "_dev", "w") as writer:
        #     num_written_lines = 0
        #     tf.logging.info("***** dev results *****")
        #     for (i, prediction) in enumerate(result):
        #         probabilities = prediction["probabilities"]
        #         if i >= num_actual_eval_examples:
        #             break
        #         output_line = "\t".join(
        #             str(class_probability)
        #             for class_probability in probabilities) + "\n"
        #         writer.write(output_line)
        #         num_written_lines += 1
        # assert num_written_lines == num_actual_eval_examples

        """
        进行模型的predict
        """
        predict_examples = processor.get_test_examples(FLAGS.test_data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(
            predict_examples,
            label_list,
            FLAGS.max_seq_length,
            tokenizer,
            predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn, checkpoint_path=FLAGS.predict_ckpt)

        with tf.gfile.GFile(FLAGS.predict_file_write_path, "w") as writer:
            num_written_lines = 0

            tf.logging.info("***** Predict results *****")

            for (i, prediction) in enumerate(result):

                # 因为在predict的时候返回acc一直报错，所以将这行代码注释
                # tf.logging.info("***** acc *****:" + str(prediction["accuracy"]))
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


# main
if __name__ == "__main__":
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("do_train")
    flags.mark_flag_as_required("do_eval")
    flags.mark_flag_as_required("do_predict")
    flags.mark_flag_as_required("train_batch_size")
    flags.mark_flag_as_required("save_checkpoints_steps")
    flags.mark_flag_as_required("iterations_per_loop")
    flags.mark_flag_as_required("learning_rate")
    flags.mark_flag_as_required("num_train_epochs")

    tf.app.run()

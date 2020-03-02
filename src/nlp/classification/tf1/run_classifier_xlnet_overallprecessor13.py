# !/user/bin/python
# -*- coding:utf-8 -*-
"""
date：          2019-09-02
Description : run xlnet
auther : wcy
"""
# import modules
import os, sys

from os.path import join
from absl import flags
import csv
from pandas.core.frame import DataFrame
import collections
import numpy as np
from metrics import get_metrics, get_metrics_ops

import tensorflow as tf

import sentencepiece as spm

from xlnet import model_utils
from xlnet import function_builder
from xlnet.classifier_utils import PaddingInputExample
from xlnet.classifier_utils import convert_single_example
from xlnet.prepro_utils import preprocess_text, encode_ids

curr_path = os.getcwd()
sys.path.append(curr_path)

__all__ = []

# config
flags.DEFINE_string("train_data", default=os.path.join(curr_path, "data", "ccf_data_zr_deal", "train_1.csv"), help=None)
flags.DEFINE_string("dev_data", default=os.path.join(curr_path, "data", "ccf_data_zr_deal", "dev_1.csv"), help=None)
flags.DEFINE_string("test_data", default=os.path.join(curr_path, "data", "ccf_data_zr_deal", "dev_1.csv"), help=None)
flags.DEFINE_string("task_name", default="ccf", help=None)

#flags.DEFINE_string("predict_dir",
#                    default="gs://ccf_cp/predict_test_xlnet",
#                   help="Dir for saving prediction files.")

flags.DEFINE_string("predict_dir",
                    #default="gs://checkpoints_for_pretrained_lm",
                    default="I:/xlnet",
                    help="Dir for saving prediction files.")


flags.DEFINE_bool("if_predict_train", False, None)


flags.DEFINE_integer("label_num", 2, None)
flags.DEFINE_string("predict_file_write_path", None, None)

#flags.DEFINE_string("model_config_path",
#                    "gs://ccf_cp/ccf/pretrain_model/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json",
#                    help=None)
flags.DEFINE_string("model_config_path",
                    #"gs://checkpoints_for_pretrained_lm/xlnet_config.json",
                    #default="I:/xlnet/xlnet_config.json",
                    default="I:/xlnet",
                    help=None)               
                 

flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_string(
    "summary_type",
    default="last",
    help="Method used to summarize a sequence into a compact vector.")

flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,
                  help="Whether to use bfloat16.")

# data repeat
flags.DEFINE_integer("data_repeat", None, None)

# loss
flags.DEFINE_bool("use_multi_loss", False, None)
flags.DEFINE_float("ce_loss_weight", 0.3, None)
flags.DEFINE_float("fl_loss_weight", 0.3, None)
flags.DEFINE_float("fl_loss_gama", 0.15, None)
flags.DEFINE_float("hg_loss_weight", 0.3, None)

flags.DEFINE_bool("use_label_smoothing", False, None)
flags.DEFINE_float("label_smoothing_weight", 0.2, None)

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")

flags.DEFINE_string("init_checkpoint",
                    #default="gs://checkpoints_for_pretrained_lm/xlnet_model.ckpt",
                    #default="I:/xlnet/xlnet_model.ckpt",
                    default="I:/xlnet",
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")

flags.DEFINE_string("output_dir",
                    #default="gs://checkpoints_for_pretrained_lm/output_xlnet",
                    default="I:/xlnet",
                    help="Output dir for TF records.")

flags.DEFINE_string("data_dir", default="",
                    help="Directory for input data.")

flags.DEFINE_string("spiece_model_file",
                    #default="/home/ran_wang_math/ccf_data/pretrain_model/chinese_xlnet_mid_L-24_H-768_A-12/spiece.model",
                    #default="I:/xlnet/spiece.model",
                    default="I:/xlnet",
                    help="Sentence Piece model path.")

flags.DEFINE_string("model_dir",
                    #default="gs://checkpoints_for_pretrained_lm/model_dir_xlnet",
                    default="I:/xlnet",
                    help="Directory for saving the finetuned model.")

# training
flags.DEFINE_bool("do_train", default=True, help="whether to do training")

flags.DEFINE_integer("train_steps", default=100,
                     help="Number of training steps")
flags.DEFINE_integer("save_steps", default=100,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("iterations", default=100,
                     help="number of iterations per TPU training loop.")

flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=10000, help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("train_batch_size", default=32,
                     help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")

flags.DEFINE_bool("eval_only_one", default=False, help="whether to do eval")


flags.DEFINE_string("eval_ckpt", default=None, help="whether to do eval")


flags.DEFINE_bool("do_predict", default=True, help="whether to do prediction")
flags.DEFINE_float("predict_threshold", default=0,
                   help="Threshold for binary prediction.")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=32,
                     help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=32,
                     help="batch size for prediction.")
flags.DEFINE_bool("eval_all_ckpt", default=True,
                  help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string(
    "predict_ckpt",
    default=None,
    help="Ckpt path for do_predict. If None, use the last one.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# task specific
flags.DEFINE_integer("max_seq_length", default=512, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1,
                     help="Num passes for processing training data. "
                          "This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None,
                    help="Classifier layer scope.")
flags.DEFINE_bool("is_regression", default=False,
                  help="Whether it's a regression task.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=True, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer(
    "num_core_per_host",
    default=8,
    help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
         "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default="v3-8", help="TPU name.")
flags.DEFINE_string("tpu_zone", default="us-central1-b", help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")

FLAGS = flags.FLAGS


# define class
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


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
                if len(line) == 0:
                    continue
                lines.append(line)
            return lines


class OverallProcessor(DataProcessor):
    def __init__(self, use_test_b = False):
       self.use_text_b = use_test_b

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            label = line[1]
            text_a = line[2]
            if self.use_text_b is False:
                text_b = None
            else:
                text_b = line[3]
            #text_b = line[3]
            examples.append(
                InputExample(
                    guid=guid,
                    label=label,
                    text_a=text_a,
                    text_b=text_b))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            text_a = line[2]
            if self.use_text_b is False:
                text_b = None
            else:
                text_b = line[3]
            label = line[1]
            examples.append(
                InputExample(
                    guid=guid,
                    label=label,
                    text_a=text_a,
                    text_b=text_b))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            if FLAGS.if_predict_train:
                text_a = line[2]
                if self.use_text_b is False:
                    text_b = None
                else:
                    text_b = line[3]
                label = line[1]
            else:
                text_a = line[1]
                if self.use_text_b is False:
                    text_b = None
                else:
                    text_b = line[3]
                label = self.get_labels()[0]
            examples.append(
                InputExample(
                    guid=guid,
                    label=label,
                    text_a=text_a,
                    text_b=text_b))
        return examples

    def get_labels(self):
        """See base class."""
        # train_example = DataFrame(self.get_train_examples(FLAGS.data_dir))
        # return [str(i) for i in list(train_example.iloc[:,1].unique())]
        return ["0","1"]


# define function
def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenize_fn, output_file,
        num_passes=1):
    """Convert a set of `InputExample`s to a TFRecord file."""

    # do not create duplicated records
    if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
        tf.logging.info(
            "Do not overwrite tfrecord {} exists.".format(output_file))
        return

    tf.logging.info("Create new tfrecord {}.".format(output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    if num_passes > 1:
        examples *= num_passes

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example {} of {}".format(ex_index,
                                                              len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenize_fn)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(
                    value=list(values)))
            return f

        def create_float_feature(values):
            f = tf.train.Feature(
                float_list=tf.train.FloatList(
                    value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_float_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        if label_list is not None:
            features["label_ids"] = create_int_feature([feature.label_id])
        else:
            features["label_ids"] = create_float_feature(
                [float(feature.label_id)])
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
        "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    if FLAGS.is_regression:
        name_to_features["label_ids"] = tf.FixedLenFeature([], tf.float32)

    tf.logging.info("Input tfrecord file {}".format(input_file))

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params, input_context=None):
        """The actual input function."""
        if FLAGS.use_tpu:
            batch_size = params["batch_size"]
        elif is_training:
            batch_size = FLAGS.train_batch_size
        elif FLAGS.do_eval:
            batch_size = FLAGS.eval_batch_size
        else:
            batch_size = FLAGS.predict_batch_size

        d = tf.data.TFRecordDataset(input_file)
        # Shard the dataset to difference devices
        if input_context is not None:
            tf.logging.info(
                "Input pipeline id %d out of %d",
                input_context.input_pipeline_id,
                input_context.num_replicas_in_sync)
            d = d.shard(input_context.num_input_pipelines,
                        input_context.input_pipeline_id)

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
            d = d.repeat()

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def get_model_fn(n_class):
    def model_fn(features, labels, mode, params):
        #### Training or Evaluation
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Get loss from inputs
        if FLAGS.is_regression:
            (total_loss, per_example_loss, logits
             ) = function_builder.get_regression_loss(FLAGS, features, is_training)
        else:
            (total_loss, per_example_loss, logits, p
             ) = function_builder.get_classification_loss(
                FLAGS, features, n_class, is_training)

        # Check model parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # load pretrained models
        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        # Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            assert FLAGS.num_hosts == 1

            def metric_fn(
                    per_example_loss,
                    label_ids,
                    logits,
                    is_real_example, probabilities):

                if FLAGS.data_repeat:
                    raw_shape = [-1, FLAGS.data_repeat, probabilities.shape[-1].value]

                    probabilities = tf.reduce_mean(tf.reshape(probabilities, shape=raw_shape), -2)
                    predictions = tf.argmax(probabilities, axis=-1, output_type=tf.int32)

                    raw_shape = [-1, FLAGS.data_repeat]
                    label_ids = tf.reduce_mean(tf.reshape(label_ids, shape=raw_shape), -1)
                    label_ids = tf.cast(label_ids, dtype=tf.int32)

                    is_real_example_a = tf.reduce_mean(tf.reshape(is_real_example, shape=raw_shape), -1)
                    is_real_example_a = tf.cast(is_real_example_a, dtype=tf.int32)

                    conf_mat = get_metrics_ops(label_ids, predictions, 3, is_real_example_a)

                    accuracy = tf.metrics.accuracy(
                        labels=label_ids,
                        predictions=predictions,
                        weights=is_real_example_a)
                else:
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
                    'eval_accuracy': accuracy,
                    'eval_loss': loss,
                    "conf_mat": conf_mat}

            def regression_metric_fn(
                    per_example_loss, label_ids, logits, is_real_example):
                loss = tf.metrics.mean(
                    values=per_example_loss,
                    weights=is_real_example)
                pearsonr = tf.contrib.metrics.streaming_pearson_correlation(
                    logits, label_ids, weights=is_real_example)

                return {'eval_loss': loss, 'eval_pearsonr': pearsonr}

            is_real_example = tf.cast(
                features["is_real_example"], dtype=tf.float32)

            # Constucting evaluation TPUEstimatorSpec with new cache.
            label_ids = tf.reshape(features['label_ids'], [-1])

            if FLAGS.is_regression:
                metric_fn = regression_metric_fn
            else:
                metric_fn = metric_fn

            """
            注意: 只对 is_regression = False 适用， is_regression = True 时是没有 p 这个变量的
            """
            metric_args = [
                per_example_loss,
                label_ids,
                logits,
                is_real_example, p]

            if FLAGS.use_tpu:
                eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=(metric_fn, metric_args),
                    scaffold_fn=scaffold_fn)
            else:
                eval_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metric_fn(*metric_args))

            return eval_spec

        elif mode == tf.estimator.ModeKeys.PREDICT:
            label_ids = tf.reshape(features["label_ids"], [-1])

            if FLAGS.data_repeat:
                raw_shape = [-1, FLAGS.data_repeat, p.shape[-1].value]
                p = tf.reduce_mean(tf.reshape(p, shape=raw_shape), -2)

            # change
            # predictions = {
            #     "logits": logits,
            #     "labels": label_ids,
            #     "is_real": features["is_real_example"],
            #     "p": p,
            # }   -->   predictions={"p": p}
            # 很多返回的东西都没有用
            predictions = {"p": p}

            if FLAGS.use_tpu:
                output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
            else:
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predictions)
            return output_spec

        # Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(
            FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        # Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            # Creating host calls
            if not FLAGS.is_regression:
                label_ids = tf.reshape(features['label_ids'], [-1])
                predictions = tf.argmax(
                    logits, axis=-1, output_type=label_ids.dtype)
                is_correct = tf.equal(predictions, label_ids)
                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

                monitor_dict["accuracy"] = accuracy

                host_call = function_builder.construct_scalar_host_call(
                    monitor_dict=monitor_dict,
                    model_dir=FLAGS.model_dir,
                    prefix="train/",
                    reduce_fn=tf.reduce_mean)
            else:
                host_call = None

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                host_call=host_call,
                scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if FLAGS.do_predict:
        predict_dir = FLAGS.predict_dir
        if not tf.gfile.Exists(predict_dir):
            tf.gfile.MakeDirs(predict_dir)

    processor = OverallProcessor()

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval, `do_predict` or "
            "`do_submit` must be True.")

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    label_list = processor.get_labels() if not FLAGS.is_regression else None

    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        return encode_ids(sp, text)

    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn(
        len(label_list) if label_list is not None else None)

    spm_basename = os.path.basename(FLAGS.spiece_model_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            predict_batch_size=FLAGS.predict_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)

    if FLAGS.do_train:
        train_file_base = "{}.len-{}.train.tf_record".format(
            spm_basename, FLAGS.max_seq_length)
        train_file = os.path.join(FLAGS.output_dir, train_file_base)
        tf.logging.info("Use tfrecord file {}".format(train_file))

        train_examples = processor.get_train_examples(FLAGS.train_data)
        np.random.shuffle(train_examples)
        tf.logging.info("Num of train samples: {}".format(len(train_examples)))

        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            train_file, FLAGS.num_passes)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.dev_data)
        tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

        while len(eval_examples) % FLAGS.eval_batch_size != 0:
            eval_examples.append(PaddingInputExample())

        eval_file_base = "{}.len-{}.{}.eval.tf_record".format(
            spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            eval_file)

        assert len(eval_examples) % FLAGS.eval_batch_size == 0
        eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        if FLAGS.eval_only_one:
            result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=FLAGS.eval_ckpt)
            pre, rec, f1 = get_metrics(result["conf_mat"], 3)
            tf.logging.info("eval ckpt" + FLAGS.eval_ckpt)
            tf.logging.info("eval_precision: {}".format(pre))
            tf.logging.info("eval_recall: {}".format(rec))
            tf.logging.info("eval_f1: {}".format(f1))
            tf.logging.info("eval_accuracy: {}".format(result["eval_accuracy"]))
            tf.logging.info("eval_loss: {}".format(result["eval_loss"]))
            tf.logging.info("-------------------------\n\n")

        else:
            # Filter out all checkpoints in the directory
            steps_and_files = []
            filenames = tf.gfile.ListDirectory(FLAGS.model_dir)

            for filename in filenames:
                if filename.endswith(".index"):
                    ckpt_name = filename[:-6]
                    cur_filename = join(FLAGS.model_dir, ckpt_name)
                    global_step = int(cur_filename.split("-")[-1])
                    tf.logging.info("Add {} to eval list.".format(cur_filename))
                    steps_and_files.append([global_step, cur_filename])
            steps_and_files = sorted(steps_and_files, key=lambda x: x[0])

            # Decide whether to evaluate all ckpts
            if not FLAGS.eval_all_ckpt:
                steps_and_files = steps_and_files[-1:]

            result_list = list()
            for global_step, filename in sorted(
                    steps_and_files, key=lambda x: x[0]):
                result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)
                result_list.append([global_step, result])

            for step, result in result_list:
                tf.logging.info("\n\n------ step -------" + str(step))
                pre, rec, f1 = get_metrics(result["conf_mat"], 3)
                tf.logging.info("eval_precision: {}".format(pre))
                tf.logging.info("eval_recall: {}".format(rec))
                tf.logging.info("eval_f1: {}".format(f1))
                tf.logging.info("eval_accuracy: {}".format(result["eval_accuracy"]))
                tf.logging.info("eval_loss: {}".format(result["eval_loss"]))
                tf.logging.info("-------------------------\n\n")

    if FLAGS.do_predict:

        """
        首先对使用的ckpt进行eval，防止加载错模型
        """
        eval_examples = processor.get_dev_examples(FLAGS.dev_data)
        tf.logging.info("Num of eval samples: {}".format(len(eval_examples)))

        while len(eval_examples) % FLAGS.eval_batch_size != 0:
            eval_examples.append(PaddingInputExample())

        eval_file_base = "{}.len-{}.{}.eval.tf_record".format(
            spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            eval_file)

        assert len(eval_examples) % FLAGS.eval_batch_size == 0
        eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=True)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=FLAGS.predict_ckpt)
        pre, rec, f1 = get_metrics(result["conf_mat"], FLAGS.label_num)

        tf.logging.info("\n\n\n加载的模型的效果为：")
        tf.logging.info("eval_precision: {}".format(pre))
        tf.logging.info("eval_recall: {}".format(rec))
        tf.logging.info("eval_f1: {}".format(f1))
        tf.logging.info("eval_accuracy: {}".format(result["eval_accuracy"]))
        tf.logging.info("eval_loss: {}".format(result["eval_loss"]))
        tf.logging.info("-------------------------\n\n\n")

        """
        做完eval后，进行模型的predict
        """
        eval_examples = processor.get_test_examples(FLAGS.test_data)

        num_actual_predict_examples = len(eval_examples)

        if FLAGS.data_repeat:
            num_actual_predict_examples /= FLAGS.data_repeat

        eval_file_base = "{}.len-{}.{}.predict.tf_record".format(
            spm_basename, FLAGS.max_seq_length, FLAGS.eval_split)
        eval_file = os.path.join(FLAGS.output_dir, eval_file_base)

        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,
            eval_file)

        pred_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # Decide whether to evaluate all ckpts
        result = estimator.predict(input_fn=pred_input_fn, checkpoint_path=FLAGS.predict_ckpt)
        with tf.gfile.GFile(FLAGS.predict_file_write_path, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                p = prediction["p"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in p) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

        tf.logging.info("模型预测完成。。。。")


# main
if __name__ == '__main__':
    tf.app.run()
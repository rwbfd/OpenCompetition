#  coding = 'utf-8'
import csv
import os

import tensorflow as tf

from .common import InputExample


class DataProcessorBase(object):
    """Base class for data converters for sequence classification data sets."""

    def __init__(self, labels):
        self.labels = labels

    def get_train_examples(self, data_dir, curr_path, tokenization):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, curr_path, tokenization):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, curr_path, tokenization):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.labels

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return line


class DataProcessor(DataProcessorBase):

    def __init__(self, labels):
        super().__init__(labels)

    def get_train_examples(self, data_dir, curr_path, tokenization):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label))
        return examples

    def get_dev_examples(self, data_dir, curr_path, tokenization):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label))
        return examples

    def get_test_examples(self, data_dir, curr_path, tokenization):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)

            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode("0")
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label))
        return examples


class KeyProcessor(DataProcessor):
    def get_train_examples(self, data_dir, curr_path, tokenization):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label))
        return examples

    def get_dev_examples(self, data_dir, curr_path, tokenization):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label))
        return examples

    def get_test_examples(self, data_dir, curr_path, tokenization):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode("0")
            examples.append(
                InputExample(
                    guid=guid,
                    text_a=text_a,
                    text_b=text_b,
                    label=label))
        return examples

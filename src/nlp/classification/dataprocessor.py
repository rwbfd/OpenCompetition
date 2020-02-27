class OverallProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(curr_path, data_dir))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            #guid = tokenization.convert_to_unicode(line[0])
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
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
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            label = tokenization.convert_to_unicode(line[1])
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
                id = tokenization.convert_to_unicode(line[0])
                text_a = tokenization.convert_to_unicode(line[2])
                text_b = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            else:
                text_a = tokenization.convert_to_unicode(line[1])
                text_b = tokenization.convert_to_unicode(line[2])
                label = tokenization.convert_to_unicode("0")
            examples.append(
                InputExample(
                    guid=guid,
                    label=label,
                    text_a=text_a,
                    text_b=text_b))
        return examples

    def get_labels(self):
        """See base class."""
        train_example = DataFrame(self.get_train_examples())
        return list(train_example.iloc[:,1].unique())


import torchtext


class FinetuneDataset(torchtext.datasets.TranslationDataset):
    """Defines a dataset for machine translation."""

    def __init__(self, pairs, fields, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        for src_line, trg_line in pairs:
            src_line, trg_line = src_line.strip(), trg_line.strip()
            if src_line != '' and trg_line != '':
                examples.append(torchtext.data.Example.fromlist([src_line, trg_line], fields))

        super(torchtext.datasets.TranslationDataset, self).__init__(examples, fields, **kwargs)

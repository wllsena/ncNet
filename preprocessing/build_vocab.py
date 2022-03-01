import torch
from torchtext.data import BucketIterator, Field, TabularDataset
from torchtext.vocab import Vectors

from preprocessing.word_tokenize import word_tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_vocab(data_dir, db_info, batch_size, max_input_length):
    glove_path = './dataset/tuned_glove.6B.100d.txt'

    SRC = Field(tokenize=word_tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TRG = Field(tokenize=word_tokenize,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TOK_TYPES = Field(tokenize=word_tokenize,
                      init_token='<sos>',
                      eos_token='<eos>',
                      lower=True,
                      batch_first=True)

    train_data, valid_data, test_data = TabularDataset.splits(path=data_dir,
                                                              format='csv',
                                                              skip_header=True,
                                                              train='train.csv',
                                                              validation='dev.csv',
                                                              test='test.csv',
                                                              fields=[
                                                                  ('tvBench_id', None),
                                                                  ('db_id', None),
                                                                  ('chart', None),
                                                                  ('hardness', None),
                                                                  ('query', None),
                                                                  ('question', None),
                                                                  ('vega_zero', None),
                                                                  ('mentioned_columns', None),
                                                                  ('mentioned_values', None),
                                                                  ('query_template', None),
                                                                  ('src', SRC),
                                                                  ('trg', TRG),
                                                                  ('tok_types', TOK_TYPES),
                                                              ])

    db_information = TabularDataset(path=db_info,
                                    format='csv',
                                    skip_header=True,
                                    fields=[('table', TRG), ('column', TRG), ('value', TRG)])

    glove_vectors = Vectors(glove_path)

    SRC.build_vocab([glove_vectors.itos], vectors=glove_vectors)
    TRG.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)
    TOK_TYPES.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), sort=False, batch_size=batch_size, device=device)

    return SRC, TRG, TOK_TYPES, batch_size, train_iterator, valid_iterator, test_iterator, max_input_length

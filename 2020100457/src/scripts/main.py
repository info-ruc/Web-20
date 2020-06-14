import argparse
import torch
import torchtext.data as data
from torchtext.vocab import Vectors

import model
import train
import dataset

parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001,
                    help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=128,
                    help='number of epochs for train [default: 128]')
parser.add_argument('-test', type=bool, default=False,
                    help='Train or Test, default False: Train')
parser.add_argument('-predict', type=bool, default=False,
                    help='Try to input a chinese sentence about automobile, to predict its emotion!')
parser.add_argument('-batch-size', type=int, default=128,
                    help='batch size for training [default: 128]')
parser.add_argument('-save-dir', type=str, default='../../extra/snapshot',
                    help='where to save the snapshot')
parser.add_argument('-device', type=int, default=-1,
                    help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-snapshot', type=str, default=None,
                    help='filename of model snapshot [default: None]')
args = parser.parse_args()


def load_word_vectors(model_name, model_path):
    vectors = Vectors(name=model_name, cache=model_path)
    return vectors


def load_dataset(text_field, label_field, args, **kwargs):
    train_dataset, dev_dataset, test_dataset = dataset.get_dataset(
        '../dataset', text_field, label_field)
    vectors = load_word_vectors("sgns.zhihu.word", "../../extra/pretrained")
    text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    label_field.build_vocab(train_dataset, dev_dataset)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
        (train_dataset, dev_dataset, test_dataset),
        batch_sizes=(args.batch_size, len(dev_dataset), len(test_dataset)),
        sort_key=lambda x: len(x.text),
        **kwargs)
    return train_iter, dev_iter, test_iter


def load_test_dataset(text_field, label_field):
    train_dataset, dev_dataset, _ = dataset.get_dataset(
        '../dataset', text_field, label_field)
    test_data = dataset.get_test_dataset(text_field, label_field)
    vectors = load_word_vectors("sgns.zhihu.word", "../../extra/pretrained")
    text_field.build_vocab(train_dataset, dev_dataset, vectors=vectors)
    label_field.build_vocab(train_dataset, dev_dataset)
    test_iter = data.Iterator(test_data, 1)
    return test_iter


def train_test(text_field, label_field, test=False):
    train_iter, dev_iter, test_iter = load_dataset(
        text_field, label_field, args, device=-1, repeat=False, shuffle=True)

    args.vocabulary_size = len(text_field.vocab)
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors

    args.class_num = len(label_field.vocab)
    args.cuda = args.device != -1 and torch.cuda.is_available()
    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        if attr in {'vectors'}:
            continue
        print('\t{}={}'.format(attr.upper(), value))

    text_cnn = model.TextCNN(args)
    if args.snapshot:
        print('\nLoading model from {}...\n'.format(args.snapshot))
        text_cnn.load_state_dict(torch.load(args.snapshot))

    if args.cuda:
        device = torch.device("cuda", args.device)
        text_cnn = text_cnn.to(device)
    if args.test:
        try:
            train.eval(test_iter, text_cnn, args, True)
        except KeyboardInterrupt:
            print('Exiting from testing early')
    else:
        try:
            train.train(train_iter, dev_iter, text_cnn, args)
        except KeyboardInterrupt:
            print('Exiting from training early')


def own_test(text_field, label_field):
    test_iter = load_test_dataset(text_field, label_field)

    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
    args.vocabulary_size = len(text_field.vocab)
    args.class_num = 3
    args.cuda = args.device != -1 and torch.cuda.is_available()
    # print('Parameters:')
    # for attr, value in sorted(args.__dict__.items()):
    #     if attr in {'vectors'}:
    #         continue
    #     print('\t{}={}'.format(attr.upper(), value))

    text_cnn = model.TextCNN(args)
    if args.snapshot:
        print('\nLoading model from {}...\n'.format(args.snapshot))
        text_cnn.load_state_dict(torch.load(args.snapshot))
    else:
        text_cnn.load_state_dict(torch.load('model.pth'))

    if args.cuda:
        device = torch.device("cuda", args.device)
        text_cnn = text_cnn.to(device)
    try:
        train.test(test_iter, text_cnn, args)
    except KeyboardInterrupt:
        print('Exiting from training early')


if __name__ == '__main__':

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)

    if args.test:
        print('Loading data...')
        train_test(text_field, label_field, True)
    elif args.predict:
        print("Loading pretrained vectors...")
        own_test(text_field, label_field)
    else:
        print('Loading data...')
        train_test(text_field, label_field)

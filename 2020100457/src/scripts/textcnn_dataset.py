import re
from torchtext import data
import readline
import jieba
import logging
jieba.setLogLevel(logging.INFO)

# 正则表达式除了中文 英文 汉字外 其他的字符都不能输入
regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev, test = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='dev.tsv', test='test.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev, test


def get_test_dataset(text_field, label_field):
    text_field.tokenize = word_cut

    fields = [('text', text_field), ('label', label_field)]
    examples = []

    sentence = input("请输入您对汽车的评论：")
    while len(sentence) < 12:
        sentence = input("评论长度过小，请重新输入评论：")

    examples.append(data.Example.fromlist([sentence, None], fields))
    return data.Dataset(examples, fields)


if __name__ == '__main__':
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)

    # test_data = get_test_dataset(text_field, label_field)
    train_dataset, dev_dataset, test_data = get_dataset(
        'data', text_field, label_field)

    text_field.build_vocab(train_dataset, dev_dataset)
    label_field.build_vocab(train_dataset, dev_dataset)
    test_iter = data.Iterator(test_data, 1)
    iter_ = iter(test_iter)
    batch = next(iter_)
    feature, target = batch.text, batch.label
    feature.t_(), target.sub_(1)
    print(feature.size())  # 128, 26
    print(target.size())

    # train_dataset, dev_dataset, test_dataset = get_dataset('data', text_field, label_field)
    # print(test_dataset) # <torchtext.data.dataset.TabularDataset object at 0x11f7da470>

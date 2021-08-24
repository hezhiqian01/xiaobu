import re
import collections
# import json


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word in vocab:
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += 1
    return pairs


def merge_vocab(pair, v_in):
    v_out = []
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub('-'.join(pair), word)
        v_out.append(w_out)
    return v_out


# vocab = {'l o w </w>': 5, 'l o w e r </w>': 2,
#          'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
# num_merges = 10
# for i in range(num_merges):
#     pairs = get_stats(vocab)
#     best = max(pairs, key=pairs.get)
#     print(best)
#     vocab = merge_vocab(best, vocab)
#     print(vocab)

def get_vocab(train_filename, test_filename, output_train, output_test, output_merged_words, num_merges=100):
    vocab = []
    label = []
    for filename in [train_filename, test_filename]:
        with open(filename) as f:
            for line in f:
                l = line.split('\t')
                vocab.append(l[0])
                vocab.append(l[1])
                if len(l) == 3:
                    label.append(l[2])
                else:
                    label.append(-5)
    print("句子数：{}".format(len(vocab)))
    merged_word = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        best = max(pairs, key=pairs.get)
        # print(best)
        merged_word.append('-'.join(best))
        vocab = merge_vocab(best, vocab)
        # print(vocab)
        # break
    # print(vocab)
    output_train_file = open(output_train, 'w')
    output_test_file = open(output_test, 'w')

    flag = False
    line = []
    i = 0
    for key in vocab:
        if flag:
            line.append(key)
            line.append(label[i])
            if label[i] == -5:
                f = output_test_file
            else:
                f = output_train_file
            f.write("\t".join(line))
            i += 1
            flag = False
            line = []
        else:
            line.append(key)
            flag = True

    output_test_file.close()
    output_train_file.close()
    print("saved train file to {}".format(output_train))
    print("saved test file to {}".format(output_test))

    with open(output_merged_words, 'w') as f:
        for line in merged_word:
            f.write(line + '\n')
    print("saved merged words to {}".format(output_merged_words))

    return vocab, merged_word


if __name__ == '__main__':
    train_filename = '../tcdata/gaiic_track3_round1_train_20210228.tsv'
    test_filename = '../tcdata/gaiic_track3_round1_train_20210228.tsv'

    output_train = '../user_data/train-bpe-500.txt'
    output_test = '../user_data/test-bpe-500.txt'
    output_merge = '../user_data/bpe-merged-words-500.txt'

    get_vocab(train_filename, test_filename, output_train, output_test, output_merge, num_merges=500)


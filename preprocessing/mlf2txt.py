#!/usr/bin/python
import argparse

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('MLF', type=str,
                               help='Name of the source dataset')
commandLineParser.add_argument('TXT', type=str,
                               help='Name of the target dataset')
"""
This functions proccesses an MLF file into a DAT file - all utterances are turned into a single line per utterance,
pre-pended and post-pended with <s> and </s> .
"""


def main(argv=None):
    args = commandLineParser.parse_args()
    words = []
    confidences = []
    with open(args.MLF, 'r') as handle:
        for line in handle.readlines():
            if line == '#!MLF!#\n':
                continue
            line = line.replace('\n', '').split()
            if len(line) > 1:
                wrd = line[-2]
                conf = line[-1]
            else:
                if line[0] == '.':
                    wrd = '</s>\n'
                    conf = '1.0\n'
                else:
                    wrd = '<s>'
                    conf = '1.0'
            words.append(wrd)
            confidences.append(conf)

    text = ' '.join(words)
    confidences = ' '.join(confidences)
    with open(args.TXT + '.dat', 'w') as handle:
        handle.write(text)
    with open(args.TXT + '.conf', 'w') as handle:
        handle.write(confidences)

if __name__ == '__main__':
    main()

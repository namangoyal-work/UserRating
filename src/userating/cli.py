"""Command line interface. Same verbs (and argument order) as the original main.py --
train / test / cv -- plus `predict` for scoring ad-hoc text from the shell."""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import __version__
from .metrics import eval_report
from .model import SentimentClassifier, load, save
from .preprocess import transform


def read_train_csv(path, limit=None):
    df = pd.read_csv(path, header=None).dropna()
    if limit is not None:
        df = df.iloc[:limit]
    return df


def cmd_train(args):
    model = SentimentClassifier(dim=args.dim)
    train = read_train_csv(args.train_csv, args.limit)
    model.fit(list(train[0]), list(train[1]))
    save(model, args.model_path)


def cmd_test(args):
    test = pd.read_csv(args.input_csv, header=None)

    model = load(args.model_path)
    # one prediction per input row: blank/NaN reviews become empty strings, not drops
    preds = model.predict([str(x) if not pd.isna(x) else '' for x in test[0]])
    np.savetxt(args.output_path, preds, fmt='%d')


def cmd_cv(args):
    model = SentimentClassifier()
    df = read_train_csv(args.train_csv, args.limit)

    # preprocessing is stateless per review, so transform once and share across
    # folds -- no leakage, all fitted state still trains per fold
    docs = transform(list(df[0]))

    skf = StratifiedKFold(n_splits=args.folds)
    f1m = []
    f1M = []
    for i, (train, val) in enumerate(skf.split(df, df[1])):
        print(f'Fold {i}:', flush=True)
        model.fit([docs[j] for j in train], df.iloc[train][1], pretransformed=True)
        preds = model.predict([docs[j] for j in val], pretransformed=True)
        report = eval_report(df.iloc[val][1], preds)
        f1m.append(report['f1_micro'])
        f1M.append(report['f1_macro'])

    print()
    print('Averaged metrics:')
    print(f'    F1 micro: {sum(f1m) / args.folds}')
    print(f'    F1 macro: {sum(f1M) / args.folds}')
    print(f'    Final Score: {(sum(f1m) + sum(f1M)) / (2 * args.folds)}')


def cmd_predict(args):
    model = load(args.model_path)
    texts = args.text if args.text else [line.strip() for line in sys.stdin if line.strip()]
    probs = model.predict_proba(texts)
    for text, p in zip(texts, probs):
        rating = int(p.argmax()) + 1
        stars = '*' * rating
        print(f'{rating} {stars:<5} (confidence {p.max():.2f})  {text[:60]}')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='userating',
        description='Turn free-text user feedback into 1-5 star ratings.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('train', help='train a model on a labeled csv (review,rating)')
    p.add_argument('train_csv')
    p.add_argument('model_path')
    p.add_argument('--limit', type=int, default=None, help='train on the first N rows only (smoke runs)')
    p.add_argument('--dim', type=int, default=2000, help='SVD dimensionality cap')
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('test', help='predict ratings for a csv of reviews, one per line')
    p.add_argument('model_path')
    p.add_argument('input_csv')
    p.add_argument('output_path')
    p.set_defaults(func=cmd_test)

    p = sub.add_parser('cv', help='stratified k-fold cross validation with a full report')
    p.add_argument('train_csv')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--folds', type=int, default=5)
    p.set_defaults(func=cmd_cv)

    p = sub.add_parser('predict', help='score ad-hoc review text (args or stdin)')
    p.add_argument('model_path')
    p.add_argument('text', nargs='*')
    p.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()

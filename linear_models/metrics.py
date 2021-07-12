import numpy as np

THRESHOLD = 0.5


# вспомогательная функция для расчета TP, FP, FN, TN
def find_tp_fp_fn_tn(y_true, y_predict, percent=None):
    if percent is None:
        y_predict = (y_predict >= THRESHOLD)
    elif 1 <= percent <= 100:
        size_predict = y_predict.shape[0]
        size_true = y_true.shape[0]
        top_predict = int((0.01 * (100 - percent)) * size_predict)
        top_true = int((0.01 * (100 - percent)) * size_true)
        y_predict = (y_predict[top_predict:] >= THRESHOLD)
        y_true = y_true[top_true:]
    else:
        raise ValueError
    tp = sum((y_true == 1) & (y_predict == 1))
    fp = sum((y_true == 0) & (y_predict == 1))
    fn = sum((y_true == 1) & (y_predict == 0))
    tn = sum((y_true == 0) & (y_predict == 0))
    return tp, fp, fn, tn


def accuracy_score(y_true, y_predict, percent=None):
    tp, fp, fn, tn = find_tp_fp_fn_tn(y_true, y_predict, percent)
    try:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        print('ZeroDivisionError')
        accuracy = 0
    return accuracy


def precision_score(y_true, y_predict, percent=None):
    tp, fp, fn, tn = find_tp_fp_fn_tn(y_true, y_predict, percent)
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print('ZeroDivisionError')
        precision = 0
    return precision


def recall_score(y_true, y_predict, percent=None):
    tp, fp, fn, tn = find_tp_fp_fn_tn(y_true, y_predict, percent)
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print('ZeroDivisionError')
        recall = 0
    return recall


def lift_score(y_true, y_predict, percent=None):
    tp, fp, fn, tn = find_tp_fp_fn_tn(y_true, y_predict, percent)
    try:
        precision = precision_score(y_true, y_predict, percent)
        l = len(y_true)
        lift = precision / ((tp + fn) / l)
    except ZeroDivisionError:
        print('ZeroDivisionError')
        lift = 0
    return lift


def f1_score(y_true, y_predict, percent=None):
    try:
        precision = precision_score(y_true, y_predict, percent)
        recall = recall_score(y_true, y_predict, percent)
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        print('ZeroDivisionError')
        f1 = 0
    return f1


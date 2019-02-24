from collections import Counter
from datetime import datetime
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sb
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.metrics.classification import unique_labels


class ExperimentReport(object):

    def __init__(self, pre_dir='report', pdf_name=''):
        self.title = 'Experiment_' + str(datetime.today()) + pdf_name
        self.pdf = PdfPages(os.path.join(pre_dir, self.title + '.pdf'))
        self.pages = []

    def add_simple_text_page(self, text='', size=12, figsize=(8, 5), usetex=False, font='monospace'):
        plt.rc('text', usetex=usetex)
        fig = plt.figure(figsize=figsize)
        fig.text(x=0.05, y=0.9, s=text, color='black', ha='left', va='top', size=size, fontname=font)
        plt.close()
        self.pages.append(fig)
        return fig

    def add_simple_plot_page(self, x, y, style='b-', title='', figsize=(8, 5), usetex=False):
        plt.rc('text', usetex=usetex)
        fig = plt.figure(figsize=figsize)
        plt.plot(x, y, style)
        plt.title(title)
        plt.close()
        self.pages.append(fig)
        return fig

    def pdf_info(self):
        today = datetime.today()
        d = self.pdf.infodict()
        d['Title'] = self.title
        d['Author'] = 'markmo'
        d['Subject'] = 'Experiment Results'
        d['Keywords'] = 'Top-5 Accuracy'
        d['CreationDate'] = today
        d['ModDate'] = today

    def add_page(self, fig):
        self.pages.append(fig)

    def flush(self):
        for pg in self.pages:
            self.pdf.savefig(pg)

        self.pdf.close()

    def test_run(self):
        # Page 1
        self.add_simple_text_page()

        # Page 2
        self.add_simple_plot_page(x=range(7), y=[3, 1, 4, 1, 5, 9, 2], style='r-o', title='')

        # Pages 3-4
        x = np.arange(0, 5, 0.1)
        self.add_simple_plot_page(x=x, y=np.sin(x), style='b-', title='')
        self.add_simple_plot_page(x=x, y=x ** 2, style='b-', title='')


class AssessmentReport(ExperimentReport):

    def __init__(self, pre_dir='report', pdf_name=''):
        super().__init__(pre_dir, pdf_name)

    def roc_auc(self, y_score, y_true, figsize=(8, 5)):
        tpr, fpr, _ = roc_curve(y_true=y_true, y_score=1 - y_score)
        roc_auc = auc(fpr, tpr)
        if roc_auc < 0.5:
            tpr, fpr, _ = roc_curve(y_true=y_true, y_score=y_score)
            roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        self.add_page(fig)
        return fig

    # noinspection PyUnresolvedReferences
    def plot_confusion_matrix(self, confusion_mat, classes, normalize=False, figsize=(8, 5),
                              title='Confusion Matrix', color_map=plt.cm.Blues):
        """
        Prints the confusion matrix.

        Normalization can be applied with `normalize` set to `True`.

        :param confusion_mat:
        :param classes:
        :param normalize: (bool) If `True`, normalize the matrix
        :param figsize:
        :param title:
        :param color_map:
        :return:
        """
        if normalize:
            confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
            print('Normalized Confusion Matrix')
        else:
            print('Confusion Matrix without normalization')

        fig = plt.figure(figsize=figsize)
        plt.imshow(confusion_mat, interpolation='nearest', cmap=color_map)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        threshold = confusion_mat / 2
        m, n = confusion_mat.shape
        for i, j in product(range(m, n)):
            color = 'white' if confusion_mat[i, j] > threshold else 'black'
            plt.text(j, i, format(confusion_mat[i, j], fmt), horizontalalignment='center', color=color)

        plt.ylabel('True label')
        plt.xlabel('Pred label')
        self.add_page(fig)
        plt.close()


def evaluate_classifier(y_true, y_score, pdf_report, idx_class=None):
    """

    :param y_true: (1D array)
    :param y_score: (2D array) shape [n_samples, n_classes]
    :param pdf_report:
    :param idx_class:
    :return:
    """
    if idx_class is not None:
        y_true = list(map(lambda ys: idx_class[int(ys)], y_true))
        y_true = np.array(y_true)
        y_score = list(map(lambda ys: list(map(lambda s: (idx_class[int(s[0])], s[1]), enumerate(ys))), y_score))

    y_score_top_n = list(map(lambda ys: sorted(ys, key=lambda p: p[1], reverse=True), y_score))
    y_score_top_n = np.array(y_score_top_n)
    n_samples = len(y_score_top_n)
    y_true = y_true[:n_samples]
    test_dist = Counter(y_true)
    recall1 = Counter()
    recall5 = Counter()

    y_p_n = y_score_top_n.transpose([1, 0, 2])
    y_top_n_pred = np.array(y_p_n)[:, :, 0]
    top_5_pred = y_top_n_pred[:5].T
    top_1_pred = y_top_n_pred[:1].T
    for i in range(n_samples):
        if y_true[i] in top_1_pred[i]:
            recall1[y_true[i]] += 1
            recall5[y_true[i]] += 1
        elif y_true[i] in top_5_pred[i]:
            recall5[y_true[i]] += 1

    # pred_table = pd.DataFrame(np.concatenate([y_true[:, None], top_5_pred], axis=1),
    #                           columns=['lb', 'p1', 'p2', 'p3', 'p4', 'p5'])
    # print(pred_table)
    dist_r5_r1 = pd.DataFrame(columns=['label', 'n_samples', 'r1', 'r5', 'recall1', 'recall5'])
    for i, (k, v) in enumerate(test_dist.most_common()):
        dist_r5_r1.loc[i] = [k, v, recall1[k], recall5[k], recall1[k] / v, recall5[k] / v]

    top_n_acc = list(map(lambda xs: accuracy_score(y_true, xs, normalize=True), y_top_n_pred))
    top_n_acc_cum = top_n_acc.copy()
    n = len(top_n_acc)
    for i in range(1, n + 1):
        top_n_acc_cum += np.concatenate([[0] * i, top_n_acc[:-i]])

    sb.set()
    sb.set_style('dark')
    fig, ax1 = plt.subplots(figsize=(12, 15))
    plt.subplots_adjust(left=0.4, top=0.95, bottom=0.05)
    ax2 = ax1.twiny()
    ax2.set_xlabel('SET2', color='b')
    sb.barplot(x='n_samples', y='label', data=dist_r5_r1, ax=ax1, palette='Blues_d')
    sb.pointplot(x='recall1', y='label', data=dist_r5_r1, ax=ax2, linestyles='-', color='r')
    sb.pointplot(x='recall5', y='label', data=dist_r5_r1, ax=ax2, linestyles='-', color='b')
    pdf_report.add_page(fig)
    plt.close()

    fig, ax1 = plt.subplots(figsize=None)
    n = 10
    ay = top_n_acc[:n]
    tx1 = 'top_n_acc:\n' + str(ay) + '\n'
    sb.barplot(x=list(range(1, 11)), y=ay, palette='Blues_d', ax=ax1)
    pdf_report.add_page(fig)
    plt.close()

    print('top_5_acc_cum:', top_n_acc_cum[:10])
    fig, ax1 = plt.subplots(figsize=None)
    n = 10
    ay = top_n_acc_cum[:n]
    tx2 = 'top_n_acc_cum:\n' + str(ay) + '\n'
    sb.barplot(x=list(range(1, 11)), y=ay, palette='Blues_d', ax=ax1)
    pdf_report.add_page(fig)
    plt.close()

    tx3 = 'Top 5 Accuracy: ' + str(ay[4]) + '\n'
    pdf_report.add_simple_text_page(text=(tx1 + tx2 + tx3), figsize=(10, 5))
    confusion_mat = confusion_matrix(y_true=y_true, y_pred=top_1_pred)
    pdf_report.plot_confusion_matrix(confusion_mat, figsize=(20, 20), classes=unique_labels(y_true, top_1_pred))
    return top_n_acc_cum[:5]

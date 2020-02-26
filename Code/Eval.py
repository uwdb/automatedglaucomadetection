from __future__ import print_function, division


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from common.common import *
from OCT_dataset import input_fn



def plot_confusion_matrix(y_true, y_pred, classes=['0','1'],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,f=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    f.write('\n')
    f.write(np.array2string(cm,precision=4,separator=',',suppress_small=True))
    f.write('\n')
    f.write(np.array2string(cm,precision=4,separator=',',suppress_small=True))
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    return ax

def create_auc_roc(np_tru, np_pred, roc_fn, res_f, rs_path):
    # Print Area Under Curve
    false_positive_rate, recall, thresholds = roc_curve(np_tru, np_pred[:, 1])
    roc_auc = auc(false_positive_rate, recall)
    print("Area under ROC curve(%): {:.3f}".format(roc_auc * 100))
    res_f.write("Area under ROC curve(%): {:.3f}".format(roc_auc * 100))
    plt.figure()
    plt.title('ROC')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('1-Specificity')
    plt.savefig(rs_path + roc_fn)
    plt.clf()
    return


def smax(X, param=1.0, axis=None):
    y = np.atleast_2d(X)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(param)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p




def run_eval(model, test_fn, const, chkpoint=True,nclasses=2,size=224, batch_size=32):
    class_names = ['Normal', 'GL']
    eval_fn_f = const.data_dir + test_fn
    print("in eval, run_eval, eval_fn_f {}".format(eval_fn_f))
    eval_data = pd.read_table(eval_fn_f, header=None, sep='\t')
    eval_data.columns = ['fn', 'l']


    eval_data['fn'] = eval_data['fn'].apply(lambda x: "{}{}".format(const.root_dir, x))

    eval_inputs = input_fn(False, eval_data['fn'].tolist(), eval_data['l'].tolist(),
                               batch_size=batch_size, ls=False,
                                nclasses=nclasses,size=size)

    val_steps = eval_data.shape[0]//batch_size



    print('model checkpoint location {}'.format(const.checkpoint_path))
    if chkpoint:
        model.load_weights(const.checkpoint_path)
        results_fn = const.results_fn
    else:
        model.load_weights(const.weights_path)
        results_fn = const.save_path + 'end_weights.txt'


    pred = model.predict(eval_inputs, steps=val_steps)

    res = open(results_fn, 'w')
    res.write("Results: \n")


    tru = eval_data['l'].tolist()
    np_tru = np.asarray(tru)
    np_pred = pred[:np_tru.shape[0]]
    np_pred = smax(np_pred, axis=1)
    np_pred_r =np.argmax(np_pred,axis=1)



    cn = ['Normal', 'GL']
    if nclasses is 2:
        create_auc_roc(np_tru=np_tru, np_pred=np_pred, roc_fn="ROC.png", res_f=res, rs_path=const.save_path)


    plot_confusion_matrix(tru, np_pred_r, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig(const.save_path + "overall_cm.png")
    plt.clf()

    # Plot normalized confusion matrix
    plot_confusion_matrix(np_tru, np_pred_r,  classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(const.save_path + "normalized_cm.png")
    plt.clf()


    res.write('\n')
    res.write("report:\n {}".format(classification_report(tru, np_pred_r), target_names=cn))
    results_df = pd.DataFrame(np_pred)
    results_df['labels'] = tru
    results_df['fn'] = eval_data[['fn']]
    results_df.to_csv(const.results_csv, index=None, sep='\t')
    res.close()
    return




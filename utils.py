import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def calculate_metrics(ytrue, ypred,average='weighted'):
    acc = accuracy_score(ytrue, ypred)
    recall = recall_score(ytrue, ypred, average=average)
    f1 = f1_score(ytrue, ypred,average=average)
    cm = confusion_matrix(ytrue, ypred)
    fps = cm.sum(axis=0) - np.diag(cm)  # False positives
    negs = cm.sum(axis=1) - np.diag(cm)  # True negatives + False positives
    far = np.zeros_like(fps, dtype=float)
    mask = negs != 0
    far[mask] = fps[mask] / negs[mask]
    fns = cm.sum(axis=1) - np.diag(cm)  # False negatives
    poss = cm.sum(axis=1)  # True positives + False negatives
    frr = np.zeros_like(fns, dtype=float)
    mask = poss != 0
    frr[mask] = fns[mask] / poss[mask]
    metrics = {
        'accuracy': acc,
        'recall': recall,
        'f1_score': f1,
        'far': far,
        'frr': frr,
    }
    return metrics





def plot_roc(y_test,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-class Classification')
    plt.legend(loc="lower right")
    plt.savefig('./images_results/eis_structure_roc.png',dpi=300)
    plt.show()
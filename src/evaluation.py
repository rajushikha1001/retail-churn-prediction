from sklearn.metrics import confusion_matrix, roc_auc_score


def confusion_matrix_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)


def roc_auc_report(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    print("ROC-AUC Score: ", auc)

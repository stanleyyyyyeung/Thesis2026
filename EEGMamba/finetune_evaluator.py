import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, roc_auc_score, \
    precision_recall_curve, auc, r2_score, mean_squared_error
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    @torch.no_grad()
    def predict(self, model):
        """Returns y_true, y_pred as (N_windows, seq_len), in loader order."""
        model.eval()
        ys, ps = [], []
        for x, y in tqdm(self.data_loader, mininterval=1):
            pred_y = model(x.cuda()).argmax(-1).cpu().numpy()
            L = y.shape[-1]
            ps.append(pred_y.reshape(-1, L))
            ys.append(y.numpy().reshape(-1, L))
        return np.concatenate(ys), np.concatenate(ps)

    @staticmethod
    def metrics_from_arrays(truths, preds, n_classes=5):
        acc = balanced_accuracy_score(truths, preds)
        f1 = f1_score(truths, preds, average='weighted')
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds, labels=list(range(n_classes)))
        return acc, kappa, f1, cm

    def get_metrics_for_multiclass(self, model):
        y, p = self.predict(model)
        return self.metrics_from_arrays(y.ravel(), p.ravel(), self.params.num_of_classes)

    def get_metrics_for_binaryclass(self, model):
        model.eval()

        truths = []
        preds = []
        scores = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
            truths += y.long().cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()
            scores += score_y.cpu().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        scores = np.array(scores)
        acc = balanced_accuracy_score(truths, preds)
        roc_auc = roc_auc_score(truths, scores)
        precision, recall, thresholds = precision_recall_curve(truths, scores, pos_label=1)
        pr_auc = auc(recall, precision)
        cm = confusion_matrix(truths, preds)
        return acc, pr_auc, roc_auc, cm

    def get_metrics_for_regression(self, model):
        model.eval()

        truths = []
        preds = []
        for x, y in tqdm(self.data_loader, mininterval=1):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        corrcoef = np.corrcoef(truths, preds)[0, 1]
        r2 = r2_score(truths, preds)
        rmse = mean_squared_error(truths, preds) ** 0.5
        return corrcoef, r2, rmse

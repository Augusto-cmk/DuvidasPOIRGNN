from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configuration.next_poi_category_prediciton_configuration import NextPoiCategoryPredictionConfiguration

class NextPoiCategoryPredictionLoader:

    def __init__(self):
        self.next_poi_category_prediction_configuration = NextPoiCategoryPredictionConfiguration()

    def _int_to_category(self, list_of_int, dataset_name):

        categories = []

        for i in list_of_int:
            categories.append(self.next_poi_category_prediction_configuration.INT_TO_CATEGORIES[1][dataset_name]['7_categories'][str(i)])

        return categories

    def entropy_indexes(self, outpt_dir, list_indexes, dataset_name):

        list_y_wrong_predicted = []
        list_y_wrong_labels = []
        list_y_righ_predicted = []
        list_y_right_labels = []
        list_entropy_right = []
        list_entropy_wrong = []
        for i in  range(len(list_indexes)):
            indexes = list_indexes[i]
            y_wrong_predicted, y_wrong_labels, y_righ_predicted, y_right_labels, entropy_right, entropy_wrong = indexes
            y_wrong_predicted = self._int_to_category(y_wrong_predicted.tolist(), dataset_name)
            y_righ_predicted = self._int_to_category(y_righ_predicted.tolist(), dataset_name)
            y_right_labels = self._int_to_category(y_right_labels.tolist(), dataset_name)
            y_wrong_labels = self._int_to_category(y_wrong_labels.tolist(), dataset_name)
            print("tipo: ", type(entropy_right), len(entropy_right))
            list_y_wrong_predicted += y_wrong_predicted
            list_y_wrong_labels += y_wrong_labels
            list_y_righ_predicted += y_righ_predicted
            list_y_right_labels += y_right_labels
            list_entropy_right += entropy_right.tolist()
            list_entropy_wrong += entropy_wrong.tolist()



        df_dict = {'Prediction type': ['Right']*len(list_entropy_right) + ['Wrong']*len(list_entropy_wrong), 'Entropy': list_entropy_right + list_entropy_wrong, 'Prediction': list_y_righ_predicted + list_y_wrong_predicted, 'Label': list_y_right_labels + list_y_wrong_labels}


        df = pd.DataFrame(df_dict)
        self.boxplot(output_dir=outpt_dir, name="wrong_x_right_entropy", df=df, x='Prediction type', y='Entropy')
        self.boxplot(output_dir=outpt_dir, name="wrong_x_right_entropy_x_label", df=df, x='Prediction type', y='Entropy', hue='Label')

    def boxplot(self, output_dir, name, df, x, y, hue=None):
        plt.figure()
        fig = sns.boxplot(data=df, x=x, y=y, hue=hue)

        if hue is not None:
            sns.move_legend(fig, loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=0.)
        fig = fig.get_figure()
        fig.savefig(output_dir + name + ".png", dpi=400, bbox_inches='tight')


    def plot_history_metrics(self, folds_histories, folds_reports, output_dir, n_folds, n_replications, list_indexes, dataset_name, show=False):

        # n_folds = len(folds_histories)
        # n_replications = len(folds_histories[0])
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        print("pasta: ", output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.entropy_indexes(output_dir, list_indexes, dataset_name)
        for fold_histories in folds_histories:
            for i in range(len(fold_histories)):
                h = fold_histories[i]
                file_index = "replication_" + str(i)
                plt.figure(figsize=(12, 12))
                plt.plot(h['acc'])
                plt.plot(h['val_acc'])
                plt.title('model acc')
                plt.ylabel('acc')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                if show:
                    plt.show()
                plt.savefig(output_dir + file_index+ "_history_accuracy.png")
                # summarize history for loss
                plt.figure(figsize=(12, 12))
                plt.plot(h['loss'])
                plt.plot(h['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.savefig(output_dir + file_index + "_history_loss.png")
                if show:
                    plt.show()

    def save_report_to_csv(self, output_dir, report, n_folds, n_replications, usuarios):

        precision_dict = {}
        recall_dict = {}
        fscore_dict = {}
        column_size = n_folds * n_replications
        print("final")
        print(report)
        for key in report:
            if key == 'accuracy':
                fscore_column = 'accuracy'
                fscore_dict[fscore_column] = report[key]
                continue
            elif key == 'recall' or key == 'f1-score' \
                    or key == 'support':
                continue
            elif key == 'macro avg' or key == 'weighted avg':
                precision_dict[key] = report[key]['precision']
                recall_dict[key] = report[key]['recall']
                fscore_dict[key] = report[key]['f1-score']
                continue
            fscore_column = key
            fscore_column_data = report[key]['f1-score']
            if len(fscore_column_data) < column_size:
                while len(fscore_column_data) < column_size:
                    fscore_column_data.append(np.nan)
            fscore_dict[fscore_column] = fscore_column_data

            precision_column = key
            precision_column_data = report[key]['precision']
            if len(precision_column_data) < column_size:
                while len(precision_column_data) < column_size:
                    precision_column_data.append(np.nan)
            precision_dict[precision_column] = precision_column_data

            recall_column = key
            recall_column_data = report[key]['recall']
            if len(recall_column_data) < column_size:
                while len(recall_column_data) < column_size:
                    recall_column_data.append(np.nan)
            recall_dict[recall_column] = recall_column_data

        #print("final: ", new_dict)
        precision = pd.DataFrame(precision_dict)
        print("Métricas precision: \n", precision)
        output_dir = output_dir + str(n_folds) + "_folds/" + str(n_replications) + "_replications/"
        print("pasta", output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        precision.to_csv(output_dir + "precision.csv", index_label=False, index=False)

        recall = pd.DataFrame(recall_dict)
        print("Métricas recall: \n", recall)
        recall.to_csv(output_dir + "recall.csv", index_label=False, index=False)

        fscore = pd.DataFrame(fscore_dict)
        print("Métricas fscore: \n", fscore)
        fscore.to_csv(output_dir + "fscore.csv", index_label=False, index=False)

    def wrong_predicted_samples_plots(self, output_dir, x, y_predicted, y_label):
        pass

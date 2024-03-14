import copy
import statistics as st
import math
import numpy as np
import json
from scipy.stats import entropy
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from keras import utils as np_utils
from sklearn.model_selection import KFold
from spektral.layers.convolutional import GCNConv, DiffusionConv
from extractor.file_extractor import FileExtractor
from foundation.util.next_poi_category_prediction_util import sequence_to_x_y,sequence_tuples_to_spatial_temporal_and_feature8_ndarrays, remove_hour_from_sequence_y
from foundation.util.nn_preprocessing import one_hot_decoding
from models.poi_rgnn import MFA_RNN
from loader.next_poi_category_prediction_loader import NextPoiCategoryPredictionLoader
import ast
import math


class NextPoiCategoryPredictionDomain:


    def __init__(self, dataset_name, distance_sigma, duration_sigma):
        self.file_extractor = FileExtractor()
        self.next_poi_category_prediction_loader = NextPoiCategoryPredictionLoader()
        self.dataset_name = dataset_name
        self.distance_sigma = distance_sigma
        self.duration_sigma = duration_sigma
        self.count=0

    def read_sequences(self, filename, n_splits, step_size, dataset_name):
        df = self.file_extractor.read_csv(filename)
        df['sequence'] = df['sequence'].apply(lambda e: ast.literal_eval(e))
        df['total'] = df['sequence'].apply(lambda x:len(x))
        df = df.sort_values(by='total', ascending=False)

        if dataset_name == "gowalla":
            minimum = 40
        else:
            minimum = 300
        
            
        df = df[df['total'] >= minimum]

        # reindex ids
        df['id'] = np.array([i for i in range(len(df))])

        users_ids = df['id'].tolist()
        sequences = df['sequence'].tolist()
        x_list = []
        y_list = []
        countries = {}
        max_country = 0
        max_distance = 0
        max_duration = 0

        distance_list = []
        duration_list = []

        maior_mes = 0
        for i in range(len(users_ids)):

            user_id = users_ids[i]
            sequence = sequences[i]
            new_sequence = []

            if len(sequence) < minimum:
                x_list.append([])
                continue

            size = len(sequence)
            for j in range(size):
                location_category_id = sequence[j][0]
                hour = sequence[j][1]
                country = sequence[j][2]
                distance = sequence[j][3]
                duration = sequence[j][4]
                if j < len(sequence) -1:
                    if duration > 72 and sequence[j+1][4] > 72:
                        continue
                week_day = sequence[j][5]
                poi_id = sequence[j][7]
                month = sequence[j][8]
                if month > maior_mes:
                    maior_mes = month

                if distance > 50:
                    distance = 50
                if duration > 72:
                    duration = 72
                distance_list.append(distance)
                duration_list.append(duration)
                countries[country] = 0
                if country > max_country:
                    max_country = country
                if distance > max_distance:
                    max_distance = distance
                if duration > max_duration:
                    max_duration = duration
                distance = self._distance_importance(distance)
                duration = self._duration_importance(duration)
                new_sequence.append([location_category_id, hour, country, distance, duration, week_day, user_id, poi_id, month])

            x, y = sequence_to_x_y(new_sequence, step_size)
            y = remove_hour_from_sequence_y(y)

            x_list.append(x)
            y_list.append(y)

        df['x'] = x_list
        df['y'] = y_list
        df = df[['id', 'x', 'y']]


        # remove users that have few samples
        ids_remove_users = []
        ids_ = df['id'].tolist()
        x_list = df['x'].tolist()
        for i in range(df.shape[0]):
            user = x_list[i]
            if len(user) < n_splits or len(user) < int(minimum/step_size):
                ids_remove_users.append(ids_[i])
                continue

        # remove users that have few samples
        df = df[['id', 'x', 'y']].query("id not in " + str(ids_remove_users))

        x_users = df['x'].tolist()
        kf = KFold(n_splits=n_splits)
        users_train_indexes = [None] * n_splits
        users_test_indexes = [None] * n_splits
        for i in range(len(x_users)):
            user = x_users[i]

            j = 0

            for train_indexes, test_indexes in kf.split(user):
                if users_train_indexes[j] is None:
                    users_train_indexes[j] = [train_indexes]
                    users_test_indexes[j] = [test_indexes]
                else:
                    users_train_indexes[j].append(train_indexes)
                    users_test_indexes[j].append(test_indexes)
                j += 1

        max_userid = len(df)
        df['id'] = np.array([i for i in range(len(df))])
        ids_list = df['id'].tolist()
        x_list = x_users
        y_list = df['y'].tolist()
        for i in range(len(x_list)):
            sequences_list = x_list[i]


            for j in range(len(sequences_list)):
                sequence = sequences_list[j]
                for k in range(len(sequence)):

                    if sequence[k][6] != ids_list[i]:
                        exit()


                sequences_list[j] = sequence
            x_list[i] = sequences_list

        ids = df['id'].tolist()
        x = x_list
        y = y_list

        return {'ids': ids, 'x': x, 'y': y}, users_train_indexes, users_test_indexes, max_userid
    
    def run_tests_one_location_output_k_fold(self,
                                             users_list,
                                             users_train_index,
                                             users_test_index,
                                             n_replications: int,
                                             k_folds,
                                             epochs,
                                             class_weight,
                                             sequences_size,
                                             base_report,
                                             number_of_categories,
                                             batch,
                                             num_users,
                                             parameters,
                                             output_dir):

        folds_histories = []
        histories = []
        iteration = 0
        seeds = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0}
        list_wrong_samples = []
        list_y_wrong_predicted = []
        list_y_right_labels = []
        list_indexes = []
        for i in range(k_folds):
            tf.random.set_seed(seeds[iteration])
            X_train, X_test, y_train, y_test = self.extract_train_test_from_indexes_k_fold_v2(users_list=users_list,
                                                                                           users_train_indexes=
                                                                                           users_train_index[i],
                                                                                           users_test_indexes=
                                                                                           users_test_index[i],
                                                                                           number_of_categories=number_of_categories,
                                                                                           seed=seeds[iteration])

            for j in range(n_replications):
                model = MFA_RNN().build(sequences_size,
                                        location_input_dim=number_of_categories,
                                        num_users=num_users,
                                        time_input_dim=48,
                                        seed=seeds[iteration])
                history, report, indexes = self._train_and_evaluate_model(model,
                                                                 X_train,
                                                                 y_train,
                                                                 X_test,
                                                                 y_test,
                                                                 epochs,
                                                                 batch,
                                                                 parameters,
                                                                )
                base_report = self._add_location_report(base_report, report)
                iteration+=1
                histories.append(history)
                list_indexes.append(indexes)
        folds_histories.append(histories)

        return folds_histories, base_report, list_wrong_samples, list_y_wrong_predicted, list_y_right_labels, list_indexes
    
    def extract_train_test_from_indexes_k_fold_v2(self,
                                               users_list,
                                               users_train_indexes,
                                               users_test_indexes,
                                               number_of_categories,
                                               seed):

        y_train_concat = []
        y_test_concat = []
        ids = users_list['ids']
        x_list = users_list['x']
        y_list = users_list['y']
        x_train_spatial = []
        x_train_temporal = []
        x_train_distance = []
        x_train_duration = []
        x_train_ids = []
        x_train_pois_ids = []
        x_train_adjacency = []
        x_train_directed_adjacency = []
        x_train_distances_matrix = []
        x_train_temporal_matrix = []
        x_train_durations_matrix = []
        x_train_score = []
        x_train_poi_category_probabilities = []
        x_test_spatial = []
        x_test_temporal = []
        x_test_distance = []
        x_test_duration = []
        x_test_ids = []
        x_test_pois_ids = []
        x_test_adjacency = []
        x_test_directed_adjacency = []
        x_test_distances_matrix = []
        x_test_temporal_matrix = []
        x_test_durations_matrix = []
        x_test_poi_category_probabilities = []
        x_test_score = []
        x_train_week_adjacency = []
        x_train_weekend_adjacency = []
        x_test_week_adjacency = []
        x_test_weekend_adjacency = []

        usuario_n = 0
        for i in range(len(ids)):
            usuario_n +=1
            user_x = np.array(x_list[i])
            user_y = np.array(y_list[i])
            X_train = list(user_x[users_train_indexes[i]])
            X_test = list(user_x[users_test_indexes[i]])
            y_train = list(user_y[users_train_indexes[i]])
            y_test = list(user_y[users_test_indexes[i]])
            maximum_train = 100
            maximum_test = int(maximum_train/4)
            if len(X_train) > maximum_train:
                X_train = X_train[:maximum_train]
                y_train = y_train[:maximum_train]
            if len(X_test) > maximum_test:
                X_test = X_test[:maximum_test]
                y_test = y_test[:maximum_test]
            if len(y_train) == 0 or len(y_test) == 0:
                continue

            # x train
            spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train, month_train = sequence_tuples_to_spatial_temporal_and_feature8_ndarrays(X_train)
            # x_test
            spatial_test, temporal_test, country_test, distance_test, duration_test, week_day_test, ids_test, pois_ids_test, month_test = sequence_tuples_to_spatial_temporal_and_feature8_ndarrays(X_test)

            x = [spatial_train, temporal_train, country_train, distance_train, duration_train, week_day_train, ids_train, pois_ids_train, month_train]
            adjacency_matrix_train, adjacency_matrix_test, distances_matrix_train, temporal_matrix_train, durations_matrix_train, score_train, poi_category_probabilities_train, score_test, poi_category_probabilities_test, directed_adjacency_matrix_train, adjacency_week_matrix_train, adjacency_weekend_matrix_train, distance_week_matrix_train, distance_weekend_matrix_train, durations_week_matrix_train, durations_weekend_matrix_train = self._generate_train_test_graph_matrices(x, spatial_test, spatial_train, pois_ids_test, number_of_categories)
            
            x_train_week_adjacency += adjacency_week_matrix_train
            x_train_weekend_adjacency += adjacency_weekend_matrix_train
            x_train_adjacency += adjacency_matrix_train
            x_train_directed_adjacency += directed_adjacency_matrix_train
            x_train_distances_matrix += distances_matrix_train
            x_train_temporal_matrix += temporal_matrix_train
            x_train_durations_matrix += durations_matrix_train
            x_train_poi_category_probabilities += poi_category_probabilities_train
            x_train_score += score_train
            x_test_adjacency += adjacency_matrix_test
            x_test_directed_adjacency += [directed_adjacency_matrix_train[0]]*len(spatial_test)
            x_test_distances_matrix += [distances_matrix_train[0]]*len(spatial_test)
            x_test_temporal_matrix += [temporal_matrix_train[0]]*len(spatial_test)
            x_test_durations_matrix += [durations_matrix_train[0]]*len(spatial_test)
            x_test_poi_category_probabilities += poi_category_probabilities_test
            x_test_score += score_test
            x_test_week_adjacency += [adjacency_week_matrix_train[0]]*len(spatial_test)
            x_test_weekend_adjacency += [adjacency_weekend_matrix_train[0]]*len(spatial_test)

            x_train_spatial += spatial_train
            x_train_temporal += temporal_train
            x_train_distance += distance_train
            x_train_duration += duration_train
            x_train_ids += ids_train
            x_train_pois_ids += pois_ids_train
            x_test_spatial += spatial_test
            x_test_temporal += temporal_test
            x_test_distance += distance_test
            x_test_duration += duration_test
            x_test_ids += ids_test
            x_test_pois_ids += pois_ids_test

            if len(y_train) == 0:
                continue

            y_train_concat = y_train_concat + y_train
            y_test_concat = y_test_concat + y_test

        
        X_train = [np.array(x_train_spatial), 
                   np.array(x_train_temporal),
                    np.array(x_train_distance), 
                    np.array(x_train_duration),
                    np.array(x_train_ids), 
                    np.array(x_train_pois_ids),
                    np.array(x_train_adjacency),
                    np.array(x_train_score),
                    np.array(x_train_poi_category_probabilities),
                    np.array(x_train_week_adjacency), 
                    np.array(x_train_weekend_adjacency),
                    np.array(x_train_durations_matrix),  
                    np.array(x_train_directed_adjacency)]
        
        X_test = [np.array(x_test_spatial), 
                  np.array(x_test_temporal),
                    np.array(x_test_distance), 
                    np.array(x_test_duration),
                    np.array(x_test_ids), 
                    np.array(x_test_pois_ids),
                    np.array(x_test_adjacency),
                    np.array(x_test_score), 
                    np.array(x_test_poi_category_probabilities),
                    np.array(x_test_week_adjacency), 
                    np.array(x_test_weekend_adjacency),
                    np.array(x_test_durations_matrix), 
                    np.array(x_test_directed_adjacency)]

        y_train = y_train_concat
        y_test = y_test_concat

        X_train, y_train = self._shuffle(X_train, y_train, seed, 7)
        X_test, y_test = self._shuffle(X_test, y_test, seed, 7)


        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        y_train = np_utils.to_categorical(y_train, num_classes=number_of_categories)
        y_test = np_utils.to_categorical(y_test, num_classes=number_of_categories)

        y_train = [y_train]
        y_test = [y_test]

        return X_train, X_test, y_train, y_test
    
    def _add_location_report(self, location_report, report):
        for l_key in report:
            if l_key == 'accuracy':
                location_report[l_key].append(report[l_key])
                continue
            for v_key in report[l_key]:
                location_report[l_key][v_key].append(report[l_key][v_key])

        return location_report
    
    def _train_and_evaluate_model(self,
                                  model,
                                  X_train,
                                  y_train,
                                  X_test,
                                  y_test,
                                  epochs,
                                  batch,
                                  parameters):
        
        model.compile(optimizer=parameters['optimizer'], loss=parameters['loss'],
                        metrics= keras.metrics.CategoricalAccuracy(name="acc"))
        hi = model.fit(X_train,
                        y_train,
                        validation_data=(X_test, y_test),
                        batch_size=batch,
                        epochs=epochs,
                        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

        y_predict_location = model.predict(X_test, batch_size=batch)

        entropies = self.entropy_of_predictions(y_predict_location)

        # To transform one_hot_encoding to list of integers, representing the locations
        y_predict_location = one_hot_decoding(y_predict_location)
        y_test_location = one_hot_decoding(y_test[0])
        right_indexes = self.indexes_of_right_predicted_samples(y_predict_location, y_test_location)

        report = skm.classification_report(y_test_location, y_predict_location, output_dict=True)
        wrong_indexes, right_indexes = self.indexes_of_wrong_predicted_samples(y_predict_location, y_test_location)
        y_wrong_predicted = y_predict_location[wrong_indexes]
        y_wrong_labels = y_test_location[wrong_indexes]
        y_righ_predicted = y_predict_location[right_indexes]
        y_right_labels = y_test_location[right_indexes]
        entropy_right = entropies[right_indexes]
        entropy_wrong = entropies[wrong_indexes]
        return hi.history, report, [y_wrong_predicted, y_wrong_labels, y_righ_predicted, y_right_labels, entropy_right, entropy_wrong]
    
    def entropy_of_predictions(self, predictions):

        entropies = []
        for i in range(len(predictions)):
            maximum = max(predictions[i])
            minimum = min(predictions[i])
            normalized = []
            for j in range(len(predictions[i])):
                value = ((predictions[i][j]-minimum)/(maximum - minimum))
                normalized.append(value)
            out = entropy(normalized)
            entropies.append(out)

        return np.array(entropies)
    
    def indexes_of_wrong_predicted_samples(self, y_predicted, y_label):

        indexes = []
        right_indexes = []

        for i in range(len(y_predicted)):

            predicted = y_predicted[i]
            label = y_label[i]
            if predicted != label:
                indexes.append(i)
            else:
                right_indexes.append(i)

        return indexes, right_indexes
    
    def indexes_of_right_predicted_samples(self, y_predicted, y_label):

        indexes = []

        for i in range(len(y_predicted)):

            predicted = y_predicted[i]
            label = y_label[i]
            if predicted == label:
                indexes.append(i)

        return indexes
    
    def output_dir(self, output_base_dir, dataset_type, category_type,model_name):

        return output_base_dir+dataset_type+category_type+model_name
    
    def _generate_train_week_weekend_duration_distance_graph_matrices(self, n_categories, category, temporal, distance, duration):

        adjacency_week = [[0 for j in range(n_categories)] for i in range(n_categories)]
        adjacency_weekend = [[0 for j in range(n_categories)] for i in range(n_categories)]

        distance_week = [[[] for j in range(n_categories)] for i in range(n_categories)]
        distance_weekend = [[[] for j in range(n_categories)] for i in range(n_categories)]

        duration_week = [[[] for j in range(n_categories)] for i in range(n_categories)]
        duration_weekend = [[[] for j in range(n_categories)] for i in range(n_categories)]

        for i in range(len(category)):

            category_sequence = category[i]
            temporal_sequence = temporal[i]
            distance_sequence = distance[i]
            duration_sequence = duration[i]

            for j in range(1, len(temporal_sequence)):

                hour = temporal_sequence[j]
                from_category = int(category_sequence[j-1])
                to_category = int(category_sequence[j])
                distance_value = distance_sequence[j]
                duration_value = duration_sequence[j]
                if hour < 24:

                    adjacency_week, distance_week, duration_week = self._generate_train_week_weekend_graph_matrix(from_category, to_category, adjacency_week, distance_week, duration_week, distance_value, duration_value)

                else:

                    adjacency_weekend, distance_weekend, duration_weekend = self._generate_train_week_weekend_graph_matrix(from_category, to_category, adjacency_weekend, distance_weekend, duration_weekend, distance_value, duration_value)

        distance_week = self._summarize_categories_distance_matrix(distance_week)
        distance_weekend = self._summarize_categories_distance_matrix(distance_weekend)
        duration_week = self._summarize_categories_distance_matrix(duration_week)
        duration_weekend = self._summarize_categories_distance_matrix(duration_weekend)

        return adjacency_week, adjacency_weekend, distance_weekend, distance_weekend, duration_week, duration_weekend
    
    def _generate_train_week_weekend_graph_matrix(self, from_category, to_category, adjacency_matrix, distance_matrix, duration_matrix, distance_value, duration_value):

        # direct
        adjacency_matrix[from_category][to_category] += 1
        distance_matrix[from_category][to_category].append(distance_value)
        duration_matrix[from_category][to_category].append(duration_value)
        # undirect
        adjacency_matrix[to_category][from_category] += 1
        distance_matrix[to_category][from_category].append(distance_value)
        duration_matrix[to_category][from_category].append(duration_value)

        return adjacency_matrix, distance_matrix, duration_matrix



    def _generate_train_test_graph_matrices(self, x_train, spatial_test, sequence_spatial_train, pois_ids_test, n_categories):

        minimum = 0.001
        spatial, temporal, country, distance, duration, week_day, ids, pois_ids, month = x_train

        # week weekend matrices
        adjacency_week, adjacency_weekend, distance_week, distance_weekend, duration_week, duration_weekend = self._generate_train_week_weekend_duration_distance_graph_matrices(n_categories, spatial, temporal, distance, duration)

        # PoiXCategory
        unique_pois_ids = pd.Series(np.array(pois_ids).flatten()).unique().tolist()
        pois_categories_matrix = {int(unique_pois_ids[i]): [0 for j in range(n_categories)] for i in range(len(unique_pois_ids))}

        for i in range(len(pois_ids)):

            categories = spatial[i]
            pois_id = pois_ids[i]


            for j in range(len(categories)):
                category = int(categories[j])
                poi_id = int(pois_id[j])
                pois_categories_matrix[poi_id][category] += 1

        for i in range(len(unique_pois_ids)):

            poi_id = unique_pois_ids[i]
            total = sum(pois_categories_matrix[poi_id])
            pois_categories_matrix[poi_id] = list(np.array(pois_categories_matrix[poi_id])/total)

        sequences_poi_category_train = []
        for i in range(len(pois_ids)):

            pois_id = pois_ids[i]
            sequence = []
            for j in range(len(pois_id)):
                poi_id = int(pois_id[j])
                sequence.append(pois_categories_matrix[poi_id])

            sequences_poi_category_train.append(sequence)

        sequences_poi_category_test = []
        for i in range(len(pois_ids_test)):

            pois_id = pois_ids_test[i]
            sequence = []
            for j in range(len(pois_id)):
                poi_id = int(pois_id[j])
                if pois_id not in list(pois_categories_matrix.keys()):
                    pois_categories_matrix[poi_id] = [1/n_categories for j in range(n_categories)]
                sequence.append(pois_categories_matrix[poi_id])

            sequences_poi_category_test.append(sequence)

        score_train, poi_category_probabilities_train = self.poi_category_matrix(sequences_poi_category_train, spatial)
        score_test, poi_category_probabilities_test = self.poi_category_matrix(sequences_poi_category_test, spatial)




        categories_distances_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_durations_matrix = [[[] for j in range(n_categories)] for i in range(n_categories)]
        categories_adjacency_matrix = [[0 for j in range(n_categories)] for i in range(n_categories)]
        categories_directed_adjacency_matrix = [[0 for j in range(n_categories)] for i in range(n_categories)]
        categories_temporal_matrix = [[0 for j in range(48)] for i in range(n_categories)]

        original_size = len(spatial)
        spatial = np.array(spatial, dtype='int').flatten()
        temporal = np.array(temporal, dtype='int').flatten()
        distance = np.array(distance).flatten()
        duration = np.array(duration).flatten()
        for i in range(1, len(spatial)):
            category = spatial[i]
            hour = temporal[i]

            pre_category = spatial[i - 1]
            categories_distances_matrix[category][pre_category].append(distance[i])
            categories_distances_matrix[pre_category][category].append(distance[i])
            categories_adjacency_matrix[category][pre_category] += 1
            categories_directed_adjacency_matrix[pre_category][category] += 1
            categories_adjacency_matrix[pre_category][category] += 1
            categories_temporal_matrix[category][hour] += 1
            categories_durations_matrix[category][pre_category].append(duration[i])
            categories_durations_matrix[pre_category][category].append(duration[i])

        categories_distances_matrix = self._summarize_categories_distance_matrix(categories_distances_matrix)
        categories_durations_matrix = self._summarize_categories_distance_matrix(categories_durations_matrix)

        categories_adjacency_matrix = np.array(categories_adjacency_matrix) + minimum
        categories_directed_adjacency_matrix = np.array(categories_directed_adjacency_matrix) + minimum
        adjacency_week = np.array(adjacency_week) + minimum
        adjacency_weekend = np.array(adjacency_weekend) + minimum

        # weight adjacency matrix based on each sequence
        list_weighted_adjacency_matrices_train = []
        list_weighted_adjacency_matrices_test = []
        list_weighted_directed_adjacency_matrices = []
        list_weighted_adjacency_week_matrices = []
        list_weighted_adjacency_weekend_matrices = []
        for i in range(0, len(sequence_spatial_train)):
            sequence = sequence_spatial_train[i]
            weighted_adjacency_matrix_train = copy.copy(categories_adjacency_matrix)

            value  = 1
            for j in range(1, len(sequence)):
                category = int(sequence[j])
                pre_category = int(sequence[j-1])
                weighted_adjacency_matrix_train[pre_category][category] = weighted_adjacency_matrix_train[pre_category][category]*value
                weighted_adjacency_matrix_train[category][pre_category] = weighted_adjacency_matrix_train[pre_category][category]*value

            weighted_adjacency_matrix_train = DiffusionConv.preprocess(weighted_adjacency_matrix_train)
            categories_directed_adjacency_matrix = DiffusionConv.preprocess(categories_directed_adjacency_matrix)
            adjacency_week = GCNConv.preprocess(adjacency_week)
            adjacency_weekend = GCNConv.preprocess(adjacency_weekend)
            list_weighted_adjacency_matrices_train.append(weighted_adjacency_matrix_train)
            list_weighted_directed_adjacency_matrices.append(categories_directed_adjacency_matrix)
            list_weighted_adjacency_week_matrices.append(adjacency_week)
            list_weighted_adjacency_weekend_matrices.append(adjacency_weekend)

        for i in range(0, len(spatial_test)):
            sequence = spatial_test[i]
            weighted_adjacency_matrix_test = copy.copy(categories_adjacency_matrix)
            value = 1
            for j in range(1, len(sequence)):
                category = int(sequence[j])
                pre_category = int(sequence[j - 1])
                weighted_adjacency_matrix_test[pre_category][category] = weighted_adjacency_matrix_test[pre_category][category]* value
                weighted_adjacency_matrix_test[category][pre_category] = weighted_adjacency_matrix_test[pre_category][category]* value

            
            weighted_adjacency_matrix_test = DiffusionConv.preprocess(weighted_adjacency_matrix_test)
            list_weighted_adjacency_matrices_test.append(weighted_adjacency_matrix_test)

        adjacency_matrix = list_weighted_adjacency_matrices_train

        distances_matrix = [categories_distances_matrix]*original_size
        temporal_matrix = [categories_temporal_matrix]*original_size
        durations_matrix = [categories_durations_matrix]*original_size
        distance_week = [distance_week]*original_size
        distance_weekend = [distance_weekend]*original_size
        duration_week = [duration_week]*original_size
        duration_weekend = [duration_weekend]*original_size

        return [adjacency_matrix, list_weighted_adjacency_matrices_test, distances_matrix, temporal_matrix, durations_matrix, score_train,
                poi_category_probabilities_train, score_test, poi_category_probabilities_test,
                list_weighted_directed_adjacency_matrices, list_weighted_adjacency_week_matrices,
                list_weighted_adjacency_weekend_matrices,  distance_week, distance_weekend, duration_week, duration_weekend]
    

    def poi_category_matrix(self, sequences_poi_category, categories):
        scores = []
        lasts_rows = []

        for i in range(len(sequences_poi_category)):
            # list of visited categories
            local_categories = categories[i]
            local_categories = local_categories[1:]

            matrix_poi_category = sequences_poi_category[i]
            lasts_rows.append(matrix_poi_category)
            probabilities_to_category = []
            for j in range(len(matrix_poi_category)):
                probabilities_to_category.append(np.argmax(matrix_poi_category[j]))

            probabilities_to_category = probabilities_to_category[:2]
            score = skm.accuracy_score(local_categories, probabilities_to_category)
            scores.append(score)

        return scores, lasts_rows
    
    def _summarize_categories_distance_matrix(self, categories_distances_matrix):
        for row in range(len(categories_distances_matrix)):

            category_distances_list = []
            for column in range(len(categories_distances_matrix[row])):

                values = categories_distances_matrix[row][column]

                if len(values) == 0:
                    categories_distances_matrix[row][column] = 0
                    category_distances_list.append(0)
                else:

                    d_cc = st.median(values)
                    categories_distances_matrix[row][column] = d_cc

        return categories_distances_matrix
    
    def _distance_importance(self, distance):

        distance = distance * distance
        distance = -(distance / (self.distance_sigma * self.distance_sigma))
        distance = math.exp(distance)

        return distance
    
    def _duration_importance(self, duration):

        duration = duration * duration
        duration = -(duration / (self.duration_sigma * self.duration_sigma))
        duration = math.exp(duration)

        return duration
    
    def _shuffle(self, x, y, seed, score_column_index):

        columns = [i for i in range(len(x))]
        data_dict = {}
        for i in range(len(x)):
            feature = x[i].tolist()

            if i == score_column_index:
                for j in range(len(feature)):
                    feature[j] = str(feature[j])
                    data_dict[columns[i]] = feature
                continue

            for j in range(len(feature)):
                feature[j] = str(list(feature[j]))

            data_dict[columns[i]] = feature

        data_dict['y'] = y


        df = pd.DataFrame(data_dict).sample(frac=1., random_state=seed)

        y = df['y'].to_numpy()
        x_new = []

        for i in range(len(columns)):
            feature = df[columns[i]].tolist()
            for j in range(len(feature)):
                feature[j] = json.loads(feature[j])
            x_new.append(np.array(feature))

        return x_new, y
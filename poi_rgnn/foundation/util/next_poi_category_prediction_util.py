import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot
from keras.preprocessing.sequence import pad_sequences

CATEGORIES = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12],
              [0, 13], [0, 14], [0, 15], [0, 16], [0, 17], [0, 18], [0, 19], [0, 20], [0, 21], [0, 22], [0, 23],
              [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12],
              [1, 13], [1, 14], [1, 15], [1, 16], [1, 17], [1, 18], [1, 19], [1, 20], [1, 21], [1, 22], [1, 23],
              [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9], [2, 10], [2, 11], [2, 12],
              [2, 13], [2, 14], [2, 15], [2, 16], [2, 17], [2, 18], [2, 19], [2, 20], [2, 21], [2, 22], [2, 23]
              ]

def sequence_to_x_y(list_events: list, step_size):
    x = []
    y = []
    step = []
    for i in range(step_size, len(list_events)-1):
        step = list_events[i-step_size:i]
        x.append(step)
        y.append(list_events[i+1])

    return x, y

def sequence_to_x_y_v1(list_events: list, step_size):
    x = []
    y = []
    step = []
    cont = 0
    # [location_category_id, hour, day_type, user_id]
    for e in list_events:
        if cont < step_size:
            step.append(e)
            cont = cont + 1
        else:
            y.append(e)
            x.append(step)
            step = []
            cont = 0

    return x, y

def sequence_to_x_y_v2(list_events: list, step_size):
    x = []
    y = []
    step = []
    cont = 0
    # [location_category_id, hour, day_type, user_id]
    # for e in list_events:
    #     if cont < step_size:
    #         step.append(e)
    #         cont = cont + 1
    #     else:
    #         y.append(e)
    #         x.append(step)
    #         step = []
    #         cont = 0

    for i in range(step_size, len(list_events) - 1):
        e = list_events[i]
        step = list_events[i - step_size:i]
        x.append(step)
        y.append(list_events[i + 1])

    return x, y

    return x, y

def remove_hour_from_sequence_x(list_events: list):

    locations = []
    for events in list_events:
        step = []
        for event in events:
            step.append(event[0])

        locations.append(step)

    return locations

def remove_hour_from_sequence_y(list_events: list):

    locations = []
    for e in list_events:
        locations.append(e[0])

    return locations

def return_hour_from_sequence_y(list_events: list):

    hours = []
    for e in list_events:
        hours.append(e[1])

    return np.asarray(hours)

def sequence_tuples_to_ndarray_x(list_events: list):

    locations = []
    for e in list_events:
        locations.append([np.asarray([e[0][0], e[0][1]]), np.asarray([e[1][0], e[1][1]]), np.asarray([e[2][0], e[2][1]])])

    return locations

def sequence_tuples_to_ndarray_y(list_events: list):

    locations = []
    for e in list_events:
        locations.append(np.asarray(e[0]))

    return locations

def one_hot_decoding(data):

    new = []
    for e in data:
        new.append(np.argmax(e))

    return new

def hours_shift_x(list_events, shift = 3):
    """
     Sum "shift" to each hour, in order to change theirs' numbers representation and diferentiates it from locations ids
    :param shift:
    :return:
    """
    locations = []
    for events in list_events:
        step = []
        for event in events:
            step.append([event[0], event[1] + shift])

        locations.append(step)

    return locations

def one_hot_encoding_dense(x):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(CATEGORIES)
    enc.transform(x)

def sequence_tuples_to_spatial_and_temporal_ndarrays(list_events: list, id = None, deep_move = False):

    spatial = []
    temporal = []
    if id == None:
        for e in list_events:
            spatial.append(np.asarray([e[0][0], e[1][0], e[2][0], e[3][0]]))
            temporal.append(np.asarray([e[0][1], e[1][1], e[2][1], e[3][1]]))

        return [np.asarray(spatial), np.asarray(temporal)]
    else:
        ids = []
        datetimes = []
        first_last_datetimes = []
        seconds_between_sequences = []
        for e in list_events:
            spatial.append(np.asarray([e[j][0] for j in range(len(e))]))
            temporal.append(np.asarray([e[j][1] for j in range(len(e))]))
            ids.append(np.asarray([e[j][2] for j in range(len(e))]))
            #ids.append(id)
            if deep_move:
                datetimes.append(np.asarray([e[j][3] for j in range(len(e))]))
                if len(first_last_datetimes) > 0:
                    if first_last_datetimes[-1][2] == e[0][2]:
                        seconds_between_sequences.append((e[-1][3] - first_last_datetimes[-1][0]).seconds)
                    else:
                        seconds_between_sequences.append(-1)
                first_last_datetimes.append((e[0][3], e[-1][3], e[0][2]))

        if deep_move:
            pad_size = 9
            set_spatial_history, set_temporal_history, set_ids_history = sequence_to_historical_and_current(spatial,
                                                                                                            temporal,
                                                                                                            ids,
                                                                                                            datetimes,
                                                                                                            first_last_datetimes,
                                                                                                            seconds_between_sequences,
                                                                                                            pad_size)
            spatial = pad_sequences(spatial, pad_size)
            temporal = pad_sequences(temporal, pad_size)
            ids = pad_sequences(spatial, pad_size)
            return [np.asarray(set_spatial_history), np.asarray(set_temporal_history), np.asarray(set_ids_history),
                    np.asarray(spatial), np.asarray(temporal), np.asarray(ids)]
        else:
            return [np.asarray(spatial), np.asarray(temporal), np.asarray(ids)]

def sequence_to_historical_and_current(spatial, temporal, ids, datetimes, first_last_datetimes, seconds_between_sequences,
                                       pad_size):

    set_spatial_history = [spatial[0].tolist()]
    set_temporal_history = [temporal[0].tolist()]
    set_ids_history = [ids[0].tolist()]
    set_sequences_to_use = []
    user_start_index = 0
    #print("segundos: ", seconds_between_sequences)
    print("tamanho: ", len(spatial), " np: ", np.asarray(spatial).shape)
    for i in range(1, len((spatial))):
        spatial_history = []
        temporal_history = []
        ids_history = []
        ind = 4
        # if i < ind:
        #     print("indice se: ", i)
        for j in reversed(range(i)):
            if seconds_between_sequences[j] != -1:
                # if i < ind:
                #     print("sub: ", [seconds_between_sequences[k] for k in range(j, i)], " index: ", [k for k in range(j, i)],
                #           " sum: ", sum([seconds_between_sequences[k] for k in range(j, i)]))
                if sum([seconds_between_sequences[k] for k in range(j, i)]) <= 86400:
                    spatial_history.append(spatial[j].tolist())
                    temporal_history.append(temporal[j].tolist())
                    ids_history.append(ids[j].tolist())
                else:
                    # exceded time
                    # if i < ind:
                    #     print("f 1: ", spatial_history)
                    if len(spatial_history) > 0:
                        spatial_history = flat(spatial_history)
                        set_spatial_history.append(spatial_history)
                        temporal_history = flat(temporal_history)
                        set_temporal_history.append(temporal_history)
                        ids_history = flat(ids_history)
                        set_ids_history.append(ids_history)
                        spatial_history = []
                        temporal_history = []
                        ids_history = []
                    else:
                        # add itself
                        set_spatial_history.append(spatial[i].tolist())
                        set_temporal_history.append(temporal[i].tolist())
                        set_ids_history.append(ids[i].tolist())
                    break
            else:
                # other user
                user_start_index = i
                if len(spatial_history) > 0:
                    # fill with what was found
                    if i < ind:
                        print("f 2: ", spatial_history)
                    spatial_history = flat(spatial_history)
                    set_spatial_history.append(spatial_history)
                    temporal_history = flat(temporal_history)
                    set_temporal_history.append(temporal_history)
                    ids_history = flat(ids_history)
                    set_ids_history.append(ids_history)
                    spatial_history = []
                    temporal_history = []
                    ids_history = []
                    break
                else:
                    # if none history was found. the history is equal to the current
                    if i < ind:
                        print("f 3: ", spatial[i])
                    set_spatial_history.append(spatial[i].tolist())
                    set_temporal_history.append(temporal[i].tolist())
                    set_ids_history.append(ids[i].tolist())
                break

        if len(spatial_history) > 0:
            # fill with what was found
            if i < ind:
                print("f 4: ", spatial_history)
            spatial_history = flat(spatial_history)
            set_spatial_history.append(spatial_history)
            temporal_history = flat(temporal_history)
            set_temporal_history.append(temporal_history)
            ids_history = flat(ids_history)
            set_ids_history.append(ids_history)
            spatial_history = []
            temporal_history = []
            ids_history = []

    #print("historico: ", set_spatial_history)
    pad_value = 0.0
    max_leght = pad_size
    set_spatial_history = pad_sequences(set_spatial_history, max_leght, value=pad_value)
    set_temporal_history = pad_sequences(set_temporal_history, max_leght, value=pad_value)
    set_ids_history = pad_sequences(set_ids_history, max_leght, value=pad_value)
    # set_spatial_history = [np.asarray(set_spatial_history[i]) for i in range(len(set_spatial_history))]
    # set_temporal_history = [np.asarray(set_temporal_history[i]) for i in range(len(set_temporal_history))]
    # set_ids_history = [np.asarray(set_ids_history[i]) for i in range(len(set_ids_history))]
    #print("historico: ", set_spatial_history)
    print("Final: ", "spatial his: ", len(set_spatial_history), " temporal his: ", len(set_temporal_history),
          " ids his: ", len(set_ids_history))
    print("Primeiro: ", " set_spatial his: ", set_spatial_history[0], " spatial: ", spatial[0])
    print("segundo: ", " set_spatial his: ", set_spatial_history[1], " spatial: ", spatial[1])
    print("terceiro: ", " set_spatial his: ", set_spatial_history[2], " spatial: ", spatial[2])
    print("quarto: ", " set_spatial his: ", set_spatial_history[3], " spatial: ", spatial[3])
    return set_spatial_history, set_temporal_history, set_ids_history
    # set_spatial_history = []
    # set_temporal_history = []
    # set_ids_history = []
    # set_sequences_to_use = []
    #
    # set_spatial_history.append(spatial[0])
    # set_temporal_history.append(temporal[0])
    # set_ids_history.append(ids[0])
    # for i in range(1, len(spatial)):
    #     current_datetime = datetimes[i][-1]
    #     current_id = ids[i][-1]
    #     sequences_to_use = []
    #     spatial_history = np.asarray([])
    #     temporal_history = np.asarray([])
    #     ids_history = np.asarray([])
    #
    #     # It process previous sequences
    #     for j in reversed(range(i)):
    #         ids_sequence = ids[j]
    #         if ids_sequence[0] != current_id:
    #             break
    #         spatial_sequence = spatial[j]
    #         temporal_sequence = temporal[j]
    #         datetimes_sequence = datetimes[j]
    #         use_sequence = True
    #         for datetime in [datetimes_sequence[0], datetimes_sequence[-1]]:
    #             if datetime > current_datetime or (current_datetime - datetime).seconds > 86400:
    #                 use_sequence = False
    #                 break
    #
    #         if use_sequence:
    #             sequences_to_use.append(j)
    #             spatial_history = np.concatenate((spatial_history, spatial_sequence))
    #             temporal_history = np.concatenate((temporal_history, temporal_sequence))
    #             ids_history = np.concatenate((ids_history, ids_sequence))
    #
    #     if len(spatial_history) == 0 and i != 0:
    #         spatial_history = spatial_sequence
    #         temporal_history = temporal_sequence
    #         ids_history = ids_sequence
    #
    #     set_sequences_to_use.append(sequences_to_use)
    #     set_spatial_history.append(spatial_history)
    #     set_temporal_history.append(temporal_history)
    #     set_ids_history.append(ids_history)
    #
    # print("Final: ", "spatial his: ", set_spatial_history, " temporal his: ", set_temporal_history,
    #       " ids his: ", set_ids_history)
    # return set_spatial_history, set_temporal_history, set_ids_history

def pad_sequences_wrap(sequences, max_lengh = 50):

    sequence = flat(sequences)

    sequence = pad_sequences(sequence, maxlen=max_lengh)

    return sequence

def flat(sequences):

    flat_sequence = []
    for sequence in sequences:
        if type(sequence) is not list:
            flat_sequence = flat_sequence + sequence.tolist()
        else:
            flat_sequence = flat_sequence + sequence


    return flat_sequence


def sequence_tuples_to_spatial_temporal_and_feature3_ndarrays(list_events: list):

    spatial = []
    temporal = []
    ids = []
    day_type = []
    for e in list_events:
        spatial.append(np.asarray([e[j][0] for j in range(len(e))]))
        temporal.append(np.asarray([e[j][1] for j in range(len(e))]))
        day_type.append(np.asarray([e[j][2] for j in range(len(e))]))
        ids.append(np.asarray([e[j][3] for j in range(len(e))]))

    return [np.asarray(spatial), np.asarray(temporal), np.array(day_type), np.asarray(ids)]

def sequence_tuples_to_spatial_temporal_and_feature4_ndarrays(list_events: list):

    spatial = []
    temporal = []
    ids = []
    country = []
    distance = []
    for e in list_events:
        spatial.append(np.asarray([e[j][0] for j in range(len(e))]))
        temporal.append(np.asarray([e[j][1] for j in range(len(e))]))
        country.append(np.asarray([e[j][2] for j in range(len(e))]))
        distance.append(np.asarray([e[j][3] for j in range(len(e))]))
        ids.append(np.asarray([e[j][4] for j in range(len(e))]))

    return [np.asarray(spatial), np.asarray(temporal), np.array(country), np.array(distance), np.asarray(ids)]

def sequence_tuples_to_spatial_temporal_and_feature5_ndarrays(list_events: list):

    spatial = []
    temporal = []
    ids = []
    country = []
    distance = []
    duration = []
    for e in list_events:
        spatial.append(np.asarray([e[j][0] for j in range(len(e))]))
        temporal.append(np.asarray([e[j][1] for j in range(len(e))]))
        country.append(np.asarray([e[j][2] for j in range(len(e))]))
        distance.append(np.asarray([e[j][3] for j in range(len(e))]))
        duration.append(np.asarray([e[j][4] for j in range(len(e))]))
        ids.append(np.asarray([e[j][5] for j in range(len(e))]))

    return [np.asarray(spatial), np.asarray(temporal), np.array(country), np.array(distance), np.array(duration), np.asarray(ids)]

def sequence_tuples_to_spatial_temporal_and_feature6_ndarrays(list_events: list):

    spatial = []
    temporal = []
    ids = []
    country = []
    distance = []
    duration = []
    week_day = []
    for e in list_events:
        spatial.append([e[j][0] for j in range(len(e))])
        temporal.append([e[j][1] for j in range(len(e))])
        country.append([e[j][2] for j in range(len(e))])
        distance.append([e[j][3] for j in range(len(e))])
        duration.append([e[j][4] for j in range(len(e))])
        week_day.append([e[j][5] for j in range(len(e))])
        ids.append([e[j][6] for j in range(len(e))])

    # print("Maior country: ", max(np.array(country).flatten()))
    # print("Maior distance: ", max(np.array(distance).flatten()))
    # print("Maior duration: ", max(np.array(duration).flatten()))
    # print("Maior week day: ", max(np.array(week_day).flatten()))

    return [spatial, temporal, country, distance, duration, week_day, ids]

def sequence_tuples_to_spatial_temporal_and_feature7_ndarrays(list_events: list):

    spatial = []
    temporal = []
    ids = []
    country = []
    distance = []
    duration = []
    week_day = []
    poi = []
    for e in list_events:
        spatial.append([e[j][0] for j in range(len(e))])
        temporal.append([e[j][1] for j in range(len(e))])
        country.append([e[j][2] for j in range(len(e))])
        distance.append([e[j][3] for j in range(len(e))])
        duration.append([e[j][4] for j in range(len(e))])
        week_day.append([e[j][5] for j in range(len(e))])
        ids.append([e[j][6] for j in range(len(e))])
        poi.append([e[j][7] for j in range(len(e))])

    # print("Maior country: ", max(np.array(country).flatten()))
    # print("Maior distance: ", max(np.array(distance).flatten()))
    # print("Maior duration: ", max(np.array(duration).flatten()))
    # print("Maior week day: ", max(np.array(week_day).flatten()))

    return [spatial, temporal, country, distance, duration, week_day, ids, poi]

def sequence_tuples_to_spatial_temporal_and_feature8_ndarrays(list_events: list):

    spatial = []
    temporal = []
    ids = []
    country = []
    distance = []
    duration = []
    week_day = []
    poi = []
    month = []
    for e in list_events:
        spatial.append([e[j][0] for j in range(len(e))])
        temporal.append([e[j][1] for j in range(len(e))])
        country.append([e[j][2] for j in range(len(e))])
        distance.append([e[j][3] for j in range(len(e))])
        duration.append([e[j][4] for j in range(len(e))])
        week_day.append([e[j][5] for j in range(len(e))])
        ids.append([e[j][6] for j in range(len(e))])
        poi.append([e[j][7] for j in range(len(e))])
        month.append([e[j][8] for j in range(len(e))])

    # print("Maior country: ", max(np.array(country).flatten()))
    # print("Maior distance: ", max(np.array(distance).flatten()))
    # print("Maior duration: ", max(np.array(duration).flatten()))
    # print("Maior week day: ", max(np.array(week_day).flatten()))
    # print("maior poi: ", max(np.array(poi).flatten()))
    # print("Maior mes de: ", max(np.array(month).flatten()))

    return [spatial, temporal, country, distance, duration, week_day, ids, poi, month]

def plot_history_metrics(h: pd.DataFrame, metrics_names: list, figure_dir, show=False):

    for metric_name in metrics_names:
        pyplot.figure(figsize=(12, 12))
        pyplot.title(metric_name[2])
        pyplot.plot(h[metric_name[0]])
        pyplot.plot(h[metric_name[1]])
        pyplot.ylabel(metric_name[2])
        pyplot.xlabel('Epoch')
        pyplot.legend(['Training', 'Validation'], loc='upper right')
        pyplot.savefig(figure_dir + metric_name[2] + ".png")
        if show:
            print(h)
            pyplot.show()

def save_report(report_name, n_tests, epochs, report: dict, dir):
    # f = open(dir + report_name, 'w')
    # f.write(str(report))
    # f.close()

    df = pd.DataFrame(report)
    df.to_csv(dir + report_name + "_" + str(n_tests) + ".csv", index_label="metric")


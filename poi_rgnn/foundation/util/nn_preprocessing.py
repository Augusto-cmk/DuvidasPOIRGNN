import numpy as np

def one_hot_decoding(data):

    new = []
    for e in data:
        new.append(np.argmax(e))

    return np.array(new)

def one_hot_decoding_predicted(data):

    new = []
    for e in data:
        node_label = []
        for node in e:
            node_label.append(np.argmax(node))
        new.append(node_label)

    new = np.array(new).flatten()
    return new

def top_k_rows(data, k):

    row_sum = []
    for i in range(len(data)):
        row_sum.append([np.sum(data[i]), i])

    row_sum = sorted(row_sum, reverse=True, key=lambda e:e[0])
    # if len(row_sum) > k:
    # if row_sum[k][0] < 4:
    #     print("ola")
    row_sum = row_sum[:k]

    row_sum = [e[1] for e in row_sum]

    return np.array(row_sum)

def filter_data_by_valid_category(user_matrix, user_category, osm_categories):

    idx = []
    print("Tamanho user cate: ",  user_category.shape)
    print("Tamanho user matr: ", user_matrix.shape)
    for i in range(len(user_category)):
        if user_category[i] == "" or user_category[i] == " ":
            continue
        elif user_category[i] not in osm_categories:
            continue
        else:
            idx.append(i)
    idx = np.array(idx)
    if len(idx) == 0:
        return np.array([]), np.array([])
    print(idx)
    user_matrix = user_matrix[idx[:, None], idx]
    user_category = user_category[idx]
    return user_matrix, user_category

def weighted_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask
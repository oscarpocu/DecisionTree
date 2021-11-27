import numpy as np
"""Hola"""

def precision(predictions, y):
    prediction_classes = np.unique(y)
    if prediction_classes.shape[0] != 2:
        print("Only supported binary classes")
        exit(-1)
    correct = predictions[predictions == y]
    false = predictions[predictions != y]
    true_positive = correct[correct == prediction_classes[0]]
    false_positive = false[false == prediction_classes[0]]
    return true_positive.shape[0] / (true_positive.shape[0] + false_positive.shape[0])


def accuracy(predictions, y):
    return y[(predictions == y)].shape[0] / y.shape[0]


def recall(predictions, y):
    prediction_classes = np.unique(y)
    if prediction_classes.shape[0] != 2:
        print("Only supported binary classes")
        exit(-1)
    correct = predictions[predictions == y]
    false = predictions[predictions != y]
    true_positive = correct[correct == prediction_classes[0]]
    false_negative = false[false == prediction_classes[1]]
    return true_positive.shape[0] / (true_positive.shape[0] + false_negative.shape[0])


def cross_val_score(model, x, y, cv=5, scoring="accuracy"):
    if scoring == "accuracy":
        scoring_func = accuracy
    elif scoring == "precision":
        scoring_func = precision
    elif scoring == "recall":
        scoring_func = recall
    else:
        scoring_func = accuracy

    y = np.reshape(y, (y.shape[0], 1))
    data = np.hstack((x, y))
    np.random.shuffle(data)

    _x = data[:, :-1]
    _y = data[:, -1]
    n_rows = int(np.trunc(x.shape[0] / cv))
    x_partitions = []
    y_partitions = []

    i = 0
    for _ in range(cv - 1):
        x_partitions.append(_x[i * n_rows:(i + 1) * n_rows, :])
        y_partitions.append(_y[i * n_rows:(i + 1) * n_rows])
        i += 1

    x_partitions.append(_x[i * n_rows:, :])
    y_partitions.append(_y[i * n_rows:])

    scores = []
    index_set = np.array([x for x in range(cv)])

    for it_index in range(cv):
        x_test = x_partitions[it_index]
        x_train_partitions = np.array(x_partitions, dtype=object)[index_set != it_index]
        y_test = y_partitions[it_index]
        y_train_partitions = np.array(y_partitions, dtype=object)[index_set != it_index]

        x_train = x_train_partitions[0]
        y_train = y_train_partitions[0]

        for j in range(1, cv-1):
            x_train = np.vstack((x_train, x_train_partitions[j]))
            y_train = np.concatenate((y_train, y_train_partitions[j]))

        model.fit(x_train, y_train)
        scores.append(scoring_func(model.predict(x_test), y_test))

    return np.mean(np.array(scores))




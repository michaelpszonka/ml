from math import sqrt, pow
import collections
#reads input file
def get_data(file_name):
    data = list()
    file = open(f'../data/{file_name}', 'r')
    for line in file.readlines():
        row = line.split(',')
        processed_row = list()
        for i in range(len(row) - 1):
            processed_row.append(float(row[i]))
        processed_row.append((row[-1].strip()))
        data.append(processed_row)
    return data


def mean(column):
    return sum(column)/len(column)

def mode(labels):
    most_common_label = collections.Counter(labels).most_common()
    return most_common_label[0][0]


def euclidean_distance(query, regression_data):
    squared_distance = 0
    for feature in range(len(regression_data) - 1):
        squared_distance += pow(regression_data[feature] - query[feature], 2)
    return sqrt(squared_distance)

def knn(query, regression_data, k, choice_fn):
    neighbor_values = []
    for idx, row in enumerate(regression_data, 0):
        distance = euclidean_distance(query, row)
        neighbor_values.append((distance, idx))

    sorted_neighbor_values = sorted(neighbor_values)
    k_nearest_neighbors_and_indicies = sorted_neighbor_values[:k]
    k_nearest_labels = [regression_data[idx][1] for distance, idx in k_nearest_neighbors_and_indicies]


    knn_prediction = choice_fn(k_nearest_labels)

    return knn_prediction, k_nearest_neighbors_and_indicies






if __name__ == '__main__':
    #height to weight
    #regression data
    # height_weight_regression_data = [
    #    [65.75, 112.99],
    #    [71.52, 136.49],
    #    [69.40, 153.03],
    #    [68.22, 142.34],
    #    [67.79, 144.30],
    #    [68.70, 123.30],
    #    [69.80, 141.49],
    #    [70.01, 136.46],
    #    [67.90, 112.37],
    #    [66.49, 127.45],
    # ]
    # 
    # query = [68]
    # knn_weight, k_nearest_neighbors = knn(query, height_weight_regression_data, 3, choice_fn=mean)
    # print(f'Likely weight for someone who is {query[0]} inches tall: {knn_weight}')

    #clasification_data
    clf_data = [
        [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0],
    ]

    query_clf = [1]
    knn_weight_clf_data, k_nearest_neighbors_clf_data = knn(query_clf, clf_data, 3, choice_fn=mode)
    print(knn_weight_clf_data)

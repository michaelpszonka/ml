from math import pi, sqrt, exp, pow

#reads input file
def get_data(file_name):
    data = list()
    file = open(f'../naive_bayes/data/{file_name}', 'r')
    for line in file.readlines():
        row = line.split(',')
        processed_row = list()
        for i in range(len(row) - 1):
            processed_row.append(float(row[i]))
        processed_row.append((row[-1].strip()))
        data.append(processed_row)
    return data


def mean(feature):
    return sum(feature) / len(feature)

def std(feature):
    mu = mean(feature)
    var = sum([pow(val-mu, 2) for val in feature]) / len(feature)

    return sqrt(var)

def seperate_by_class(dataset):
    classes = dict()
    for row in dataset:
        label = row[-1]
        if label not in classes:
            classes[label] = list()
        classes[label].append(row[0:-1])
    return classes

def summarize(dataset):
    parsed_data = seperate_by_class(dataset)
    summaries = dict()
    for classes, data in parsed_data.items():
        summaries[classes] = [(mean(feature), std(feature), len(feature)) for feature in zip(*data)]

    return summaries

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def predict(training_summary, test_data, total_rows):
    probabilities = dict()
    feature_rows = len(test_data)
    p = 0
    for label in training_summary:
        for f in range(feature_rows):
            p += calculate_probability(test_data[f], training_summary[label][f][0], training_summary[label][f][1]) * float(training_summary[label][f][2] / total_rows)
        probabilities[label] = p
        p = 0

    return probabilities





if __name__ == '__main__':
    test_data = (get_data('iris_flowers'))
    test_value = [5.1,3.5,1.4,0.2]
    data_sum = summarize(test_data)
    res = predict(data_sum, test_value, len(test_data))

    print(res)
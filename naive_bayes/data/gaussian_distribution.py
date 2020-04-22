from math import sqrt, pi, exp

#reads input file
def get_data(file_name):
    file = open(f'./data/{file_name}', 'r')
    data = [line.strip().split(',') for line in file.readlines()]
    return list(data)

#seperates each class into a dict with its class label as a key, and its associate data as a list of values
def seperate_by_class(dataset):
    seperated_data = dict()
    for row in dataset:
        class_label = row[-1]
        if(class_label not in seperated_data):
            seperated_data[class_label] = list()
        seperated_data[class_label].append(row[:-1])
    return seperated_data

#calculate mean of a column of data
def mean(column):
    return sum(column) / len(column)

#calculate standard deviation of a colum of data
def std_dev(column):
    mu = mean(column)
    variance = sum([(mu - d)**2 for d in column])/(len(column) - 1)
    std = sqrt(variance)
    return std

# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

#given a full dataset, return the mean and standard deviation of each column in the dataset
def summarize_dataset(dataset):
    summary = [(mean(column), std_dev(column)) for column in zip(*dataset)]
    return summary

#provide summary of dataset by column and by class
def summarize_by_class(dataset):
    summaries = dict()
    data_by_class = seperate_by_class(dataset)
    for class_value, rows in data_by_class.items():
        summaries[class_value] = summarize_dataset(rows)

    return summaries

if __name__ == "__main__":
    training_data = get_data('iris_flowers')
    test_data = [[3.393533211,2.331273381,0],
                [3.110073483,1.781539638,0],
                [1.343808831,3.368360954,0],
                [3.582294042,4.67917911,0],
                [2.280362439,2.866990263,0],
                [7.423436942,4.696522875,1],
                [5.745051997,3.533989803,1],
                [9.172168622,2.511101045,1],
                [7.792783481,3.424088941,1],
                [7.939820817,0.791637231,1]]

    # processed_data = seperate_by_class(training_data)
    summary = summarize_by_class(test_data)
    print(summary)


    # summary = summarize_dataset(test_data)
#
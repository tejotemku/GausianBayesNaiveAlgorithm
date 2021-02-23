import random
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


class BayesNaiveAlgorithm:
    def __init__(self, data, nr_of_columns, percentage, graphs_on=False):
        self.training_samples_length = len(data)//nr_of_columns
        self.training_samples = {}
        self.testing_samples = []
        self.testing_samples_length = 0
        self.graphs_on = graphs_on
        self.percentage = percentage
        self.result = 0

        for i in range(0, self.training_samples_length):
            sample = data[i*nr_of_columns: (i*nr_of_columns) + nr_of_columns]
            class_value = data[(i*nr_of_columns) + nr_of_columns - 1]
            if class_value not in self.training_samples:
                self.training_samples[class_value] = []
            self.training_samples[class_value].append(sample)
        self.divide_data()
        self.algorithm()

    # cross validation by percentage of initial data being picked for testing
    def divide_data(self):
        self.testing_samples_length = int(self.training_samples_length * self.percentage)
        self.training_samples_length -= self.testing_samples_length

        for value in self.training_samples:
            for i in range(self.testing_samples_length // 3):
                index = random.randrange(len(self.training_samples[value]))
                self.testing_samples.append(self.training_samples[value].pop(index))

    def class_probs(self, summaries, sample):
        total_rows = self.testing_samples_length
        probs = {}
        for class_value, summary in summaries.items():
            probs[class_value] = summaries[class_value][0][2]/total_rows
            for i in range(len(summary)):
                avg, stdev, length = summary[i]
                probs[class_value] *= (1 / (math.sqrt(2 * math.pi) * stdev)) * math.exp(-((sample[i]-avg)**2 / (2 * stdev**2 )))
        return probs

    def predict(self, summaries, sample):
        probabilities = self.class_probs(summaries, sample)
        best_label, best_prob = None, -1
        sum = 0
        for class_value, prob in probabilities.items():
            sum += prob
            if best_label is None or prob > best_prob:
                best_prob = prob
                best_label = class_value
        return best_label, best_prob/sum

    @staticmethod
    def stdev(column):
        avg = sum(column)/len(column)
        return math.sqrt(sum([(x-avg)**2 for x in column]) / float(len(column)))

    def algorithm(self):
        # calculating avg, stdev, size
        summaries = {}
        for class_value, samples in self.training_samples.items():
            summaries[class_value] = [
                (sum(column)/len(column),
                 self.stdev(column),
                 len(column)
                 ) for column in zip(*list(s[:-1] for s in samples))]

        # testing
        correct_results = 0
        graphs = {}
        error_matrix = {}
        for sample in self.testing_samples:
            p = self.predict(summaries, sample[:-1])
            correct_results += int(sample[-1] == p[0])

            if sample[-1] not in error_matrix:
                error_matrix[sample[-1]] = {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 0}
            error_matrix[sample[-1]][p[0]] += 1
            if self.graphs_on:
                print(sample, p)
            if sample[-1] not in graphs:
                graphs[sample[-1]] = [[], []]
            graphs[sample[-1]][0].append(int(sample[-1] == p[0]))
            graphs[sample[-1]][1].append(p[1])
        self.result = correct_results / self.testing_samples_length
        if self.graphs_on:
            for e in error_matrix:
                print(error_matrix[e].values())
            print(correct_results, self.testing_samples_length)
            plt.figure()
            for s in graphs:
                fpr, tpr, tresholds = roc_curve(graphs[s][0], graphs[s][1])
                plt.plot(fpr, tpr, label=s)
            plt.title(f'Roc curve - {int(self.percentage*100)}% data is tested')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(graphs)
            plt.grid(True)
            plt.show()


# data in table in format [ value, value, ..., class, value, value, .... class, ... ]
data = [ ]

results = []
probs = []
for i in range(8):
    prob = 0.1 * (i + 1)
    avg = 0
    for j in range(1000):
        b = BayesNaiveAlgorithm(data, 5, prob)
        avg += b.result
    probs.append(prob*100)
    results.append(avg/1000)
plt.figure()
plt.plot(probs, results)
plt.xlabel("Probability - %")
plt.ylabel("Avg Succes rate")
plt.grid(True)
plt.show()
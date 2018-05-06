class NaiveBayes(object):
    def __init__(self, legalLabels):
        pass

    def train(self, data):
        frequency_table = {}
        labels = []
        occurrences = {}

        for image_data, label in data:
            if label not in frequency_table:
                frequency_table[label] = {}
                labels.append(label)
                occurrences[label] = 0

            occurrences[label] += 1

            for pixel in image_data:
                if pixel not in frequency_table[label]:
                    frequency_table[label][pixel] = {
                        'black': 0,
                        'white': 0
                    }

                if image_data[pixel] == 1:
                    frequency_table[label][pixel]['black'] += 1
                else:
                    frequency_table[label][pixel]['white'] += 1

        self.frequency_table = frequency_table
        self.labels = labels
        self.total_images = len(data)
        self.occurrences = occurrences

    def classify(self, data):
        guesses = []

        for image in data:
            highest_probability = -2147483647
            best_guess = self.labels[0]

            for label in self.labels:
                probability = self.calculatePosterior(label, image)

                if probability > highest_probability:
                    highest_probability = probability
                    best_guess = label

            guesses.append(best_guess)

        return guesses

    def calculatePosterior(self, label, data):
        trainingData = self.frequency_table[label]

        import math

        prior = math.log(float(self.occurrences[label]) / float(self.total_images))

        for pixel in data:
            total_values = float(trainingData[pixel]['black'] + trainingData[pixel]['white'])

            if data[pixel] == 1:
                if trainingData[pixel]['black'] == 0:
                    prior += math.log(float(1) / (total_values + 1))
                else:
                    prior += math.log(float(trainingData[pixel]['black'] + 1) / (total_values + 1))
            else:
                if trainingData[pixel]['white'] == 0:
                    prior += math.log(float(1) / (total_values + 1))
                else:
                    prior += math.log(float(trainingData[pixel]['white'] + 1) / (total_values + 1))

        return prior
class NaiveBayes(object):
    def __init__(self):
        pass

    def train(self, data):
        frequency_table = {}
        labels = []

        for label, image_data in data:
            if label not in frequency_table:
                frequency_table[label] = {}
                labels.append(label)

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

        self.frequency_taqble = frequency_table
        self.labels = labels
        self.total_images = len(data)

    def classify(self, data):
        guesses = []

        for image in data:
            highest_probability = 0
            best_guess = self.labels[0]

            for label in self.labels:
                probability = self.calculatePosterior(label, image)

                if probability > highest_probability:
                    highest_probability = probability
                    best_guess = label

            guesses.append(best_guess)

        return guesses

    def calculatePosterior(self, label, data):
        trainingData = self.frequency_taqble[label]

        probability = 1

        for pixel in data:
            total_values = float(trainingData[pixel]['black'] + trainingData[pixel]['white'])

            if data[pixel] == 1:
                if trainingData[pixel]['black'] == 0:
                    probability *= float(1) / total_values
                    # print 'multiplying by', float(1) / total_values
                else:
                    probability *= float(trainingData[pixel]['black']) / total_values
                    # print 'multiplying by', float(trainingData[pixel]['black']) / total_values
            else:
                if trainingData[pixel]['white'] == 0:
                    probability *= float(1) / total_values
                    # print 'multiplying by', float(1) / total_values
                else:
                    probability *= float(trainingData[pixel]['white']) / total_values
                    # print 'multiplying by', float(trainingData[pixel]['white']) / total_values

        return probability
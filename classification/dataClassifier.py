import samples
from naiveBayes import NaiveBayes
from perceptron import PerceptronClassifier

DIGIT_DATUM_WIDTH = 28
DIGIT_DATUM_HEIGHT = 28
FACE_DATUM_WIDTH = 60
FACE_DATUM_HEIGHT = 70


MODE_DIGITS = 0
MODE_FACES = 1
legalLabels = []

# Change these lines.
algorithm = NaiveBayes
mode = MODE_FACES
num_training = 450
num_testing = 150


def basicFeatureExtractorDigit(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is white (0) or gray/black (1)
    """
    a = datum.getPixels()

    features = {}
    for x in range(DIGIT_DATUM_WIDTH):
        for y in range(DIGIT_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0

    return features


def basicFeatureExtractorFace(datum):
    """
    Returns a set of pixel features indicating whether
    each pixel in the provided datum is an edge (1) or no edge (0)
    """
    a = datum.getPixels()

    features = {}
    for x in range(FACE_DATUM_WIDTH):
        for y in range(FACE_DATUM_HEIGHT):
            if datum.getPixel(x, y) > 0:
                features[(x, y)] = 1
            else:
                features[(x, y)] = 0

    return features


if mode == MODE_DIGITS:
    rawTrainingData = samples.loadDataFile("digitdata/trainingimages", num_training, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", num_training)
    rawTestData = samples.loadDataFile("digitdata/testimages", num_testing, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", num_testing)
    legalLabels = [0,1,2,3,4,5,6,7,8,9]

    feature_function = basicFeatureExtractorDigit
else:
    rawTrainingData = samples.loadDataFile("facedata/facedatatrain", num_training, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", num_training)
    rawTestData = samples.loadDataFile("facedata/facedatatest", num_testing, FACE_DATUM_WIDTH, FACE_DATUM_HEIGHT)
    testLabels = samples.loadLabelsFile("facedata/facedatatestlabels", num_testing)
    legalLabels = [0,1]

    feature_function = basicFeatureExtractorFace

trainingData = map(feature_function, rawTrainingData)
testData = map(feature_function, rawTestData)

classifier = algorithm(legalLabels)

import time
start = time.time()
classifier.train(zip(trainingData, trainingLabels))
end = time.time()

print end - start

guesses = classifier.classify(testData)
correct = [guesses[i] == testLabels[i] for i in range(len(testLabels))].count(True)

print str(correct), ("correct out of " + str(len(testLabels)) + " (%.1f%%).") % (100.0 * correct / len(testLabels))

import random
image = random.randint(0, len(testLabels))

print rawTestData[image]
print 'actual:', testLabels[image], 'guessed:', classifier.classify([feature_function(rawTestData[image])])[0]

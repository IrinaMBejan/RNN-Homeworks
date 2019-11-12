#! /usr/bin/env python

# Tema 2 Retele Neuronale
# Bejan Irina Madalina 

import cPickle
import gzip
import random

import numpy


PATH = "mnist.pkl.gz"
LEARNING_RATE = 0.2
ITER_MAX = 5
TRAINSET_SIZE = 50000
TESTSET_SIZE = 10000


def load_data(path):
    with gzip.open(path, "rb") as stream:
        train, valid, test = cPickle.load(stream)
    return {
        "train": train,
        "valid": valid,
        "test": test,
    }


def activation(value):
    """Activation function, returns 1 if the value is over a certain value."""
    return float(value > 0)


def train_digit(digit, train_data):
    iter_count = 0
    bias = 0
    classified_all = False

    # initialize weights
    weights = numpy.random.uniform(0, 1, len(train_data[0][0]))

    while not classified_all and iter_count < ITER_MAX:
        for idx in range(TRAINSET_SIZE):
            label, pixels = map(lambda pos: train_data[pos][idx], (1, 0))
            expect = float(label == digit)            
            total = numpy.add(numpy.dot(weights, pixels), bias)
            trigger = activation(total)
            adjust = (expect - trigger) * LEARNING_RATE
            bias += adjust
            if expect == trigger:
                classified_all = True

            weights = numpy.add(weights, numpy.dot(pixels, adjust))
        iter_count += 1

    return bias, weights


def print_results(digits, total):
    for idx, ok in enumerate(digits[0]):
        print("Digit {}: {}".format(idx, float(ok) / digits[1][idx]))
    print("Average: {}".format(float(total[0]) / total[1]))


def compute_score(perceptron, pixels):
    bias, weights = perceptron
    total = numpy.add(numpy.dot(weights, pixels), bias)
    return total


def test(perceptrons, data):
    digit_correct = [0] * 10
    digit_all = [0] * 10
    correct_count = 0
    
    test_data = data["test"]
    for idx in range(TESTSET_SIZE):
        label, pixels = map(lambda pos: test_data[pos][idx], (1, 0))
        total = compute_score(perceptrons[label], pixels)
        if activation(total):
            digit_correct[label] += 1
            correct_count += 1
        digit_all[label] += 1
    print_results((digit_correct, digit_all), (correct_count, TESTSET_SIZE))


def recognize(perceptrons, data):
    test_data = data["test"]
    indexes = random.sample(range(TESTSET_SIZE), 10)
    for idx in indexes:
        label, pixels = map(lambda pos: test_data[pos][idx], (1, 0))
        perceptron_dict = {}
        for idx, perceptron in enumerate(perceptrons):
            perceptron_dict[idx] = compute_score(perceptron, pixels)
        guess = sorted(perceptron_dict, key=lambda arg: perceptron_dict[arg], reverse=True)
        print("{}: {}".format(label, ", ".join(map(str, guess))))


def main():
    data = load_data(PATH)
    
    # Train a perceptron for each digit
    perceptrons = [train_digit(d, data["train"]) for d in range(10)]
    
    test(perceptrons, data)
    recognize(perceptrons, data)


if __name__ == "__main__":
    main()

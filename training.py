#!/usr/bin/python

from pyfann import libfann
import sys

if __name__ == "__main__":
    connection_rate = 1
    learning_rate = 0.7
    num_input = 20
    num_hidden = 3
    num_output = 1

    desired_error = 0.0001
    max_iterations = 100000
    iterations_between_reports = 1000

    ann = libfann.neural_net()
    ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
    ann.set_learning_rate(learning_rate)
    ann.set_activation_function_hidden(libfann.ELLIOT_SYMMETRIC)
    ann.set_activation_function_output(libfann.ELLIOT_SYMMETRIC)

    ann.train_on_file(sys.argv[1], max_iterations, iterations_between_reports, desired_error)

    ann.save(sys.argv[1]+".net")

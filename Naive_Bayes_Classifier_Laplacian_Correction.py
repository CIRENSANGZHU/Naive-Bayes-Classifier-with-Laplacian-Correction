# A Naive Bayes Classifier with Laplacian Correction
# The training data set is the Watermelon Data Set 3.0, in page 84 of our text book
# The testing data is the watermelon sample in page 151

import xlrd
import numpy as np
import math
import copy
from copy import deepcopy


def load_training_data(filename):
    """
    :param filename: the file name of the excel file including training data set, a string, the data should be on sheet1.
    eg: 'Watermelon_Dataset_3.0.xlsx'

    :return attributes: the names and corresponding possible values of all attributes, it's a dictionary.
    eg: {'color': ['green', 'black', 'white'], 'root': ['curled', 'little_curled', 'straight'], ...}
    :return training_samples: all training samples attributes, values and labels, each sample is a dictionary, and combined to a list.
    eg: [{'color': 'green', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.697, 'sugar_content': 0.46, 'label': 'good'}, ...}
    """

    # open the sheet of the excel file
    training_data_list = xlrd.open_workbook(filename)  # open the excel file of training data set
    sheet1 = training_data_list.sheets()[0]  # the data set is in sheet1 of the excel file

    # row and column number start at 0
    sample_num = sheet1.nrows - 1  # the number of training samples, the first row in the excel are attribute names
    attribute_num = sheet1.ncols - 2  # the number of attributes, the first column is No. and the last is label

    # process the attributes information
    attributes = {}  # the list of all attributes names and corresponding values
    for i in range(attribute_num):
        # i = 0, 1, ..., (attribute_num - 1)
        attribute_name = sheet1.cell(0, i+1).value  # the name of current attribute
        attribute_value = []  # possible values of current attribute
        attribute = {}  # the complete information of current attribute, including attribute_name and attribute_value
        for j in range(sample_num):
            # j = 0, 1, ..., (sample_num - 1)
            value = sheet1.cell(j+1, i+1).value
            if value not in attribute_value:
                attribute_value.append(value)
        attributes[attribute_name] = attribute_value

    # read training samples data
    training_samples = []  # list all training samples, each sample is a dictionary
    for i in range(sample_num):
        # i = 0, 1, ..., (sample_num - 1)
        current_sample = {}  # each sample
        for j in range(attribute_num):
            # j = 0, 1, ..., (attribute_num - 1)
            attribute_name = sheet1.cell(0, j+1).value  # each attribute
            attribute_value = sheet1.cell(i+1, j+1).value  # the value of this sample at this attribute
            current_sample[attribute_name] = attribute_value
        current_sample['label'] = sheet1.cell(i+1, attribute_num+1).value
        training_samples.append(current_sample)

    return attributes, training_samples


def class_prior_probability_estimation(training_samples):
    """
    Notice: Laplacian Correction Adopted. Line 85 And Line 90.

    :param training_samples: data of all training samples, the format are shown in load_training_data function.

    :return class_prior_probability: the estimation of each possible label, a dictionary.
    eg: {'good': 0.4705882352941177, 'bad': 0.5294117647058824}
    """

    # get all possible values of label
    label_type = []  # list all possible values of label shown in the training samples
    for each_sample in training_samples:
        each_label = each_sample['label']
        if each_label not in label_type:
            label_type.append(each_label)

    # initialize class_prior_probability
    class_prior_probability = {}  # a dictionary, keys are classes, values are corresponding probability
    label_value_num = len(label_type)  # the number of possible values of label
    sample_num = len(training_samples)
    for i in range(label_value_num):
        # i = 0, 1, ..., (label_num - 1)
        label_value = label_type[i]
        class_prior_probability[label_value] = 1/(sample_num + label_value_num)  # Laplacian Correction

    # estimate the class_prior_probability
    for each_sample in training_samples:
        each_label = each_sample['label']
        class_prior_probability[each_label] = class_prior_probability[each_label] + 1/(sample_num + label_value_num)  # Laplacian Correction

    return class_prior_probability


def normal_distribution_parameter_estimation(data):
    """
    Notice: Unbiased Estimation Adopted. Line 115.

    :param data: a list, each element is a real number, the value of some attribute
    eg: [0.46, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211]

    :return miu: the estimation of miu of the normal distribution based on 'data'
    eg: 0.27875
    :return sigma: the estimation of sigma of the normal distribution based on 'data'
    eg: 0.10092394590553255
    """

    miu = np.mean(data)  # estimate miu of the normal distribution

    sigma = 0  # initial sigma
    data_num = len(data)  # the number of data
    # estimate sigma of the normal distribution
    for each_data in data:
        sigma = sigma + (each_data-miu) ** 2
    sigma = sigma/(data_num-1)  # unbiased estimation adopted!!
    sigma = sigma ** 0.5

    return miu, sigma


def num_of_sample_of_different_label(training_samples):
    """
    :param training_samples: data of all training samples, the format are shown in load_training_data function.

    :return num_of_each_label: number of samples of each kind of label, a dictionary.
    eg: {'good': 8, 'bad': 9}
    """

    # get all possible values of label
    label_type = []  # list all possible values of label shown in the training samples
    for each_sample in training_samples:
        each_label = each_sample['label']
        if each_label not in label_type:
            label_type.append(each_label)

    # initialize num_of_each_label
    num_of_each_label = {}
    for label in label_type:
        num_of_each_label[label] = 0

    # get num_of_each_label
    for each_sample in training_samples:
        label = each_sample['label']
        num_of_each_label[label] += 1

    return num_of_each_label


def discrete_attribute_conditional_probability_estimation(training_samples, attributes):
    """
    Notice: Laplacian Correction Adopted. Line 194 And Line 202.

    :param training_samples: data of all training samples, the format are shown in load_training_data function.
    :param attributes: the names and corresponding possible values of all attributes, a dictionary, shown in load_training_data function.

    :return discrete_attribute_conditional_probability: a dictionary.
    eg: {'good': {'green': 0.375, 'black': 0.5,...}, 'bad': {'green': 0.3333333333333333, 'black': 0.2222222222222222,...}}
    """

    # get all possible values of label
    label_type = []  # list all possible values of label shown in the training samples
    for each_sample in training_samples:
        each_label = each_sample['label']
        if each_label not in label_type:
            label_type.append(each_label)

    # get all discrete attributes name
    all_discrete_attributes_name = list(attributes.keys())  # a list
    all_discrete_attributes_name.remove('density')
    all_discrete_attributes_name.remove('sugar_content')  # remove continuous attributes

    # # get all possible values of all discrete attributes
    # attributes_values = []
    # for attribute in all_discrete_attributes_name:
    #     for each_attribute_value in attributes[attribute]:
    #         attributes_values.append(each_attribute_value)
    #
    # # estimate discrete attribute conditional probability
    # discrete_attribute_conditional_probability = {}  # initialize
    # for label in label_type:
    #     discrete_attribute_conditional_probability[label] = {}
    #     for each_attribute_value in attributes_values:
    #         discrete_attribute_conditional_probability[label][each_attribute_value] = 0

    num_of_each_label = num_of_sample_of_different_label(training_samples)  # possible label and corresponding numbers shown in training data

    # initialize
    discrete_attribute_conditional_probability = {}
    for label in label_type:
        discrete_attribute_conditional_probability[label] = {}
        for attribute in all_discrete_attributes_name:
            attribute_value_num = len(attributes[attribute])  # possible number of value corresponding to each attribute
            for each_attribute_value in attributes[attribute]:
                discrete_attribute_conditional_probability[label][each_attribute_value] = 1/(num_of_each_label[label] + attribute_value_num)  # Laplacian Correction

    # estimate discrete attribute conditional probability
    for each_sample in training_samples:
        label = each_sample['label']
        for each_discrete_attribute in all_discrete_attributes_name:
            value = each_sample[each_discrete_attribute]
            attribute_value_num = len(attributes[each_discrete_attribute])  # possible number of value corresponding to each attribute
            discrete_attribute_conditional_probability[label][value] += 1/(num_of_each_label[label] + attribute_value_num)  # Laplacian Correction

    return discrete_attribute_conditional_probability


def continuous_attribute_distribution_parameter_estimation(training_samples):
    """
    Notice: all continuous attributes are assumed to fit a normal distribution.
    Notice: Unbiased Estimation Adopted. See Function normal_distribution_parameter_estimation.

    :param training_samples: data of all training samples, the format are shown in load_training_data function.

    :return parameters: the normal distribution parameters, a dictionary.
    eg: {'good': {'density': {'miu': 0.57375, 'sigma': 0.12921051483086482}, ...}, 'bad': {'density': {'miu': 0.49611111111111117, 'sigma': 0.19471867170641627}, ...}}
    """

    # get all possible values of label
    label_type = []  # list all possible values of label shown in the training samples
    for each_sample in training_samples:
        each_label = each_sample['label']
        if each_label not in label_type:
            label_type.append(each_label)

    # get all continuous attributes name
    all_continuous_attributes_name = list(attributes.keys())  # a list
    all_continuous_attributes_name.remove('color')
    all_continuous_attributes_name.remove('root')
    all_continuous_attributes_name.remove('knock_sound')
    all_continuous_attributes_name.remove('texture')
    all_continuous_attributes_name.remove('belly_button')
    all_continuous_attributes_name.remove('touch')  # remove discrete attributes

    # calculate parameters
    parameters = {}
    for each_label in label_type:
        parameters[each_label] = {}
        for each_continuous_attribute in all_continuous_attributes_name:
            parameters[each_label][each_continuous_attribute] = {}
            data = []
            for each_sample in training_samples:
                if each_sample['label'] == each_label:
                    value = each_sample[each_continuous_attribute]
                    data.append(value)

            miu, sigma = normal_distribution_parameter_estimation(data)
            parameters[each_label][each_continuous_attribute]['miu'] = miu
            parameters[each_label][each_continuous_attribute]['sigma'] = sigma

    return parameters


def load_testing_data(filename):
    """
    :param filename: the file name of the excel file including testing data set, a string, the data should be on sheet1.
    eg: 'Test_Dataset.xlsx'

    :return testing_samples: all testing samples attributes and values, each sample is a dictionary, and combined to a list.
    eg: [{'color': 'green', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.697, 'sugar_content': 0.46}, ...]
    """

    # open the sheet of the excel file
    testing_data_list = xlrd.open_workbook(filename)  # open the excel file of training data set
    sheet1 = testing_data_list.sheets()[0]  # the data set is in sheet1 of the excel file

    # row and column number start at 0
    sample_num = sheet1.nrows - 1  # the number of testing samples, the first row in the excel are attribute names
    attribute_num = sheet1.ncols - 2  # the number of attributes, the first column is No. and the last is label

    # read testing samples data
    testing_samples = []  # list all testing samples, each sample is a dictionary
    for i in range(sample_num):
        # i = 0, 1, ..., (sample_num - 1)
        current_sample = {}  # each sample
        for j in range(attribute_num):
            # j = 0, 1, ..., (attribute_num - 1)
            attribute_name = sheet1.cell(0, j + 1).value  # each attribute
            attribute_value = sheet1.cell(i + 1, j + 1).value  # the value of this sample at this attribute
            current_sample[attribute_name] = attribute_value
        testing_samples.append(current_sample)

    return testing_samples


def bayes_score(testing_sample, class_prior_probability, discrete_attribute_conditional_probability, parameters):
    """
    Notice: Deepcopy Adopted. Line 303 And Line 308.

    :param testing_sample: one sample for testing, a dictionary.
    eg: {'color': 'green', 'root': 'curled', 'knock_sound': 'turbidity', 'texture': 'clear', 'belly_button': 'hollow', 'touch': 'hard', 'density': 0.697, 'sugar_content': 0.46}
    :param class_prior_probability: see function class_prior_probability_estimation.
    :param discrete_attribute_conditional_probability: see function discrete_attribute_conditional_probability_estimation.
    :param parameters: see function continuous_attribute_distribution_parameter_estimation.

    :return score: the score of each label for testing sample, a dictionary.
    eg: {'good': 0.025631024529740684, 'bad': 7.722360621178054e-05}
    """

    # get all attributes names of a sample
    all_attributes = list(testing_sample.keys())

    # get all discrete attributes names of a sample
    all_discrete_attributes = deepcopy(all_attributes)
    all_discrete_attributes.remove('density')
    all_discrete_attributes.remove('sugar_content')  # remove continuous attributes

    # get all continuous attributes names of a sample
    all_continuous_attributes = deepcopy(all_attributes)
    all_continuous_attributes.remove('color')
    all_continuous_attributes.remove('root')
    all_continuous_attributes.remove('knock_sound')
    all_continuous_attributes.remove('texture')
    all_continuous_attributes.remove('belly_button')
    all_continuous_attributes.remove('touch')  # remove discrete attributes

    # initialize score
    possible_label = list(class_prior_probability.keys())  # a list of possible labels
    score = {}  # a dictionary, keys are possible labels, values are score of each label
    for label in possible_label:
        score[label] = class_prior_probability[label]

    # print(possible_label)
    # print(all_discrete_attributes)
    # print(all_continuous_attributes)

    # calculate discrete attributes score
    for label in possible_label:
        for discrete_attribute in all_discrete_attributes:
            sample_value = testing_sample[discrete_attribute]
            score[label] = score[label] * discrete_attribute_conditional_probability[label][sample_value]

    # calculate continuous attributes score
    for label in possible_label:
        for continuous_attribute in all_continuous_attributes:
            sample_value = testing_sample[continuous_attribute]
            miu = parameters[label][continuous_attribute]['miu']
            sigma = parameters[label][continuous_attribute]['sigma']
            temp_score = (((1/2/math.pi) ** 0.5)/sigma) * (np.exp(-((sample_value-miu)**2)/(2*(sigma**2))))
            score[label] = score[label] * temp_score

    return score


def classify(score):
    score_list = []  # convert "score" into a list, eg: [['good', 0.6], ['bad', 0.4]].
    all_possible_label = list(score.keys())  # all possible label
    possible_label_num = len(all_possible_label)  # number of possible labels
    for i in range(possible_label_num):
        # i = 0, 1, ..., (possible_label_num - 1)
        label_score_pair = [all_possible_label[i], score[all_possible_label[i]]]
        score_list.append(label_score_pair)

    # print(score_list)

    # initialize
    classify_result = ''
    biggest_score = 0

    # find the most probable class for the sample
    for pair in score_list:
        if pair[1] > biggest_score:
            biggest_score = pair[1]
            classify_result = pair[0]

    return classify_result


if __name__ == '__main__':
    train_file = 'Watermelon_Dataset_3.0.xlsx'
    attributes, training_samples = load_training_data(train_file)
    class_prior_probability = class_prior_probability_estimation(training_samples)
    discrete_attribute_conditional_probability = discrete_attribute_conditional_probability_estimation(training_samples, attributes)
    parameters = continuous_attribute_distribution_parameter_estimation(training_samples)

    test_file = 'Test_Dataset.xlsx'
    testing_samples = load_testing_data(test_file)
    testing_sample_1 = testing_samples[0]  # only test the first sample in the testing samples file
    score = bayes_score(testing_sample_1, class_prior_probability, discrete_attribute_conditional_probability, parameters)
    result = classify(score)

    print('The score of Test Sample No.1 is: ' + str(score))
    print('Test Sample No.1 is predicted to be a ' + result + ' watermelon.')



#!/usr/bin/env python
# encoding: utf-8

from generate_feature import *
from model import *

train_file = ''
test_file = ''
enrollment_train_file = ''
enrollment_test_file = ''
object_file = ''
truth_train_file = ''
val_train = 'internal1/enrollment_train.csv'
val_test = 'internal1/enrollment_test.csv'

# read in data
enroll_train = read_enrollment(enrollment_train_file)
enroll_test = read_enrollment(enrollment_test_file)
train_data = read_log_data(enroll_train, train_file, object_file)
test_data = read_log_data(enroll_test, test_file, object_file)

# get course information
all_data = train_data.append(test_data)
course_info = get_course_info(all_data)

# get user information
user_info = get_user_info(all_data)

# generate feature for our current best result
train_X = generate_feature(train_data, course_info,
                                    user_info, enroll_train)
test_X = generate_feature(test_data, course_info,
                                    user_info, enroll_test)

# genereate daily feature
train_daily = get_daily_activity(train_data, course_info)
test_daily = get_daily_activity(test_data, course_info)
train_daily = train_daily.as_matrix()
test_daily = test_daily.as_matrix()

# run model
label_data = pd.io.parsers.read_csv(truth_train_file, header = None)
val_train_data = pd.io.parsers.read_csv(val_train)
val_test_data = pd.io.parsers.read_csv(val_test)
model(train_X, label_data, val_train_data, val_test_data, 240)





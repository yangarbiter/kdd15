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

all_data = train_data.append(test_data)
all_enroll = enroll_train.append(enroll_test)

# get course information
course_info = get_course_info(all_data)

# get session information
session_info, session = get_session_data(all_data, all_enroll)

# get user information
user_info = get_user_info(all_data, session)

#  get leak feature
leak_info = get_leak_feature(all_data, course_info)

# generate feature for our current best result
train_X = generate_feature(train_data, course_info, user_info,
                leak_info, session_info, enroll_train)
train_X = train_X.as_matrix()
test_X = generate_feature(test_data, course_info, user_info,
                leak_info, session_info, enroll_test)
test_X = test_X.as_matrix()

# genereate daily feature
train_daily = get_daily_activity(train_data, course_info)
train_daily = train_daily.as_matrix()
test_daily = get_daily_activity(test_data, course_info)
test_daily = test_daily.as_matrix()

# genereate daily feature with exact date
daily_id, daily_user = get_activity_exact_date(train_data)

# run model
label_data = pd.io.parsers.read_csv(truth_train_file, header = None)
val_train_data = pd.io.parsers.read_csv(val_train)
val_test_data = pd.io.parsers.read_csv(val_test)
model(train_X, label_data, val_train_data, val_test_data, 240)






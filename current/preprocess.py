#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import numpy as np

def read_object_data(object_file):
    """read object data and split children into multiple rows"""
    object_data = []
    with open(object_file, 'r') as file:
        for line in file:
            line = line.strip().split(',')
            if line[3] == '':
                object_data.append(line)
            else:
                children = line[3].split(' ')[:-1]
                for child in children:
                    object_data.append(line[:3] + [child] + line[4:])
    object_data = pd.DataFrame(object_data)
    object_data = object_data.drop_duplicates().reset_index(drop = True)
    object_data.columns = ['course_id', 'module_id',
                            'category', 'children', 'time']
    return object_data

def preprocess_object_data(object_data):
    return object_data

def read_log_data(enrollment, log_file, object_file):
    enrollment.index = enrollment['enrollment_id']
    log_data = pd.io.parsers.read_csv(log_file)
    log_data.drop_duplicates(inplace = True)
    log_data = log_data.join(enrollment[['username', 'course_id']],
                                        on = 'enrollment_id')
    log_data['time'] = pd.to_datetime(log_data['time'])
    log_data['date'] = [t.date() for t in log_data['time']]
    log_data['hour'] = [t.time() for t in log_data['time']]
    object_data = read_object_data(object_file)
    object_data.index = object_data['module_id']
    object_data = object_data.rename(columns = {'time' : 'release_time'})
    log_data = log_data.join(object_data[['category', 'release_time']], on = 'object')
    return log_data

def read_enrollment(enroll_file):
    enrollment = pd.io.parsers.read_csv(enroll_file)
    enrollment = enrollment[enrollment['enrollment_id'] != 139669]
    return enrollment

def normalize(feature, enrollment):
    enrollment.index = enrollment['enrollment_id']
    columns = feature.columns
    feature['course_id'] = enrollment['course_id']
    for c in columns:
        print c
        group_mean = feature.groupby('course_id')[c].mean()
        group_std = feature.groupby('course_id')[c].std()
        feature['mean'] = [group_mean[course] for course in feature['course_id']]
        feature['std'] = [group_std[course] for course in feature['course_id']]
        feature[c] = (feature[c] - feature['mean']) / feature['std']
    feature.drop(['mean', 'std', 'course_id'], axis = 1, inplace = True)
    feature.fillna(0, inplace = True)
    return feature

def smooth(feature, by = 3):
    feature_smooth = pd.DataFrame()
    col = feature.columns
    for c in range(len(col) - by + 1):
        data = feature[col[c: c + by]].mean(axis = 1)
        feature_smooth[col[c]] = data
    return feature_smooth

def to_one(feature):
    f = lambda x : int(x > 0)
    return feature.applymap(f)

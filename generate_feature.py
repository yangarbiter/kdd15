#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import numpy as np

def read_log_data(log_file):
    log_data = pd.io.parsers.read_csv(log_file)
    log_data.drop_duplicates(inplace = True)
    log_data['date'] = [t.date() for t in pd.to_datetime(log_data['time'])]
    log_data['time'] = [t.time() for t in pd.to_datetime(log_data['time'])]
    return log_data

def read_enrollment(enroll_file):
    enrollment = pd.io.parsers.read_csv(enroll_file)
    enrollment = enrollment[enrollment['enrollment_id'] != 139669]
    return enrollment

def get_course_info(all_data, video_info = False, problem_info = False):
    """
    return a course dict contains the following course information:
    - start_date: start date
    - end_date: end date
    - n_enroll: number of enrollment
    - d_activity: daily number of view
    - n_video: number of released video
    - video_info: total number of view,
                  first appearance day,
                  daily number of view of each video
    - n_problem: number of problem
    - pro_info: total number of view,
                first appearance day,
                daily number of view of each problem
    """
    course_info = {c: {} for c in set(all_data['course_id'])}

    # basic information
    course_data = all_data[['course_id', 'username','date']]
    course_data.drop_duplicates(inplace = True)
    course_by_date = course_data.groupby('course_id')['date'].value_counts()
    course_by_user = course_data.groupby('course_id')['username'].value_counts()

    # video information
    course_video = all_data[(all_data['event'] == 'video')][['course_id', 'username', 'date', 'object']]
    course_video.drop_duplicates(inplace = True)
    course_by_video = course_video.groupby('course_id')['object'].value_counts()
    course_by_video_date = course_video.groupby(['course_id', 'object'])['date'].value_counts()

    # problem information
    course_pro = all_data[(all_data['event'] == 'problem')][['course_id', 'username', 'date', 'object']]
    course_pro.drop_duplicates(inplace = True)
    course_by_pro = course_pro.groupby('course_id')['object'].value_counts()
    course_by_pro_date = course_pro.groupby(['course_id', 'object'])['date'].value_counts()

    for c in course_info:
        course_info[c]['start_date'] = min(course_by_date[c].index)
        course_info[c]['end_date'] = max(course_by_date[c].index)
        course_info[c]['n_enroll'] = len(course_by_user[c])
        course_info[c]['d_activity'] = {k:v for k, v in
                            zip(course_by_date[c].index, course_by_date[c])}
        course_info[c]['n_video'] = len(course_by_video[c])
        course_info[c]['n_problem'] = len(course_by_pro[c])
        if(video_info):
            video = {v:{} for v in course_by_video[c].index}
            for v in video:
                video[v]['n_view'] = course_by_video[c][v]
                video[v]['appear_date'] = min(course_by_video_date[c][v].index)
                video[v]['d_activity'] = {k:v for k, v in
                           zip(course_by_video_date[c][v].index, course_by_video_date[c][v])}
            course_info[c]['video_info'] = video
        if(problem_info):
            pro = {p:{} for p in course_by_pro[c].index}
            for p in pro:
                pro[p]['n_view'] = course_by_pro[c][p]
                pro[p]['appear_date'] = min(course_by_pro_date[c][p].index)
                pro[p]['d_activity'] = {k:v for k, v in
                           zip(course_by_pro_date[c][p].index, course_by_pro_date[c][p])}
            course_info[c]['pro_info'] = video
    return course_info

def get_total_activity(log_data):
    """
    - n_records
    - first_day
    - last_day
    - event_source pair count(9)
    - n_video_object
    - n_problem_object
    """
    data = pd.DataFrame()
    data['n_records'] = log_data.groupby('enrollment_id')['time'].count()
    data['first_day'] = log_data.groupby('enrollment_id').first()['date']
    data['last_day'] = log_data.groupby('enrollment_id').last()['date']

    log_server = log_data[log_data['source'] == 'server']
    data['n_server_navigate'] = log_server[log_server['event'] == 'nagivate'] \
                                    .groupby('enrollment_id')['time'].count()
    data['n_server_access'] = log_server[log_server['event'] == 'access']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_server_problem'] = log_server[log_server['event'] == 'problem']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_server_discussion'] = log_server[log_server['event'] == 'discussion']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_server_wiki'] = log_server[log_server['event'] == 'wiki']\
                                    .groupby('enrollment_id')['time'].count()

    log_browser = log_data[log_data['source'] == 'browser']
    data['n_browser_access'] = log_browser[log_browser['event'] == 'access']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_browser_problem'] = log_browser[log_browser['event'] == 'problem']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_browser_pageclose'] = log_browser[log_browser['event'] == 'page_close']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_browser_video'] = log_browser[log_browser['event'] == 'video']\
                                    .groupby('enrollment_id')['time'].count()

    log_video = log_data[log_data['event'] == 'video']\
                            [['enrollment_id', 'object']]
    log_video.drop_duplicates(inplace = True)
    data['n_video_object'] = log_video.groupby('enrollment_id').count()

    log_problem = log_data[log_data['event'] == 'problem']\
                            [['enrollment_id', 'object']]
    log_problem.drop_duplicates(inplace = True)
    data['n_problem_object'] = log_problem.groupby('enrollment_id').count()
    data.fillna(0, inplace = True)
    return data

def generate_feature(log_data, course_info, enrollment):
    log_feature = get_total_activity(log_data)
    log_feature['course_end'] = [course_info[c]['end_date'] for c in enrollment['course_id']]
    log_feature['s_first_day'] = [d.days for d in
                            (log_feature['course_end'] - log_feature['first_day'])]
    log_feature['s_last_day'] = [d.days for d in
                            (log_feature['course_end'] - log_feature['last_day'])]

    log_feature['course_n_video'] = [course_info[c]['n_video']
                                        for c in enrollment['course_id']]
    log_feature['finish_video'] = log_feature['n_video_object'] \
                                        / log_feature['course_n_video']
    log_feature['course_n_pro'] = [course_info[c]['n_problem']
                                        for c in enrollment['course_id']]
    log_feature['click_pro'] = log_feature['n_problem_object'] \
                                        / log_feature['course_n_pro']
    log_feature.drop('first_day', axis=1, inplace=True)
    log_feature.drop('last_day', axis=1, inplace=True)
    log_feature.drop('course_end', axis=1, inplace=True)
    return log_feature

if __name__ == "__main__":
    train_file = ''
    test_file = ''
    enrollment_file = ''
    train_data = read_log_data(train_file)
    test_data = read_log_data(test_file)
    enrollment = read_enrollment(enrollment_file)
    all_data = train_data.append(test_data)
    course_info = get_course_info(all_data)
    train_X = generate_feature(train_data, course_info, enrollment)
    train_X = train_X.as_matrix()


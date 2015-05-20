#!/usr/bin/env python
# encoding: utf-8

from preprocess import *

def get_course_info(all_data, video_info = False, problem_info = False):
    """
    Return a course dict contains the following course information:
    - start_date: start date of the course
    - end_date: end date of the course
    - n_enroll: number of enrollment
    - n_source_event_pair(9)
    - d_activity: daily number of view
    - n_video: number of released video (collect from log file)
    - n_problem: number of problem (collect from log file)
    - n_chapter: number of released chapter (collect from log file)
    - n_sequential: number of released sequential (collect from log file)
    - s_per_chapter: number of sequential per chapter
    - v_per_chapter: number of video per chapter
    - p_per_chapter: number of problem per chapter
    - v_release_fre: video release frequency
    - video_info: total number of view,
                  first appearance day,
                  daily number of view of each video
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
    course_event = all_data.groupby(['course_id', 'source','event'])['time'].count()

    # video information
    course_video = all_data[(all_data['event'] == 'video')]\
                        [['course_id', 'username', 'date', 'object']]
    course_video.drop_duplicates(inplace = True)
    course_first_video = course_video.groupby(['course_id', 'object'])['date'].min()
    course_by_video = course_video.groupby('course_id')['object'].value_counts()
    course_by_video_date = course_video.groupby(['course_id', 'object'])\
                                        ['date'].value_counts()

    # problem information
    course_pro = all_data[(all_data['event'] == 'problem')]\
                        [['course_id', 'username', 'date', 'object']]
    course_pro.drop_duplicates(inplace = True)
    course_by_pro = course_pro.groupby('course_id')['object'].value_counts()
    course_by_pro_date = course_pro.groupby(['course_id', 'object'])\
                                        ['date'].value_counts()

    # sequential information
    course_seq = all_data[all_data['category'] == 'sequential']\
                                        [['course_id', 'object']]
    course_seq.drop_duplicates(inplace = True)
    course_by_seq = course_seq.groupby('course_id').count()

    # chapter information
    course_cha = all_data[all_data['category'] == 'chapter']\
                                        [['course_id', 'object']]
    course_cha.drop_duplicates(inplace = True)
    course_by_cha = course_cha.groupby('course_id').count()
    for c in course_info:
        course_info[c]['start_date'] = min(course_by_date[c].index)
        course_info[c]['end_date'] = max(course_by_date[c].index)
        course_info[c]['n_enroll'] = len(course_by_user[c])
        course_info[c]['n_server_navigate'] = course_event[c]['server']\
                                                .get('nagivate', 0)
        course_info[c]['n_server_access'] = course_event[c]['server']\
                                                .get('access', 0)
        course_info[c]['n_server_problem'] = course_event[c]['server']\
                                                .get('problem', 0)
        course_info[c]['n_server_discussion'] = course_event[c]['server']\
                                                .get('discussion', 0)
        course_info[c]['n_server_wiki'] = course_event[c]['server']\
                                                .get('wiki', 0)
        course_info[c]['n_browser_access'] = course_event[c]['browser']\
                                                .get('access', 0)
        course_info[c]['n_browser_problem'] = course_event[c]['browser']\
                                                .get('problem', 0)
        course_info[c]['n_browser_pageclose'] = course_event[c]['browser']\
                                                .get('page_close', 0)
        course_info[c]['n_browser_video'] = course_event[c]['browser']\
                                                .get('video', 0)
        # course_info[c]['d_activity'] = {k:v for k, v in
                            # zip(course_by_date[c].index, course_by_date[c])}
        course_info[c]['n_video'] = len(course_by_video[c])
        course_info[c]['n_problem'] = len(course_by_pro[c])
        course_info[c]['n_chapter'] = course_by_cha.ix[c][0]
        course_info[c]['n_sequential'] = course_by_seq.ix[c][0]
        course_info[c]['s_per_chapter'] = course_info[c]['n_sequential'] \
                                                / course_info[c]['n_chapter']
        course_info[c]['v_per_chapter'] = course_info[c]['n_video'] \
                                                / course_info[c]['n_chapter']
        course_info[c]['p_per_chapter'] = course_info[c]['n_problem'] \
                                                / course_info[c]['n_chapter']
        course_info[c]['v_release_fre'] = len(set(course_first_video[c].values))
        # for video detail information
        if(video_info):
            video = {v:{} for v in course_by_video[c].index}
            for v in video:
                video[v]['n_view'] = course_by_video[c][v]
                video[v]['appear_date'] = min(course_by_video_date[c][v].index)
                video[v]['d_activity'] = {k:v for k, v in
                           zip(course_by_video_date[c][v].index, course_by_video_date[c][v])}
            course_info[c]['video_info'] = video
        # for problem detail information
        if(problem_info):
            pro = {p:{} for p in course_by_pro[c].index}
            for p in pro:
                pro[p]['n_view'] = course_by_pro[c][p]
                pro[p]['appear_date'] = min(course_by_pro_date[c][p].index)
                pro[p]['d_activity'] = {k:v for k, v in
                           zip(course_by_pro_date[c][p].index, course_by_pro_date[c][p])}
            course_info[c]['pro_info'] = video
    return course_info

def get_user_info(all_data):
    """
    Return a pandas DataFrame contains information about each user
    - n_records: number of log
    - n_enroll: number of enrollment of this user
    - online_day: total number of online day
    """
    user_info = pd.DataFrame()
    user_info['n_records'] = all_data.groupby('username')['time'].count()

    user_enroll = all_data[['username', 'enrollment_id']]
    user_enroll.drop_duplicates(inplace = True)
    user_info['n_enroll'] = user_enroll.groupby('username').count()

    user_date = all_data[['username', 'date']]
    user_date.drop_duplicates(inplace = True)
    user_info['online_day'] = user_date.groupby('username').count()
    # user_info['first_day'] = user_date.groupby('username')['date'].min()
    # user_info['last_day'] = user_date.groupby('username')['date'].max()
    user_info.fillna(0, inplace = True)
    return user_info

def get_daily_activity(log_data, course_info):
    """
    Return a pandas DataFrame contains daily activity information for
    each enrollment_id
    - n_records: number of log each day
    - n_source_event_pair(9)
    - n_video_object: number of video viewed each day
    - n_problem_object: number of problem clicked each day
    - n_seq_object: number of sequential viewed each day
    Each day has 13 features, so daily features has 13 * 30 features.
    """
    daily = pd.DataFrame()
    log_data['c_start'] = [course_info[c]['start_date']
                                for c in log_data['course_id']]
    log_data['days'] = log_data['date'] - log_data['c_start']
    daily['total'] = log_data.groupby('enrollment_id')['time'].count()
    for i in range(30):
        d = pd.Timedelta(str(i) + ' days')
        log_day = log_data[log_data['days'] == d]
        d_name = 'n_' + str(i) + '_'
        daily[d_name + 'records'] = log_day.groupby('enrollment_id')['time'].count()
        log_server = log_day[log_day['source'] == 'server']
        daily[d_name + 's_navigate'] = log_server[log_server['event'] == 'nagivate'] \
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 's_access'] = log_server[log_server['event'] == 'access']\
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 's_problem'] = log_server[log_server['event'] == 'problem']\
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 's_discussion'] = log_server[log_server['event'] == 'discussion']\
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 's_wiki'] = log_server[log_server['event'] == 'wiki']\
                                        .groupby('enrollment_id')['time'].count()

        log_browser = log_day[log_day['source'] == 'browser']
        daily[d_name + 'b_access'] = log_browser[log_browser['event'] == 'access']\
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 'b_problem'] = log_browser[log_browser['event'] == 'problem']\
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 'b_pageclose'] = log_browser[log_browser['event'] == 'page_close']\
                                        .groupby('enrollment_id')['time'].count()
        daily[d_name + 'b_video'] = log_browser[log_browser['event'] == 'video']\
                                        .groupby('enrollment_id')['time'].count()

        log_day_video = log_day[log_day['event'] == 'video']\
                                [['enrollment_id', 'object']]
        log_day_video.drop_duplicates(inplace = True)
        daily[d_name + 'video_object'] = log_day_video.groupby('enrollment_id').count()

        log_day_pro = log_day[log_day['event'] == 'problem']\
                                [['enrollment_id', 'object']]
        log_day_pro.drop_duplicates(inplace = True)
        daily[d_name + 'pro_object'] = log_day_pro.groupby('enrollment_id').count()

        log_day_seq = log_day[log_day['category'] == 'sequential']\
                                [['enrollment_id', 'object']]
        log_day_seq.drop_duplicates(inplace = True)
        daily[d_name + 'seq_object'] = log_day_seq.groupby('enrollment_id').count()
    daily.fillna(0, inplace = True)
    daily.drop('total', axis = 1, inplace = True)
    return daily

def get_total_activity(log_data):
    """
    Return a pandas DataFrame contains summary features of each enrollment_id
    - n_records: number of log
    - first_day: first day online
    - last_day: last day online
    - online_day: total number of online day
    - event_source pair count(9)
    - n_video_object: number of video viewed by certain enrollment_id
    - n_problem_object: number of problem clicked by certain enrollment_id
    - n_chapter_object: number of chapter clicked by certain enrollment_id
    - n_sequential_object: number of sequential clicked by certain enrollment_id
    """
    data = pd.DataFrame()
    data['n_records'] = log_data.groupby('enrollment_id')['time'].count()
    data['first_day'] = [pd.to_datetime(t)
                for t in log_data.groupby('enrollment_id')['time'].min()]
    data['last_day'] = [pd.to_datetime(t)
                for t in log_data.groupby('enrollment_id')['time'].max()]

    log_online = log_data[['enrollment_id', 'date']]
    log_online.drop_duplicates(inplace = True)
    data['online_day'] = log_online.groupby('enrollment_id').count()

    log_server = log_data[log_data['source'] == 'server']
    data['n_server_navigate'] = log_server[log_server['event'] == 'nagivate'] \
                                    .groupby('enrollment_id')['time'].count()
    data['n_server_access'] = log_server[log_server['event'] == 'access']\
                                    .groupby('enrollment_id')['time'].count()
    # data['n_server_access_cha'] = log_server[(log_server['event'] == 'access') \
                                            # &(log_server['category'] == 'chapter')] \
                                    # .groupby('enrollment_id')['time'].count()
    # data['n_server_access_seq'] = log_server[(log_server['event'] == 'access') \
                                            # &(log_server['category'] == 'sequential')] \
                                    # .groupby('enrollment_id')['time'].count()
    data['n_server_problem'] = log_server[log_server['event'] == 'problem']\
                                    .groupby('enrollment_id')['time'].count()
    data['n_server_discussion'] = log_server[log_server['event'] == 'discussion']\
                                    .groupby('enrollment_id')['time'].count()
    # data['n_server_wiki'] = log_server[log_server['event'] == 'wiki']\
                                    # .groupby('enrollment_id')['time'].count()

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

    log_chapter = log_data[log_data['category'] == 'chapter']\
                            [['enrollment_id', 'object']]
    log_chapter.drop_duplicates(inplace = True)
    data['n_chapter_object'] = log_chapter.groupby('enrollment_id').count()

    log_sequential = log_data[log_data['category'] == 'sequential']\
                            [['enrollment_id', 'object']]
    log_sequential.drop_duplicates(inplace = True)
    data['n_sequential_object'] = log_sequential.groupby('enrollment_id').count()
    data.fillna(0, inplace = True)
    return data

def generate_feature(log_data, course_info, user_info, enrollment):
    """
    Return a pandas DataFrame contains three kinds of features:
    - log_feature: activity of each enrollment from log file
    - course_info
    - user_info
    """
    log_feature = get_total_activity(log_data)
    enrollment.index = enrollment['enrollment_id']
    enrollment['course_end'] = [pd.to_datetime(course_info[c]['end_date'] + pd.Timedelta('1 days'))
                                    for c in enrollment['course_id']]
    log_feature['first_day'] = [int(d.total_seconds()/60/60/24) for d in
                            (enrollment['course_end'] - log_feature['first_day'])]
    log_feature['last_day'] = [int(d.total_seconds()/60/60/24) for d in
                            (enrollment['course_end'] - log_feature['last_day'])]

    # truth.index = truth[0]
    # enrollment['drop'] = truth[1]
    # dropout = enrollment.groupby('course_id')['drop'].mean()
    # log_feature['course_drop'] = [dropout[c] for c in enrollment['course_id']]
    log_feature['course_s_access'] = [course_info[c]['n_server_access']
                                        for c in enrollment['course_id']]
    # log_feature['c_s_navigate'] = [course_info[c]['n_server_navigate']
                                        # for c in enrollment['course_id']]
    # log_feature['c_s_problem'] = [course_info[c]['n_server_problem']
                                        # for c in enrollment['course_id']]
    # log_feature['c_s_discussion'] = [course_info[c]['n_server_discussion']
                                        # for c in enrollment['course_id']]
    # log_feature['c_b_access'] = [course_info[c]['n_browser_access']
                                        # for c in enrollment['course_id']]
    # log_feature['c_b_problem'] = [course_info[c]['n_browser_problem']
                                        # for c in enrollment['course_id']]
    # log_feature['c_b_pageclose'] = [course_info[c]['n_browser_pageclose']
                                        # for c in enrollment['course_id']]
    # log_feature['c_b_video'] = [course_info[c]['n_browser_video']
                                        # for c in enrollment['course_id']]
    log_feature['course_enroll'] = [course_info[c]['n_enroll']
                                        for c in enrollment['course_id']]
    enrollment['course_n_video'] = [course_info[c]['n_video']
                                        for c in enrollment['course_id']]
    log_feature['course_n_problem'] = [course_info[c]['n_problem']
                                        for c in enrollment['course_id']]
    log_feature['course_n_chapter'] = [course_info[c]['n_chapter']
                                        for c in enrollment['course_id']]
    log_feature['course_n_sequential'] = [course_info[c]['n_sequential']
                                        for c in enrollment['course_id']]
    log_feature['course_video_release'] = [course_info[c]['v_release_fre']
                                        for c in enrollment['course_id']]
    # log_feature['course_s_per_chapter'] = [course_info[c]['s_per_chapter']
                                        # for c in enrollment['course_id']]
    # log_feature['course_v_per_chapter'] = [course_info[c]['v_per_chapter']
                                        # for c in enrollment['course_id']]
    # log_feature['course_p_per_chapter'] = [course_info[c]['p_per_chapter']
                                        # for c in enrollment['course_id']]

    log_feature['click_video_rate'] = log_feature['n_video_object'] \
                                        / enrollment['course_n_video']
    log_feature['click_problem_rate'] = log_feature['n_problem_object'] \
                                        / log_feature['course_n_problem']
    # log_feature['click_chapter_rate'] = log_feature['n_chapter_object'] \
                                        # / log_feature['course_n_chapter']
    log_feature['click_sequential_rate'] = log_feature['n_sequential_object'] \
                                        / log_feature['course_n_sequential']
    # log_feature['finish_rate'] = log_feature['finish_video'] \
                                    # / log_feature['online_day']
    # log_feature.drop(['n_problem_object', 'n_video_object', 'n_sequential_object', 'n_chapter_object'], axis = 1, inplace = True)

    user_enroll = log_data[['enrollment_id', 'username', 'course_id']]
    user_enroll.drop_duplicates(inplace = True)
    user_enroll = user_enroll.join(user_info, on = 'username')
    user_enroll.index = user_enroll['enrollment_id']
    log_feature['u_n_records'] = user_enroll['n_records']
    log_feature['u_n_enroll'] = user_enroll['n_enroll']
    log_feature['u_online_day'] = user_enroll['online_day']
    return log_feature
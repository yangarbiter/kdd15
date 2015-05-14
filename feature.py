
import numpy as np
import proc
import cPickle
from joblib import Parallel, delayed

def logs2features(logs):
    ret = []

    #num of events
    ret.append(len(logs))

    #first and last event
    ret.append(logs[-1][0]-logs[0][0])

    #num of each source type
    FET_NUM = len(ret)
    for event in proc.SOURCE:
        ret.append(0)
    for log in logs:
        ret[FET_NUM + log[1]] += 1

    #num of each event type
    FET_NUM = len(ret)
    for event in proc.EVENTS:
        ret.append(0)
    for log in logs:
        ret[FET_NUM + log[2]] += 1

    return ret

def course_one_hot(data):
    allcourses = set([])
    for i in data.keys():
        allcourses.add(data[i]['course_id'])
    allcourses = list(allcourses)

    ret = []
    for i in data.keys():
        _ = np.zeros((len(allcourses)))
        _[allcourses.index(data[i]['course_id'])] = 1
        ret.append(_)

    return np.array(ret)

def build_feature(data):
    X = Parallel(n_jobs=20, verbose=5)(delayed(logs2features)(data[i]['logs']) for i in data.keys())

    c_one_hot = course_one_hot(data)
    print np.shape(X), np.shape(c_one_hot)
    X = np.hstack((X, c_one_hot))
    return X

def main():
    with open(proc.DATA_PATH_PREFIX+'data.pkl', 'r') as f:
        data = cPickle.load(f)
    for i in data.keys():
        if data[i]['logs'] == []:
            print i, 'empty_log'
    del data[139669]
    X = np.array(build_feature(data))


if __name__ == "__main__":
    main()

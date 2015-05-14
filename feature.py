
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
    for source in proc.SOURCE:
        ret.append(0)
    for log in logs:
        ret[FET_NUM + log[1]] += 1

    #num of each event type
    FET_NUM = len(ret)
    for event in proc.EVENTS:
        ret.append(0)
    for log in logs:
        ret[FET_NUM + log[2]] += 1

    #TODO event source pair
    FET_NUM = len(ret)
    for source in proc.SOURCE:
        for event in proc.EVENTS:
            ret.append(0)
    for log in logs:
        ret[FET_NUM + log[1]*len(proc.EVENTS) + log[2]] += 1

    #TODO ratio


    #TODO course count


    return ret

def last_course_time(data):
    course_time = {}
    for i in sorted(data.keys()):
        if data[i]['course_id'] not in course_time:
            course_time[data[i]['course_id']] = 0
        else:
            course_time[data[i]['course_id']] =\
                max(course_time[data[i]['course_id']], data[i]['logs'][-1][0])
    return course_time

def course_one_hot(data):
    allcourses = set([])
    for i in sorted(data.keys()):
        allcourses.add(data[i]['course_id'])
    allcourses = list(allcourses)

    ret = []
    for i in sorted(data.keys()):
        _ = np.zeros((len(allcourses)))
        _[allcourses.index(data[i]['course_id'])] = 1
        ret.append(_)

    return np.array(ret)

def build_feature(data):
    X = Parallel(n_jobs=20, verbose=5)(delayed(logs2features)(data[i]['logs'])
            for i in sorted(data.keys()))

    c_one_hot = course_one_hot(data)
    print np.shape(X), np.shape(c_one_hot)

    course_time = last_course_time(data)
    X_course_time = [course_time[data[i]['course_id']] for i in data.keys()]
    X_course_time = np.reshape(np.array(X_course_time), (-1, 1))

    X = np.hstack((X, c_one_hot))
    X = np.hstack((X, X_course_time))
    return X

def main():
#output training pkl
    #with open(proc.DATA_PATH_PREFIX+'data.pkl', 'r') as f:
    #    data = cPickle.load(f)
    #del data[139669] #data without log
    #ans = proc.loadans()

    #X = build_feature(data)
    #y = [ ans[i] for i in sorted(data.keys()) ]

    #with open('xy.pkl', 'w') as f:
    #    cPickle.dump((X, y), f)


#output testing pkl
    with open(proc.DATA_PATH_PREFIX+'testdata.pkl', 'r') as f:
        testdata = cPickle.load(f)
    idxs = sorted(testdata.keys())
    X = build_feature(testdata)

    with open('testx.pkl', 'w') as f:
        cPickle.dump((idxs, X), f)



if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
import csv, time, cPickle

DATA_PATH_PREFIX = "../data/"

EVENTS = ['problem', 'video', 'access', 'wiki', 'discussion', 'nagivate', 'page_close']
SOURCE = ['server', 'browser']

def loaddat():
    data = {}

    with open(DATA_PATH_PREFIX + 'enrollment_train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if row[0] == 'enrollment_id': continue
            enrole_id = int(row[0])
            data[enrole_id] = {
                'logs': [],
                'user_id': row[1],
                'course_id': row[2],
            }


    with open(DATA_PATH_PREFIX + 'log_train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            if row[0] == 'enrollment_id': continue
            x = []
            enrole_id = int(row[0])
            #time, source, event, object
            x.append(int(time.mktime(time.strptime(row[3],'%Y-%m-%dT%H:%M:%S'))))
            x.append(SOURCE.index(row[4]))
            x.append(EVENTS.index(row[5]))
            x.append(row[6])
            data[enrole_id]['logs'].append(x)

    return data

def loadans():
    ans = {}
    with open(DATA_PATH_PREFIX + 'truth_train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            enrole_id = int(row[0])
            ans[enrole_id] = int(row[1])
    return ans

def main():
    """
    load data from pickle file
    """
    with open(DATA_PATH_PREFIX+'data.pkl', 'r') as f:
        data = cPickle.load(f)
    ans = loadans()
    #print set(data.keys()) - set(loadans().keys())

    """
    plot the time between first and last event.
    """
    diff_time = np.array(
            [data[k]['logs'][-1][0] - data[k]['logs'][0][0] for k in ans.keys()]
            )
    y = np.array([ans[k] for k in ans.keys()])
    _temp, bins = np.histogram(diff_time[y==1], bins=30)
    _temp2, bins = np.histogram(diff_time[y==0], bins=bins)
    plt.plot(_temp/float(np.sum(_temp)))
    plt.plot(_temp2/float(np.sum(_temp2)))
    plt.show(block=False)
    _ = raw_input('type something')
    plt.close()


    """
    plot the last event time distribution.
    """
    last_time = np.array(
            [data[k]['logs'][-1][0] for k in ans.keys()]
            )
    m = min(last_time)
    _temp, bins = np.histogram(last_time[y==1]-m, bins=30)
    _temp2, bins = np.histogram(last_time[y==0]-m, bins=bins)
    plt.plot(_temp/float(np.sum(_temp)))
    plt.plot(_temp2/float(np.sum(_temp2)))
    plt.show(block=False)
    _ = raw_input('type something')
    plt.close()

    """
    plot number of event distribution.
    """
    num_logs = np.array(
            [len(data[k]['logs']) for k in ans.keys()]
            )
    _temp, bins = np.histogram(num_logs[y==1], bins=30)
    _temp2, bins = np.histogram(num_logs[y==0], bins=bins)
    plt.plot(_temp/float(np.sum(_temp)))
    plt.plot(_temp2/float(np.sum(_temp2)))
    plt.show(block=False)
    _ = raw_input('type something')
    plt.close()

    """
    save data to pickle file
    """
    #data = loaddat()
    #with open(DATA_PATH_PREFIX+'data.pkl', 'w') as f:
    #    cPickle.dump(data, f)


if __name__ == "__main__":
    main()

import numpy as np
import os, urllib.request

# Data URLs
urlfiles = {
    'agvxh': 'steinmetz_part1.npz',
    'uv3mw': 'steinmetz_part2.npz',
    'ehmw2': 'steinmetz_part3.npz'}

# trials are all 2500ms, with 500 ms ITI and stim onset at 500 ms or 50 bins
# (https://neurostars.org/t/steinmetz-et-al-2019-dataset-questions/14539/8)
# input e.g:  dat['response_time']

def timePoints2Bins(timePoints):
    x = timePoints * 100  # seconds to ms in 10 ms bins
    bins = np.floor(x)
    return (bins)


# for a single session, input the time bin of the reference timepoint (e.g. response time),
# plus or minus duration in ms (e.g. 300ms)
# if method = 'add','minus'

def cutSpikeTimes(referenceTime_bin, spikes, method, msDuration):
    msIdx = int(msDuration / 10)
    # initialize new 3D matrix
    newSpikes = np.zeros((spikes.shape[0], spikes.shape[1], int(msIdx)))

    if np.min(referenceTime_bin) < msIdx:
        raise IndexError('index falls outside of the dataframe')

    for iTrial, x in enumerate(referenceTime_bin):
        if method == 'add':
            newSpikes[:, iTrial, :] = spikes[:, iTrial, int(x):int(x) + msIdx]
        elif method == 'minus':
            newSpikes[:, iTrial, :] = spikes[:, iTrial, int(x) - msIdx:int(x)]
    return (newSpikes)



# 450 ms before response time:responseTime
# cutSpikeTimes(referenceTime_bin, spikes, 'minus', 450)
def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0):
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)



def concatSpikesPerSession(sessionIdx):
    alldatTmp = alldat.copy()
    brainRegionIdx = np.concatenate([alldatTmp[x]['brain_area'] for x in sessionIdx])
    maxTrials = np.max([alldatTmp[x]['spks'].shape[1] for x in sessionIdx])

    spikesAllSessions = []
    for num, i in enumerate(sessionIdx):
        # alldatTmp[i]['spks'] = pad_along_axis(alldatTmp[i]['spks'], 300, axis=1)
        if num == 0:
            stacked = pad_along_axis(alldatTmp[i]['spks'], 300, axis=1)
        else:
            stacked = np.vstack((stacked, pad_along_axis(alldatTmp[i]['spks'], 300, axis=1)))

    return (stacked, brainRegionIdx)
    # print(alldatTmp[i]['spks'].shape)
    # print(stacked.shape)




def load_neural_data(subject=11):
    alldat = []
    # get spike data
    for aurl in urlfiles:
        file = urlfiles[aurl]
        if not os.path.exists(file):
            urllib.request.urlretrieve('https://osf.io/{}/download'.format(aurl),
                                       filename=file)
        alldat = np.hstack((alldat, np.load(file, allow_pickle=True)['dat']))
        print('Loaded: {}'.format(file))
    dat = alldat[subject]


    # spike data: dat['spks']
    # spike data is a 3D matrix: neurons x trials x time
    spikes = dat['spks']
    _, trialLen, _ = spikes.shape

    # for cutting spikes from stim onset to 300 ms
    stimDuration = spikes[:, :, 50:80]

    referenceTime_bin = timePoints2Bins(dat['response_time'])

    mouseList = [alldat[x]['mouse_name'] for x in range(len(alldat))]

    # index dataframe by mouse
    indices = [i for i, elem in enumerate(mouseList) if 'Cori' in elem]

    # NeuronsxtrialsxTime
    stackedCori, brainRegion_Cori = concatSpikesPerSession(indices)

    np.sum(brainRegion_Cori == 'MOs')

    Cori_MOs = stackedCori[brainRegion_Cori == 'MOs', :, :]

    Cori_MOs = Cori_MOs.transpose(2, 1, 0) # time x batch x neurons

    return Cori_MOs
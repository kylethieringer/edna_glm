import os
import glob
import h5py
import random
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import sklearn
from sklearn.metrics import f1_score
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import multiprocessing
import scipy.signal
from scipy.stats import zscore

sys.path.append(os.path.dirname(os.path.realpath('')))
from utils import bases

ftrList = ['mFV', 'mLS', 'fFV', 'fLS', 'mfDist', 'mfAng', 'fmAng', 'pulse', 'sine']


def fill_missing(x, kind="nearest", **kwargs):
    if x.ndim == 3:
        return np.stack([fill_missing(xi, kind=kind, **kwargs) for xi in x], axis=0)
    return pd.DataFrame(x).interpolate(kind=kind, axis=0, limit_direction='both', **kwargs).to_numpy()


def filter_trace(trace, threshold=None, medianFilt='simple', kernel_size=3, return_idxs=False):
    if threshold is None and medianFilt != 'median':
        raise ValueError("please set threshold to do simple filtering")

    if threshold is not None:
        idxs = np.argwhere(abs(trace) > threshold)
        trace[idxs] = np.nan
        if return_idxs:
            return idxs

    if medianFilt == 'median':
        trace = scipy.signal.medfilt(trace, kernel_size=kernel_size)

    return trace


def load_feature(filepath, feature, fill=False, filt=None, normalize=False, **kwargs):
    """
    load kinematic features from h5py file
    Args:
        normalize: whether to zscore the feature after loading
        filepath: path to .h5 file
        feature: str name of feature to load
        fill: remove nans
        filt: None, 'simple', or 'median' to do low level smoothing of outliers

    Returns: numpy array of feature

    """
    with h5py.File(filepath, 'r') as f:
        if feature not in f.keys():
            ftr = []
            return ftr
        ftr = np.array(f[feature])
    if filt is not None:
        ftr = filter_trace(ftr, medianFilt=filt, **kwargs)
    if fill:
        ftr = fill_missing(ftr).flatten()

    if normalize:
        ftr = zscore(ftr)

    return ftr


def splitTrainTest(x, y, testFrac=0.2, oversample=True):
    """Split data into train and test set."""

    sampleIDs = np.arange(len(x))  # indices for each row of data
    #         random.seed(0)
    random.shuffle(sampleIDs)  # shuffle indices
    x, y = x[sampleIDs], y[sampleIDs]  # reorder data
    zeroInds, oneInds = np.argwhere(y == 0).flatten(), np.argwhere(y == 1).flatten()
    minClass = min(len(zeroInds), len(oneInds))
    numTest = int(testFrac * minClass)  # number of test samples for each class

    # Split data into train and test sets
    xTest = np.concatenate((x[zeroInds][:numTest], x[oneInds][:numTest]))
    yTest = np.concatenate((y[zeroInds][:numTest], y[oneInds][:numTest]))
    xTrain = np.concatenate((x[zeroInds][numTest:], x[oneInds][numTest:]))
    yTrain = np.concatenate((y[zeroInds][numTest:], y[oneInds][numTest:]))

    # Randomly oversample training data to balance classes
    if oversample:
        ros = RandomOverSampler()
        xTrain, yTrain = ros.fit_resample(xTrain, yTrain)

    # data is shuffled however it is organized into zeros then ones.
    # Shuffling again so model doesn't just learn structure of dataset
    trainIDs = np.arange(len(xTrain))
    random.shuffle(trainIDs)
    xTrain, yTrain = xTrain[trainIDs], yTrain[trainIDs]

    testIDs = np.arange(len(xTest))
    random.shuffle(testIDs)
    xTest, yTest = xTest[testIDs], yTest[testIDs]

    # Make data dictionary
    data = {'xTrain': xTrain, 'xTest': xTest, 'yTrain': yTrain, 'yTest': yTest}

    # Data indices
    testInds = list(sampleIDs[zeroInds][:numTest]) + list(sampleIDs[oneInds][:numTest])

    return data, testInds


def design(exptList, singleFtr=False, threshold=15, mfDist_type=1, mfDist_thresh=10, mfAng_thresh=60, window=450):
    assert os.path.isdir(exptList[0]), "exptList should contain experiment directories"
    if singleFtr:
        assert singleFtr in ftrList, "single feature must be in designated feature list"
    else:
        print('fitting GLM with multiple inputs')

    output = []
    features = []

    for exptDir in exptList:
        fly = os.path.basename(exptDir)
        ftrfile = os.path.join(exptDir, fly + '.h5')
        songftrs = os.path.join(exptDir, fly + '_song.h5')
        if fly == '220809_162800_16276625_rig2_1':  # tracking errors
            continue
        if not os.path.exists(ftrfile) or not os.path.exists(songftrs):
            continue

        # load song features
        pulse_samp = load_feature(songftrs, 'pulseStEn')
        sine_samp = load_feature(songftrs, 'sineStEn')
        frame_at_sample = load_feature(ftrfile, 'frame_at_sample')
        pulse = np.zeros(int(frame_at_sample[-1]))
        sine = np.zeros(int(frame_at_sample[-1]))

        if pulse_samp.any():
            pulse_frame = frame_at_sample[pulse_samp].astype(int)
            for st, en in pulse_frame:
                pulse[st:en] = 1
        if sine_samp.any():
            sine_frame = frame_at_sample[sine_samp].astype(int)
            for st, en in sine_frame:
                sine[st:en] = 1

        # load all other features
        mFV = load_feature(ftrfile, 'mFV', fill=True, filter='median', threshold=threshold, normalize=True)
        fFV = load_feature(ftrfile, 'fFV', fill=True, filter='median', threshold=threshold, normalize=True)
        mLS = load_feature(ftrfile, 'mLS', fill=True, filter='median', threshold=threshold, normalize=True)
        fLS = load_feature(ftrfile, 'fLS', fill=True, filter='median', threshold=threshold, normalize=True)

        # for mfDist we are thresholding based on the change in distance (if there are big jumps)
        if mfDist_type == 1:
            mfDist = load_feature(ftrfile, 'mfDist', fill=True)
        else:
            mfDist = load_feature(ftrfile, 'mfDist_mHead_fAbd', fill=True)

        idxs = filter_trace(np.diff(mfDist), threshold=threshold, return_idxs=True)
        mfDist[idxs] = np.nan
        mfDist = fill_missing(mfDist).flatten() / 30  # convert from pixels to mm
        z_mfDist = zscore(mfDist)

        mfAng = load_feature(ftrfile, 'mfAng', fill=True, filter='median')
        z_mfAng = load_feature(ftrfile, 'mfAng', fill=True, filter='median', normalize=True)
        z_fmAng = load_feature(ftrfile, 'fmAng', fill=True, filter='median', normalize=True)

        oeStarts = load_feature(ftrfile, 'oeStarts')
        # oeEnds = load_feature(ftrfile, 'oeEnds')

        # set up design matrix
        usedIdxs = np.zeros(len(mfDist))
        vidlen = range(mfDist.shape[0])

        # prediction ovipositor extension bout starts
        for st in oeStarts:
            start = st - window
            stop = st

            if start < 0 or stop >= vidlen[-1]:  # if window is out of range don't use
                continue

            if np.mean(mfDist[start:stop]) > mfDist_thresh:
                continue  # excluding when the animals are far apart
            if np.mean(np.abs(mfAng[start:stop])) > mfAng_thresh:
                continue  # excluding when male is not facing female

            if np.sum(usedIdxs[start:stop]) > 0:  # no double-dipping
                continue
            if np.all(np.isnan(mfDist[start:stop])):
                continue

            trial = vidlen[start:stop]

            # mFV, mLS, fFV, fLS, mfDist, mfAng, fmAng, pulse, sine
            if not singleFtr:
                allftrs = np.concatenate([mFV[trial], mLS[trial],
                                          fFV[trial], fLS[trial],
                                          z_mfDist[trial],
                                          z_mfAng[trial], z_fmAng[trial],
                                          pulse[trial], sine[trial]])
                features.append(allftrs)
            else:
                allftrs = {'mFV': mFV[trial], 'mLS': mLS[trial],
                           'fFV': fFV[trial], 'fLS': fLS[trial],
                           'mfDist': z_mfDist[trial],
                           'mfAng': z_mfAng[trial], 'fmAng': z_fmAng[trial],
                           'pulse': pulse[trial], 'sine': sine[trial]}
                features.append(allftrs[singleFtr])

            output.append(1)
            usedIdxs[start:stop + 1] = 1  # prevent double usage of data for negative class

        # predicting other times
        unusedIdxs = np.where(usedIdxs == 0)[0]
        random.shuffle(unusedIdxs)
        # cycle through randomly selected frames that have not been used yet
        for st in unusedIdxs:
            start = st - window
            stop = st

            if start < 0 or stop >= vidlen[-1]:  # must be within video limits
                continue

            if np.mean(mfDist[start:stop]) > mfDist_thresh:
                continue  # excluding when the animals are far apart
            if np.mean(np.abs(mfAng[start:stop])) > mfAng_thresh:
                continue  # excluding when male is not fac

            # make sure we are not overlapping with previously used trials
            if np.sum(usedIdxs[start - window: stop + window]) > 0:
                continue
            if np.all(np.isnan(mfDist[start:stop])):
                continue

            trial = vidlen[start:stop]
            usedIdxs[start:stop + 1] = 1

            if not singleFtr:
                allftrs = np.concatenate([mFV[trial], mLS[trial],
                                          fFV[trial], fLS[trial],
                                          z_mfDist[trial],
                                          z_mfAng[trial], z_fmAng[trial],
                                          pulse[trial], sine[trial]])
                features.append(allftrs)
            else:
                allftrs = {'mFV': mFV[trial], 'mLS': mLS[trial],
                           'fFV': fFV[trial], 'fLS': fLS[trial],
                           'mfDist': z_mfDist[trial],
                           'mfAng': z_mfAng[trial], 'fmAng': z_fmAng[trial],
                           'pulse': pulse[trial], 'sine': sine[trial]}
                features.append(allftrs[singleFtr])

            output.append(0)

    output = np.array(output)
    features = np.array(features)

    return features, output


def fit_glm(features, output, groupname, window=450):
    B = bases.raised_cosine(0, 12, [0, 325], 150, window)

    numfeatures = features.shape[1] // window
    if numfeatures > 1:
        B = bases.multifeature_basis(B, numfeatures)

    clr = sklearn.linear_model.LogisticRegressionCV(Cs=5, max_iter=10_000)
    # clr = sklearn.linear_model.RidgeClassifier(alpha=0.001)
    df = pd.DataFrame(columns=['group', 'score', 'f1', 'f1_train', 'filter_norms', 'group_filters'])
    for i in range(1000):
        data, testinds = splitTrainTest(features, output)
        xTrain = data['xTrain']
        yTrain = data['yTrain']
        xTest = data['xTest']
        yTest = data['yTest']

        Xtrain = np.dot(xTrain, B)
        Xtest = np.dot(xTest, B)

        clr.fit(Xtrain, yTrain)
        probClasses = clr.predict(Xtest)
        fitment = clr.predict(Xtrain)
        score = clr.score(Xtest, yTest)
        f1 = f1_score(yTest, probClasses)
        f1_train = f1_score(yTrain, fitment)

        basis_weights = clr.coef_
        filters = np.dot(basis_weights, B.T)
        filters = filters.reshape((-1, window))
        filter_norms = np.linalg.norm(filters, axis=1)
        df.loc[len(df.index)] = [groupname, score, f1, f1_train, filter_norms, filters]

    return df


def add_features_toFit(features, output, groupname, prevFeaturesNames=None, prevFeatures=None):
    if prevFeatures is None:
        prevFeatures = []
    if prevFeaturesNames is None:
        prevFeaturesNames = []

    outputdf = pd.DataFrame(columns=['group', 'ftr', 'score', 'f1', 'f1_train', 'filter_norms', 'filters'])

    for i, ftr in enumerate(ftrList):
        if ftr in prevFeaturesNames:
            continue
        print(f"WORKING ON {ftr}")
        t0 = time.time()
        nextfeature = features[:, i, :]
        if len(prevFeatures):
            # import pdb;pdb.set_trace()
            nextfeature = np.concatenate([prevFeatures, nextfeature], axis=1)
        df = fit_glm(nextfeature, output, groupname)
        df['ftr'] = ftr
        print(df.mean(numeric_only=True))
        t1 = time.time()
        print(f"DONE ({round(t1 - t0, 2)}s)")
        outputdf = pd.concat([outputdf, df])

    # import pdb; pdb.set_trace()
    performanceRank = outputdf.groupby('ftr').mean().sort_values('f1', ascending=False)
    topFtr = performanceRank.index[0]
    topFtrScore = performanceRank.f1[0]
    topFtrScore_train = performanceRank.f1_train[0]

    topFtr = prevFeaturesNames + [topFtr]

    print(f"BEST FEATURE PERFORMANCE IS:  {topFtr}  ;  F1 SCORE  : {topFtrScore} ")

    topIdxs = [ftrList.index(ftr) for ftr in topFtr]
    topfeatures = features[:, topIdxs, :]
    topfeatures = topfeatures.reshape((topfeatures.shape[0], -1))
    return topFtr, topFtrScore, topfeatures, topFtrScore_train


def model_selection(groupname):
    print(groupname)
    savepath = r'/tigress/MMURTHY/Kyle/code/edna'
    featurepath = fr'/tigress/MMURTHY/Kyle/code/edna/results/{groupname}/features.pkl'
    outputpath = fr'/tigress/MMURTHY/Kyle/code/edna/results/{groupname}/output.pkl'

    with open(featurepath, 'rb') as pkl:
        features = pickle.load(pkl)
    with open(outputpath, 'rb') as pkl:
        output = pickle.load(pkl)
    features = features.reshape((-1, 9, 450))

    performance = pd.DataFrame(columns=['rank', 'feature', 'score', 'train'])
    # FIND THE TOP PERFORMING FEATURE
    topFtrs, topScore, nextfeatures, topTrain = add_features_toFit(features, output, groupname)
    performance.loc[0] = [0, topFtrs[-1], topScore, topTrain]

    # FIND THE NEXT TOP PERFORMING FEATURES
    for i in range(5):
        topFtrs, topScore, nextfeatures, topTrain = add_features_toFit(features, output, groupname,
                                                                       prevFeaturesNames=topFtrs,
                                                                       prevFeatures=nextfeatures)
        performance.loc[i + 1] = [i + 1, topFtrs[-1], topScore, topTrain]

    plt.figure(figsize=(6,4))
    plt.title(groupname)
    sns.pointplot(data=performance, x='feature', y='score', c='r', palette=['r'])
    sns.pointplot(data=performance, x='feature', y='train', c='gray', palette=['gray'])
    # plt.ylim(0.6, .9)
    plt.show()
    plt.savefig(os.path.join(savepath, f'results/{groupname}/add_ftrs.png'))
    plt.close()
    return performance


def main(groupname):
    exptList = glob.glob(rf'/cup/murthy/Kyle/data/edna/{groupname}/**')
    savepath = r'/tigress/MMURTHY/Kyle/code/edna'
    print(groupname, len(exptList))

    features, output = design(exptList)
    saveFtr = os.path.join(savepath, f'results/{groupname}/features.pkl')
    saveoutput = os.path.join(savepath, f'results/{groupname}/output.pkl')
    with open(saveFtr, 'wb') as pkl:
        pickle.dump(features, pkl)
    with open(saveoutput, 'wb') as pkl:
        pickle.dump(output, pkl)

    groupdf = fit_glm(features, output, groupname)
    saveFile = os.path.join(savepath, rf'results/{groupname}/multi_results2.pkl')

    with open(saveFile, 'wb') as pkl:
        pickle.dump(groupdf, pkl)

    print(groupname, "FINISHED AND SAVED")


if __name__ == "__main__":
    exptGroups = [os.path.basename(i) for i in glob.glob(r'/cup/murthy/Kyle/data/edna/*')]
    exptGroups.remove('control')
    random.seed(47)
    # performance = model_selection(exptGroups[0])

    with multiprocessing.Pool(4) as pool:
        pool.map(model_selection, [exptfolder for exptfolder in exptGroups])

import os
import glob
import h5py
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from imblearn.over_sampling import RandomOverSampler
import sklearn
from sklearn.metrics import log_loss, f1_score

from utils import bases, song_utils

import multiprocessing


def load_feature(filepath, ftr, fill=False):
    with h5py.File(filepath, 'r') as f:
        if ftr in f.keys():
            feature = np.copy(f[ftr])
        else:
            feature = []
            return feature
    if ftr.endswith('V'):
        feature[np.where(feature > 5)[0]] = np.nan
        feature[np.where(feature < -5)[0]] = np.nan
        feature = fill_missing(feature)
    if fill:
        feature = fill_missing(feature).flatten()

    return feature


def fill_missing(x, kind="nearest", **kwargs):
    if x.ndim == 3:
        return np.stack([fill_missing(xi, kind=kind, **kwargs) for xi in x], axis=0)
    return pd.DataFrame(x).interpolate(kind=kind, axis=0, limit_direction='both', **kwargs).to_numpy()


def splitTrainTest(x, y, testFrac=0.2, oversample=True):
    '''Split data into train and test set.'''

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


def design(ftrfile, feat, window, mfDist_thresh, fmAng_thresh, px2mm=30):
    output = []
    D = []

    # predicting ovipositer bout starts
    oeStarts = load_feature(ftrfile, 'oeStarts')
    oeEnds = load_feature(ftrfile, 'oeEnds')
    mfDist = load_feature(ftrfile, 'mfDist', fill=True)  # units = px (29px/mm)
    fmAng = load_feature(ftrfile, 'fmAng', fill=True)

    # set up design matrix
    usedIdxs = np.zeros(len(mfDist))
    vidlen = range(mfDist.shape[0])

    for st in oeStarts:
        start = st - window
        stop = st

        if start < 0 or stop >= vidlen[-1]:  # if window is out of range don't use
            continue

        if np.mean(mfDist[start:stop]) > mfDist_thresh * px2mm: continue  # excluding when the animals are far apart
        if np.mean(np.abs(fmAng[start:stop])) > fmAng_thresh: continue  # excluding when male is not facing female

        if np.sum(usedIdxs[start:stop]) > 0:  # no double-dipping
            continue
        if np.all(np.isnan(mfDist[start:stop])):
            continue
        if np.nanmean(mfDist[start:stop]) > 300:
            continue

        trial = vidlen[start:stop]

        usedIdxs[start:stop + 1] = 1  # prevent double usage of data for negative class

        output.append(1)
        D.append(feat[trial])

    # prevent use of data during OE or very close to OE
    for st, en in zip(oeStarts, oeEnds):
        usedIdxs[st - window * 2: en + window * 2] = 1

    unusedIdxs = np.where(usedIdxs == 0)[0]
    random.shuffle(unusedIdxs)
    nStarts = 0
    # import pdb; pdb.set_trace()
    # cycle through randomly selected frames that have not been used yet
    for st in unusedIdxs:
        # if nStarts==len(oeStarts*5):
        #     continue
        start = st - window
        stop = st

        if start < 0 or stop >= vidlen[-1]:  # must be within video limits
            continue

        if np.mean(mfDist[start:stop]) > mfDist_thresh * px2mm: continue  # excluding when the animals are far apart
        if np.mean(np.abs(fmAng[start:stop])) > fmAng_thresh: continue  # excluding when male is not fac

        # make sure we are not overlapping with previously used trials
        if np.sum(usedIdxs[start - window:stop + window]) > 0:
            continue
        if np.all(np.isnan(mfDist[start:stop])):
            continue

        trial = vidlen[start:stop]
        usedIdxs[start:stop + 1] = 1

        output.append(0)
        D.append(feat[trial])
        nStarts += 1
    y = np.array(output)
    x = np.array(D)
    return x, y


def load_design(exptList, ftr, window, mfDist_thresh=10, fmAng_thresh=60, px2mm=30):
    fullDesignMatrix = []
    fullOutput = []

    for exptDir in exptList:
        #         initialize feature and Zscore it
        fly = os.path.basename(exptDir)
        # print(fly)
        ftrfile = os.path.join(exptDir, fly + '_smoothed.h5')
        # print(ftrfile)
        songftrs = os.path.join(exptDir, fly + '_song.h5')
        if fly == '220809_162800_16276625_rig2_1':  # weird tracks
            continue
        if ftr == 'wings':
            wingL = load_feature(ftrfile, 'wingML', fill=True)
            wingR = load_feature(ftrfile, 'wingMR', fill=True)
            wings = np.stack((wingL, wingR)).T
            feat = abs(np.max(wings, axis=1))
            feat = gaussian_filter1d(feat, sigma=2)
            feat = zscore(feat, nan_policy='omit')
        elif ftr == 'song':
            pulse = load_feature(songftrs, 'pulseStEn')
            sine = load_feature(songftrs, 'sineStEn')
            frame_at_sample = load_feature(ftrfile, 'frame_at_sample')
            feat = np.zeros(int(frame_at_sample[-1]))
            if pulse.any():
                pulse = frame_at_sample[pulse].astype(int)
                for st, en in pulse:
                    feat[st:en] = 1
            if sine.any():
                sine = frame_at_sample[sine].astype(int)
                for st, en in sine:
                    feat[st:en] = 2

        elif ftr != 'pulse' and ftr != 'sine':
            feat = load_feature(ftrfile, ftr, fill=True)
            feat = gaussian_filter1d(feat, sigma=2)
            feat = zscore(feat, nan_policy='omit')
            if np.isnan(feat).any():
                continue

        else:  # feature = 'sine' or 'pulse'
            ftrname = ftr + 'StEn'
            feature = load_feature(songftrs, ftrname)
            frame_at_sample = load_feature(ftrfile, 'frame_at_sample')
            feat = np.zeros(int(frame_at_sample[-1]))
            if feature.any():
                feature = frame_at_sample[feature].astype(int)
                for st, en in feature:
                    feat[st:en] = 1

        x, y = design(ftrfile, feat, window, mfDist_thresh, fmAng_thresh, px2mm)

        fullDesignMatrix.append(x)
        fullOutput.append(y)
        # import pdb; pdb.set_trace();

    fullDesignMatrix = np.array([trace for array in fullDesignMatrix for trace in array])
    fullOutput = np.array([output for array in fullOutput for output in array])

    print(f"shape of x: {fullDesignMatrix.shape}")
    print(f"shape of y: {fullOutput.shape}")

    return fullDesignMatrix, fullOutput


def pcor(x, y, window):
    fraction_correct = []
    logloss = []
    filtershape = []
    filter_norms = []
    predicted = []
    # variance_explained = []
    f1_value = []
    deviance = []

    # creating the cosine bases depending on the history window
    # these are all somewhat trial and error.
    # for the most part the shapes don't matter too much, they only effect the filter shapes
    # plot with plt.plot(B)
    if window == 1500:  # 10 seconds if 150fps
        B = bases.raised_cosine(0, 12, [0, 1000], 200, window)
    elif window == 300:  # 2 seconds if 150fps
        B = bases.raised_cosine(0, 12, [0, 220], 75, window)
    elif window == 750:
        B = bases.raised_cosine(0, 12, [0, 375], 10, window)
    elif window == 450:
        B = bases.raised_cosine(0, 12, [0, 325], 150, window)
    elif window == 225:
        B = bases.raised_cosine(0, 12, [0, 175], 75, window)
    elif window == 150:
        B = bases.raised_cosine(0, 12, [0, 120], 25, window)
    else:
        print("PLEASE DEFINE WINDOW")
        return

    clr = sklearn.linear_model.LogisticRegression(C=.001, max_iter=10_000)

    i = 0
    while i <= 1000:
        data, testinds = splitTrainTest(x, y)
        xTrain = data['xTrain']
        yTrain = data['yTrain']
        xTest = data['xTest']
        yTest = data['yTest']

        Xtrain = np.dot(xTrain, B)
        Xtest = np.dot(xTest, B)

        # import pdb; pdb.set_trace()
        clr.fit(Xtrain, yTrain)
        probs = clr.predict_proba(Xtest)
        predicted.append(probs)  # the model's predicted classes for the test set
        probClasses = clr.predict(Xtest)
        # import pdb; pdb.set_trace()
        score = clr.score(Xtest, yTest)  # returns the accuracy
        # r2 = r2_score(yTest, probClasses)  # finds the r^2 value
        ll = log_loss(yTest, probs)  # finds the log loss
        f1 = f1_score(yTest, probClasses)

        fraction_correct.append(score)
        # variance_explained.append(r2)
        logloss.append(ll)
        f1_value.append(f1)

        dev = 2 * log_loss(yTest, probs, normalize=False)
        deviance.append(dev)

        filt = B @ clr.coef_[:, None].flatten()  # reconstructs the filter to time-domain from model coefficients
        filtershape.append(filt)

        filter_norm = np.linalg.norm(
            filt)  # the test score is proportional to the norm of the filters if all on same scale
        filter_norms.append(filter_norm)

        i += 1

    filtershape = np.array(filtershape)
    return fraction_correct, f1_value, logloss, filtershape, filter_norms, deviance


def plot_filters(filtershapes, ftr, savepath):
    plt.figure(figsize=(8, 3))

    plt.title(ftr)
    plt.plot(filtershapes.T, alpha=0.01, c='k')
    plt.plot(np.mean(filtershapes, axis=0), c='r')
    xticks = np.arange(0, filtershapes.shape[-1] + 1, 75)
    xlabels = np.arange(-filtershapes.shape[-1] / 150, 0.1, .5)
    plt.xticks(xticks, xlabels)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def main(exptfolder, window=1500, overwrite=False):
    cup = song_utils.getParentDirectory('cup')
    tigress = song_utils.getParentDirectory('tigress')

    exptList = glob.glob(os.path.join(cup, rf'Kyle/data/edna/{exptfolder}/**'))

    savepath = os.path.join(tigress, r'Kyle/code/edna')
    print(fr"Saving results to : /results/{exptfolder}")

    if not os.path.exists(os.path.join(savepath, f'results/{exptfolder}')):
        os.makedirs(os.path.join(savepath, f'results/{exptfolder}'))

    ftrList = [
        "mfDist",
        "mFV", "fFV",
        "mFA", "fFA",
        "mLV", "fLV",
        "mLS", "fLS",
        "mLA", "fLA",
        "mRS", "fRS",
        "mfAng", "fmAng",
        "song", "pulse", "sine"
    ]

    random.seed(15)
    results = pd.DataFrame(columns=['feature', 'pCor', 'f1_score', 'logloss', 'filterNorms', 'deviance'])
    saveFile = os.path.join(savepath, rf'results/{exptfolder}/{window}_results.csv')
    if os.path.exists(saveFile) and overwrite:
        print("overwriting previous results")
    elif os.path.exists(saveFile) and not overwrite:
        print("results csv file exists but will not overwrite features")
        results = pd.read_csv(saveFile, index_col=0)
        ftrList = [ftr for ftr in ftrList if ftr not in results['feature'].unique()]

    for ftr in ftrList:
        print("working on {}".format(ftr))

        x, y = load_design(exptList, ftr, window, mfDist_thresh=10)
        fracCorr, f1value, lloss, fshapes, fnorms, dev = pcor(x, y, window)

        print("plotting {} filters".format(ftr))
        filterSavePath = os.path.join(savepath, f'results/{exptfolder}/{window}_{ftr}_filters.png')
        plot_filters(fshapes, ftr, filterSavePath)

        for pc, f1, ll, fn, d in zip(fracCorr, f1value, lloss, fnorms, dev):
            results.loc[len(results.index)] = [ftr, pc, f1, ll, fn, d]
        print("done with {}\n".format(ftr))
        results.to_csv(saveFile)

    print("saving final dataframe\n\n\n")
    results.to_csv(saveFile)

    print("All done")


if __name__ == "__main__":

    exptPath = r'/run/user/1000/gvfs/smb-share:server=cup.pni.princeton.edu,share=murthy/Kyle/data/edna/'

    with multiprocessing.Pool(5) as pool:
        pool.map(main, [expt_folder for expt_folder in os.listdir(exptPath)])

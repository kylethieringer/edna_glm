import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import seaborn as sns
import scipy.stats
from multiprocessing import Pool
import sys
sys.path.append(os.path.dirname(os.path.realpath('')))
from utils import features
plt.style.use("default")


### list of features to plot
ftrlist = ['fFV','fLV','fRS','fmAng',
           'mFV','mLV','mRS','mfAng',
           'mfDist','mfDist_mHead_fAbd',
           'pulse', 'sine']

### which groups belong in which plot
manipulated = ['blind', 'blind_deaf', 'deaf', 'AD_control_BD']
silenced = ['AD_control', 'LC31_Kir', 'vpoEN_Kir']

### setting speed limits to get rid of outliers
speedLimit_mms = 50  # mm/s
speedLimit = speedLimit_mms *30/150  # px/frame

### how much time before / after to plot
window_sec = 10  # seconds
window = window_sec*150  # frames

### color scheme to use
colors = {'AD_control_BD':'k',
          'AD_control':'k',
          'blind':'deepskyblue',
          'LC31_Kir':'deepskyblue',
          'deaf':'forestgreen',
          'vpoEN_Kir':'forestgreen',
          'blind_deaf':'magenta'}


def load_feature(filepath,ftr, remove_outliers = False, outlier_limit = None, fill=False):

    with h5py.File(filepath, 'r') as f:
        feature = np.array(f[ftr])
    if remove_outliers:
        idxs = np.where(abs(feature)>outlier_limit)[0]
        feature[idxs] = np.nan
    if fill:
        feature = features.fill_missing(feature)
    return feature


def load_songftr(songpath, ftrpath, ftr):

    if not ftr.endswith('StEn'):
        ftr = ftr+'StEn'
    with h5py.File(songpath, 'r') as f:
        featureStEn = np.array(f[ftr])
    with h5py.File(ftrpath, 'r') as f:
        frame_at_sample = np.array(f['frame_at_sample'])
        trxM = np.array(f['trxM'])

    lastframe = len(trxM)
    feature = np.zeros(lastframe)
    if len(featureStEn)==0:
        return feature

    featureStEn = frame_at_sample[featureStEn].astype(int)
    for st, en in featureStEn:
        feature[st:en] = 1

    return feature


def get_all_data(exptGroups):
    df = pd.DataFrame(columns=['group', 'fly', 'ftr', 'window'])

    for groupDir in exptGroups:
        group = os.path.basename(groupDir)
        print("GROUP: ", group)
        exptList = glob.glob(os.path.join(groupDir, '2*'))
        for exptDir in exptList:
            fly = os.path.basename(exptDir)
            print("FLY: ", fly)
            ftrfile = os.path.join(exptDir, fly + '.h5')
            songfile = os.path.join(exptDir, fly+'_song.h5')
            if not os.path.exists(ftrfile) or not os.path.exists(songfile):
                continue
            try:
                oeStarts = load_feature(ftrfile, 'oeStarts')
            except KeyError:
                print("OE doesnt exist in : ", exptDir)
                continue
            for ftr in ftrlist:
                if ftr.endswith('V') or ftr.endswith('S'):
                    remove_outliers = True
                    outlier_limit = speedLimit
                else:
                    remove_outliers = False
                    outlier_limit = None
                if ftr != 'pulse' and ftr != 'sine':
                    feature = load_feature(ftrfile, ftr, remove_outliers=remove_outliers, outlier_limit=outlier_limit)
                else:
                    feature = load_songftr(songfile, ftrfile, ftr+'StEn')

                for oe in oeStarts:
                    if oe < window or len(feature) - oe < window:  # make sure we are in range
                        continue
                    idxrange = range(oe - window, oe + window)

                    df.loc[len(df.index)] = [group, fly, ftr, feature[idxrange]]
    return df


def get_group_data(groupDir):
    df = pd.DataFrame(columns=['group', 'fly', 'ftr', 'window'])

    group = os.path.basename(groupDir)
    print("GROUP: ", group)
    exptList = glob.glob(os.path.join(groupDir, '2*'))
    for exptDir in exptList:
        fly = os.path.basename(exptDir)
        print("FLY: ", fly)
        ftrfile = os.path.join(exptDir, fly + '.h5')
        songfile = os.path.join(exptDir, fly+'_song.h5')
        if not os.path.exists(ftrfile) or not os.path.exists(songfile):
            continue
        try:
            oeStarts = load_feature(ftrfile, 'oeStarts')
        except KeyError:
            print("OE doesnt exist in : ", exptDir)
            continue
        for ftr in ftrlist:
            if ftr.endswith('V') or ftr.endswith('S'):
                remove_outliers = True
                outlier_limit = speedLimit
            else:
                remove_outliers = False
                outlier_limit = None
            if ftr != 'pulse' and ftr != 'sine':
                feature = load_feature(ftrfile, ftr, remove_outliers=remove_outliers, outlier_limit=outlier_limit)
            else:
                feature = load_songftr(songfile, ftrfile, ftr+'StEn')

            for oe in oeStarts:
                if oe < window or len(feature) - oe < window:  # make sure we are in range
                    continue
                idxrange = range(oe - window, oe + window)

                df.loc[len(df.index)] = [group, fly, ftr, feature[idxrange]]
    return df

def split_all_data(df):
    manipulatedDF = pd.DataFrame(columns=['group', 'ftr', 'avg', 'sem'])
    silencedDF = pd.DataFrame(columns=['group', 'ftr', 'avg', 'sem'])

    for group in df.group.unique():
        groupDF = df[df.group == group]
        for ftr in ftrlist:
            ftrarr = np.stack(groupDF[groupDF.ftr == ftr]['window'].to_numpy()).T
            ftravg = np.nanmean(ftrarr, axis=1)
            ftrsem = scipy.stats.sem(ftrarr, axis=1, nan_policy='omit').data

            if group in manipulated:
                manipulatedDF.loc[len(manipulatedDF.index)] = [group, ftr, ftravg, ftrsem]
            elif group in silenced:
                silencedDF.loc[len(silencedDF.index)] = [group, ftr, ftravg, ftrsem]
            else:
                print("error")

    return manipulatedDF, silencedDF


def plot_df(df, saveDir, plotName, window=window_sec, overwrite=False):
    for ftr in ftrlist:
        featureDF = df[df.ftr == ftr]
        plt.figure(figsize=(8, 3))
        for group in df.group.unique():
            avg = featureDF[featureDF.group == group]['avg'].to_numpy()[0]
            sem = featureDF[featureDF.group == group]['sem'].to_numpy()[0]
            plt.plot(avg, label=group, color=colors[group])
            plt.fill_between(np.arange(0, len(avg)), avg - sem, avg + sem, alpha=0.2, color=colors[group])
            plt.legend()
            plt.xticks(np.arange(0, len(avg) + 1, 300),
                       labels=(np.arange(-len(avg) / 2, len(avg) / 2 + 1, 300) / 150).astype(int))
            plt.axvline(len(avg) / 2, c='k', ls='--', alpha=0.5)
            plt.xlabel('time since OE onset (s)')
            plt.title(ftr)
            sns.despine()
            plt.tight_layout()

            savepath = os.path.join(saveDir, f'{plotName}_{ftr}_{window}s.pdf')
            if not os.path.exists(savepath):
                plt.savefig(savepath)
            elif os.path.exists(savepath) and overwrite:
                plt.savefig(savepath)

        plt.show()
        plt.close()


def plot_combined(df1, df2, saveDir, window=window_sec, overwrite=False):
    for ftr in ftrlist:
        feature_s_DF = df1[df1.ftr == ftr]
        feature_m_DF = df2[df2.ftr == ftr]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for group in df1.group.unique():
            avg_s = feature_s_DF[feature_s_DF.group == group]['avg'].to_numpy()[0]
            sem_s = feature_s_DF[feature_s_DF.group == group]['sem'].to_numpy()[0]
            axs[0].plot(avg_s, label=group, color=colors[group], lw=0.5)
            axs[0].fill_between(np.arange(0, len(avg_s)), avg_s - sem_s, avg_s + sem_s, alpha=0.2, color=colors[group])
            axs[0].set_xticks(np.arange(0, len(avg_s) + 1, 300),
                              labels=(np.arange(-len(avg_s) / 2, len(avg_s) / 2 + 1, 300) / 150).astype(int))
            axs[0].axvline(len(avg_s) / 2, c='k', ls='--', alpha=0.5)
            axs[0].set_xlabel('time since OE onset (s)')
            axs[0].legend()
        for group in df2.group.unique():
            avg_m = feature_m_DF[feature_m_DF.group == group]['avg'].to_numpy()[0]
            sem_m = feature_m_DF[feature_m_DF.group == group]['sem'].to_numpy()[0]
            axs[1].plot(avg_m, label=group, color=colors[group], lw=0.5)
            axs[1].fill_between(np.arange(0, len(avg_m)), avg_m - sem_m, avg_m + sem_m, alpha=0.2, color=colors[group])
            axs[1].set_xticks(np.arange(0, len(avg_m) + 1, 300),
                              labels=(np.arange(-len(avg_m) / 2, len(avg_m) / 2 + 1, 300) / 150).astype(int))
            axs[1].axvline(len(avg_m) / 2, c='k', ls='--', alpha=0.5)
            axs[1].set_xlabel('time since OE onset (s)')
            axs[1].legend()
        plt.suptitle(ftr)

        sns.despine()
        plt.tight_layout()

        savepath = os.path.join(saveDir, f'combined_{ftr}_{window}s.pdf')
        if not os.path.exists(savepath):
            plt.savefig(savepath)
        elif os.path.exists(savepath) and overwrite:
            plt.savefig(savepath)
        plt.show()
        plt.close()


def main():
    exptGroups = glob.glob(r'/cup/murthy/Kyle/data/edna/*')
    exptGroups.remove('/cup/murthy/Kyle/data/edna/control')

    with Pool(5) as pool:
        groupdfs = pool.map(get_group_data, exptGroups)
    df = pd.concat(groupdfs)

    manipulatedDF, silencedDF = split_all_data(df)
    saveDir = os.path.join(os.path.dirname(os.path.realpath("")), 'results/featureplots')

    plot_df(manipulatedDF, saveDir, plotName="manipulated")
    plot_df(silencedDF, saveDir, plotName="silenced")
    plot_combined(silencedDF, manipulatedDF, saveDir)


if __name__=="__main__":
    main()
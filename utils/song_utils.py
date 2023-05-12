import os
import h5py
import numpy as np
from scipy.io import loadmat
import pandas as pd
import sys
import glob
import multiprocessing

def getParentDirectory(fileserver):
    """ check which operating system then direct to tigress or cup"""
    if sys.platform == 'linux':
        pDir = r'/run/user/1000/gvfs/'
        tigress = r'smb-share:server=tigress-cifs.princeton.edu,share=fileset-mmurthy'
        cup = r'smb-share:server=cup.pni.princeton.edu,share=murthy'
    elif sys.platform == 'win32':
        pDir = ''
        tigress = r'M:/'
        cup = r'W:/'
    elif sys.platform == 'darwin':
        pDir = r'/Volumes/'
        tigress = r'fileset-mmurthy'
        cup = r'murthy'
    else:
        print('dont know which operating system you are using...')
        return
    if fileserver == 'tigress':
        fullDirectory = os.path.join(pDir, tigress)
    elif fileserver == 'cup':
        fullDirectory = os.path.join(pDir, cup)
    else:
        print('please choose either "cup" or "tigress" as input')
        return

    return fullDirectory


def getValid_sineBouts(boutsten, sineStEn, boutType):
    """ filter sine bInf Sine bouts with the valid sines"""
    sineBouts = []
    if len(sineStEn) > 0:
        sinBoutsStEn = boutsten[np.where(boutType == 'Sin')[0]]
        for i in range(len(sinBoutsStEn)):
            st = sinBoutsStEn[i][0]
            en = sinBoutsStEn[i][1]

            validSine = sineStEn[np.logical_and(sineStEn[:, 0] >= st, sineStEn[:, 1] <= en)]
            sineBouts.append(validSine)
        if len(sineBouts) != 0:
            sineBouts = np.vstack(sineBouts)
        else:
            sineBouts = np.array(sineBouts)

    return sineBouts


def get_IPI(boutsten, sineStEn, wc):
    """ return IPI of pulse trains from song segmenter """
    IPI = []
    pulseTrains = []

    for st, en in boutsten:

        # is there any sine in this bout
        if len(sineStEn) > 0:
            sineInBout = sineStEn[np.logical_and(sineStEn[:, 0] >= st, sineStEn[:, 1] <= en)]
        else:
            sineInBout = np.array([])
        # where are pulse centers in this bout
        pulseInBout = wc[np.logical_and(wc >= st, wc <= en)]

        if pulseInBout.any() and ~sineInBout.any():
            IPI.append(np.diff(pulseInBout))
            pulseTrains.append(pulseInBout)

        elif pulseInBout.any() and sineInBout.any():
            # pulse train before first sine train
            pulBeforeSin = pulseInBout[pulseInBout <= sineInBout[0, 0]]
            IPI.append(np.diff(pulBeforeSin))
            pulseTrains.append(pulBeforeSin)

            # pulse trains between sine trains
            for i in range(len(sineInBout)):
                if i + 1 >= len(sineInBout): continue
                pulTrains = pulseInBout[
                    np.logical_and(pulseInBout > sineInBout[i, 1], pulseInBout < sineInBout[i + 1, 0])]
                IPI.append(np.diff(pulTrains))
                pulseTrains.append(pulTrains)

            # pulse train after last sine train
            pulAfterSin = pulseInBout[pulseInBout >= sineInBout[-1, 1]]
            IPI.append(np.diff(pulAfterSin))
            pulseTrains.append(pulAfterSin)

    interim_pulseTrains = []
    for idx in range(len(IPI)):
        gap = np.where(IPI[idx] > 1000)[0]  # 100ms (.1 seconds * 10000 samples/second)

        if len(gap) > 0:

            for i in range(len(gap)):
                if i == 0:
                    interim_pulseTrains.append(pulseTrains[idx][0:gap[i] + 1])

                if i == len(gap) - 1:
                    interim_pulseTrains.append(pulseTrains[idx][gap[i] + 1:])
                else:
                    interim_pulseTrains.append(pulseTrains[idx][gap[i - 1]:gap[i] + 1])

        else:
            interim_pulseTrains.append(pulseTrains[idx])

    filtered_IPI = []
    filtered_pulseTrains = []
    for train in interim_pulseTrains:
        if len(train) >= 3:
            ipi = np.diff(train) / 10000
            filtered_IPI.append(ipi)
            filtered_pulseTrains.append(train)

    return filtered_IPI, filtered_pulseTrains


def get_occ(trxfile):
    """ load track occupancy from tracking data 
        TODO: add this to feature file so we can just load it in with everything else
    """
    with h5py.File(trxfile, 'r') as t:
        trx_occ = np.copy(t['track_occupancy']).T
    totOcc = np.sum(trx_occ, axis=0)
    first = np.where(totOcc)[0][0]
    last = np.where(totOcc)[0][-2]
    return first, last


def fill_missing(x, kind="nearest", **kwargs):
    """Fill missing values in a timeseries.
    Author: TP
    Args:
        x: Timeseries of shape (time, _) or (_, time, _).
        kind: Type of interpolation to use. Defaults to "nearest".
    Returns:
        Timeseries of the same shape as the input with NaNs filled in.

    Notes:
        This uses pandas.DataFrame.interpolate and accepts the same kwargs.
    """
    if x.ndim == 3:
        return np.stack([fill_missing(xi, kind=kind, **kwargs) for xi in x], axis=0)
    return pd.DataFrame(x).interpolate(kind=kind, axis=0, limit_direction='both', **kwargs).to_numpy()


def filter_Sine(sineStEn, frame_at_sample, wingL, wingR, min_sine_wing_ang):
    """ remove false positives of sine segmentation based on wing angle """
    valid = []
    for st, en in sineStEn:
        fst = int(frame_at_sample[st])
        fen = int(frame_at_sample[en])
        wings = np.vstack([wingL[fst:fen], wingR[fst:fen]])
        if (np.nanmean(wings, axis=1) > min_sine_wing_ang).any():
            valid.append(True)
        else:
            valid.append(False)
    if len(valid) > 0:
        valid = np.stack(valid)
    sineStEn = sineStEn[valid]

    return sineStEn


def get_songFtrs(ftrfile, song, min_sine_wing_ang=30, min_sine_noise=0):
    """ pull out song features from the segmenter
    Args:
        ftrfile = path to file containing features
        song = mat array containing song segmentation
                or path to song file

    Returns: 
        stats (dict) : containing
            wc = pulse wave centers
            sineStEn = start and end of sine filtered with wing angle
            IPIs = distance between pulse centers in each pulse train
            pulTrainLens = length of pulse trains (calculated by sum of IPIs)
            numPulTrains = number of pulses in each pulse train
            percentSine = lengths of sine bouts divided by length of experiment (before copulation)
            percentPul = pulTrainLens divided by length of experiment (before copulation)
            percentSong = length of song bouts divided by length of experiment (before copulation)
    """
    trxpath = os.path.join(os.path.dirname(ftrfile), '000000.mp4.inference.cleaned.proofread.tracking.h5')
    exptDir = os.path.dirname(ftrfile)
    fly = os.path.basename(exptDir)

    if os.path.exists(trxpath):
        f0, f1 = get_occ(trxpath)
        trx_occ = True
    else:

        trxpath = os.path.join(exptDir, fly + '.000000.mp4.inference.cleaned.tracking.h5')
        if os.path.exists(trxpath):
            f0, f1 = get_occ(trxpath)
            trx_occ = True
        else:
            trx_occ = False

    if type(song) == str:
        loadvars = ['pInf', 'bInf', 'wInf', 'oneSong', 'noiseSample']
        song = loadmat(song, variable_names=loadvars)

    with h5py.File(ftrfile, 'r') as f:
        wingL = np.copy(f['wingML'])
        wingR = np.copy(f['wingMR'])
        frame_at_sample = np.copy(f['frame_at_sample'])

        if not trx_occ:
            # track occupancy based on male tracks
            f0 = 0  # assuming tracking starts at first frame (best would be to use track occupancy from ~.tracking.h5)
            f1 = f['trxM'].shape[0]

        if 'sample_at_frame' in f.keys():
            # first and last sample 
            s0 = f['sample_at_frame'][f0].astype(int)
            s1 = f['sample_at_frame'][f1].astype(int)

        else:
            print(fly, "ERROR with loading sample bounds")
            return

    wingL = fill_missing(wingL).flatten()
    wingR = fill_missing(wingR).flatten()

    oneSong = song['oneSong']
    nAmp = np.mean(abs(np.ptp(song['noiseSample'], axis=0)))

    # pulse wave centers
    wc = song['pInf']['wc'][0][0]
    wc = wc[np.logical_and(wc >= s0, wc <= s1)]

    # extracting song from bouts
    bouts = song['bInf']['stEn'][0][0]
    boutsten = bouts[np.logical_and(bouts[:, 0] >= s0, bouts[:, 1] <= s1)]

    sineStEn = song['wInf']['stEn'][0][0]
    sineStEn = sineStEn[np.logical_and(sineStEn[:, 0] >= s0, sineStEn[:, 1] <= s1)]

    # filter out false positive sines
    stEn = filter_Sine(sineStEn, frame_at_sample, wingL, wingR, min_sine_wing_ang)

    # filter out sines with amplitude less than 1.5 x amplitude of noise
    sineStEn = []
    for st, en in stEn:
        sAmp = abs(np.ptp(oneSong[st:en]))
        if sAmp > nAmp * min_sine_noise:
            sineStEn.append([st, en])

    sineStEn = np.array(sineStEn)

    # Analyze Pulse
    _, pulseTrains = get_IPI(boutsten, sineStEn, wc)

    pulseStEn = []
    for t in pulseTrains:
        if t.any():
            pulseStEn.append([t[0], t[-1]])
    if len(pulseStEn) > 0:
        pulseStEn = np.stack(pulseStEn)

    song_ftrs = {
        'wc': wc,
        'sineStEn': sineStEn,
        'pulseStEn': pulseStEn,
        'boutStEn': boutsten,
        'oneSong': oneSong

    }

    return song_ftrs


def save_songFtrs(song_path, output_path, overwrite=False):
    print(f"will save song variables to {output_path}")
    if not overwrite and os.path.exists(output_path):
        print("FILE EXISTS WILL NOT OVERWRITE: ")
        return output_path
    fly = os.path.basename(os.path.dirname(song_path))
    ftrfile = os.path.join(os.path.dirname(output_path), fly + '_smoothed.h5')

    if not os.path.exists(ftrfile):
        print("FEATURES DONT EXISTS CANNOT PROCEED: ")
        return output_path
    song_ftrs = get_songFtrs(ftrfile, song_path)

    origSineSegPath = os.path.join(os.path.dirname(song_path), 'sine.csv')
    if os.path.exists(origSineSegPath):
        origSine = pd.read_csv(origSineSegPath).to_numpy().flatten()
    else:
        origSine = []
    origPulseSegPath = os.path.join(os.path.dirname(song_path), 'pulse.csv')
    if os.path.exists(origPulseSegPath):
        origPulse = pd.read_csv(origSineSegPath).to_numpy().flatten()
    else:
        origPulse = []

    expt_name = os.path.basename(os.path.dirname(output_path))
    with h5py.File(output_path, 'w') as f:
        f.create_dataset("expt_name", data=expt_name)
        f.create_dataset("pulseWC", data=song_ftrs['wc'], compression=1)
        f.create_dataset("sineStEn", data=song_ftrs['sineStEn'], compression=1)
        f.create_dataset("pulseStEn", data=song_ftrs['pulseStEn'], compression=1)
        f.create_dataset("boutStEn", data=song_ftrs['boutStEn'], compression=1)
        f.create_dataset("oneSong", data=song_ftrs['oneSong'], compression=1)
        f.create_dataset("origSine", data=origSine, compression=1)
        f.create_dataset("origPulse", data=origPulse, compression=1)

    print("done")
    return output_path


# def main(expt_Dir):
#     if expt_Dir=='control':
#         return
#     rawDataDict = {
#         "AD_control": "experimental_sleapOE_AK_combined_to_analyze",
#         "AD_control_BD": "manipulated_summer22_AK_controls",
#         "blind":"manipulated_summer22_AK_blind",
#         "deaf":"manipulated_summer22_AK_deaf",
#         "blind_deaf":"manipulated_summer22_AK_blindanddeaf",
#         "LC31_Kir":"experimental_sleapOE_fall21_LK",
#         "vpoEN_Kir":"experimental_sleapOE_VK_combined_to_analyze"
#     }
#
#     exptPath = glob.glob(
#         fr'/run/user/1000/gvfs/smb-share:server=tigress-cifs.princeton.edu,share=fileset-mmurthy/eDNA/behavior/{rawDataDict[expt_Dir]}/**')
#     outputDir = f'/run/user/1000/gvfs/smb-share:server=cup.pni.princeton.edu,share=murthy/Kyle/data/edna/{expt_Dir}'
#
#     for expt_folder in exptPath:
#         fly = os.path.basename(expt_folder)
#         if fly.endswith('.txt'):
#             continue
#         output_pathDir = os.path.join(outputDir, fly)
#         song_path = os.path.join(expt_folder, 'daq_segmentation_new.mat')
#         save_songpath = os.path.join(output_pathDir, f"{fly}_song.h5")
#
#         save_songFtrs(song_path, save_songpath, overwrite=False)
#
#
# if __name__ == "__main__":
#     exptPaths = os.listdir('/run/user/1000/gvfs/smb-share:server=cup.pni.princeton.edu,share=murthy/Kyle/data/edna/')
#
#     with multiprocessing.Pool(8) as pool:
#         pool.map(main, [expt_folder for expt_folder in exptPaths])

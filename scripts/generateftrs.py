import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.realpath('')))
from utils import features, song_utils
from multiprocessing import Pool


def main(item):
    savepath, rawpath = item

    # rawDir = r'/run/user/1000/gvfs/smb-share:server=tigress-cifs.princeton.edu,share=fileset-mmurthy/eDNA/behavior'
    # saveDir = '/run/user/1000/gvfs/smb-share:server=cup.pni.princeton.edu,share=murthy/Kyle/data/edna/'
    rawDir = r'/tigress/MMURTHY/eDNA/behavior'
    saveDir = r'/scratch/gpfs/kt1303/edna'
    exptList = glob.glob(os.path.join(rawDir, rawpath, '2*'))

    for exptDir in exptList:
        fly = os.path.basename(exptDir)
        if fly.endswith('.txt'): continue
        output_pathDir = os.path.join(saveDir, savepath, fly)
        output_path = os.path.join(output_pathDir, f"{fly}_smoothed.h5")

        song_path = os.path.join(exptDir, 'daq_segmentation_new.mat')
        save_songpath = os.path.join(output_pathDir, f"{fly}_song.h5")

        features.make_expt_dataset(exptDir, output_path=output_path, overwrite=True, smoothTrx=True)
        song_utils.save_songFtrs(song_path, save_songpath, overwrite=True)


if __name__=="__main__":

    DataDict = {'AD_control': 'experimental_sleapOE_AK_combined_to_analyze',
                'vpoEN_Kir': 'experimental_sleapOE_VK_combined_to_analyze',
                'LC31_Kir': 'experimental_sleapOE_fall21_LK',
                'AD_control_BD': 'manipulated_summer22_AK_controls',
                'blind': 'manipulated_summer22_AK_blind',
                'deaf': 'manipulated_summer22_AK_deaf',
                'blind_deaf': 'manipulated_summer22_AK_blindanddeaf'}

    with Pool(8) as pool:
        pool.map(main, [sp_rp for sp_rp in DataDict.items()])
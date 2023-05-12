# lots of code from Shruthi R. & Talmo P. unless otherwise noted
# many functions + edits made by Kyle T. throughout

import os
import h5py
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.io
import scipy.interpolate
import glob
import multiprocessing
import sys
sys.path.append(os.path.dirname(os.path.realpath('')))
from utils import adskalman


fly_nodes = [
    'head',
    'thorax',
    'abdomen',
    'wingL',
    'wingR',
    'forelegL4',
    'forelegR4',
    'midlegL4',
    'midlegR4',
    'hindlegL4',
    'hindlegR4',
    'eyeL',
    'eyeR']


def load_tracks(expt_folder):
    """Load proofread and exported pose tracks.
    Args:
        expt_folder: Path to experiment folder containing inference.cleaned.proofread.tracking.h5.
    Returns:
        Tuple of (tracks, node_names).
        tracks contain the pose estimates in an array of shape (frame, joint, xy, fly).
        The last axis is ordered as [female, male].
        node_names contains a list of string names for the joints.
    """
    if os.path.isdir(expt_folder):
        track_file = os.path.join(expt_folder, "inference.cleaned.proofread.tracking.h5")
        if not os.path.exists(track_file):
            track_file = os.path.join(expt_folder, '000000.mp4.inference.cleaned.proofread.tracking.h5')
        if not os.path.exists(track_file):
            print('No proofread tracking file found. Using ".cleaned.tracking.h5" instead')
            track_file = os.path.join(expt_folder,
                                      os.path.basename(expt_folder) + '.000000.mp4.inference.cleaned.tracking.h5')
    else:
        track_file = expt_folder

    with h5py.File(track_file, "r") as f:
        tracks = np.transpose(f["tracks"][:])  # (frame, joint, xy, fly)
        node_names = f["node_names"][:]
        node_names = [x.decode() for x in node_names]

    # Crop to valid range.
    last_fidx = np.argwhere(np.isfinite(tracks.reshape(len(tracks), -1)).any(axis=-1)).squeeze()[-1]
    tracks = tracks[:last_fidx]

    return tracks, node_names


def find_female_track(expt_folder):
    if os.path.isdir(expt_folder):
        track_file = os.path.join(expt_folder, '000000.mp4.inference.cleaned.proofread.tracking.h5')
    else:
        raise ValueError("tracking file not found")

    with h5py.File(track_file, "r") as f:
        track_names = np.copy(f['track_names']).astype(str)

    if 'female' in track_names:
        femaleIdx = np.where(track_names == 'female')[0][0]
    else:
        femaleIdx = 0  # default
    return femaleIdx


def get_expt_sync(expt_folder):
    """Computes the sample/frame maps from experiment synchronization.
    Args:
        expt_folder: Path to experiment folder with daq.h5.
    Returns:
        frame_daq_sample: A vector of the length of the number of frames where each
            element is the estimated DAQ sample index.
        daq_frame_idx: A vector of the lenght of the number of samples where each
            element is the estimated video frame index.
    """
    with h5py.File(os.path.join(expt_folder, "daq.h5"), "r") as f:
        try:
            trigger = f["sync"][:]
        except KeyError:
            trigger = f["Sync"][:]

    # Threshold exposure signal.
    trigger[trigger[:] < 1.5] = 0
    trigger[trigger[:] > 1.5] = 1

    # Find connected components.
    daq2frame, n_frames = scipy.ndimage.label(trigger)

    # Compute sample at each frame.
    frame_idx, frame_time, count = np.unique(daq2frame, return_index=True,
                                             return_counts=True)
    frame_daq_sample = frame_time[1:] + (count[1:] - 1) / 2

    # Interpolate frame at each sample.
    f = scipy.interpolate.interp1d(
        frame_daq_sample,
        np.arange(frame_daq_sample.shape[0]),
        kind="nearest",
        fill_value="extrapolate"
    )
    daq_frame_idx = f(np.arange(trigger.shape[0]))

    return frame_daq_sample, daq_frame_idx


def h5read(filename, dataset):
    """Load a single dataset from HDF5 file.

    Args:
        filename: Path to HDF5 file.
        dataset: Name of the dataset.

    Returns:
        The dataset data loaded in.
    """
    with h5py.File(filename, "r") as f:
        return f[dataset][:]


def fill_missing(x, kind="nearest", **kwargs):
    """Fill missing values in a timeseries.

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


def normalize_to_egocentric(x, rel_to=None, scale_factor=1, ctr_ind=1, fwd_ind=0, fill=True, return_angles=False):
    """Normalize pose estimates to egocentric coordinates.

    Args:
        x: Pose of shape (joints, 2) or (time, joints, 2)
        rel_to: Pose to align x with of shape (joints, 2) or (time, joints, 2). Defaults
            to x if not specified.
        scale_factor: Spatial scaling to apply to coordinates after centering.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.
        fill: If True, interpolate missing ctr and fwd coordinates. If False, timesteps
            with missing coordinates will be all NaN. Defaults to True.
        return_angles: If True, return angles with the aligned coordinates.

    Returns:
        Egocentrically aligned poses of the same shape as the input.

        If return_angles is True, also returns a vector of angles.
    """

    if rel_to is None:
        rel_to = x

    is_singleton = (x.ndim == 2) and (rel_to.ndim == 2)

    if x.ndim == 2:
        x = np.expand_dims(x, axis=0)
    if rel_to.ndim == 2:
        rel_to = np.expand_dims(rel_to, axis=0)

    # Find egocentric forward coordinates.
    ctr = rel_to[..., ctr_ind, :]  # (t, 2)
    fwd = rel_to[..., fwd_ind, :]  # (t, 2)
    if fill:
        ctr = fill_missing(ctr, kind="nearest")
        fwd = fill_missing(fwd, kind="nearest")
    ego_fwd = fwd - ctr

    # Compute angle.
    ang = np.arctan2(ego_fwd[..., 1], ego_fwd[..., 0])  # arctan2(y, x) -> radians in [-pi, pi]
    ca = np.cos(ang)  # (t,)
    sa = np.sin(ang)  # (t,)

    # Build rotation matrix.
    rot = np.zeros([len(ca), 3, 3], dtype=ca.dtype)
    rot[..., 0, 0] = ca
    rot[..., 0, 1] = -sa
    rot[..., 1, 0] = sa
    rot[..., 1, 1] = ca
    rot[..., 2, 2] = 1

    # Center and scale.
    x = x - np.expand_dims(ctr, axis=1)
    x /= scale_factor

    # Pad, rotate and crop.
    x = np.pad(x, ((0, 0), (0, 0), (0, 1)), "constant", constant_values=1) @ rot
    x = x[..., :2]

    if is_singleton:
        x = x[0]

    if return_angles:
        return x, ang
    else:
        return x


def compute_wing_angles(x, left_ind=3, right_ind=4):
    """Returns the wing angles in degrees from normalized pose.

    Args:
        x: Egocentric pose of shape (..., joints, 2). Use normalize_to_egocentric on the
            raw pose coordinates before passing to this function.
        left_ind: Index of the left wing. Defaults to 3.
        right_ind: Index of the right wing. Defaults to 4.

    Returns:
        Tuple of (thetaL, thetaR) containing the left and right wing angles.

        Both are in the range [-180, 180], where 0 is when the wings are exactly aligned
        to the midline (thorax to head axis).

        Positive angles denote extension away from the midline in the direction of the
        wing. For example, a right wing extension may have thetaR > 0.
    """
    xL, yL = x[..., left_ind, 0], x[..., left_ind, 1]
    xR, yR = x[..., right_ind, 0], x[..., right_ind, 1]
    thetaL = np.rad2deg(np.arctan2(yL, xL)) + 180
    thetaL[np.greater(thetaL, 180, where=np.isfinite(thetaL))] -= 360
    thetaR = np.rad2deg(np.arctan2(yR, xR)) + 180
    thetaR[np.greater(thetaR, 180, where=np.isfinite(thetaR))] -= 360
    thetaR = -thetaR

    return thetaL, thetaR


def compute_wing_arc_angles(XM, XF):
    """Compute angle formed between male wing midpoint and female head.

    Args:
        XM: Male pose tracks of shape (time, joints, 2).
        XF: Female pose tracks of shape (time, joints, 2).

    Returns:
        A tuple of (arcThetaL, arcThetaR) both of shape (time,).

        These represent the angle (in degrees) formed between the vector perpendicular
        to male's wing and the vector formed from the male's wing midpoint and the
        female's head.

        The closer this is to 0, the more aligned the male's wing is to the female's
        head, maximizing the effectiveness of the acoustic signal.
    """
    # Get male features.
    XM_Th = XM[:, 1]  # thorax
    XM_WL = XM[:, 3]  # left wing
    XM_WR = XM[:, 4]  # right wing

    # Get female features.
    XF_H = XF[:, 0]  # head
    XF_Th = XF[:, 1]  # thorax

    # Fill missing values.
    XM_Th = fill_missing(XM_Th, kind="nearest")
    XM_WL = fill_missing(XM_WL, kind="nearest")
    XM_WR = fill_missing(XM_WR, kind="nearest")
    XF_H = fill_missing(XF_H, kind="nearest")
    XF_Th = fill_missing(XF_Th, kind="nearest")

    # Compute wing midpoints.
    XM_WRm = (XM_Th + XM_WR) / 2
    XM_WLm = (XM_Th + XM_WL) / 2

    # Compute offset of wing midpoint to tip.
    XM_WRm_to_WR = XM_WR - XM_WRm
    XM_WLm_to_WL = XM_WL - XM_WLm

    # Compute angle of relative midpoint offset.
    angWR = np.rad2deg(np.arctan2(XM_WRm_to_WR[:, 1], XM_WRm_to_WR[:, 0])) % 360
    angWL = np.rad2deg(np.arctan2(XM_WLm_to_WL[:, 1], XM_WLm_to_WL[:, 0])) % 360

    # Compute arc angle between wing midpoint and female head.
    A = XF_H - XM_WRm
    B = np.stack([np.cos(np.deg2rad(angWR - 90)), np.sin(np.deg2rad(angWR - 90))], axis=-1)
    C = ((A.reshape((-1, 1, 2)) @ B.reshape((-1, 1, 2)).transpose([0, 2, 1])).squeeze() /
         (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1)))
    arcThetaR = np.rad2deg(np.arccos(np.clip(C, -1, 1)))

    A = XF_H - XM_WLm
    B = np.stack([np.cos(np.deg2rad(angWL + 90)), np.sin(np.deg2rad(angWL + 90))], axis=-1)
    C = ((A.reshape((-1, 1, 2)) @ B.reshape((-1, 1, 2)).transpose([0, 2, 1])).squeeze() /
         (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1)))
    arcThetaL = np.rad2deg(np.arccos(np.clip(C, -1, 1)))

    return arcThetaL, arcThetaR


def signed_angle(a, b):
    """Finds the signed angle between two 2D vectors a and b.

    Args:
        a: Array of shape (n, 2).
        b: Array of shape (n, 2).

    Returns:
        The signed angles in degrees in vector of shape (n, 2).

        This angle is positive if a is rotated clockwise to align to b and negative if
        this rotation is counter-clockwise.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    theta = np.arccos(np.around(np.sum(a * b, axis=1), decimals=4))
    cross = np.cross(a, b, axis=1)
    sign = np.zeros(cross.shape)
    sign[cross >= 0] = -1
    sign[cross < 0] = 1
    return np.rad2deg(theta) * sign


def compute_features(fThx, mThx, fHd, mHd, fAbd):
    """Extract behavioral features given head and thorax coordinates.

    Args:
        fThx: Female thorax coordinates in array of shape (timesteps, 2).
        mThx: Male thorax coordinates in array of shape (timesteps, 2).
        fHd: Female head coordinates in array of shape (timesteps, 2).
        mHd: Female head coordinates in array of shape (timesteps, 2).
        fAbd: Female abdomen coordinates in array of shape (timesteps, 2).
    Returns:
        A dictionary of classical features with keys:

        mfDist: Euclidean distance between the male and female thorax.
        mFV: Forward velocity - magnitude of the velocity in the direction of heading (male).
        fFV: Forward velocity - magnitude of the velocity in the direction of heading (female).
        mFA: Forward acceleration (male).
        fFA: Forward acceleration (female).
        mLV: Lateral velocity - signed magnitude of the velocity perpendicular to the forward velocity (male).
        fLV: Lateral velocity - signed magnitude of the velocity perpendicular to the forward velocity (female).
        mLS: Lateral speed - absolute magnitude of perpendicular velocity (male).
        fLS: Lateral speed - absolute magnitude of perpendicular velocity (female).
        mLA: Lateral acceleration (male).
        fLA: Lateral acceleration (female).
        mRS: Rotational speed - change in the heading (male).
        fRS: Rotational speed - change in the heading (female).
        mfAng: Angle subtended by one fly on the other fly (male to female).
        fmAng: Angle subtended by one fly on the other fly (female to male).
        mfFV: Velocity in the direction of the other fly (male towards female).
        fmFV: Velocity in the direction of the other fly (female towards male).
        mfLS: Lateral speed of fly in perpendicular direction of the other fly (male towards female).
        fmLS: Lateral speed of fly in perpendicular direction of the other fly (female towards male).

    Notes:
        Based off of Junyu Li's implementation (/tigress/MMURTHY/junyu/code/alignFeature/compute_features.py).
    """
    # Fill missing values.
    fThx = fill_missing(fThx, kind="nearest")
    mThx = fill_missing(mThx, kind="nearest")
    fHd = fill_missing(fHd, kind="nearest")
    mHd = fill_missing(mHd, kind="nearest")

    # Euclidean distance between the male and female thorax.
    mfDist = np.sqrt(np.sum((fThx - mThx) ** 2, axis=1))
    # Euclidean distance between male head and female abdomen.
    mfDist_mHead_fAbd = np.sqrt(np.sum((fAbd - mHd) ** 2, axis=1))

    # Vector joining the thorax points in consecutive frames.
    mV_vec = np.diff(mThx, axis=0)
    mV_vec = np.pad(mV_vec, ((0, 1), (0, 0)), mode="edge")
    fV_vec = np.diff(fThx, axis=0)
    fV_vec = np.pad(fV_vec, ((0, 1), (0, 0)), mode="edge")

    # Velocity - the Euclidean distance moved by the thorax in each frame.
    # mV = np.sqrt(np.sum(mV_vec ** 2, axis=1))
    # fV = np.sqrt(np.sum(fV_vec ** 2, axis=1))

    # Unit vector joining the head to the thorax or heading.
    mDir = mHd - mThx
    fDir = fHd - fThx
    mDir_unit = mDir / np.linalg.norm(mDir, axis=1, keepdims=True)
    fDir_unit = fDir / np.linalg.norm(fDir, axis=1, keepdims=True)

    # Angle made by the body axis with the x-axis.
    # mTheta = np.rad2deg(np.arctan2(mDir[:, 1], mDir[:, 0]))
    # fTheta = np.rad2deg(np.arctan2(fDir[:, 1], fDir[:, 0]))

    # Forward velocity - magnitude of the velocity in the direction of heading.
    mFV = np.sum(mV_vec * mDir_unit, axis=1)
    mFA = np.diff(mFV, axis=0)
    mFA = np.pad(mFA, (0, 1), mode="edge")
    fFV = np.sum(fV_vec * fDir_unit, axis=1)
    fFA = np.diff(fFV, axis=0)
    fFA = np.pad(fFA, (0, 1), mode="edge")

    # Lateral velocity - magnitude of the velocity perpendicular to the forward velocity.
    mLV = np.sum(mV_vec * np.stack([-mDir_unit[:, 1], mDir_unit[:, 0]], axis=1), axis=1)
    fLV = np.sum(fV_vec * np.stack([-fDir_unit[:, 1], fDir_unit[:, 0]], axis=1), axis=1)

    # Lateral acceration.
    mLA = np.diff(mLV)
    mLA = np.pad(mLA, (0, 1), mode="edge")
    fLA = np.diff(fLV)
    fLA = np.pad(fLA, (0, 1), mode="edge")

    # Rotational speed - change in the heading of the male
    delt = 1
    mRS = signed_angle(mDir[0:(-1 - delt), :], mDir[delt:-1, :])
    mRS = np.pad(mRS, (1, 1), mode="edge")
    fRS = signed_angle(fDir[0:(-1 - delt), :], fDir[delt:-1, :])
    fRS = np.pad(fRS, (1, 1), mode="edge")

    # Vector joining one fly's thorax to the other's
    mfDir = fThx - mHd
    fmDir = mThx - fHd

    fmDir_unit = fmDir / np.linalg.norm(fmDir, axis=1, keepdims=True)
    mfDir_unit = mfDir / np.linalg.norm(mfDir, axis=1, keepdims=True)

    # Angle subtended by one fly on the other fly
    mfAng = signed_angle(mDir, mfDir)
    fmAng = signed_angle(fDir, fmDir)

    # Velocity in the direction of the other fly.
    fmFV = np.sum(fV_vec * fmDir_unit, axis=1)
    mfFV = np.sum(mV_vec * mfDir_unit, axis=1)

    # Male lateral speed in female direction: mfLS
    mfDir_unit_perp = np.stack([-mfDir_unit[:, 1], mfDir_unit[:, 0]], axis=1)
    mfLS = np.abs(np.sum(mV_vec * mfDir_unit_perp, axis=1))

    # Female lateral speed in male direction: fmLS
    fmDir_unit_perp = np.stack([-fmDir_unit[:, 1], fmDir_unit[:, 0]], axis=1)
    fmLS = np.abs(np.sum(fV_vec * fmDir_unit_perp, axis=1))

    ftrs = dict()
    ftrs["mfDist"] = mfDist
    ftrs["mFV"] = mFV
    ftrs["fFV"] = fFV
    ftrs["mFA"] = mFA
    ftrs["fFA"] = fFA
    ftrs["mLV"] = mLV
    ftrs["fLV"] = fLV
    ftrs["mLS"] = abs(mLV)
    ftrs["fLS"] = abs(fLV)
    ftrs["mLA"] = mLA
    ftrs["fLA"] = fLA
    ftrs["mRS"] = mRS
    ftrs["fRS"] = fRS
    ftrs["mfAng"] = mfAng
    ftrs["fmAng"] = fmAng
    ftrs["mfFV"] = mfFV
    ftrs["fmFV"] = fmFV
    ftrs["mfLS"] = mfLS
    ftrs["fmLS"] = fmLS
    ftrs["mfDist_mHead_fAbd"] = mfDist_mHead_fAbd

    return ftrs


def smooth_trx(trx, fps):
    """code from Shruthi R.
    """
    initial_shape = trx.shape
    trx = trx.reshape((initial_shape[0], -1))
    trx_smooth = np.zeros_like(trx)

    dt = 1 / fps
    px_noise = 3
    px_err = px_noise ** 2

    for i in range(trx.shape[-1]):
        trx_smooth[:, i] = smooth_trx_kalman(trx[:, i], dt, px_err)

    trx_smooth = trx_smooth.reshape(initial_shape)

    print('Smoothed and reshaped')

    return trx_smooth


def smooth_trx_kalman(ts, dt=1 / 100, px_err=10):
    """code from Shruthi R.
    """
    motion_model = np.array([[1, dt], [0, 1]])
    observation_model = np.array([[1, 0]])

    # motion_noise_cov = 0.5*(dt**2)*np.eye(2)
    # motion_noise_cov = 1*(dt**2)*np.eye(2)
    motion_noise_cov = 0.5 * np.array([[dt ** 2, 0], [0, 1]])

    observation_noise_cov = np.array([[px_err]])

    F = motion_model
    H = observation_model
    Q = motion_noise_cov
    R = observation_noise_cov

    initx = np.array([ts[0], ts[1] - ts[0]])

    initV = 0.1 * np.eye(2)

    print('Running Kalman smoother')

    ts_smooth, Vsmooth = adskalman.kalman_smoother(ts, F, H, Q, R, initx, initV)

    print('Smoothed')

    return ts_smooth[:, 0]


def connected_components1d(x, return_limits=False):
    """Return the indices of the connected components in a 1D logical array.

    Args:
        x: 1d logical (boolean) array.
        return_limits: If True, return indices of the limits of each component rather
            than every index. Defaults to False.

    Returns:
        If return_limits is False, a list of (variable size) arrays are returned, where
        each array contains the indices of each connected component.

        If return_limits is True, a single array of size (n, 2) is returned where the
        columns contain the indices of the starts and ends of each component.
    """
    L, n = scipy.ndimage.label(x.squeeze())
    ccs = scipy.ndimage.find_objects(L)
    starts = [cc[0].start for cc in ccs]
    ends = [cc[0].stop for cc in ccs]
    if return_limits:
        return np.stack([starts, ends], axis=1)
    else:
        return [np.arange(i0, i1, dtype=int) for i0, i1 in zip(starts, ends)]


def encode_hdf5_strings(S):
    """Encodes a list of strings for writing to a HDF5 file.

    Args:
        S: List of strings.

    Returns:
        List of numpy arrays that can be written to HDF5.
    """
    return [np.string_(x) for x in S]


def make_expt_dataset(expt_folder, output_path=None, overwrite=False, ctr_ind=1, fwd_ind=0, smoothTrx=False):
    """Gather experiment data into a single file.

    Args:
        expt_folder: Full absolute path to the experiment folder.
        output_path: Path to save the resulting dataset to. Can be specified as a folder
            or full path ending with ".h5". Defaults to saving to current folder. If a
            folder is specified, the dataset filename will be the experiment folder
            name with ".h5".
        overwrite: If True, overwrite even if the output path already exists. Defaults
            to False.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.

    Returns:
        Path to output dataset.
    """

    expt_name = os.path.basename(expt_folder)
    print(f"Starting with: {expt_name}\n")
    print(f"Raw Data Path: {expt_folder}\n")

    if output_path is None:
        output_path = os.getcwd()

    if not output_path.endswith(".h5"):
        output_path = os.path.join(output_path, f"{expt_name}.h5")

    if os.path.exists(output_path) and not overwrite:
        print(f"output path already exists and overwrite is set to False\n\n")
        return output_path

    print(f"will save features to {output_path}")
    # Load synchronization.
    sample_at_frame, frame_at_sample = get_expt_sync(expt_folder)

    # check if sleap tracks .h5 file exists
    if not any(fname.endswith('tracking.h5') for fname in os.listdir(expt_folder)):
        print(f"no tracking file exists in experiment folder")
        print("skipping for now\n\n")
        return expt_name

    # Load tracking.
    tracks, node_names = load_tracks(expt_folder)

    femaleIDX = find_female_track(expt_folder)
    maleIDX = 1 - femaleIDX

    # Compute tracking-related features.
    trxF = tracks[..., femaleIDX]
    trxM = tracks[..., maleIDX]
    if smoothTrx:
        # trxF = fill_missing(trxF)
        # trxM = fill_missing(trxM)
        trxF[:, 0:4, :] = smooth_trx(trxF[:, 0:4, :], fps=150)
        trxM[:, 0:4, :] = smooth_trx(trxM[:, 0:4, :], fps=150)
    egoF = normalize_to_egocentric(trxF)
    egoM = normalize_to_egocentric(trxM)
    egoFrM = normalize_to_egocentric(trxF, rel_to=trxM, ctr_ind=ctr_ind, fwd_ind=fwd_ind)
    egoMrF = normalize_to_egocentric(trxM, rel_to=trxF, ctr_ind=ctr_ind, fwd_ind=fwd_ind)
    wingFL, wingFR = compute_wing_angles(egoF)
    wingML, wingMR = compute_wing_angles(egoM)
    arcThetaL, arcThetaR = compute_wing_arc_angles(trxM, trxF)

    abd_ind = 3
    # Compute standard classical features.
    feats = compute_features(trxF[:, ctr_ind, :], trxM[:, ctr_ind, :], trxF[:, fwd_ind, :], trxM[:, fwd_ind, :], trxF[:,abd_ind, :])

    oe_startsPath = os.path.join(expt_folder, 'OE_all_start_frames_aka_indices_for_this_file_variable.npy')
    if os.path.exists(oe_startsPath):
        oeStarts = np.load(oe_startsPath)
    oe_endPath = os.path.join(expt_folder, 'OE_all_end_frames_aka_indices_for_this_file_variable.npy')
    if os.path.exists(oe_endPath):
        oeEnds = np.load(oe_endPath)
    print("features created\n")

    # Ensure output folder exists.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("saving to output file\n")
    # Save.
    with h5py.File(output_path, "w") as f:
        f.create_dataset("expt_name", data=expt_name)
        f.create_dataset("expt_folder", data=expt_folder)
        f.create_dataset("node_names", data=encode_hdf5_strings(node_names))

        f.create_dataset("sample_at_frame", data=sample_at_frame, compression=1)
        f.create_dataset("frame_at_sample", data=frame_at_sample, compression=1)

        f.create_dataset("trxF", data=trxF, compression=1)
        f.create_dataset("trxM", data=trxM, compression=1)
        f.create_dataset("egoF", data=egoF, compression=1)
        f.create_dataset("egoM", data=egoM, compression=1)
        f.create_dataset("egoFrM", data=egoFrM, compression=1)
        f.create_dataset("egoMrF", data=egoMrF, compression=1)
        f.create_dataset("wingFL", data=wingFL, compression=1)
        f.create_dataset("wingFR", data=wingFR, compression=1)
        f.create_dataset("wingML", data=wingML, compression=1)
        f.create_dataset("wingMR", data=wingMR, compression=1)
        f.create_dataset("arcThetaL", data=arcThetaL, compression=1)
        f.create_dataset("arcThetaR", data=arcThetaR, compression=1)

        for k, v in feats.items():
            f.create_dataset(k, data=v, compression=1)

        if os.path.exists(oe_startsPath):
            f.create_dataset("oeStarts", data=oeStarts, compression=1)
        if os.path.exists(oe_endPath):
            f.create_dataset("oeEnds", data=oeEnds, compression=1)
    print("done\n\n")

    return output_path

#
# def main(expt_folder):
#     outputDir = '/run/user/1000/gvfs/smb-share:server=cup.pni.princeton.edu,share=murthy/Kyle/data/edna/blind_deaf'
#     fly = os.path.basename(expt_folder)
#     if fly.endswith('.txt'):
#         return
#     output_pathDir = os.path.join(outputDir, fly)
#     output_path = os.path.join(output_pathDir, f"{fly}_smoothed.h5")
#
#     make_expt_dataset(expt_folder, output_path=output_path, overwrite=False, smoothTrx=True)
#
#
# if __name__ == "__main__":
#     exptPath = glob.glob(
#         '/run/user/1000/gvfs/smb-share:server=tigress-cifs.princeton.edu,share=fileset-mmurthy/eDNA/behavior/manipulated_summer22_AK_blindanddeaf/**')
#
#     with multiprocessing.Pool(3) as pool:
#         pool.map(main, [expt_folder for expt_folder in exptPath])

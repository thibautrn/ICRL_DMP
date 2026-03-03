import logging
import threading
import torch
import queue

import numpy as np

from abc import abstractmethod
from datetime import datetime
from wear_mocap_ape.data_types.bone_map import BoneMap
from wear_mocap_ape.estimate import estimate_joints, compose_msg
from wear_mocap_ape.utility import data_stats
from wear_mocap_ape.utility.names import NNS_INPUTS, NNS_TARGETS


class Estimator:
    def __init__(self,
                 x_inputs: NNS_INPUTS,
                 y_targets: NNS_TARGETS,
                 normalize: bool = True,
                 smooth: int = 1,
                 seq_len: int = 1,
                 add_mc_samples: bool = True,
                 bonemap: BoneMap = None,
                 tag: str = "Estimator"):

        self.__tag = tag
        self._active = False

        self._y_targets = y_targets
        self._x_inputs = x_inputs

        # load normalized data stats if required
        self._normalize = normalize
        if normalize:
            stats = data_stats.get_norm_stats(
                x_inputs=self._x_inputs,
                y_targets=self._y_targets
            )
            # data is normalized and has to be transformed with pre-calculated mean and std
            self._xx_m, self._xx_s = stats["xx_m"], stats["xx_s"]
            self._yy_m, self._yy_s = stats["yy_m"], stats["yy_s"]

        # average over multiple time steps
        self._smooth = max(1, smooth)  # smooth should not be smaller 1
        self._smooth_hist = []

        # monte carlo predictions
        self._last_msg = None
        # if this flag is true, the output messages contain all mc predictions
        self._add_mc_samples = add_mc_samples

        self._row_hist = []
        self._sequence_len = max(1, seq_len)  # seq_len should not be smaller 1

        # use body measurements for transitions
        if bonemap is None:
            self._larm_vec = np.array([-BoneMap.DEFAULT_LARM_LEN, 0, 0])
            self._uarm_vec = np.array([-BoneMap.DEFAULT_UARM_LEN, 0, 0])
            self._uarm_orig = BoneMap.DEFAULT_UARM_ORIG_RH
        else:
            # get values from bone map
            self._larm_vec = np.array([-bonemap.right_lower_arm_length, 0, 0])
            self._uarm_vec = np.array([-bonemap.right_upper_arm_length, 0, 0])
            self._uarm_orig = bonemap.right_upper_arm_origin_rh

        # for quicker access we store a matrix with relevant body measurements for quick multiplication
        self._body_measurements = np.r_[self._larm_vec, self._uarm_vec, self._uarm_orig][np.newaxis, :]

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_norm_stats(self, stats: dict):
        """overwrites the default norm stats loaded during the initialization"""
        # data is normalized and has to be transformed with pre-calculated mean and std
        self._xx_m, self._xx_s = stats["xx_m"], stats["xx_s"]
        self._yy_m, self._yy_s = stats["yy_m"], stats["yy_s"]
        logging.info("Replaced norm stats xx m+/-s and yy m+/-s")

    def get_last_msg(self):
        return self._last_msg

    def is_active(self):
        return self._active

    def terminate(self):
        self._active = False

    def reset(self):
        self._active = False
        self._row_hist = []
        self._smooth_hist = []

    def add_xx_to_row_hist_and_make_prediction(self, xx) -> np.array:
        self._row_hist.append(xx)
        # if not enough data is available yet, simply repeat the input as a first estimate
        while len(self._row_hist) < self._sequence_len:
            self._row_hist.append(xx)
        # if the history is too long, delete old values
        while len(self._row_hist) > self._sequence_len:
            del self._row_hist[0]
        xx_hist = np.vstack(self._row_hist)

        if self._normalize:
            xx_hist = (xx_hist - self._xx_m) / self._xx_s

        pred = self.make_prediction_from_row_hist(xx_hist)

        if self._normalize:
            pred = pred * self._yy_s + self._yy_m

        # store est in history if smoothing is required
        if self._smooth > 1:
            self._smooth_hist.append(pred)
            while len(self._smooth_hist) < self._smooth:
                self._smooth_hist.append(pred)
            while len(self._smooth_hist) > self._smooth:
                del self._smooth_hist[0]
            pred = np.vstack(self._smooth_hist)

        return pred
    



    def vec_unityLH_to_rosRH(self,v):
        """
        LH global basis is (X=Right, Y=Up, Z=Forward).
        RH ENU basis is (X=East,  Y=North, Z=Up).
        Mapping: [x_e, y_n, z_u] = [x_r, z_f, y_u].
        Works on shape (3,) or (N,3).
        """
        v = np.asarray(v)
        if v.ndim == 1:
            x,y,z = v
            return np.array([z, -x, y], dtype=float)
        else:
            x,y,z = v[:,0], v[:,1], v[:,2]
            return np.column_stack([z, -x, y]).astype(float)

    def quat_unityLH_to_rosRH(self,q_wxyz):
        """
        Inverse of android_quat_to_global_no_north.
        Input q in LH global [w,x,y,z] (X Right, Y Up, Z Fwd).
        Output RH ENU [w,x,y,z] (X East, Y North, Z Up).
        Mapping mirrors your forward function's reorder/sign:
        android→global_no_north:  qg = [-wa, xa, za, ya]
        global→android_no_north: qa = [-wg, xg, yg_swap, z_swap]
        So here: qa = [-w_g, x_g, z_g, y_g]
        """
        q = np.asarray(q_wxyz, dtype=float)
        w,x,y,z = q
        # quat -> rot (wxyz)
        n = w*w + x*x + y*y + z*z
        s = 2.0/n if n > 1e-12 else 0.0
        wx, wy, wz = w*x*s, w*y*s, w*z*s
        xx, xy, xz = x*x*s, x*y*s, x*z*s
        yy, yz = y*y*s, y*z*s
        zz = z*z*s
        R = np.array([
            [1-(yy+zz), xy-wz,     xz+wy],
            [xy+wz,     1-(xx+zz), yz-wx],
            [xz-wy,     yz+wx,     1-(xx+yy)]
        ])
        # Unity->ROS basis change
        P = np.array([[0,0,1],
                    [-1,0,0],
                    [0,1,0]], dtype=float)
        R2 = P @ R @ P.T
        # rot -> quat (stable eigen method), returns wxyz
        K = np.array([
            [R2[0,0]-R2[1,1]-R2[2,2], 0, 0, 0],
            [R2[1,0]+R2[0,1], R2[1,1]-R2[0,0]-R2[2,2], 0, 0],
            [R2[2,0]+R2[0,2], R2[2,1]+R2[1,2], R2[2,2]-R2[0,0]-R2[1,1], 0],
            [R2[1,2]-R2[2,1], R2[2,0]-R2[0,2], R2[0,1]-R2[1,0], R2[0,0]+R2[1,1]+R2[2,2]]
        ]) / 3.0
        wvals, V = np.linalg.eigh(K)
        q2 = V[:, np.argmax(wvals)]
        if q2[0] < 0: q2 = -q2
        return np.array([q2[3], q2[0], q2[1], q2[2]], dtype=float)


    def convert_msg_lh_to_rh(self,msg):
        m = list(msg)

        POS_IDXS = {
            "hand": (0,1,2),
            "larm": (3,4,5),
            "uarm": (6,7,8),
            # add hips if it has a position, etc.
        }
        QUAT_IDXS = {
            "hand": ( 9,10,11,12),  # w,x,y,z indices — adjust to your layout!
            "larm": (13,14,15,16),
            "uarm": (17,18,19,20),
            "hips": (21,22,23,24),
        }

        # positions
        for name,(i,j,k) in POS_IDXS.items():
            if max(i,j,k) < len(m):
                x,y,z = m[i], m[j], m[k]
                xr,yr,zr = self.vec_unityLH_to_rosRH([x,y,z])
                m[i],m[j],m[k] = float(xr),float(yr),float(zr)

        # quaternions
        for name,(i,j,k,l) in QUAT_IDXS.items():
            if max(i,j,k,l) < len(m):
                q = [m[i],m[j],m[k],m[l]]  # wxyz
                qr = self.quat_unityLH_to_rosRH(q)
                m[i],m[j],m[k],m[l] = [float(v) for v in qr]

        return m

    def msg_from_pred(self, pred: np.array, add_mc_samples: bool) -> np.array:
        est = estimate_joints.arm_pose_from_nn_targets(
            preds=pred,
            body_measurements=self._body_measurements,
            y_targets=self._y_targets
        )
        msg = compose_msg.msg_from_nn_targets_est(est, self._body_measurements, self._y_targets)

        self._last_msg = msg.copy()
        if add_mc_samples:
            msg = list(msg)
            # now we attach the monte carlo predictions for XYZ positions
            if est.shape[0] > 1:
                for e_row in est:
                    msg += list(e_row[:6])
        return msg

    def process_in_thread(self, sensor_q: queue):
        msg_q = queue.Queue()
        t = threading.Thread(target=self.processing_loop, args=(sensor_q, msg_q))
        t.start()
        return msg_q

    def processing_loop(self, sensor_q: queue, msg_q: queue):
        logging.info(f"[{self.__tag}] wearable streaming loop")

        # used to estimate delta time and processing speed in Hz
        start = datetime.now()
        dat = 0

        # this loops while the socket is listening and/or receiving data
        self.reset()
        self._active = True
        while self._active:

            try:
                # get the most recent smartwatch data row from the queue
                row = sensor_q.get(timeout=2)
                while sensor_q.qsize() > 5:
                    row = sensor_q.get(timeout=2)
            except queue.Empty:
                logging.info(f"[{self.__tag}] no data")
                continue

            # processing speed output
            now = datetime.now()
            if (now - start).seconds >= 5:
                start = now
                logging.info(f"[{self.__tag}] {dat / 5} Hz")
                dat = 0

            # finally get predicted positions etc
            xx = self.parse_row_to_xx(row)
            pred = self.add_xx_to_row_hist_and_make_prediction(xx)
            msg = self.msg_from_pred(pred, self._add_mc_samples)
            # msg = self.convert_msg_lh_to_rh(msg)
            msg_q.put(msg)
            dat += 1

    @abstractmethod
    def make_prediction_from_row_hist(self, xx_hist: np.array) -> np.array:
        return

    @abstractmethod
    def parse_row_to_xx(self, row) -> np.array:
        return

    @property
    def sequence_len(self):
        return self._sequence_len

    @property
    def body_measurements(self):
        return self._body_measurements

    @property
    def uarm_orig(self):
        return self._uarm_orig

    @property
    def uarm_vec(self):
        return self._uarm_vec

    @property
    def larm_vec(self):
        return self._larm_vec

    @property
    def device(self):
        return self._device

    @property
    def x_inputs(self):
        return self._x_inputs

    @property
    def y_targets(self):
        return self._y_targets

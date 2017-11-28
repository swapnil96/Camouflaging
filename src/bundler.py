import numpy as np, os, cv2
from functions import *

def parse_bundler(fname,img_shape):
    with open(fname, "r") as f:
        f.readline()                                        # Read Bundler Version Line
        ncams = int(f.readline().split()[0])
        focals, Rt, pt_tracks = [], [], []
        for idx in range(ncams):
            focals.append(float(f.readline().split()[0]))
            R = np.array([list(map(float, f.readline().split())) for x in range(3)])
            t = np.array(list(map(float, f.readline().split())))
            Rt.append((R, t))
        while True:
            line = f.readline()
            if line is None or len(line.rstrip()) == 0:
                break
            X = np.array(list(map(float, line.split())))
            f.readline()                                    # Ignore color values
            projs = f.readline().split()
            track = []
            for idx in range(int(projs[0])):
                frame = int(projs[1 + 4*idx])                  # Ignore SIFT keypoint number
                x = img_shape[1]/2. + float(projs[3 + 4*idx])
                y = img_shape[0]/2. + float(projs[4 + 4*idx])
                track.append((frame, np.array([x,y], np.float64)))
            pt_tracks.append((X, track))
    return focals, Rt, pt_tracks

class Bundler:
    def __init__(self, bundler_file, img_files, img_shape, max_dim = 1000, frame_subset = None, cams_file = None):
        self.bundler_file = bundler_file
        self.img_shape = img_shape
        if max_dim is None:
            self.scale = 1.
        else:
            self.scale = min(float(max_dim+0.4) / np.array(self.img_shape[:2], np.float64))
        if not os.path.exists(self.bundler_file):
            raise RuntimeError("Bundler Path does not exist: {}".format(self.bundler_file))
        focals, Rt, tracks = parse_bundler(self.bundler_file, self.img_shape)
        if len(focals) != len(img_files):
            raise RuntimeError("Bundler camera count ({0}) not agreeing with specified camera count ({1})".format(len(focals),len(img_files)))
        frames_ok = np.nonzero(focals)[0].tolist()
        if frame_subset is not None:
            frames_ok = list(set(frames_ok).intersection(frame_subset))
        if cams_file is not None:
            if not os.path.exists(cams_file):
                raise RuntimeWarning("Not using good cams - File not found: {}".format(cams_file))
            with open(cams_file,"r") as camf:
                good_cams = list(map(int,camf.readlines()))
                frames_ok = list(set(frames_ok).intersection(good_cams))
        frames_ok = sorted(frames_ok)
        self.img_files = [img_files[idx] for idx in frames_ok]
        self.imgs = dict([(f,cv2.resize(load_img(f,1), (0,0), fx=self.scale, fy=self.scale)) for f in self.img_files])
        if len(img_files) == 0:
            raise RuntimeError('SfM failed: No good cameras')
        self.Ks = np.zeros((len(img_files), 3, 3))
        self.Rs = np.zeros((len(img_files), 3, 3))
        self.ts = np.zeros((len(img_files), 3))
        for idx, frame in enumerate(frames_ok):
            K = -np.array([[-self.scale*focals[frame], 0., -0.5 + self.scale*self.img_shape[1]/2.,],[0., self.scale*focals[frame], -0.5 + self.scale*self.img_shape[0]/2.,],[0., 0., 1.]])
            self.Ks[idx] = K
            self.Rs[idx], self.ts[idx] = Rt[frame]

    def get_img(self, frame): return self.imgs[self.img_files[frame]].copy()
    
    def P(self, frame): return np.dot(self.Ks[frame], np.hstack([self.Rs[frame], self.ts[frame, :, np.newaxis]]))

    def center(self, frame): return -np.dot(self.Rs[frame].T, self.ts[frame])

    def project(self, frame, X):
        X = np.asarray(X.T)
        homog_X = np.concatenate([X, [1.]]) if X.ndim == 1 else np.vstack([X, np.ones(X.shape[1])])
        x = np.dot(self.P(frame), homog_X)
        y = x[:-1] / x[-1]
        return y.T

if __name__ == "__main__":
    data_path = "./Test Data/walden-tree3/"
    Bundler(data_path + "bundle/bundle.out", get_jpg_files(data_path), (3888,2592,3))
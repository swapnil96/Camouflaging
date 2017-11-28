import numpy as np, scipy.io, cPickle as pickle, sys
from functions import *

class Box:
    def __init__(self, face_idx, box_pts):
        self.texsize = texsize = 256
        self.face_idx = np.asarray(face_idx)
        self.nfaces = len(face_idx)
        self.box_pts = np.asarray(box_pts)
        self.face_planes, self.face_center, self.face_edges, self.face_pts, self.face_juv = [], [], [], [], []
        u_grid, v_grid = [np.array(x, 'd') for x in np.mgrid[:texsize, :texsize]]
        uf = u_grid.flatten()
        vf = v_grid.flatten()
        for idx in range(len(face_idx)):
            p1 = box_pts[face_idx[idx][0]]
            p2 = box_pts[face_idx[idx][1]]
            p3 = box_pts[face_idx[idx][3]]              # Index ?
            e1 = -p1 + p3
            e2 = -p1 + p2
            n = normalized(np.cross(e1, e2))
            d = -np.dot(p1, n)
            self.face_planes.append(np.concatenate([n, [d]]))
            pts = p1 + (uf/(texsize - 1.))[:, None]*e1[None, :] + (vf/(texsize - 1.))[:, None]*e2[None, :]
            juv = np.zeros((len(uf), 3), np.int64)
            juv[:, 0] = idx
            juv[:, 1] = uf
            juv[:, 2] = vf
            self.face_juv.append(juv)
            self.face_pts.append(pts)
            self.face_edges.append((p1, np.array([e1, e2])))
            self.face_center.append(p1 + 0.5*e1 + 0.5*e2)
        self.face_center = np.array(self.face_center)
        self.tex2juv = np.vstack(self.face_juv)
        self.juv2tex = np.zeros((len(self.face_idx), texsize, texsize), 'l')
        self.juv2tex[self.tex2juv[:, 0], self.tex2juv[:, 1], self.tex2juv[:, 2]] = range(len(self.tex2juv))
        self.texel_pts = np.vstack(self.face_pts)
        self.face_planes = np.array(self.face_planes)
        self.on_border = (self.tex2juv[:, 2] == 0) + (self.tex2juv[:, 1] == self.texsize-1) + (self.tex2juv[:, 2] == self.texsize-1) + (self.tex2juv[:, 1] == 0)
        self.ntexels = len(self.tex2juv)

    def backproject_rays(self, center, rays):
        rays = rays / np.sqrt(np.sum(rays**2, axis = 1)).reshape(-1,1)
        def backproj(face_visible):
            # (alpha*r + t)*n + d = 0
            # alpha*r'n + t'*n + d = 0
            # (-d - t'*n)/(r'*n)
            ray_juv = -1 + np.zeros(rays.shape, np.float64)
            best_dist = np.inf + np.zeros(rays.shape[0])
            if len(rays) == 0:
                return ray_juv, best_dist
            for idx in np.nonzero(face_visible)[0]:
                plane = self.face_planes[idx]
                dist = (-plane[3] - np.dot(center, plane[:3]))/np.dot(rays, plane[:3])
                pts = dist[:, None]*rays + center
                p1, edges = self.face_edges[idx]
                uv = np.linalg.lstsq(edges.T, (pts - p1).T)[0].T
                in_bounds = (0 <= uv[:, 0]) * (uv[:, 0] <= 1) * (0 <= uv[:, 1]) * (uv[:, 1] <= 1)
                uv = np.array((self.texsize-1)*uv)
                visible = (in_bounds) * (0 <= dist) * (dist < best_dist)
                best_dist[visible] = dist[visible]
                ray_juv[visible, 0] = idx
                ray_juv[visible, 1] = uv[visible, 0]
                ray_juv[visible, 2] = uv[visible, 1]
            return ray_juv, best_dist
        ray_juv, best_dist = backproj(np.ones(self.nfaces))
        visible = np.zeros(self.nfaces)
        rerun = False
        for idx in range(self.nfaces):
            ok = ray_juv[:, 0] == idx
            if np.sum(ok) == 0:
                visible[idx] = False
            else:
                visible[idx] = not np.all((ray_juv[ok, 1] == 0) * (ray_juv[ok, 1] == self.texsize-1) * (ray_juv[ok, 2] == 0) * (ray_juv[ok, 2] == self.texsize-1))
                if not visible[idx]:
                    rerun = True
                    print('rerunning!')
        if rerun:
            ray_juv, best_dist = backproj(visible)
        return ray_juv, best_dist

    def backproject_im(self, bndl, frame):
        shape = bndl.get_img(frame).shape
        ray_dirs = mult_im(bndl.Rs[frame].T, ray_directions(bndl.Ks[frame], shape))
        if 1:
            rays = ray_dirs.reshape((-1,ray_dirs.shape[2]))
        if 0:
            x, y = tuple(map(int, bndl.project(frame, self.mesh_pts[0])))
            rays = ray_dirs[y, x][None, :]
            print(bndl.project(frame, bndl.center(frame) + rays[0]), (x, y))
        ray_juv, dist = self.backproject_rays(bndl.center(frame), rays)
        ray_juv = ray_juv.reshape(shape)
        dist = dist.reshape(shape[:2])
        return ray_juv, dist

    def texel_visible(self, bndl, frame):
        ray_juv, dist = self.backproject_im(bndl, frame)
        face_visible = np.zeros(self.nfaces, 'bool')
        face_visible[np.array(ray_juv[ray_juv[:, :, 0] >= 0, 0], np.int64)] = 1
        proj = np.array(bndl.project(frame, self.texel_pts), np.int64)
        ok = (0 <= proj[:, 0]) * (proj[:, 0] < ray_juv.shape[1]) * (0 <= proj[:, 1]) * (proj[:, 1] < ray_juv.shape[0])
        visible = np.zeros(len(self.texel_pts), 'bool')
        visible[ok] = face_visible[self.tex2juv[ok, 0]]
        return visible

    def index_as_juv(self, xs):
        if np.ndim(xs) == 1:
            a = np.zeros(self.juv2tex.shape[:3], dtype = xs.dtype)
        else:
            a = np.zeros(self.juv2tex.shape[:3] + (xs.shape[1],), dtype = xs.dtype)
        a[self.tex2juv[:, 0], self.tex2juv[:, 1], self.tex2juv[:, 2]] = xs
        return a

    def index_as_flat(self, xs):
        return xs[self.tex2juv[:, 0], self.tex2juv[:, 1], self.tex2juv[:, 2]]

    # render with antialiasing
    def render(self, bndl, frame, texel_colors, antialias = True, alias_steps = 5, img = None, mask = None):
        if img is None:
            img = bndl.get_img(frame)
        if np.ndim(frame) == 0:
            R = bndl.Rs[frame]
            P = bndl.P(frame)
            K = bndl.Ks[frame]
            center = bndl.center(frame)
        else:
            R, P, K, center = frame
        # multiple rays per pixel
        if 1:
            X = np.asarray(self.box_pts.T)
            homog_X = np.concatenate([X, [1.]]) if X.ndim == 1 else np.vstack([X, np.ones(X.shape[1])])
            x = np.dot(P, homog_X)
            ov = np.array(ut.bbox2d((x[:-1] / x[-1]).T), np.float64)
            ov[:2] = np.floor(ov[:2])
            ov[2:] = np.ceil(ov[2:])
            ov = map(int, ov)
            ov = rect_intersect((0, 0, img.shape[1], img.shape[0]), ov)
            sigma = 0.25
            xx, yy, cyx, sample_weights = [], [], [], []
            for cx in range(ov[0], ov[0] + ov[2]):
                for cy in range(ov[1], ov[1] + ov[3]):
                    d = alias_steps/2
                    for dxi in range(-d, d+1):
                        for dyi in range(-d, d+1):
                            dxf = 0.5*float(dxi)/max(0.00001, d)
                            dyf = 0.5*float(dyi)/max(0.00001, d)
                            xx.append(cx + dxf)
                            yy.append(cy + dyf)
                            sample_weights.append(np.exp((-dxf**2 - dyf**2)/(2.*sigma**2)))
                            cyx.append([cy, cx])
            xx = np.array(xx, 'd')
            yy = np.array(yy, 'd')
            sample_weights = np.array(sample_weights, 'd')
            flat_idx = np.ravel_multi_index(ut.ensure_col(cyx, 2, 'l').T, img.shape[:2])
        rays = np.dot(np.dot(R.T, np.linalg.inv(K)),np.array([xx, yy, np.ones_like(xx)])).T
        if len(rays) == 0:
            return img
        ray_juv, _ = self.backproject_rays(center, rays)
        colors_juv = self.index_as_juv(texel_colors)
        ray_colors = np.zeros_like(ray_juv)
        for j in [-1] + range(self.nfaces):
            ok = (ray_juv[:, 0] == j)
            if j == -1:
                ray_colors[ok] = lookup_bilinear(img, xx[ok], yy[ok])
            else:
                ray_colors[ok] = lookup_bilinear(colors_juv[j], ray_juv[ok, 2], ray_juv[ok, 1])
        # average together each pixel's samples
        color_sum = np.zeros(img.shape)
        for c in range(img.shape[2]):
            color_sum[:, :, c] = np.bincount(flat_idx, weights = ray_colors[:, c]*sample_weights, minlength = img.shape[0]*img.shape[1]).reshape(img.shape[:2])
        counts = np.bincount(flat_idx, weights = sample_weights, minlength = img.shape[0]*img.shape[1]).reshape(img.shape[:2])
        res_im = img.copy()
        filled = (counts != 0)
        res_im[filled] = color_sum[filled]/counts[filled][:,None]
        res_im[~filled] = img[~filled]
        # anti-aliasing will slightly blur things, even for pixels that don't actually see the box
        # fix this by setting all off-cube pixels equal to their original color
        hit_count = np.bincount(flat_idx, weights = (ray_juv[:, 0] >= 0), minlength = img.shape[0]*img.shape[1]).reshape(img.shape[:2])
        res_im[hit_count == 0] = img[hit_count == 0]
        #ray_count = np.bincount(flat_idx, minlength = im.shape[0]*im.shape[1]).reshape(im.shape[:2])
        if mask is not None:
            res_im = res_im*(1.-mask[:,:,np.newaxis]) + img*mask[:,:,np.newaxis]
        return res_im

def box_mask(bndl, mesh, frame):
    ray_juv, dist = mesh.backproject_im(bndl, frame)
    return ray_juv[:, :, 0] >= 0

def faces_visible(bndl, mesh, frame):
    ray_juv, dist = mesh.backproject_im(bndl, frame)
    js = np.array(ray_juv[:, :, 0].flatten(), 'l')
    return np.bincount(js[js >= 0], minlength = 6) > 0

def load_from_mat(fname, whois = 0):
    
    if whois == 0:
        m = scipy.io.loadmat(fname)
        if 'faces' in m:
            if np.ndim(m['faces']) == 2 and m['faces'].shape[1] == 4:
                face_idx = -1 + np.array(m['faces'], np.int64)
            else:
                face_idx = -1 + np.array([m['faces'][0][i][0].flatten() for i in range(len(m['faces'][0]))], np.int64)
        else:
            face_idx = -1 + np.array([[1, 2, 4, 3], np.array([1, 2, 4, 3])+4, [1, 2, 2+4, 1+4], [2, 4, 4+4, 2+4], [4, 3, 3+4, 4+4], [3, 1, 1+4, 3+4]])
        if np.ndim(m['world_pos']) == 2 and m['world_pos'].shape[1] == 3:
            box_pts = np.array(m['world_pos'], np.float64)
        else:
            box_pts = np.array([m['world_pos'].squeeze()[i].flatten() for i in range(len(m['world_pos'].squeeze()))], np.float64)
        print face_idx.shape, box_pts.shape
        print face_idx, box_pts
        return Box(face_idx, box_pts)
    
    else:
        with open(fname, "rb") as myFile:
            m = pickle.load(myFile)

        face_idx = m['faces']
        box_pts = m['world_pos']
        print face_idx.shape, box_pts.shape
        print face_idx, box_pts
        return Box(face_idx, face_pts)

if __name__ == "__main__":
    b = load_from_mat("./Test Data/walden-tree3/cube.mat")

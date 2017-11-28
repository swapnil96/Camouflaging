import numpy as np, cv2, glob, util as ut, imtable, scipy.ndimage

def load_img(name, mode):
    return cv2.imread(name, mode)

def show_img(title, img, wait):
    cv2.imshow(title, img)
    k = cv2.waitKey(wait)
    cv2.destroyWindow(title)
    return k

def show_multiple_img(title, img, wait):
    if type(title) != list or len(title) == 0:
        title = [str(x) for x in range(len(img))]
    for x, y in zip(title, img):
        cv2.imshow(x, y)
    k = cv2.waitKey(wait)
    for x in title:
        cv2.destroyWindow(x)
    return k

def save_img(name, img):
    if type(name) == str:
        cv2.imwrite("out/" + name + ".jpg", img)
    else:
        [cv2.imwrite("out/" + n + ".jpg", i) for n, i in zip(name, img)]

def get_jpg_files(dir):
    return sorted(glob.glob(dir + '*.jpg'))

def normalized(v):
    n = np.sqrt(np.sum(v**2))
    return v.copy() if n == 0 else v/n

def normalize_im(a):
    norms = np.sqrt(np.sum(a ** 2, axis = 2))
    norms[norms == 0] = 1.                          # avoid divide-by-0 warning
    return a / np.tile(norms[:, :, np.newaxis], (1,1,3))

def mult_im(A, im):
    new_im = np.dot(A, np.vstack([im[:, :, 0].flatten(), im[:, :, 1].flatten(), im[:, :, 2].flatten()])).astype(np.float64)
    def r(x): return x.reshape(im.shape[:2])[:, :, np.newaxis]
    return np.concatenate(list(map(r, [new_im[0, :], new_im[1, :], new_im[2, :]])), axis = 2)

def ray_directions(K, im_shape, R = np.eye(3), normalize = True):
    h, w = im_shape[:2]
    y, x = np.mgrid[:h, :w]
    y = np.single(y)
    x = np.single(x)
    rays = np.dot(np.dot(R.T, np.linalg.inv(K)), np.array([x.flatten(), y.flatten(), np.ones(x.size)]))
    rays = rays.reshape((3, h, w)).transpose([1, 2, 0])
    assert np.allclose(rays[20, 30], np.dot(np.dot(R.T, np.linalg.inv(K)), np.array([30, 20, 1.])))
    if normalize:
        rays = normalize_im(rays)
    return rays

def knnsearch(N, X, k = 1, method = 'brute'):
    if method == 'kd':
        from scipy.spatial import cKDTree
        kd = cKDTree(N)
        return kd_query(kd, X, k = k)
    elif method == 'brute':
        import scipy.spatial.distance
        D = scipy.spatial.distance.cdist(X, N)
        if k == 1:
            I = np.argmin(D, 1)[:, np.newaxis]
        else:
            I = np.argsort(D)[:, :k]
        return D[np.arange(D.shape[0])[:, np.newaxis], I], I 
    else:
        fail('Unknown search method: %s' % method)

def lookup_bilinear(im, x, y, order = 1, mode = 'constant', cval = 0.0):
    yx = np.array([y, x])
    if np.ndim(im) == 2:
        return scipy.ndimage.map_coordinates(im, yx, order = order, mode = mode, cval = cval)
    else:
        return np.concatenate([scipy.ndimage.map_coordinates(im[:, :, i], yx, order = order, mode = mode)[:, np.newaxis] for i in xrange(im.shape[2])], axis = 1)

def bbox2d(pts):
    pts = np.asarray(pts)
    if len(pts) == 0:
        raise RuntimeError("no pts")
    return rect_from_pts(np.min(pts[:,0]),np.min(pts[:,1]),np.max(pts[:,0]),np.max(pts[:,1]))

def rect_intersect(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])
    return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

def render_scene(scene, texel_colors, bndl, mesh):
    imtable.show([('cycle', [mesh.render(bndl, frame, texel_colors), bndl.get_img(frame)]) for frame in range(len(bndl.img_files))])
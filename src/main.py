import numpy as np, networkx as nx
import scipy.ndimage
from bundler import *
from box import *
from functions import *

try:
    import mrf      # mrf is a Cython module that depends on the gco graph cuts library
except ImportError:
    print('mrf import failed! Cannot use the MRF-based models.  Try compiling it with "make". ')

class VisInfo:
    def __init__(self, bndl, mesh):
        self.visible = np.array([mesh.texel_visible(bndl, frame) for frame in range(len(bndl.img_files))])

    def texel_visible(self, frame):
        return self.visible[frame]

class MRF:
    pass

def make_graph(mesh):
    u1, v1 = np.mgrid[:mesh.texsize, :mesh.texsize]
    u1, v1 = u1.flatten(), v1.flatten()
    adj = []
    dus = [0, 1]
    dvs = [1, 0]
    for idx in range(mesh.nfaces):
        for du, dv in zip(dus, dvs):
            u2 = u1 + du
            v2 = v1 + dv
            ok = (0 <= u2) * (0 <= v2) * (u2 < mesh.juv2tex.shape[1]) * (v2 < mesh.juv2tex.shape[2])
            inds1 = mesh.juv2tex[idx, u1[ok], v1[ok]]
            inds2 = mesh.juv2tex[idx, u2[ok], v2[ok]]
            adj += zip(inds1, inds2)
    for idx in range(mesh.nfaces):
        face_border = np.logical_and(mesh.on_border, mesh.tex2juv[:, 0] == idx)
        other_border = np.logical_and(mesh.on_border, mesh.tex2juv[:, 0] != idx)
        dist, knn_inds = knnsearch(mesh.texel_pts[other_border], mesh.texel_pts[face_border], k = 3)
        inds1 = np.nonzero(face_border)[0]
        inds2 = np.nonzero(other_border)[0]
        i, idx = np.nonzero(dist <= 0.000001)
        adj += zip(inds1[i], inds2[knn_inds[i, idx]])
    graph = nx.Graph()
    graph.add_nodes_from(range(mesh.ntexels))
    graph.add_edges_from(adj)
    return graph

def make_labels(bndl, shift_dim, shift_dist):
    labels = []
    for frame in range(len(bndl.img_files)):
        for x in range(-shift_dim, shift_dim+1):
            for y in range(-shift_dim, shift_dim+1):
                labels.append((frame, shift_dist*x, shift_dist*y))
    return np.array(labels, np.int64)

def project_texels(bndl, frame, mesh, img, geom, shift = np.zeros(2), invisible_colors = False):
    if invisible_colors:
        vis = np.ones(mesh.ntexels, 'bool')
    else:
        vis = geom.texel_visible(frame)
    shift = np.asarray(shift)
    proj = shift + bndl.project(frame, mesh.texel_pts[vis])
    # Check for x-y reversing if any unexpected errors
    colors = np.concatenate([scipy.ndimage.map_coordinates(img[:, :, i], np.array([proj[:, 1], proj[:, 0]]), order = 1, mode = 'reflect')[:, np.newaxis] for i in range(img.shape[2])], axis = 1)
    all_colors = np.zeros((mesh.ntexels,) + colors.shape[1:], dtype = colors.dtype)
    all_colors[vis] = colors
    return vis, all_colors

def label_colors(bndl, mesh, geom, labels, sigma = 0, invisible_colors = False):
    colors = np.zeros((mesh.ntexels, len(labels), 3), np.float32)
    imgs = {}
    for frame in np.unique(labels[:, 0]):
        if sigma == 0:
            imgs[frame] = cv2.cvtColor(bndl.get_img(frame),cv2.COLOR_BGR2LAB)
        else:
            imgs[frame] = cv2.GaussianBlur(cv2.cvtColor(bndl.get_img(frame),cv2.COLOR_BGR2LAB), (sigma,sigma), 0)
    for p, (frame, dx, dy) in enumerate(labels):
        vis, c = project_texels(bndl, frame, mesh, imgs[frame], geom, (dx, dy), invisible_colors = invisible_colors)
        colors[vis, p] = c[vis]
    label_valid = np.zeros((mesh.ntexels, len(labels)), 'bool')
    # for a given texel, a label is valid if the corresponding frame sees the texel
    for p, (frame, _, _) in enumerate(labels):
        vis = geom.texel_visible(frame)
        label_valid[vis, p] = True
    return colors, label_valid

def stability_costs(bndl, mesh, labels, scale = 10., thresh = 2., max_stable = 1e4):
    J_intra_view = np.zeros((len(bndl.img_files), len(bndl.img_files), mesh.nfaces, 2, 2))
    face_vis = np.array([faces_visible(bndl, mesh, frame) for frame in range(len(bndl.img_files))])
    for frame1 in range(len(bndl.img_files)):
        for frame2 in range(len(bndl.img_files)):
            for idx in range(mesh.nfaces):
                if not face_vis[frame1, idx] or not face_vis[frame2, idx]:
                    J_intra_view[frame1, frame2, idx] = np.eye(2)
                else:
                    J_i = np.eye(2)
                    J_ip = np.eye(2)
                    for d in (0, 1):
                        delta = np.array([1., 0]) if d == 0 else np.array([0., 1.])
                        center = np.array([mesh.texsize/2., mesh.texsize/2.])
                        uv1 = center - delta
                        uv2 = center + delta
                        h = abs(uv2[0] - uv1[0]) + abs(uv2[1] - uv1[1])
                        pt1 = mesh.texel_pts[mesh.juv2tex[idx, int(uv1[0]), int(uv1[1])]]
                        pt2 = mesh.texel_pts[mesh.juv2tex[idx, int(uv2[0]), int(uv2[1])]]
                        f1_i = bndl.project(frame1, pt1)
                        f2_i = bndl.project(frame1, pt2)
                        f1_ip = bndl.project(frame2, pt1)
                        f2_ip = bndl.project(frame2, pt2)
                        J_i[:, d] = (f2_i - f1_i)/h
                        J_ip[:, d] = (f2_ip - f1_ip)/h
                    # use Jacobian chain rule to eliminate [du dv] variables
                    J_intra_view[frame1, frame2, idx] = np.dot(J_ip, np.linalg.pinv(J_i))
    # penalty for assigning a texel on face idx a color from frame1
    stretch_penalty = np.zeros((len(bndl.img_files), mesh.nfaces))
    for frame1 in range(len(bndl.img_files)): #label
        for idx in range(mesh.nfaces): # face
            costs = []
            for frame2 in range(len(bndl.img_files)):
                if face_vis[frame1, idx] and face_vis[frame2, idx]:
                    J = J_intra_view[frame1, frame2, idx]
                    eigs = np.abs(np.linalg.svd(J)[1])
                    costs.append(np.minimum(30., scale*np.sum(np.maximum(eigs - thresh, 0)**2)))
            costs = np.array(costs, np.float64)
            stretch_penalty[frame1, idx] = min(np.sum(costs), max_stable)/float(len(bndl.img_files))
    return stretch_penalty[labels[:, 0][None, :], mesh.tex2juv[:, 0][:, None]]

def outline_mask(bndl, mesh, frame, thresh):
    mask = box_mask(bndl, mesh, frame)
    D = scipy.ndimage.distance_transform_edt(1-mask)
    return (1 <= D) * (D <= thresh)
  
def occlusion_mask(bndl, mesh, frame, thresh = 2.):
    mask = box_mask(bndl, mesh, frame)
    D = scipy.ndimage.distance_transform_edt(mask)
    return D <= thresh

def occlusion_texels(bndl, mesh, frame, thresh = 1., only_border = True):
    occ_mask = np.array(occlusion_mask(bndl, mesh, frame, thresh = thresh), np.float64)
    vis = mesh.texel_visible(bndl, frame)
    proj = np.array(bndl.project(frame, mesh.texel_pts), np.int64)
    occ = np.zeros(mesh.ntexels, 'bool')
    occ[vis] = occ_mask[proj[vis, 1], proj[vis, 0]]
    if only_border:
        occ = occ * mesh.on_border  
    return occ

def occlusion_costs(bndl, mesh, labels, geom, sigma = 5, weight_by_frame = True, full_occ = False):
    occ_border = int(0.1*mesh.texsize)
    label_color, label_valid = label_colors(bndl, mesh, geom, labels, sigma = sigma, invisible_colors = False)
    occ_samples = np.zeros((mesh.ntexels, len(bndl.img_files), 3))
    has_sample = np.zeros((mesh.ntexels, len(bndl.img_files)), 'bool')
    for frame in range(len(bndl.img_files)):
        if full_occ:
            frame_occ = geom.texel_visible(frame)
        else:
            frame_occ = occlusion_texels(bndl, mesh, frame)
            if occ_border is not None:
                as_juv = mesh.index_as_juv(frame_occ).copy()
                for idx in range(as_juv.shape[0]):
                    dist, ind = scipy.ndimage.distance_transform_edt(1 - as_juv[idx], return_indices = True)
                    dist[ind[0] < 0] = np.inf
                    as_juv[idx, dist <= occ_border] = True
                frame_occ = np.logical_and(geom.texel_visible(frame), mesh.index_as_flat(as_juv))
      
        vis, colors = project_texels(bndl, frame, mesh, cv2.GaussianBlur(cv2.cvtColor(bndl.get_img(frame),cv2.COLOR_BGR2LAB), (sigma,sigma), 0), geom)    
        occ_samples[frame_occ, frame, :] = colors[frame_occ]
        has_sample[frame_occ, frame] = True
    is_occ = np.any(has_sample, axis = 1)
    costs = np.zeros((mesh.ntexels, label_valid.shape[1]), np.float64)
    frame_nobs = np.array(np.sum(has_sample, axis = 0), np.float64)
    nvisible = np.sum(np.any([geom.texel_visible(frame) for frame in range(len(bndl.img_files))], axis = 0))
    frame_weights = float(nvisible)/len(bndl.img_files)*(1./frame_nobs)
    for frame in range(len(bndl.img_files)):
        frame_samples = occ_samples[:, frame]
        dist_from_sample = np.sqrt(np.sum((label_color[is_occ, :, :] - frame_samples[is_occ][:, None, :])**2, axis = -1))
        dist_from_sample[~label_valid[is_occ]] = 0
        dist_from_sample[~has_sample[is_occ, frame]] = 0
        costs[is_occ, :] += frame_weights[frame]*dist_from_sample
    costs = np.array(costs, 'float32')
    return costs

def interior_mrf(bndl, mesh, geom, shift_dim = 1, shift_dist = 30.,data_weight = 1., full_occ = 0,occ_border = None, occ_weight = 0.5):
    """ Runs the Interior MRF camouflage method """
    # controls the importance of smoothness vs. the local evidence
    data_weight = 1./4*(64./mesh.texsize)
    labels = make_labels(bndl, shift_dim, shift_dist)
    label_color, label_valid = label_colors(bndl, mesh, geom, labels, invisible_colors = True)
    stable_cost = stability_costs(bndl, mesh, labels)
    print("Calculated Stability Costs")
    occ_cost = occ_weight*occlusion_costs(bndl, mesh, labels, geom)
    print("Calculated Occlusion Costs")
    face_weight = np.ones(mesh.nfaces) 
    node_visible = np.any(label_valid, axis = 1)
    fw = face_weight[mesh.tex2juv[:, 0]][:, None]
    data_cost = np.array(fw*(occ_cost + stable_cost), 'float32')
    data_cost[~label_valid] = 1e5
    data_cost[~node_visible] = 0
    data_cost *= data_weight
    m = MRF()
    m.edges = make_graph(mesh).edges()
    m.data_cost = data_cost
    m.labels = labels
    m.label_color = label_color
    m.node_visible = node_visible
    m.sameview_prior = np.zeros(mesh.ntexels, 'float32')
    m.smooth_prior = 1.
    results = mrf.solve_mrf(m)
    occ_total = np.sum((fw*occ_cost)[range(len(results)), results])
    stable_total = np.sum((fw*stable_cost)[range(len(results)), results])
    print('Occlusion cost:', occ_total)
    print('Stability cost:', stable_total)
    print(len(results)," - ",results)
    print(np.array(label_color[np.newaxis, range(len(results)), results], dtype = np.uint8).shape)
    colors = np.squeeze(cv2.cvtColor(np.array(label_color[np.newaxis, range(len(results)), results], dtype = np.uint8), cv2.COLOR_LAB2BGR))
    return colors, results, labels, (occ_total, stable_total)

def mismatched(mrf, results):
    e = np.array(mrf.edges)
    ok_edges = mrf.node_visible[e[:, 0]]*mrf.node_visible[e[:, 1]]
    return results[e[ok_edges, 0]] != results[e[ok_edges, 1]]

def boundary_mrf(scan, mesh, geom, shift_dim = 0, shift_dist = 1, max_stable = 1e4,occ_weight = 0.5, stable_weight = 1.,per_term_stable = np.inf, use_face_mrf = True):
    """ Run the Boundary MRF model """
    labels = make_labels(scan, shift_dim, shift_dist)
    label_color, label_valid = label_colors(scan, mesh, geom, labels)
    stable_cost = stable_weight*stability_costs(scan, mesh, labels)
    print("Calculated Stability Costs")
    if occ_weight == 0:
        print 'no occ cost!'
        occ_cost = np.zeros((mesh.ntexels, len(labels)))
    else:
        occ_cost = occ_weight*occlusion_costs(scan, mesh, labels, geom)
    print("Calculated Occlusion Costs")
    node_visible = np.any(label_valid, axis = 1)
    data_cost = np.array(occ_cost + stable_cost, 'float32')
    data_cost[~label_valid] = 1e5
    # no valid label? any label will do
    data_cost[~node_visible] = 0
    mismatch_cost = 1e5
    sameview_prior = 1e7 + np.zeros(mesh.ntexels, 'float32')
    sameview_prior[mesh.on_border] = mismatch_cost
    m = MRF()
    m.edges = make_graph(mesh).edges()
    m.data_cost = data_cost
    m.labels = labels
    m.label_color = label_color
    m.node_visible = node_visible
    m.sameview_prior = sameview_prior
    m.smooth_prior = 0.
  
    def en(r):
        u = np.sum(np.array(data_cost, 'd')[range(len(r)), r])
        s = np.sum(mismatched(m, r)*mismatch_cost)
        # 2* is to deal w/ bug in gc energy estimate
        print u + 2*s, (u, 2*s)
  
    if use_face_mrf:
        # Use a brute-force solver (written in Cython)
        results = mrf.solve_face_mrf(mesh, m)
        print 'Energy for brute-force solver', en(results)
        if 0:
            # verify that the brute-force solver gets a better result than alpha-expansion
            results2 = mrf.solve_mrf(m)
            print 'Energy for alpha-expansion solver:', en(results2)
    else:
        # Solve using alpha-expansion (requires gco)
        results = mrf.solve_mrf(m)
    occ_total = np.sum(occ_cost[range(len(results)), results])
    stable_total = np.sum(stable_cost[range(len(results)), results])
    print('Occlusion cost:', occ_total)
    print('Stability cost:', stable_total)
    print(len(results)," - ",results)
    colors = np.squeeze(cv2.cvtColor(np.array(label_color[np.newaxis, range(len(results)), results], dtype = np.uint8), cv2.COLOR_LAB2BGR))
    return colors, results, labels, (occ_total, stable_total)

def camo(scene = "mit-26", method = 'interior'):
    data_path = "../Test Data/" + scene + "/"
    img_files = get_jpg_files(data_path)
    if len(img_files) == 0: raise RuntimeError("Image files not found")
    img_shape = load_img(img_files[0],1).shape
    bndl = Bundler(data_path + "bundle/bundle.out", img_files, img_shape)
    print("Parsed bundler output")
    mesh = load_from_mat(data_path + "cube.mat", 0)
    print("Added box to scene")
    geom = VisInfo(bndl, mesh)
    print("Cached visual info of scenes")
    if method == 'interior':
        print("Starting Interior MRF model")
        texel_colors = interior_mrf(bndl, mesh, geom)
    elif method == 'boundary':
        print("Starting Boundary MRF model")
        texel_colors = boundary_mrf(bndl, mesh, geom)
    render_scene(scene, texel_colors[0], bndl, mesh)
    
if __name__ == "__main__":
    scene = "mit-12"
    method = "boundary"
    camo(scene, method)
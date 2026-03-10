from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import igl


EPS = 1e-12


def _as_numpy(V, F):
    try:
        import torch
        if isinstance(V, torch.Tensor):
            V = V.detach().cpu().numpy()
        if isinstance(F, torch.Tensor):
            F = F.detach().cpu().numpy()
    except Exception:
        pass

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F)

    if V.ndim == 3:
        V = np.squeeze(V, axis=0)
    if F.ndim == 3:
        F = np.squeeze(F, axis=0)

    if V.shape[-1] > 3:
        V = V[..., :3]

    F = F.astype(np.int64, copy=False)
    return V, F


def vertex_lumped_mass_from_faces(V, F, face_areas=None):
    V, F = _as_numpy(V, F)
    n_verts = V.shape[0]
    if face_areas is None:
        double_area = np.asarray(igl.doublearea(V, F), dtype=np.float64)
        face_areas = 0.5 * double_area
    else:
        face_areas = np.asarray(face_areas, dtype=np.float64)

    mass = np.zeros(n_verts, dtype=np.float64)
    contrib = np.repeat(face_areas / 3.0, 3)
    np.add.at(mass, F.reshape(-1), contrib)
    return mass


def face_mass_matrix_from_double_area(double_area):
    double_area = np.asarray(double_area, dtype=np.float64)
    return sp.diags(np.repeat(double_area, 3)).tocsc()


def convert_igl_grad_to_interleaved_sparse_tuple(grad_csc: sp.csc_matrix):
    """
    Convert libigl grad layout from [all x rows, all y rows, all z rows] to
    per-face interleaved rows [fx, fy, fz, fx, fy, fz, ...].

    Returns tuple (indices, data, n_rows, n_cols) compatible with SparseMat.from_M().
    """
    if not sp.isspmatrix_csc(grad_csc):
        grad_csc = grad_csc.tocsc()

    n_rows, n_cols = grad_csc.shape
    if n_rows % 3 != 0:
        raise ValueError(f"Expected libigl grad to have 3F rows, got {n_rows}.")
    n_faces = n_rows // 3

    Gcoo = grad_csc.tocoo()
    row = Gcoo.row.copy()
    col = Gcoo.col.astype(np.int64, copy=False)
    data = Gcoo.data

    block = row // n_faces  # 0:x,1:y,2:z
    face_row = row % n_faces
    row_interleaved = 3 * face_row + block

    order = np.argsort(row_interleaved, kind='mergesort')
    row_interleaved = row_interleaved[order]
    col = col[order]
    data = data[order]

    inds = np.vstack((row_interleaved, col)).astype(np.int64)
    return inds, data, n_rows, n_cols


def compute_mesh_differential_primitives(V, F):
    """
    Shared mesh differential operators used by both DiffusionNet and NJF/Poisson code.

    Returns
    -------
    b1, b2, b3 : (#F,3) face local basis from libigl.local_basis
    grad_csc    : (3F,V) libigl gradient operator (xyz stacked by blocks)
    double_area : (#F,) libigl double area
    face_mass3  : (3F,3F) diagonal face mass matrix diag([2A,2A,2A])
    laplace     : (V,V) stiffness matrix grad^T * face_mass3 * grad (positive semidefinite)
    rhs         : (V,3F) grad^T * face_mass3
    """
    V, F = _as_numpy(V, F)
    if F.size == 0:
        raise ValueError("Mesh differential primitives require triangular faces (point cloud not supported).")

    b1, b2, b3 = igl.local_basis(V, F)
    grad = igl.grad(V, F).tocsc()
    double_area = np.asarray(igl.doublearea(V, F), dtype=np.float64)
    face_mass3 = face_mass_matrix_from_double_area(double_area)

    laplace = (grad.T @ face_mass3 @ grad).tocsc()
    rhs = (grad.T @ face_mass3).tocsc()
    return b1, b2, b3, grad, double_area, face_mass3, laplace, rhs


def _split_igl_grad_xyz(grad_csc: sp.csc_matrix):
    if not sp.isspmatrix_csc(grad_csc):
        grad_csc = grad_csc.tocsc()
    n_rows, _ = grad_csc.shape
    if n_rows % 3 != 0:
        raise ValueError("Expected grad operator with 3F rows.")
    n_faces = n_rows // 3
    return grad_csc[:n_faces, :].tocsc(), grad_csc[n_faces:2*n_faces, :].tocsc(), grad_csc[2*n_faces:, :].tocsc()


def _vertex_face_area_average_matrix(n_verts: int, F: np.ndarray, face_areas: np.ndarray, vertex_mass: np.ndarray):
    n_faces = F.shape[0]
    rows = F.reshape(-1)
    cols = np.repeat(np.arange(n_faces, dtype=np.int64), 3)
    data = np.repeat(face_areas / 3.0, 3)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n_verts, n_faces)).tocsc()
    inv_mass = 1.0 / np.maximum(vertex_mass, EPS)
    return sp.diags(inv_mass) @ A


def build_diffusionnet_vertex_gradient_ops_from_igl(V, F, vertex_frames, grad_csc=None, double_area=None, vertex_mass=None):
    """
    Build per-vertex tangent gradient operators (gradX, gradY) from libigl's face gradient.

    The pipeline is:
      scalar vertex function -> per-face 3D gradients (libigl.grad)
      -> area-weighted averaging to vertices -> projection on vertex tangent frame.
    """
    V, F = _as_numpy(V, F)
    vertex_frames = np.asarray(vertex_frames, dtype=np.float64)
    n_verts = V.shape[0]

    if grad_csc is None:
        grad_csc = igl.grad(V, F).tocsc()
    else:
        grad_csc = grad_csc.tocsc()

    if double_area is None:
        double_area = np.asarray(igl.doublearea(V, F), dtype=np.float64)
    else:
        double_area = np.asarray(double_area, dtype=np.float64)

    face_areas = 0.5 * double_area
    if vertex_mass is None:
        vertex_mass = vertex_lumped_mass_from_faces(V, F, face_areas=face_areas)
    else:
        vertex_mass = np.asarray(vertex_mass, dtype=np.float64)

    Gx_f, Gy_f, Gz_f = _split_igl_grad_xyz(grad_csc)
    A_vf = _vertex_face_area_average_matrix(n_verts, F, face_areas, vertex_mass)

    Gx_v = (A_vf @ Gx_f).tocsc()
    Gy_v = (A_vf @ Gy_f).tocsc()
    Gz_v = (A_vf @ Gz_f).tocsc()

    basisX = vertex_frames[:, 0, :]
    basisY = vertex_frames[:, 1, :]

    gradX = (
        sp.diags(basisX[:, 0]) @ Gx_v +
        sp.diags(basisX[:, 1]) @ Gy_v +
        sp.diags(basisX[:, 2]) @ Gz_v
    ).tocsc()

    gradY = (
        sp.diags(basisY[:, 0]) @ Gx_v +
        sp.diags(basisY[:, 1]) @ Gy_v +
        sp.diags(basisY[:, 2]) @ Gz_v
    ).tocsc()

    return gradX, gradY


def generalized_eigendecomposition(L_csc, massvec, k_eig):
    massvec = np.asarray(massvec, dtype=np.float64)
    n = L_csc.shape[0]

    if k_eig <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((n, 0), dtype=np.float64)

    if n <= 2:
        k_use = min(k_eig, max(n - 1, 0))
    else:
        k_use = min(k_eig, n - 2)
    if k_use <= 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((n, 0), dtype=np.float64)

    eps = EPS
    L_eigsh = (L_csc + sp.identity(n, format='csc') * eps).tocsc()
    Mmat = sp.diags(massvec)

    failcount = 0
    while True:
        try:
            evals, evecs = sla.eigsh(L_eigsh, k=k_use, M=Mmat, sigma=eps, tol=1e-20)
            evals = np.clip(evals, a_min=0.0, a_max=np.inf)
            break
        except Exception:
            failcount += 1
            if failcount > 4:
                raise
            L_eigsh = L_eigsh + sp.identity(n, format='csc') * (eps * (10 ** failcount))

    if k_use < k_eig:
        evals_pad = np.zeros((k_eig,), dtype=np.float64)
        evecs_pad = np.zeros((n, k_eig), dtype=np.float64)
        evals_pad[:k_use] = evals
        evecs_pad[:, :k_use] = evecs
        return evals_pad, evecs_pad

    return evals, evecs


def compute_diffusionnet_mesh_operators(V, F, vertex_frames, k_eig):
    """
    Shared mesh-operator builder for DiffusionNet mesh inputs using the same libigl gradient/Laplacian
    primitives as the Poisson/NJF path.
    """
    V, F = _as_numpy(V, F)
    b1, b2, b3, grad_csc, double_area, face_mass3, L_csc, rhs_csc = compute_mesh_differential_primitives(V, F)
    massvec = vertex_lumped_mass_from_faces(V, F, face_areas=0.5 * double_area)
    massvec = massvec + EPS * max(float(np.mean(massvec)), EPS)
    evals, evecs = generalized_eigendecomposition(L_csc, massvec, k_eig)
    gradX, gradY = build_diffusionnet_vertex_gradient_ops_from_igl(
        V, F, vertex_frames=vertex_frames, grad_csc=grad_csc, double_area=double_area, vertex_mass=massvec
    )
    return massvec, L_csc, evals, evecs, gradX, gradY

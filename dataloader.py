from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import importlib

import numpy as np
import torch
from torch.utils import data
import trimesh
from tqdm import tqdm
import sys

sys.path.append("./models")
sys.path.append("./")

from models.PoissonSystem import poisson_system_matrices_from_mesh


def _import_diffusionnet_module():
    """Import DiffusionNet package from common project layouts.

    Tries (in order):
      - models.diffusion_net   (original project layout)
      - diffusion_net          (standalone package layout)
    """
    candidates = ("models.diffusion_net", "diffusion_net")
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover (layout dependent)
            last_err = exc
    raise ImportError(
        "Could not import DiffusionNet package. Expected one of: "
        f"{candidates}. Ensure your unified package (geometry/layers/utils + mesh_ops_unified.py) "
        "is available on PYTHONPATH."
    ) from last_err


diffusion_net = _import_diffusionnet_module()

DEFAULT_MESH_EXTS: Tuple[str, ...] = (".ply", ".obj", ".off", ".stl", ".glb", ".gltf")
PathLike = Union[str, Path]


# ----------------------------
# Path helpers
# ----------------------------
def as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def ensure_dir(p: PathLike, name: str) -> Path:
    pp = as_path(p).expanduser()
    if not pp.exists():
        raise FileNotFoundError(f"{name} does not exist: {pp}")
    if not pp.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {pp}")
    return pp


# ----------------------------
# IO utilities
# ----------------------------
def iter_mesh_files(folder: Path, exts: Tuple[str, ...] = DEFAULT_MESH_EXTS) -> List[Path]:
    folder = ensure_dir(folder, "sequence_dir")
    exts_lc = tuple(e.lower() for e in exts)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts_lc])


def iter_template_meshes(templates_dir: Path, exts: Tuple[str, ...] = DEFAULT_MESH_EXTS) -> List[Path]:
    templates_dir = ensure_dir(templates_dir, "templates_dir")
    files: List[Path] = []
    for ext in exts:
        files.extend(templates_dir.glob(f"*{ext}"))
        files.extend(templates_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.name.lower())


def find_matching_sequence_dirs(seqs_root: Path, template_key: str) -> List[Path]:
    seqs_root = seqs_root#ensure_dir(seqs_root, "deformations_dir")
    key = template_key.lower()
    print(key)
    matches = [d for d in seqs_root.iterdir() if d.name.lower().startswith(key)]
    return sorted(matches, key=lambda p: p.name.lower())


# ----------------------------
# Dataset wrapper
# ----------------------------
class Dataset(data.Dataset):
    """Sequence dataset yielding one sample = (mesh sequence, template, operators, landmarks)."""

    def __init__(self, data_list: List[Dict[str, Any]], subjects_dict: Dict[str, List[str]], data_type: str = "train"):
        self.data = data_list
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type

    def __getitem__(self, index: int):
        sample = self.data[index]
        return (
            sample["name"],
            torch.as_tensor(sample["verts_src"], dtype=torch.float32),
            torch.as_tensor(sample["normals_src"], dtype=torch.float32),
            torch.as_tensor(np.asarray(sample["mass_src"]), dtype=torch.float32),
            sample["L_src"].float(),
            torch.as_tensor(np.asarray(sample["evals_src"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(sample["evecs_src"]), dtype=torch.float32),
            sample["gradX_src"].float(),
            sample["gradY_src"].float(),
            torch.as_tensor(sample["faces_src"], dtype=torch.int64),
            torch.as_tensor(sample["verts_tgt"], dtype=torch.float32),
            torch.as_tensor(sample["normals_tgt"], dtype=torch.float32),
            torch.as_tensor(np.asarray(sample["mass_tgt"]), dtype=torch.float32),
            sample["L_tgt"].float(),
            torch.as_tensor(np.asarray(sample["evals_tgt"]), dtype=torch.float32),
            torch.as_tensor(np.asarray(sample["evecs_tgt"]), dtype=torch.float32),
            sample["gradX_tgt"].float(),
            sample["gradY_tgt"].float(),
            torch.as_tensor(sample["faces_tgt"], dtype=torch.int64),
            sample["poisson_solver"],
        )

    def __len__(self) -> int:
        return self.len


# ----------------------------
# Split helpers
# ----------------------------
def _subjects_dict_from_args(args) -> Dict[str, List[str]]:
    return {
        "train": [i for i in str(args.train_subjects).split(" ") if i],
        "val": [i for i in str(args.val_subjects).split(" ") if i],
        "test": [i for i in str(args.test_subjects).split(" ") if i],
    }


def _append_by_subject(
    samples: Sequence[Dict[str, Any]],
    subjects_dict: Dict[str, List[str]],
    subject_key_fn: Callable[[str], str],
    train_data: List[Dict[str, Any]],
    valid_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
) -> None:
    for s in samples:
        sid = subject_key_fn(s["name"])
        if sid in subjects_dict["train"]:
            train_data.append(s)
        if sid in subjects_dict["val"]:
            valid_data.append(s)
        if sid in subjects_dict["test"]:
            test_data.append(s)


# ----------------------------
# Operator and sequence loading
# ----------------------------
def _compute_operators(vertices: np.ndarray, faces: np.ndarray, k_eig: int):
    # Unified compute path: uses geometry.compute_operators(), which in your package now calls mesh_ops_unified.py
    _, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
        torch.as_tensor(vertices, dtype=torch.float32),
        faces=torch.as_tensor(faces, dtype=torch.int64),
        k_eig=k_eig,
    )
    return mass, L, evals, evecs, gradX, gradY


def _load_template_and_targets(
    templates_dir: PathLike,
    targets_dir: PathLike,
    k_eig: int,
    *,
    exts: Tuple[str, ...] = DEFAULT_MESH_EXTS,
) -> List[Dict[str, Any]]:
    templates_dir = ensure_dir(templates_dir, "templates_dir")
    targets_dir = ensure_dir(targets_dir, "targets_dir")

    results: List[Dict[str, Any]] = []
    for tmpl_path in iter_template_meshes(templates_dir, exts):
        template_name = tmpl_path.stem
        template_mesh = trimesh.load(str(tmpl_path), process=False)

        temp = np.asarray(template_mesh.vertices)
        faces_src = np.asarray(template_mesh.faces, dtype=np.int64)
        normals_src = np.asarray(template_mesh.vertex_normals)

        mass_src, L_src, evals_src, evecs_src, gradX_src, gradY_src = _compute_operators(temp, faces_src, k_eig)
        # Precompute Poisson/Jacobian-field operators ONCE per template mesh.
        # These are reused during NJF decoding so we don't rebuild igl.grad / Laplacian every iteration.
        poisson_sys = poisson_system_matrices_from_mesh(V=temp, F=faces_src, cpuonly=False)
        poisson_solver = poisson_sys.create_poisson_solver()

        subject_id = "_".join(template_name.split("_")[:1])
        matching_id = find_matching_sequence_dirs(targets_dir, subject_id)
        L = len(matching_id)

        ### Iterate through targets
        for i, tgt_path in tqdm(enumerate(matching_id), total=L):
            mesh = trimesh.load(str(tgt_path), process=False)
            verts_tgt = np.asarray(mesh.vertices)
            faces_tgt = np.asarray(mesh.faces)
            normals_tgt = np.asarray(mesh.vertex_normals)
            ### Compute operators for tgt
            mass_tgt, L_tgt, evals_tgt, evecs_tgt, gradX_tgt, gradY_tgt = _compute_operators(verts_tgt, faces_tgt, k_eig)

            results.append(
                {
                    "name": tgt_path.name[:-4],
                    "verts_src": temp,
                    "normals_src": normals_src,
                    "mass_src": mass_src,
                    "L_src": L_src,
                    "evals_src": evals_src,
                    "evecs_src": evecs_src,
                    "gradX_src": gradX_src,
                    "gradY_src": gradY_src,
                    "faces_src": faces_src,
                    "verts_tgt": verts_tgt,
                    "normals_tgt": normals_tgt,
                    "mass_tgt": mass_tgt,
                    "L_tgt": L_tgt,
                    "evals_tgt": evals_tgt,
                    "evecs_tgt": evecs_tgt,
                    "gradX_tgt": gradX_tgt,
                    "gradY_tgt": gradY_tgt,
                    "faces_tgt": faces_tgt,
                    "poisson_solver": poisson_solver,
                }
            )

    return results


# ----------------------------
# Main loader
# ----------------------------
def read_data(args):
    print("Loading data...")

    subjects_dict = _subjects_dict_from_args(args)
    train_data: List[Dict[str, Any]] = []
    valid_data: List[Dict[str, Any]] = []
    test_data: List[Dict[str, Any]] = []

    # MANO
    MANO_samples = _load_template_and_targets(
        args.templates_dir,
        args.targets_dir,
        args.k_eig,
    )
    _append_by_subject(
        MANO_samples,
        subjects_dict,
        subject_key_fn=lambda name: "_".join(name.split("_")[:1]),
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


# ----------------------------
# Collate function (handles sparse tensors returned by diffusion operators)
# ----------------------------
def custom_collate(batch):
    """Custom collate_fn that preserves sparse operators as lists.

    PyTorch's default collate stacks tensors into a batch dimension; that is convenient for dense tensors,
    but sparse operator tensors (L, gradX, gradY) are easier/safer to keep as per-sample objects here.
    """
    if len(batch) == 0:
        return batch

    fields = list(zip(*batch))
    out = []
    for items in fields:
        first = items[0]
        if isinstance(first, torch.Tensor):
            layout_name = str(getattr(first, "layout", ""))
            is_sparse_any = bool(first.is_sparse) or ("sparse" in layout_name and "strided" not in layout_name)
            if is_sparse_any:
                out.append(list(items))
                continue

            same_shape = all(isinstance(x, torch.Tensor) and tuple(x.shape) == tuple(first.shape) for x in items)
            out.append(torch.stack(list(items), dim=0) if same_shape else list(items))
        else:
            out.append(list(items))
    return tuple(out)



def get_dataloader(args):
    dataset: Dict[str, data.DataLoader] = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    train_ds = Dataset(train_data, subjects_dict, "train")
    valid_ds = Dataset(valid_data, subjects_dict, "val")
    test_ds = Dataset(test_data, subjects_dict, "test")

    # Sparse operators + multiprocessing can be fragile on some PyTorch versions, so default to num_workers=0.
    num_workers = int(getattr(args, "num_workers", 0) or 0)

    dataset["train"] = data.DataLoader(
        dataset=train_ds,
        batch_size=int(args.batch_size),
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    dataset["valid"] = data.DataLoader(
        dataset=valid_ds,
        batch_size=1,
        collate_fn=custom_collate,
        shuffle=True,
        num_workers=num_workers,
    )
    dataset["test"] = data.DataLoader(
        dataset=test_ds,
        batch_size=1,
        collate_fn=custom_collate,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return dataset

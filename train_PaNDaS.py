from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import trimesh
import sys

sys.path.append("models")
sys.path.append("./")
from tqdm import tqdm

from dataloader import get_dataloader


def _import_model_class():
    candidates = [
        ("PaNDaS_deformer", "DiffusionNetAutoencoder"),
        ("models.PaNDaS_deformer", "DiffusionNetAutoencoder"),
    ]
    last_err = None
    for module_name, cls_name in candidates:
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, cls_name)
        except Exception as exc:  # pragma: no cover (layout dependent)
            last_err = exc
    raise ImportError(
        "Could not import DiffusionNetAutoencoder. Tried: "
        + ", ".join(m for m, _ in candidates)
        + ". Put PaNDaS_deformer.py on PYTHONPATH (or keep legacy models path)."
    ) from last_err


def _import_diffusionnet_module():
    candidates = ("models.diffusion_net", "diffusion_net")
    last_err = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise ImportError(
        "Could not import DiffusionNet package (models.diffusion_net or diffusion_net)."
    ) from last_err


DiffusionNetAutoencoder = _import_model_class()


def _ensure_checkpoint_dir(model_path: str) -> None:
    Path(model_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _save_checkpoint(args, model, optim, epoch: int) -> None:
    _ensure_checkpoint_dir(args.model_path)
    torch.save(
        {
            "epoch": epoch,
            "autoencoder_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict() if optim is not None else None,
        },
        args.model_path,
    )


def _load_checkpoint_if_available(args, model, optim=None) -> int:
    if not getattr(args, "load_model", False):
        return 0
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["autoencoder_state_dict"])
    if optim is not None and checkpoint.get("optimizer_state_dict") is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def _first_sparse_operator(field, device: torch.device) -> torch.Tensor:
    """Field comes from custom_collate and is usually [sparse_tensor] for batch_size=1."""
    if isinstance(field, list):
        if len(field) != 1:
            raise ValueError(
                "This training script currently expects batch_size=1 for sparse operators. "
                f"Got {len(field)} operators."
            )
        op = field[0]
    else:
        op = field
    op = op.to(device)
    if op.dim() == 2:
        op = op.unsqueeze(0)
    return op


def _squeeze_b1(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.dim() > 0 and x.shape[0] == 1:
        return x[0]
    raise ValueError(f"Expected leading batch dimension 1 for {name}, got shape {tuple(x.shape)}")


def _prepare_model_inputs(sample, device: torch.device) -> Dict[str, Any]:
    """Convert custom-collated sample tuple into tensors expected by forward_latent_njf().

    This script is kept intentionally conservative and uses batch_size=1 because the sparse operators
    are collated as Python lists (one operator per sample) and the current NJF decoder path is not
    written for heterogeneous topologies inside a batch.
    """
    # Dense fields are stacked by custom_collate; sparse fields are lists of length B.
    names = sample[0]  # list[str]
    verts_src = sample[1].to(device)           # [B, V, 3]
    normals_src = sample[2].to(device)
    mass_src = sample[3].to(device)      # [B, V]
    L_src = _first_sparse_operator(sample[4], device)
    evals_src = sample[5].to(device)     # [B, K]
    evecs_src = sample[6].to(device)     # [B, V, K]
    gradX_src = _first_sparse_operator(sample[7], device)
    gradY_src = _first_sparse_operator(sample[8], device)
    faces_src = sample[9].to(device)  # [B, F, 3]
    verts_tgt = sample[10].to(device)  # [B, V, 3]
    normals_tgt = sample[11].to(device)
    mass_tgt = sample[12].to(device)  # [B, V]
    L_tgt = _first_sparse_operator(sample[13], device)
    evals_tgt = sample[14].to(device)  # [B, K]
    evecs_tgt = sample[15].to(device)  # [B, V, K]
    gradX_tgt = _first_sparse_operator(sample[16], device)
    gradY_tgt = _first_sparse_operator(sample[17], device)
    faces_tgt = sample[18].to(device)  # [B, F, 3]

    poisson_solver = sample[19][0] if isinstance(sample[19], (list, tuple)) else sample[19]


    return {
        "name": names[0] if isinstance(names, (list, tuple)) else names,
        "verts_src": verts_src,
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


def _forward_batch(model, batch: Dict[str, Any]) -> torch.Tensor:
    return model.forward(
        verts_src=batch["verts_src"],
        mass_src=batch["mass_src"],
        L_src=batch["L_src"],
        evals_src=batch["evals_src"],
        evecs_src=batch["evecs_src"],
        gradX_src=batch["gradX_src"],
        gradY_src=batch["gradY_src"],
        faces_src=batch["faces_src"],
        verts_tgt=batch["verts_tgt"],
        mass_tgt=batch["mass_tgt"],
        L_tgt=batch["L_tgt"],
        evals_tgt=batch["evals_tgt"],
        evecs_tgt=batch["evecs_tgt"],
        gradX_tgt=batch["gradX_tgt"],
        gradY_tgt=batch["gradY_tgt"],
        faces_tgt=batch["faces_tgt"],
        poisson_solver=batch["poisson_solver"],
    )


def _export_predicted_mesh(vertices_pred: torch.Tensor, faces_template: torch.Tensor, vertices_gt: torch.Tensor,
                            faces_gt: torch.Tensor, pred_dir: Path, tgt_dir: Path, name) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    faces_pred_np = faces_template[0].detach().cpu().numpy()
    pred_mesh = trimesh.Trimesh(vertices_pred[0].detach().cpu().numpy(), faces_pred_np, process=False)

    pred_mesh.export(str(pred_dir / f"{name}.ply"))

    faces_gt_np = faces_gt[0].detach().cpu().numpy()
    gt_mesh = trimesh.Trimesh(vertices_gt[0].detach().cpu().numpy(), faces_gt_np, process=False)
    gt_mesh.export(str(tgt_dir / f"{name}.ply"))


def train(args):
    model = DiffusionNetAutoencoder(args).to(args.device)
    criterion = nn.MSELoss()
    criterion_val = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = _load_checkpoint_if_available(args, model, optim)

    dataset = get_dataloader(args)
    for epoch in range(start_epoch, args.epochs):
        # ---------------- Validation (every eval_every epochs) ----------------
        if args.eval_every > 0 and epoch > 0 and epoch % args.eval_every == 0:
            model.eval()
            valid_losses = []
            with torch.no_grad():
                pbar = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]), desc=f"VAL {epoch+1}")
                for b, sample in pbar:
                    batch = _prepare_model_inputs(sample, args.device)
                    vertices_pred = _forward_batch(model, batch)
                    loss_val = criterion_val(vertices_pred, batch["verts_tgt"])
                    valid_losses.append(float(loss_val.item()))
                    pbar.set_description(f"(Epoch {epoch + 1}) VAL LOSS:{np.mean(valid_losses):.10f}")

                    if args.export_val_meshes:
                        pred_dir = Path(args.results_path) / "Meshes_Val" / str(epoch) / "preds" / batch["name"]
                        tgt_dir = Path(args.results_path) / "Meshes_Val" / "targets" / batch["name"]
                        _export_predicted_mesh(
                            vertices_pred, batch["faces_src"], batch["verts_tgt"],
                            batch["faces_tgt"], pred_dir, tgt_dir, name=batch["name"]
                        )
            _save_checkpoint(args, model, optim, epoch)

        # ---------------- Training ----------------
        model.train()
        train_losses = []
        pbar = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]), desc=f"TRAIN {epoch+1}")
        for b, sample in pbar:
            batch = _prepare_model_inputs(sample, args.device)
            vertices_pred = _forward_batch(model, batch)

            optim.zero_grad(set_to_none=True)
            loss = criterion(vertices_pred, batch["verts_tgt"])
            loss.backward()
            optim.step()

            train_losses.append(float(loss.item()))
            pbar.set_description(f"(Epoch {epoch + 1}) TRAIN LOSS:{np.mean(train_losses):.10f}")

        _save_checkpoint(args, model, optim, epoch)


def test(args):
    dataset = get_dataloader(args)
    model = DiffusionNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint["autoencoder_state_dict"])
    metric = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        losses = []
        pbar = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]), desc="TEST")
        for _, sample in pbar:
            batch = _prepare_model_inputs(sample, args.device)
            vertices_pred = _forward_batch(model, batch)
            loss_val = metric(vertices_pred, batch["vertices_target_seq"])
            losses.append(float(loss_val.item()))
            pbar.set_description(f"TEST LOSS:{np.mean(losses):.10f}")

            if args.export_test_meshes:
                pred_dir = Path(args.results_path) / "Meshes_test" / "preds" / batch["name"]
                tgt_dir = Path(args.results_path) / "Meshes_test" / "targets" / batch["name"]
                _export_predicted_mesh(
                    vertices_pred, batch["faces_src"], batch["vertices_tgt"],
                    batch["faces_tgt"], pred_dir, tgt_dir, name=batch["name"]
                )



def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description="Train/test/infer DiffusionNet + NJF model")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=_default_device())
    parser.add_argument("--eval_every", type=int, default=5)

    parser.add_argument("--export_val_meshes", type=_str2bool, default=True)
    parser.add_argument("--export_test_meshes", type=_str2bool, default=True)

    # data args
    parser.add_argument('--template_file', type=str, default='./data/MANO/templates_aligned/01_01r.ply')
    parser.add_argument('--templates_dir', type=str, default='./data/MANO/templates_small')
    parser.add_argument('--targets_dir', type=str, default='../datasets/MANO_ALIGNED_0')
    parser.add_argument('--train_subjects', type=str, default="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18"
                                                              " 19 20 21 22 23 24 25 26 27 28 29 30 31"
                                                              " 32 33 34 35 36 37 38 39 40 41 42 43 44 45"
                                                              " 46 47")
    parser.add_argument('--val_subjects', type=str, default="48 01")
    parser.add_argument('--test_subjects', type=str, default="49 50 01")
    parser.add_argument('--results_path', type=str, default='../Data/PaNDaS/PaNDaS_MANO')

    # checkpoint args
    parser.add_argument('--load_model', type=_str2bool, default=False)
    parser.add_argument('--models_dir', type=str, default='../Data/PaNDaS/Models')
    parser.add_argument('--model_path', type=str, default='../Data/PaNDaS/Models/PaNDaS_MANO.pth.tar')

    # model hyperparameters
    parser.add_argument('--latent_channels', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--k_eig', type=int, default=64)
    parser.add_argument('--n_points', type=int, default=778)  # 778
    parser.add_argument('--n_faces', type=int, default=1538)  # 1538
    parser.add_argument('--batchnorm_encoder', type=str, default='GROUPNORM')
    parser.add_argument('--batchnorm_decoder', type=str, default='GROUPNORM')
    parser.add_argument('--shuffle_triangles', type=_str2bool, default=False)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    print(args.device)

    train(args)
    test(args)


if __name__ == "__main__":
    main()

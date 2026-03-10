import sys, os, glob
import trimesh
import numpy as np
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm

sys.path.append('./model/diffusion-net/src')
import models.diffusion_net as diffusion_net
from dataloader import get_dataloader
from models.PaNDaS_deformer import DiffusionNetAutoencoder


def train(args):
    model = DiffusionNetAutoencoder(args).to(args.device)

    criterion = nn.MSELoss()
    criterion_n = nn.CosineSimilarity(dim=1, eps=1e-6)
    criterion_val = nn.MSELoss()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    template_mesh = trimesh.load(args.template_file)

    starting_epoch = 0
    if args.load_model:
        checkpoint = torch.load(args.model_path, map_location=args.device)  # args.model_path
        model.load_state_dict(checkpoint['autoencoder_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print(starting_epoch)
    dataset = get_dataloader(args)

    train_losses = []
    val_losses = []
    for epoch in range(starting_epoch, args.epochs):
        valid_loss_log = []
        if epoch%20==0 and epoch>0:
            model.eval()
            with torch.no_grad():
                t_test_loss = 0
                pbar_talk = tqdm(enumerate(dataset["valid"]), total=len(dataset["valid"]))
                for b, sample in pbar_talk:
                    vertices = sample[0].to(args.device)
                    template = sample[1].to(args.device)
                    mass = sample[2].to(args.device)
                    L = sample[3].to(args.device)
                    evals = sample[4].to(args.device)
                    evecs = sample[5].to(args.device)
                    gradX = sample[6].to(args.device)
                    gradY = sample[7].to(args.device)
                    name = sample[8][0]
                    faces = sample[9].to(args.device)
                    mass_template = sample[10].to(args.device)
                    L_template = sample[11].to(args.device)
                    evals_template = sample[12].to(args.device)
                    evecs_template = sample[13].to(args.device)
                    gradX_template = sample[14].to(args.device)
                    gradY_template = sample[15].to(args.device)
                    faces_template = sample[16].to(args.device)

                    if args.use_normals:
                        normals, normals_template = sample[17].to(args.device), sample[18].to(args.device)
                        in_features = torch.cat((vertices, normals), dim=2)
                        in_features_template = torch.cat((template, normals_template), dim=2)
                        vertices_pred, latent = model.forward_val_latent_njf(in_features_template, in_features, mass, L, evals, evecs,
                                                          gradX, gradY, faces,
                                                          mass_template, L_template, evals_template, evecs_template,
                                                          gradX_template, gradY_template, faces_template, name)
                    else:
                        vertices_pred, latent = model.forward_val_latent_njf(template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                                                  mass_template, L_template, evals_template, evecs_template,
                                                  gradX_template, gradY_template, faces_template, name)

                    t_test_loss += criterion_val(vertices_pred, vertices).item()

                    os.makedirs(f'{args.results_path}/Meshes_Val/{str(epoch)}/preds', exist_ok=True)
                    os.makedirs(f'{args.results_path}/Meshes_Val/targets', exist_ok=True)
                    mesh_template = trimesh.Trimesh(template.cpu().detach().numpy()[0], template_mesh.faces)
                    mesh_template.export(f'{args.results_path}/Meshes_Val/template_val.ply')

                    mesh = trimesh.Trimesh(vertices_pred[:,:,:3].cpu().detach().numpy()[0], template_mesh.faces)
                    mesh.export(f'{args.results_path}/Meshes_Val/{str(epoch)}/preds/{sample[8][0][:-4]}.ply')
                    mesh = trimesh.Trimesh(vertices[:,:,:3].cpu().detach().numpy()[0], template_mesh.faces)
                    mesh.export(f'{args.results_path}/Meshes_Val/targets/{sample[8][0][:-4]}.ply')

                    pbar_talk.set_description(
                        "(Epoch {}) VAL LOSS:{:.10f}".format((epoch + 1), (t_test_loss) / (b + 1)))
                    valid_loss_log.append(np.mean(t_test_loss))
                current_loss = np.mean(valid_loss_log)
                val_losses.append(current_loss)

        loss_log = []
        model.train()
        tloss = 0

        pbar_talk = tqdm(enumerate(dataset["train"]), total=len(dataset["train"]))
        for b, sample in pbar_talk:
            vertices = sample[0].to(args.device)
            template = sample[1].to(args.device)
            mass = sample[2].to(args.device)
            L = sample[3].to(args.device)
            evals = sample[4].to(args.device)
            evecs = sample[5].to(args.device)
            gradX = sample[6].to(args.device)
            gradY = sample[7].to(args.device)
            name = sample[8][0]
            faces = sample[9].to(args.device)
            mass_template = sample[10].to(args.device)
            L_template = sample[11].to(args.device)
            evals_template = sample[12].to(args.device)
            evecs_template = sample[13].to(args.device)
            gradX_template = sample[14].to(args.device)
            gradY_template = sample[15].to(args.device)
            faces_template = sample[16].to(args.device)

            if args.use_normals:
                normals, normals_template = sample[17].to(args.device), sample[18].to(args.device)
                in_features = torch.cat((vertices, normals), dim=2)
                in_features_template = torch.cat((template, normals_template), dim=2)
                vertices_pred, latent, pred_jac, pred_jac_restricted = model.forward_latent_njf(in_features_template, in_features, mass, L, evals, evecs,
                                                  gradX, gradY, faces,
                                                  mass_template, L_template, evals_template, evecs_template,
                                                  gradX_template, gradY_template, faces_template, name)
            else:
                vertices_pred, latent, pred_jac, pred_jac_restricted = model.forward_latent_njf(template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                                          mass_template, L_template, evals_template, evecs_template,
                                          gradX_template, gradY_template, faces_template, name)

            optim.zero_grad()
            #C, L, normals_faces = get_center_length_normal_batch(faces.to(dtype=torch.int32), vertices)
            #C_pred, L_pred, normals_faces_pred = get_center_length_normal_batch(faces.to(dtype=torch.int32), vertices_pred)
            pred_jac_restricted_batched = pred_jac_restricted.view(pred_jac_restricted.shape[0]*pred_jac_restricted.shape[1], pred_jac_restricted.shape[-2], pred_jac_restricted.shape[-1])
            print(pred_jac_restricted_batched.shape)
            normals_pred_batch = torch.cross(pred_jac_restricted_batched[:, :, 0], pred_jac_restricted_batched[:, :, 1])
            normals_pred = normals_pred_batch.view(args.batch_size, normals_faces.shape[1], normals_faces.shape[2])
            # areas = torch.det(torch.bmm(pred_jac_restricted_batched, pred_jac_restricted_batched.transpose(1, 2)))
            loss = criterion(vertices_pred, vertices) + 0.00001*(1 - criterion_n(normals_pred, normals_faces).mean())
            loss.backward()
            optim.step()
            tloss += loss.item()
            loss_log.append(loss.item())
            pbar_talk.set_description(
                "(Epoch {}) TRAIN LOSS:{:.10f}".format((epoch + 1), tloss / (b + 1)))
        train_losses.append(np.mean(loss_log))

        torch.save({'epoch': epoch,
                    'autoencoder_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    }, args.model_path)


def test(args):
    template_mesh = trimesh.load(args.template_file)
    dataset = get_dataloader(args)
    model = DiffusionNetAutoencoder(args).to(
        args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    metric = nn.MSELoss()

    epochs = checkpoint['epoch'] + 1
    print(epochs)

    model.eval()
    with torch.no_grad():
        t_test_loss = 0
        pbar_talk = tqdm(enumerate(dataset["test"]), total=len(dataset["test"]))
        for b, sample in pbar_talk:
            vertices = sample[0].to(args.device)
            template = sample[1].to(args.device)
            mass = sample[2].to(args.device)
            L = sample[3].to(args.device)
            evals = sample[4].to(args.device)
            evecs = sample[5].to(args.device)
            gradX = sample[6].to(args.device)
            gradY = sample[7].to(args.device)
            name = sample[8][0]
            faces = sample[9].to(args.device)
            mass_template = sample[10].to(args.device)
            L_template = sample[11].to(args.device)
            evals_template = sample[12].to(args.device)
            evecs_template = sample[13].to(args.device)
            gradX_template = sample[14].to(args.device)
            gradY_template = sample[15].to(args.device)
            faces_template = sample[16].to(args.device)
            if args.use_hks:
                hks = sample[10].to(args.device)
                vertices_pred = model.forward(template, hks, mass, L, evals, evecs, gradX, gradY, faces)
            else:
                if args.use_normals:
                    normals, normals_template = sample[17].to(args.device), sample[18].to(args.device)
                    in_features = torch.cat((vertices, normals), dim=2)
                    in_features_template = torch.cat((template, normals_template), dim=2)
                    vertices_pred, latent = model.forward_val_latent_njf(in_features_template, in_features, mass, L, evals, evecs,
                                                      gradX, gradY, faces,
                                                      mass_template, L_template, evals_template, evecs_template,
                                                      gradX_template, gradY_template, faces_template, name)
                else:
                    vertices_pred, latent = model.forward_val_latent_njf(template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                                                  mass_template, L_template, evals_template, evecs_template,
                                                  gradX_template, gradY_template, faces_template, name)
            t_test_loss += metric(vertices_pred, vertices).item()
            pbar_talk.set_description(
                "TEST LOSS:{:.10f}".format((t_test_loss) / (b + 1)))

            os.makedirs(f'{args.results_path}/Meshes_test', exist_ok=True)

            for i, name in enumerate(sample[8]):
                mesh = trimesh.Trimesh(vertices_pred[i, :, :3].detach().cpu().numpy(), template_mesh.faces)
                mesh.export(f'{args.results_path}/Meshes_test/' + str(name)[:-4] + '.ply')

            os.makedirs(f'{args.results_path}/Meshes_targets', exist_ok=True)
            for i, name in enumerate(sample[8]):
                mesh = trimesh.Trimesh(vertices[i, :, :3].detach().cpu().numpy(), template_mesh.faces)
                mesh.export(f'{args.results_path}/Meshes_targets/' + str(name)[:-4] + '.ply')


def interpolate(args):
    template_mesh = trimesh.load(args.template_file)
    n_point_template = np.array(template_mesh.vertices).shape[0]
    test_target_mesh = trimesh.load(args.test_target_mesh)
    verts_source, verts_target = np.array(template_mesh.vertices), np.array(test_target_mesh.vertices)
    template_mesh = trimesh.Trimesh(verts_source, template_mesh.faces)

    test_target_mesh = trimesh.Trimesh(verts_target, test_target_mesh.faces)

    verts_source, verts_target = torch.tensor(verts_source).to(device=args.device, dtype=torch.float), torch.tensor(verts_target).to(device=args.device, dtype=torch.float)

    model = DiffusionNetAutoencoder(args).to(
        args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    model.eval()
    with torch.no_grad():

        ### Source
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_source.to('cpu'), faces=torch.tensor(template_mesh.faces), k_eig=args.k_eig)
        mass_template = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals_template = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs_template = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L_template = L.float().to(args.device).unsqueeze(0)
        gradX_template = gradX.float().to(args.device).unsqueeze(0)
        gradY_template = gradY.float().to(args.device).unsqueeze(0)
        faces_template = torch.tensor(template_mesh.faces).to(args.device).float().unsqueeze(0)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        if args.use_hks:
            hks_test = torch.tensor(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)).to(
                args.device).float().unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(template_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_source = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_source, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_source = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass, L, evals, evecs,
                              gradX, gradY, faces,
                              mass_template, L_template, evals_template, evecs_template,
                              gradX_template, gradY_template, faces_template)
            else:
                z_source = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_source.unsqueeze(0), mass, L, evals,
                                                 evecs, gradX, gradY, faces,
                                                 mass_template, L_template, evals_template, evecs_template,
                                                 gradX_template, gradY_template, faces_template)

        ### Target
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_target.to('cpu'), faces=torch.tensor(test_target_mesh.faces), k_eig=args.k_eig)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        if args.use_hks:
            hks_test = torch.tensor(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)).to(
                args.device).float().unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(test_target_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_target = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(test_target_mesh.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_target, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_target = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass, L, evals, evecs,
                                                     gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)
            else:
                z_target = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_target.unsqueeze(0), mass, L,
                                                     evals,
                                                     evecs, gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)

        for f in glob.glob(f'{args.results_path}/interpolation/*'):
            os.remove(f)
        T = np.linspace(0, 1, 30)
        zT = [z_source]
        os.makedirs(f'{args.results_path}/cache', exist_ok=True)
        pred, _ = model.decode_njf(z_source, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0),
                                faces_template, "test.obj")
        os.makedirs(f'{args.results_path}/interpolation', exist_ok=True)
        mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
        mesh.export(f'{args.results_path}/interpolation/{str(0).zfill(3)}.ply')
        i = 1

        for t in T:
            zt = (1 - t) * z_source + t * z_target
            zT.append(zt)

            pred, _ = model.decode_njf(zt, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0),
                                    faces_template, "test.obj")
            os.makedirs(f'{args.results_path}/interpolation', exist_ok=True)
            mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
            mesh.export(f'{args.results_path}/interpolation/{str(i).zfill(3)}.ply')
            i += 1


def compute_norms(intrinsic_jacobians, vector_field, face_indices, bary_coords):
    """
    Given:
      - intrinsic_jacobians: tensor of shape (1, m, 3, 2) defined on face centroids,
      - vector_field: tensor of shape (1, N, 3) defined on vertices,
      - face_indices: tensor of shape (m, 3) defining each face by its vertex indices,
      - bary_coords: tensor of shape (m, 3, 2) containing the gradients of the barycentric coordinates per face,

    this function computes two scalar fields (each of shape (m,)):
      1. The Frobenius norm of the intrinsic jacobians.
      2. The Frobenius norm of the gradient of the vector field computed on the faces.

    Returns:
      norm_intrinsic, norm_gradient  -- each is a tensor of shape (m,)
    """
    # Compute Frobenius norm of intrinsic jacobians.
    # Square the entries, sum over the last two dimensions (3 and 2), then take square root.
    norm_intrinsic = torch.sqrt(torch.sum(intrinsic_jacobians ** 2, dim=(2, 3))).squeeze(0)  # shape: (m,)

    # Compute the gradient of the vector field on the faces.
    # This uses the mesh-dependent operator compute_face_gradient.
    face_gradient = compute_face_gradient(vector_field, face_indices, bary_coords)  # shape: (1, m, 3, 2)

    # Compute its Frobenius norm (over the 3×2 entries).
    norm_gradient = torch.sqrt(torch.sum(face_gradient ** 2, dim=(2, 3))).squeeze(0)  # shape: (m,)

    return norm_intrinsic, norm_gradient


def compute_bary_coords(vertex_positions, face_indices, eps=1e-8):
    """
    Compute the gradients of the barycentric coordinate functions for each face.

    Parameters:
        vertex_positions (torch.Tensor): Tensor of shape (1, N, 3) with vertex positions.
        face_indices (torch.LongTensor): Tensor of shape (m, 3) defining each triangular face
                                           by the indices of its three vertices.
        eps (float): A small number to avoid division by zero in degenerate cases.

    Returns:
        bary_coords (torch.Tensor): Tensor of shape (m, 3, 2) where, for each face,
                                    bary_coords[i, j, :] is the gradient (in a local 2D coordinate system)
                                    of the j-th barycentric coordinate function on that face.
    """
    # Remove batch dimension for indexing; vertices now has shape (N, 3)
    vertices = vertex_positions.squeeze(0)
    # Gather the vertex positions for each face; shape becomes (m, 3, 3)
    face_verts = vertices[face_indices]  # (m, 3, 3)

    # For each face, define a local 2D coordinate system.
    # Let v0 be the origin.
    v0 = face_verts[:, 0, :]  # (m, 3)
    v1 = face_verts[:, 1, :]  # (m, 3)
    v2 = face_verts[:, 2, :]  # (m, 3)

    # Create two edge vectors in 3D.
    p = v1 - v0  # (m, 3)
    q = v2 - v0  # (m, 3)

    # Define e1 as the normalized vector along p.
    L = torch.norm(p, dim=1) + eps  # (m,), add eps for numerical stability
    e1 = p / L.unsqueeze(1)  # (m, 3)

    # Project q onto e1 to get the coordinate 'a'
    a = torch.sum(q * e1, dim=1)  # (m,)

    # The component of q orthogonal to e1 gives the second local direction.
    q_perp = q - a.unsqueeze(1) * e1  # (m, 3)
    b = torch.norm(q_perp, dim=1) + eps  # (m,)

    # Now the local 2D coordinates of the vertices are:
    # u0 = (0, 0)
    # u1 = (L, 0)
    # u2 = (a, b)

    # The (unsigned) doubled area is L * b.
    denom = L * b  # (m,)

    # Compute the gradients of the barycentrics.
    # Using the standard formulas for a triangle in 2D:
    # grad(lambda0) = ((0 - b), (a - L)) / (L * b)
    grad_lambda0 = torch.stack((-b, a - L), dim=1) / denom.unsqueeze(1)  # (m, 2)
    # grad(lambda1) = ((b - 0), (0 - a)) / (L * b)
    grad_lambda1 = torch.stack((b, -a), dim=1) / denom.unsqueeze(1)  # (m, 2)
    # grad(lambda2) = ((0 - 0), (L - 0)) / (L * b) = (0, L) / (L*b) = (0, 1/b)
    grad_lambda2 = torch.stack((torch.zeros_like(b), L), dim=1) / denom.unsqueeze(1)  # (m, 2)

    # Stack the gradients for the three barycentrics to form a (m, 3, 2) tensor.
    bary_coords = torch.stack((grad_lambda0, grad_lambda1, grad_lambda2), dim=1)

    return bary_coords

def compute_face_gradient(vector_field, face_indices, bary_coords):
    """
    Dummy example function to compute the gradient of a vector field on faces.

    Parameters:
        vector_field (torch.Tensor): Tensor of shape (1, N, 3) defined on vertices.
        face_indices (torch.LongTensor): Tensor of shape (m, 3) with vertex indices for each face.
        bary_coords (torch.Tensor): Tensor of shape (m, 3, 2) containing, for each face,
                                    the (constant) gradients of the barycentric coordinate functions.
                                    Each face thus has three 2D vectors corresponding to grad(λ_i).

    Returns:
        torch.Tensor: A tensor of shape (1, m, 3, 2) representing the gradient (Jacobian) of the vector field on each face.

    Explanation:
        For a given face, if the vertex values are V_i (i=1,2,3) and the barycentric gradients are grad(λ_i),
        then the gradient on the face is computed as:

            grad_V = V_1 ⊗ grad(λ_1) + V_2 ⊗ grad(λ_2) + V_3 ⊗ grad(λ_3)

        where “⊗” denotes an outer product and the result is a 3×2 matrix.
    """
    # Number of faces (m)
    m = face_indices.shape[0]

    # Gather the vertex values for each face.
    # vector_field has shape (1, N, 3); face_indices has shape (m, 3).
    # We want face_vals of shape (1, m, 3, 3): for each face, three vertices each with 3 components.
    print(face_indices)
    face_vals = vector_field[:, face_indices, :]  # shape: (1, m, 3, 3)
    print(face_vals.shape)

    # Now, bary_coords is assumed to be (m, 3, 2). We need to combine the vertex values with these.
    # For each face, perform a weighted sum: for each vertex i, multiply V_i (a 3-vector) by grad(λ_i) (a 2-vector)
    # to obtain a 3x2 contribution, then sum over i.
    # One way is to do an elementwise multiplication and sum over the vertex dimension.
    # We can add a dimension to bary_coords to broadcast: (1, m, 3, 2)
    bary_coords_expanded = bary_coords.unsqueeze(0)  # shape: (1, m, 3, 2)

    # To combine the vertex values (1, m, 3, 3) with bary_coords (1, m, 3, 2),
    # we want an outer product for each vertex. One approach is to unsqueeze the vertex values:
    face_vals_expanded = face_vals.unsqueeze(-1)  # shape: (1, m, 3, 3, 1)
    # Now multiply with bary_coords_expanded (1, m, 3, 1, 2)
    bary_coords_expanded = bary_coords_expanded.unsqueeze(-2)  # shape: (1, m, 3, 1, 2)

    # The product has shape: (1, m, 3, 3, 2)
    face_grad_contributions = face_vals_expanded * bary_coords_expanded
    # Sum over the vertex (barycentric) index (dimension 2):
    face_gradient = face_grad_contributions.sum(dim=2)  # shape: (1, m, 3, 2)

    return face_gradient

def generate_array(N, indices):
    # Create an array of zeros of length N
    arr = np.zeros(N, dtype=int)
    # Set specified indices to 1
    for i in indices:
        if 0 <= i < N:
            arr[i] = 1
        else:
            print(f"Warning: Index {i} is out of bounds for array of size {N}.")
    return arr


def partial_interpolate(args):
    #idx = [199, 161, 162, 163, 198, 160, 291, 289, 290, 276, 247, 293, 292, 202, 201, 198, 197, 141, 76, 278]
    #id_out = np.arange(468, 696)  #np.array(list(np.arange(480, 696))) #np.arange(468, 696)#np.arange(580, 696)  #np.arange(468, 696)
    #id_out = np.load("./FAUST/idx_left_arm.npy")

    template_mesh = trimesh.load(args.template_file)
    test_target_mesh = trimesh.load(args.test_target_mesh)
    verts_source, verts_target = np.array(template_mesh.vertices), np.array(test_target_mesh.vertices)

    # --- Define the 3D box bounds ---
    x_min, x_max = 0.14, 10  # 0.01, 1  # -1, -0.015
    y_min, y_max = 0, 10
    z_min, z_max = -10, 10

    # --- Select vertices inside the box ---
    in_box = np.logical_and.reduce((
        verts_source[:, 0] >= x_min, verts_source[:, 0] <= x_max,
        verts_source[:, 1] >= y_min, verts_source[:, 1] <= y_max,
        verts_source[:, 2] >= z_min, verts_source[:, 2] <= z_max
    ))
    selected_vertex_indices = np.where(in_box)[0]
    mask = np.zeros(verts_source.shape[0])
    mask[selected_vertex_indices] = 1
    np.save("../RoNeD/FAUST/mask_arm.npy", mask)

    selected_face_region = []

    for i, face in enumerate(template_mesh.faces):
        if np.intersect1d(face, selected_vertex_indices).size > 0:
            selected_face_region.append(i)

    id_out = np.array(selected_face_region)

    #np.arange(468, 696)
    #id_out = np.arange(773,1230) #1230  #for face indices
    #id_out = np.array(list(np.arange(0, 773)) + list(np.arange(1230, 1310)))
    #id_out = np.array(list(np.arange(0, 480)) + list(np.arange(696, 778)))

    template_mesh = trimesh.Trimesh(verts_source, template_mesh.faces)
    test_target_mesh = trimesh.Trimesh(verts_target, test_target_mesh.faces)

    verts_source, verts_target = torch.tensor(verts_source).to(device=args.device, dtype=torch.float), torch.tensor(
        verts_target).to(device=args.device, dtype=torch.float)

    model = DiffusionNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    model.eval()
    with torch.no_grad():

        ### Source
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_source.to('cpu'), faces=torch.tensor(template_mesh.faces), k_eig=args.k_eig)
        mass_template = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals_template = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs_template = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L_template = L.float().to(args.device).unsqueeze(0)
        gradX_template = gradX.float().to(args.device).unsqueeze(0)
        gradY_template = gradY.float().to(args.device).unsqueeze(0)
        faces_template = torch.tensor(template_mesh.faces).to(args.device).float().unsqueeze(0)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        if args.use_hks:
            hks_test = torch.tensor(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)).to(
                args.device).float().unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(template_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_source = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_source, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_source = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass,
                                                     L, evals, evecs,
                                                     gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)
            else:
                z_source = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_source.unsqueeze(0), mass, L,
                                                     evals,
                                                     evecs, gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)

        ### Target
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_target.to('cpu'), faces=torch.tensor(test_target_mesh.faces), k_eig=args.k_eig)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(test_target_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_target = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(test_target_mesh.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_target, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_target = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass,
                                                     L, evals, evecs,
                                                     gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)
            else:
                z_target = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_target.unsqueeze(0), mass, L,
                                                     evals,
                                                     evecs, gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)

        z_target_partial = torch.reshape(z_source.clone(), (1, args.n_faces, args.latent_channels*3))
        z_target = torch.reshape(z_target, (1, args.n_faces, args.latent_channels*3))
        z_target_partial[:, id_out, :] = z_target[:, id_out, :]

        for f in glob.glob(f'{args.results_path}/partial_interpolation/*'):
            os.remove(f)
        T = np.linspace(0, 1, 30)
        zT = [z_source]
        pred, pred_jac_restricted = model.decode_njf(z_source, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0),
                                faces_template, "test.obj")


        os.makedirs(f'{args.results_path}/partial_interpolation', exist_ok=True)
        mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
        mesh.export(f'{args.results_path}/partial_interpolation/{str(0).zfill(3)}.ply')
        i = 1

        for t in T:
            zt = (1 - t) * z_source + t * z_target_partial
            zT.append(zt)

            pred, pred_jac_restricted = model.decode_njf(zt, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0),
                                    faces_template, "test.obj")

            os.makedirs(f'{args.results_path}/partial_interpolation', exist_ok=True)
            mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
            mesh.export(f'{args.results_path}/partial_interpolation/{str(i).zfill(3)}.ply')
            i += 1

        bary_coords = compute_bary_coords(torch.tensor(template_mesh.vertices).to(device=args.device, dtype=torch.float32).unsqueeze(0), torch.tensor(template_mesh.faces).to(device=args.device, dtype=torch.int))
        norm_intrinsic, norm_gradient = compute_norms(pred_jac_restricted, pred - verts_source, torch.tensor(template_mesh.faces).to(device=args.device, dtype=torch.int), bary_coords)
        np.save("./def_norm/norm_jac.npy", norm_intrinsic.detach().cpu().numpy())
        np.save("./def_norm/norm_grad_def.npy", norm_gradient.detach().cpu().numpy())
        mask = generate_array(1538, id_out)
        #mask = generate_array(10000, id_out)
        np.save("./def_norm/mask.npy", mask)

        i = 1

        for t in T:
            zt = (1 - t) * z_target_partial + t * z_target
            zT.append(zt)

            pred, pred_jac_restricted = model.decode_njf(zt, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0),
                                    faces_template, "test.obj")
            os.makedirs(f'{args.results_path}/partial_interpolation', exist_ok=True)
            mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
            mesh.export(f'{args.results_path}/partial_interpolation/{str(30 + i).zfill(3)}.ply')
            i += 1


def test_unregistered(args):
    os.makedirs(f'{args.results_path}/Meshes_test_scan', exist_ok=True)
    for f in glob.glob(f'{args.results_path}/Meshes_test_scan/*'):
        os.remove(f)
    os.makedirs(f'{args.results_path}/Meshes_targets_scan', exist_ok=True)
    for f in glob.glob(f'{args.results_path}/Meshes_targets_scan/*'):
        os.remove(f)

    template_base = trimesh.load(args.template_file)
    template_unseen = trimesh.load(args.template_unseen)
    #template_unseen = trimesh.Trimesh(align_MANO(template_unseen, template_base), template_unseen.faces)
    template_unseen_verts = torch.tensor(np.array(template_unseen.vertices)).to(device=args.device, dtype=torch.float)
    template_unseen.export(f'{args.results_path}/Meshes_targets_scan/template.ply')

    model = DiffusionNetAutoencoder(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    model.eval()
    with torch.no_grad():
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            template_unseen_verts.to('cpu'), faces=torch.tensor(template_unseen.faces), k_eig=args.k_eig)
        mass_template = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals_template = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs_template = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L_template = L.float().to(args.device).unsqueeze(0)
        gradX_template = gradX.float().to(args.device).unsqueeze(0)
        gradY_template = gradY.float().to(args.device).unsqueeze(0)
        faces_template = torch.tensor(template_unseen.faces).to(args.device).float().unsqueeze(0)
        for i, elt in enumerate(os.listdir(args.scans_dir)):
            mesh = trimesh.load(os.path.join(args.scans_dir, elt))
            #mesh_verts = align_MANO(mesh, template_base)
            #mesh = trimesh.Trimesh(mesh_verts, mesh.faces)
            vertices = torch.tensor(np.array(mesh.vertices)).to(device=args.device, dtype=torch.float)

            mesh_target = trimesh.Trimesh(vertices[:, :3].detach().cpu().numpy(), mesh.faces)
            mesh_target.export(f'{args.results_path}/Meshes_targets_scan/' + str(elt) + '.ply')

            frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
                vertices.to('cpu'), faces=torch.tensor(mesh.faces), k_eig=args.k_eig)
            mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
            if args.use_hks:
                hks_test = torch.tensor(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)).to(
                    args.device).float().unsqueeze(0)
            evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
            evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
            L = L.float().to(args.device).unsqueeze(0)
            gradX = gradX.float().to(args.device).unsqueeze(0)
            gradY = gradY.float().to(args.device).unsqueeze(0)
            faces = torch.tensor(mesh.faces).to(args.device).float().unsqueeze(0)

            if args.use_hks:
                z_source = model.forward(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
            else:
                if args.use_normals:
                    normals = torch.FloatTensor(np.array(mesh.vertex_normals)).to(args.device)
                    normals_template = torch.FloatTensor(np.array(template_unseen.vertex_normals)).to(args.device)

                    in_features = torch.cat((vertices, normals), dim=1)
                    in_features_template = torch.cat((template_unseen_verts, normals_template), dim=1)

                    pred, _ = model.forward_val_latent_njf(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass,
                                                         L, evals, evecs,
                                                         gradX, gradY, faces,
                                                         mass_template, L_template, evals_template, evecs_template,
                                                         gradX_template, gradY_template, faces_template, "test.obj")
                else:
                    pred, _ = model.forward_val_latent_njf(template_unseen_verts.unsqueeze(0), vertices.unsqueeze(0), mass, L,
                                             evals,
                                             evecs, gradX, gradY, faces,
                                             mass_template, L_template, evals_template, evecs_template,
                                             gradX_template, gradY_template, faces_template, "test.obj")


                mesh_pred = trimesh.Trimesh(pred[0, :, :3].detach().cpu().numpy(), template_unseen.faces)
                mesh_pred.export(f'{args.results_path}/Meshes_test_scan/' + str(elt) + '.ply')


def partial_interpolate2(args):
    # id_out = np.load("./MANO/idx_faces_pinky.npy")  # pinky
    # id_out = np.arange(773,1001)    #ring
    # id_out = np.load("./MANO/idx_faces_index.npy")
    # id_out = np.load("./MANO/idx_faces_middle.npy")
    # id_out = np.load("./MANO/idx_faces_thumb.npy")
    #id_out = np.array(list(np.load("./MANO/idx_faces_thumb.npy")) + list(np.load("./MANO/idx_faces_middle.npy")))
    #id_out2 = np.array(list(np.load("./MANO/idx_faces_pinky.npy")) + list(np.arange(773,1001)))

    template_mesh = trimesh.load(args.template_file)
    test_target_mesh, test_target_mesh2 = trimesh.load(args.test_target_mesh), trimesh.load(args.test_target_mesh2)
    verts_source, verts_target, verts_target2 = np.array(template_mesh.vertices), np.array(test_target_mesh.vertices), np.array(test_target_mesh2.vertices)

    verts_source, verts_target, verts_target2 = torch.tensor(verts_source).to(device=args.device, dtype=torch.float), torch.tensor(
        verts_target).to(device=args.device, dtype=torch.float), torch.tensor(
        verts_target2).to(device=args.device, dtype=torch.float)

    model = DiffusionNetAutoencoder(args).to(
        args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(checkpoint['autoencoder_state_dict'])
    model.eval()
    with torch.no_grad():

        ### Source
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_source.to('cpu'), faces=torch.tensor(template_mesh.faces), k_eig=args.k_eig)
        mass_template = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals_template = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs_template = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L_template = L.float().to(args.device).unsqueeze(0)
        gradX_template = gradX.float().to(args.device).unsqueeze(0)
        gradY_template = gradY.float().to(args.device).unsqueeze(0)
        faces_template = torch.tensor(template_mesh.faces).to(args.device).float().unsqueeze(0)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        if args.use_hks:
            hks_test = torch.tensor(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)).to(
                args.device).float().unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(template_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_source = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_source, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_source = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass,
                                                     L, evals, evecs,
                                                     gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)
            else:
                z_source = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_source.unsqueeze(0), mass, L,
                                                     evals,
                                                     evecs, gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)


        ### Target
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_target.to('cpu'), faces=torch.tensor(test_target_mesh.faces), k_eig=args.k_eig)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        if args.use_hks:
            hks_test = torch.tensor(diffusion_net.geometry.compute_hks_autoscale(evals, evecs, count=16)).to(
                args.device).float().unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(test_target_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_target = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(test_target_mesh.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_target, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_target = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0), mass,
                                                     L, evals, evecs,
                                                     gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)
            else:
                z_target = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_target.unsqueeze(0), mass, L,
                                                     evals,
                                                     evecs, gradX, gradY, faces,
                                                     mass_template, L_template, evals_template, evecs_template,
                                                     gradX_template, gradY_template, faces_template)

        z_target_partial = torch.reshape(z_source.clone(), (1, args.n_faces, args.latent_channels*3))
        z_target = torch.reshape(z_target, (1, args.n_faces, args.latent_channels*3))
        z_target_partial[:, id_out, :] = z_target[:, id_out, :]

        ### Target 2
        frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.compute_operators(
            verts_target2.to('cpu'), faces=torch.tensor(test_target_mesh.faces), k_eig=args.k_eig)
        mass = torch.FloatTensor(np.array(mass)).float().to(args.device).unsqueeze(0)
        evals = torch.FloatTensor(np.array(evals)).to(args.device).unsqueeze(0)
        evecs = torch.FloatTensor(np.array(evecs)).to(args.device).unsqueeze(0)
        L = L.float().to(args.device).unsqueeze(0)
        gradX = gradX.float().to(args.device).unsqueeze(0)
        gradY = gradY.float().to(args.device).unsqueeze(0)
        faces = torch.tensor(test_target_mesh.faces).to(args.device).float().unsqueeze(0)

        if args.use_hks:
            z_target = model.get_latent_features(hks_test, mass, L, evals, evecs, gradX, gradY, faces)
        else:
            if args.use_normals:
                normals = torch.FloatTensor(np.array(test_target_mesh2.vertex_normals)).to(args.device)
                normals_template = torch.FloatTensor(np.array(template_mesh.vertex_normals)).to(args.device)

                in_features = torch.cat((verts_target2, normals), dim=1)
                in_features_template = torch.cat((verts_source, normals_template), dim=1)

                z_target2 = model.get_latent_features_evecs(in_features_template.unsqueeze(0), in_features.unsqueeze(0),
                                                           mass,
                                                           L, evals, evecs,
                                                           gradX, gradY, faces,
                                                           mass_template, L_template, evals_template, evecs_template,
                                                           gradX_template, gradY_template, faces_template)
            else:
                z_target = model.get_latent_features_evecs(verts_source.unsqueeze(0), verts_target.unsqueeze(0), mass,
                                                           L,
                                                           evals,
                                                           evecs, gradX, gradY, faces,
                                                           mass_template, L_template, evals_template, evecs_template,
                                                           gradX_template, gradY_template, faces_template)

        z_target_partial2 = torch.reshape(z_target_partial.clone(), (1, args.n_faces, args.latent_channels * 3))
        z_target2 = torch.reshape(z_target2, (1, args.n_faces, args.latent_channels * 3))
        z_target_partial2[:, id_out2, :] = z_target2[:, id_out2, :]

        for f in glob.glob(f'{args.results_path}/partial_interpolation/*'):
            os.remove(f)
        T = np.linspace(0, 1, 30)
        zT = [z_source]
        pred = model.decode_njf(z_source, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0), faces_template
                               , "test.obj")

        os.makedirs(f'{args.results_path}/partial_interpolation', exist_ok=True)
        mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
        mesh.export(f'{args.results_path}/partial_interpolation/{str(0).zfill(3)}.ply')
        i = 1

        for t in T:
            zt = (1 - t) * z_source + t * z_target_partial2
            zT.append(zt)
            pred = model.decode_njf(zt, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0), faces_template
                                   , "test.obj")

            os.makedirs(f'{args.results_path}/partial_interpolation', exist_ok=True)
            mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
            mesh.export(f'{args.results_path}/partial_interpolation/{str(i).zfill(3)}.ply')
            i += 1

        i = 1

        for t in T:
            zt = (1 - t) * z_target_partial + t * z_target
            zT.append(zt)

            pred = model.decode_njf(zt, torch.tensor(template_mesh.vertices).to(args.device).float().unsqueeze(0),
                                    faces_template, "test.obj")
            os.makedirs(f'{args.results_path}/partial_interpolation', exist_ok=True)
            mesh = trimesh.Trimesh(pred[0].detach().cpu().numpy(), template_mesh.faces)
            mesh.export(f'{args.results_path}/partial_interpolation/{str(30 + i).zfill(3)}.ply')
            i += 1


def main():
    parser = argparse.ArgumentParser(description='PaNDaS')

    parser.add_argument("--use_pretrained", type=bool, default=False)
    # learning args
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=float, default=300)
    parser.add_argument('--batch_size', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")

    # data args
    parser.add_argument('--template_file', type=str,
                        default='../datasets/MANO_ALIGNED/04_01r.ply')  # './MANO/template_MANO.ply')
    parser.add_argument('--templates_dir', type=str, default='./MANO/templates_aligned')
    parser.add_argument('--template_unseen', type=str, default='./MANO/templates_aligned/49_01r.ply')
    parser.add_argument('--meshes_path', type=str, default='../datasets/MANO_ALIGNED')
    parser.add_argument('--scans_dir', type=str, default='../datasets/MANO_SCANS_test_dir_49')
    parser.add_argument('--unseen_meshes_dir', type=str, default='/media/tbesnier/T5 EVO/datasets/MANO_UNSEEN')
    parser.add_argument('--test_target_mesh', type=str, default='../datasets/MANO_ALIGNED/04_18r.ply')

    parser.add_argument('--test_target_mesh2', type=str, default='../datasets/MANO_ALIGNED/01_18r.ply')
    parser.add_argument('--train_subjects', type=str, default="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18"
                                                                            " 19 20 21 22 23 24 25 26 27 28 29 30 31"
                                                                            " 32 33 34 35 36 37 38 39 40 41 42 43 44 45"
                                                                            " 46 47 48 49 50")
    parser.add_argument('--val_subjects', type=str, default="49 50")
    parser.add_argument('--test_subjects', type=str, default="49 50")
    parser.add_argument('--results_path', type=str, default="../Data/LISC/results_default")

    # checkpoint args
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--models_dir", type=str, default="../Data/LISC/Models")
    parser.add_argument("--model_path", type=str, default="../Data/LISC/Models/LISC_MANO_njf_default.pth.tar")

    # model hyperparameters
    parser.add_argument('--latent_channels', type=int, default=42)
    parser.add_argument('--use_hks', type=bool, default=False)
    parser.add_argument('--use_normals', type=bool, default=True)
    parser.add_argument('--in_channels', type=int, default=6)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--k_eig', type=int, default=16)

    parser.add_argument('--n_points', type=int, default=5038)   #778
    parser.add_argument('--n_faces', type=int, default=1538)  #1538

    parser.add_argument('--batchnorm_encoder', type=str, default="GROUPNORM")
    parser.add_argument('--batchnorm_decoder', type=str, default="GROUPNORM")
    parser.add_argument('--shuffle_triangles', type=bool, default=False)

    args = parser.parse_args()

    #train(args)

    #test(args)

    #test_unregistered(args)

    interpolate(args)

    partial_interpolate(args)

    #compute_latent_mean(args)

if __name__ == "__main__":
    main()

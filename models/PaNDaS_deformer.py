from __future__ import annotations
import sys
import torch
import torch.nn as nn
import models.diffusion_net as diffusion_net

sys.path.append("./models/njf")
from net import njf_decoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, args):
        super(DiffusionNetAutoencoder, self).__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.latent_channels = args.latent_channels
        self.n_faces = args.n_faces
        self.device = args.device
        self.bs = args.batch_size

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                         C_out=self.latent_channels,
                                                         C_width=128,
                                                         N_block=4,
                                                         outputs_at='faces')
        self.encoder_def = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                             C_out=self.latent_channels,
                                                             C_width=128,
                                                             N_block=4,
                                                             outputs_at='global_mean')

        # decoder
        self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, 2*self.latent_channels), args=args)


    def forward(self, verts_src, mass_src, L_src, evals_src, evecs_src, gradX_src, gradY_src, faces_src,
                verts_tgt, mass_tgt, L_tgt, evals_tgt, evecs_tgt, gradX_tgt, gradY_tgt, faces_tgt,
                poisson_solver=None):

        z_template = self.encoder(verts_src, mass=mass_src, L=L_src, evals=evals_src,
                                  evecs=evecs_src,
                                  gradX=gradX_src, gradY=gradY_src, faces=faces_src)

        z = self.encoder_def(verts_tgt, mass=mass_tgt, L=L_tgt, evals=evals_tgt,
                                  evecs=evecs_tgt,
                                  gradX=gradX_tgt, gradY=gradY_tgt, faces=faces_tgt)
        z_ref = self.encoder_def(verts_src, mass=mass_src, L=L_src, evals=evals_src,
                                  evecs=evecs_src,
                                  gradX=gradX_src, gradY=gradY_src, faces=faces_src)
        z = z - z_ref
        z = z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))
        feat_field = torch.cat((z_template, z), dim=-1)

        # NJF decoder
        delta, pred_jac = self.decoder.predict_map(feat_field, source_verts=verts_src, source_faces=faces_src,
                                                   L=L_src, poisson_solver=poisson_solver)
        delta, pred_jac = delta.to(self.device), pred_jac.to(self.device)


        pred = delta + verts_src[:, :, :3]


        return pred
import sys, os
import torch
import torch.nn as nn
import models.diffusion_net as diffusion_net
sys.path.append("./models/njf")
from models.njf.net import njf_decoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DiffusionNetAutoencoder(nn.Module):
    def __init__(self, args):
        super(DiffusionNetAutoencoder, self).__init__()
        if args.use_hks:
            self.in_channels = 16
        else:
            self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.latent_channels = args.latent_channels
        self.device = args.device
        self.bs = args.batch_size
        self.n_faces = args.n_faces
        self.dataset = args.dataset

        # encoder
        self.encoder = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                         C_out=self.latent_channels // 2,
                                                         C_width=128,  # self.latent_channels*2,
                                                         N_block=4,
                                                         outputs_at='faces',
                                                         dropout=False,
                                                         normalization="None")
        self.encoder_def = diffusion_net.layers.DiffusionNet(C_in=self.in_channels,
                                                             C_out=self.latent_channels // 2,
                                                             C_width=128,  # self.latent_channels*2,
                                                             N_block=4,
                                                             outputs_at='global_mean',
                                                             dropout=False,
                                                             normalization="None")

        # decoder
        self.decoder = njf_decoder(latent_features_shape=(self.bs, self.n_faces, self.latent_channels), args=args)

        print("encoder parameters: ", count_parameters(self.encoder))
        print("decoder parameters: ", count_parameters(self.decoder))

    def forward_latent_njf(self, template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template,
                           name):
        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template, evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = self.encoder_def(vertices, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        z_ref = self.encoder_def(template, mass=mass_template, L=L_template, evals=evals_template,
                                 evecs=evecs_template, gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = z - z_ref
        z = z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))

        #cat_latent = self.cma(z_template.squeeze(0), z.squeeze(0)).unsqueeze(0)

        cat_latent = torch.cat((z_template, z), dim=-1)

        delta, pred_jac, pred_jac_restricted = self.decoder.predict_map(cat_latent, source_verts=template, source_faces=faces_template,
                                        batch=False, target_vertices=vertices)
        delta, pred_jac, pred_jac_restricted = delta.to(self.device), pred_jac.to(self.device), pred_jac_restricted.to(self.device)
        pred = delta + template[:, :, :3]
        return pred, cat_latent, pred_jac, pred_jac_restricted

    def forward_val_latent_njf(self, template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template,
                               name):
        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template, evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = self.encoder_def(vertices, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        z_ref = self.encoder_def(template, mass=mass_template, L=L_template, evals=evals_template,
                                 evecs=evecs_template, gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = z - z_ref

        z = z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))
        cat_latent = torch.cat((z_template, z), dim=-1)  ## B x F x nb_features
        #cat_latent = self.cma(z_template.squeeze(0), z.squeeze(0)).unsqueeze(0)

        delta = self.decoder.predict_map(cat_latent, source_verts=template, source_faces=faces_template,
                                         target_vertices=None)[0].to(self.device)

        pred = delta + template[:, :, :3]
        return pred, cat_latent

    def get_latent_features(self, template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template):
        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template,
                                  evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = self.encoder_def(vertices, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        z_ref = self.encoder_def(template, mass=mass_template, L=L_template, evals=evals_template,
                                 evecs=evecs_template, gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = z - z_ref
        cat_latent = torch.cat([z_template, z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))], dim=-1)
        #cat_latent = self.cma(z_template.squeeze(0), z.squeeze(0)).unsqueeze(0)
        return cat_latent

    def get_latent_features_evecs(self, template, vertices, mass, L, evals, evecs, gradX, gradY, faces,
                mass_template, L_template, evals_template, evecs_template, gradX_template, gradY_template, faces_template):
        z_template = self.encoder(template, mass=mass_template, L=L_template, evals=evals_template, evecs=evecs_template,
                                  gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = self.encoder_def(vertices, mass=mass, L=L, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY, faces=faces)
        z_ref = self.encoder_def(template, mass=mass_template, L=L_template, evals=evals_template,
                                 evecs=evecs_template, gradX=gradX_template, gradY=gradY_template, faces=faces_template)
        z = z - z_ref

        z = z.unsqueeze(1).expand((z_template.shape[0], z_template.shape[1], z.shape[-1]))
        #cat_latent = self.cma(z_template.squeeze(0), z.squeeze(0)).unsqueeze(0)

        cat_latent = torch.cat((z_template, z), dim=-1)

        return cat_latent

    def decode_njf(self, z, template, faces_template, name):

        delta, pred_jac, pred_jac_restricted = self.decoder.predict_map(z, source_verts=template, source_faces=faces_template)
        delta = delta.to(self.device)
        pred_jac_restricted = pred_jac_restricted.to(self.device)

        return delta + template[:, :, :3], pred_jac_restricted
import os

import numpy
import torch
import igl
import MeshProcessor

WKS_DIM = MeshProcessor.WKS_DIM
WKS_FACTOR = 1000


class SourceMesh:
    '''
    datastructure for the source mesh to be mapped
    '''

    def __init__(self, source_v, source_f, use_wks=False, random_centering=False,
                 cpuonly=False, L=None):
        self.__use_wks = use_wks
        self.source_v = source_v
        self.source_f = source_f
        self.centroids_and_normals = None
        self.center_source = False
        self.poisson = None
        self.__extra_keys = {'samples': True, 'samples_normals': True, 'samples_wks': True}
        self.__source_global_translation_to_original = 0
        self.__loaded_data = {}
        self.random_centering = random_centering
        self.source_mesh_centroid = None
        self.mesh_processor = None
        self.cpuonly = cpuonly
        self.L = L

    def get_vertices(self):
        return self.source_vertices

    def get_global_translation_to_original(self):
        return self.__source_global_translation_to_original

    def vertices_from_jacobians(self, d):
        return self.poisson.solve_poisson(d)

    def jacobians_from_vertices(self, v):
        return self.poisson.jacobians_from_vertices(v)

    def restrict_jacobians(self, J):
        return self.poisson.restrict_jacobians(J)

    def get_loaded_data(self, key: str):

        return self.__loaded_data.get(key)

    def get_source_triangles(self):

        return self.mesh_processor.get_faces()

    def to(self, device):
        self.poisson = self.poisson.to(device)
        self.centroids_and_normals = self.centroids_and_normals.to(device)
        for key in self.__loaded_data.keys():
            self.__loaded_data[key] = self.__loaded_data[key].to(device)
        return self

    def __init_from_mesh_data(self):
        assert self.mesh_processor is not None
        self.mesh_processor.prepare_differential_operators_for_use()  # call 1
        self.source_vertices = torch.from_numpy(self.mesh_processor.get_vertices()).type(torch.double)
        centroids = self.mesh_processor.get_centroids()
        centroid_points_and_normals = centroids.points_and_normals
        self.centroids_and_normals = torch.from_numpy(centroid_points_and_normals).type(torch.double)
        self.poisson = self.mesh_processor.diff_ops.poisson_solver

    def load(self, source_v=None, source_f=None, L=None, poisson_solver=None):
        # mesh_data = SourceMeshData.SourceMeshData.meshprocessor_from_file(self.source_dir)

        # Fast path: reuse a precomputed PoissonSolver (e.g. from the dataloader)
        # This avoids recomputing igl.grad / Laplacian / RHS inside MeshProcessor.
        if poisson_solver is not None:
            self.poisson = poisson_solver
            self.source_vertices = torch.as_tensor(source_v, dtype=torch.double)
            # keep a copy of faces if needed for downstream utilities
            self.source_f = torch.as_tensor(source_f, dtype=torch.int64) if source_f is not None else self.source_f
            return self

        if source_v is not None and source_f is not None:
            self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_array(source_v, source_f, ttype=torch.double,
                                                                                       cpuonly=self.cpuonly,
                                                                                       load_wks_samples=self.__use_wks,
                                                                                       load_wks_centroids=self.__use_wks, L=L)
        else:
            if os.path.isdir(self.source_dir):
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_directory(self.source_dir,
                                                                                               torch.double,
                                                                                               cpuonly=self.cpuonly,
                                                                                               load_wks_samples=self.__use_wks,
                                                                                               load_wks_centroids=self.__use_wks)
            else:
                self.mesh_processor = MeshProcessor.MeshProcessor.meshprocessor_from_file(self.source_dir, torch.double,
                                                                                          cpuonly=self.cpuonly,
                                                                                          load_wks_samples=self.__use_wks,
                                                                                          load_wks_centroids=self.__use_wks)
        self.__init_from_mesh_data()

    def get_point_dim(self):
        return self.centroids_and_normals.shape[1]

    def get_centroids_and_normals(self):
        return self.centroids_and_normals

    def get_mesh_centroid(self):
        return self.source_mesh_centroid


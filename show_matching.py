import polyscope as ps
import polyscope.imgui as psim
import trimesh
import trimesh as tri
import numpy as np
import os

ui_int = 0
colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    'burlywood', 'cadetblue',
    'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
    'cornsilk', 'crimson', 'cyan', 'darkblue',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
    'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
    'darkslateblue', 'darkslategray', 'darkslategrey',
    'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', '#000000'
]

def register_surface(name, mesh, x=0.0, y=0.0, z=0.0, idx_color=1, transparency=1.0, disp_vectors=None, disp_heatmap=None, scale_vectors = 0.5):
    vertices, faces = np.array(mesh.vertices), np.array(mesh.faces)
    vertices = vertices + np.stack((x*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), y*np.ones((vertices.shape[0],1)), np.zeros((vertices.shape[0],1))), axis=1)[:,:,0]
    vertices = vertices + np.stack((np.zeros((vertices.shape[0],1)), np.zeros((vertices.shape[0],1)), z*np.ones((vertices.shape[0],1))), axis=1)[:,:,0]

    mesh = ps.register_surface_mesh(name, vertices, faces, edge_width=0.0)
    mesh.set_color(tuple(int(colors[idx_color][i:i + 2], 16) / 255.0 for i in (1, 3, 5)))
    mesh.set_smooth_shade(False)
    mesh.set_transparency(transparency)

    if disp_vectors is not None:
        mesh.add_vector_quantity("displacement vectors", scale_vectors * disp_vectors, enabled=True,
                                 color=tuple(int(colors[-1][i:i + 2], 16) / 255.0 for i in (1, 3, 5)), vectortype="ambient")

    if disp_heatmap is not None:
        min_bound, max_bound = disp_heatmap.min(), disp_heatmap.max()  #
        mesh.add_scalar_quantity('Varifold signature', disp_heatmap, defined_on='vertices', enabled=True, cmap='blues', vminmax=(min_bound, max_bound))

    return mesh

# Define our callback function, which Polyscope will repeatedly execute while running the UI.
def callback():

    global ui_int, meshes, heatmap, displacement_vectors

    # == Settings

    # Note that it is a push/pop pair, with the matching pop() below.
    psim.PushItemWidth(150)

    # == Show text in the UI

    psim.TextUnformatted("Sequence of meshes")
    psim.TextUnformatted("Sequence length: {}".format(len(meshes)))
    psim.Separator()

    # Input Int Slider
    changed, ui_int = psim.SliderInt("Frame", ui_int, v_min=0, v_max=len(meshes)-1)
    if changed:
        ps.remove_all_structures()
        register_surface(name=f'Step {ui_int} Ours', mesh=meshes[ui_int], idx_color=1, disp_vectors=None, disp_heatmap=heatmap)


if __name__ == '__main__':
    GT = False
    render_vid = False

    meshes_dir = "./DFAUST/templates_dir"#'../Data/LISC/results_default_DFAUST/partial_interpolation'
    #meshes_dir = "../Data/LISC/results_default_COMA/interpolation_multiple/interp06"
    #meshes_dir = "../Data/LISC/comparisons/scans/DFAUST/NJF/preds/50027_jumping_jacks"
    #meshes_dir = "../../papers/PaNDaS/comparison_videos/partial_MANO_scan"
    l_mesh_dir = os.listdir(meshes_dir)
    l_mesh_dir.sort()

    meshes = [tri.load(os.path.join(meshes_dir, l_mesh_dir[i])) for i in range(len(l_mesh_dir)) if 'obj' in l_mesh_dir[i]]

    #mesh_ref = trimesh.load(os.path.join("../Data/LISC/results_dual_MANO_stats/partial_component/", "50_01r.ply"))

    #displacement_vectors = np.array(meshes[10].vertices - meshes[0].vertices)
    #displacement_vectors = [np.array(meshes[i].vertices - mesh_ref.vertices) for i in range(len(l_mesh_dir))]
    #heatmap = np.linalg.norm(np.array(meshes[16].vertices - meshes[0].vertices), axis=1)  #feature_field
    #heatmap = np.load("../RoNeD/FAUST/mask_arm_test.npy")
    # thumb = np.load("./MANO/idx_faces_thumb.npy")
    # index = np.load("./MANO/idx_faces_index.npy")
    # middle = np.load("./MANO/idx_faces_middle.npy")
    # ring = np.arange(773,1001)
    # np.save("./MANO/idx_faces_ring.npy", ring)
    # pinky = np.load("./MANO/idx_faces_pinky.npy")
    #
    # npy_files = [
    #     "./MANO/idx_faces_thumb.npy",
    #     "./MANO/idx_faces_index.npy",
    #     "./MANO/idx_faces_middle.npy",
    #     "./MANO/idx_faces_ring.npy",
    #     "./MANO/idx_faces_pinky.npy",
    # ]
    # # If indices are 1D (flat), set seg_shape to the target shape (e.g., (H, W) or (N,))
    # # If indices are coordinates (N,D), seg_shape is still required (e.g., (H, W) or (D,H,W))
    # seg_shape = (1538,)  # <-- change me
    # background_label = 0
    #
    #
    # def paint(seg: np.ndarray, idx: np.ndarray, label: int):
    #     idx = np.asarray(idx)
    #
    #     # Case A: flat indices (N,) or (N,1)
    #     if idx.ndim == 1 or (idx.ndim == 2 and idx.shape[1] == 1):
    #         flat = idx.reshape(-1).astype(np.int64)
    #         seg_flat = seg.reshape(-1)
    #         seg_flat[flat] = label
    #         return
    #
    #     # Case B: coordinate indices (N, D) where D == seg.ndim
    #     if idx.ndim == 2 and idx.shape[1] == seg.ndim:
    #         coords = tuple(idx[:, d].astype(np.int64) for d in range(idx.shape[1]))
    #         seg[coords] = label
    #         return
    #
    #     raise ValueError(
    #         f"Unsupported index array shape {idx.shape}. "
    #         f"Expected (N,), (N,1), or (N,{seg.ndim})."
    #     )
    #
    #
    # seg = np.full(seg_shape, background_label, dtype=np.int32)
    #
    # for label, f in enumerate(npy_files, start=1):
    #     idx = np.load(f)
    #     paint(seg, idx, label)
    #
    # # Flatten so it's "one label per element"
    # labels = seg.reshape(-1).astype(int)
    #
    # with open("./MANO/segmentation_map.txt", "w") as f:
    #     for v in labels:
    #         f.write(f"{v}\n")

    def indices_to_mask(indices1, indices2, size):
        arr = np.zeros(size, dtype=int)
        arr[indices1] = 1
        arr[indices2] = 1
        return arr

    feature_field = np.load("./FAUST/idx_left_arm.npy")
    feature_field2 = np.load("./DFAUST/left_arm_right_leg.npy")
    heatmap = indices_to_mask(feature_field, feature_field2, size=meshes[0].vertices.shape[0])

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")
    ps.set_ground_plane_height_factor(0)

    register_surface(name=f'Step {0} Ours', mesh=meshes[0], idx_color=1, disp_vectors=None, disp_heatmap=heatmap)

    ps.set_user_callback(callback)
    ps.show()

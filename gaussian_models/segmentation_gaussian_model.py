import os
import torch
import trimesh
import numpy as np
from torch import nn
import networkx as nx
import torch.nn.functional as F
from collections import Counter
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from utils.general_utils import build_scaling_rotation, inverse_sigmoid, get_expon_lr_func

class SegmentationGaussianModel:

    def __init__(self):
        self.active_sh_degree = 0
        self.max_sh_degree = 0

        self.temperature = 1
        self.label_colors = None

        self.free_xyz = False
        self.unsigned_offset = False

        self._d_xyz = None
        
        self._bary = None
        self._normal_offset = None
        self.logits = None
        self._scaling = None
        self._rotation = None
        self._opacity = None
        
        self.vertices = None
        self.faces = None
        
        self.posed_triangle_vertices = None
        self.posed_vertex_normals = None
        self.posed_base_rotation = None

        self.canonical_triangle_vertices = None
        self.canonical_vertex_normals = None
        self.canonical_base_rotation = None

        self.neighbors = None

        self.init_scaling = None

        self.labels = None
        self.refined_labels = None

        self.optimizer = None
        self.spatial_lr_scale = 0

        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        relative_rotation = self.rotation_activation(self._rotation)
        base_rotation = self.rotation_activation(self.posed_base_rotation)
        return quat_xyzw_to_wxyz(quat_product(
            quat_wxyz_to_xyzw(base_rotation),
            quat_wxyz_to_xyzw(relative_rotation)
        ))
    
    @property
    def get_canonical_rotation(self):
        relative_rotation = self.rotation_activation(self._rotation)
        base_rotation = self.rotation_activation(self.canonical_base_rotation)
        return quat_xyzw_to_wxyz(quat_product(
            quat_wxyz_to_xyzw(base_rotation),
            quat_wxyz_to_xyzw(relative_rotation)
        ))
    
    @property
    def get_xyz(self):
        if self.free_xyz:
            return self.posed_triangle_vertices.mean(dim=1) + self._d_xyz
        else:
            bary = F.softmax(self._bary, dim=-1).unsqueeze(-1)
            base = torch.sum(bary * self.posed_triangle_vertices, dim=1)
            normal = F.normalize(torch.sum(bary * self.posed_vertex_normals, dim=1), dim=-1)
            offset = F.softplus(self._normal_offset) if not self.unsigned_offset else self._normal_offset
            return base + offset * normal
    
    @property
    def get_canonical_xyz(self):
        if self.free_xyz:
            return self.canonical_triangle_vertices.mean(dim=1) + self._d_xyz
        else:
            bary = F.softmax(self._bary, dim=-1).unsqueeze(-1)
            base = torch.sum(bary * self.canonical_triangle_vertices, dim=1)
            normal = F.normalize(torch.sum(bary * self.canonical_vertex_normals, dim=1), dim=-1)
            offset = F.softplus(self._normal_offset) if not self.unsigned_offset else self._normal_offset
            return base + offset * normal
    
    @property
    def get_features(self):
        probs = F.softmax(self._logits / self.temperature, dim=-1)
        one_hot = F.one_hot(torch.argmax(probs, dim=-1), num_classes=probs.shape[-1]).float()
        probs = (one_hot - probs).detach() + probs
        blended = probs @ self.label_colors
        features = blended.unsqueeze(1)
        return features
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def training_setup(self, args):
        if args.free_xyz:
            l = [
                {'params': [self._d_xyz], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "d_xyz"},
                {'params': [self._logits], 'lr': args.feature_lr, "name": "logits"},
                {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            ]
        else:
            l = [
                {'params': [self._bary], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "bary"},
                {'params': [self._normal_offset], 'lr': args.position_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._logits], 'lr': args.feature_lr, "name": "logits"},
                {'params': [self._scaling], 'lr': args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': args.rotation_lr, "name": "rotation"},
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=args.position_lr_init*self.spatial_lr_scale,
            lr_final=args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=args.position_lr_delay_mult,
            max_steps=args.num_iterations
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ["bary", "offset"]:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z']
        if self.free_xyz:
            l.extend(['dx', 'dy', 'dz'])
        for i in range(self.label_colors.shape[1]):
            l.append(f'f_dc_{i}')
        l.append("opacity")
        for i in range(self._scaling.shape[1] + 1):
            l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rot_{i}')
        for i in range(self._rotation.shape[1]):
            l.append(f'rel_rot_{i}')
        for i in range(self._bary.shape[1]):
            l.append(f'bary_{i}')
        l.append("normal_offset")
        for i in range(self._logits.shape[1]):
            l.append(f'prob_{i}')
        l.append("labels")
        l.append("refined_labels")
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self.free_xyz:
            d_xyz = self._d_xyz.detach().cpu().numpy()

        xyz = self.get_xyz.detach().cpu().numpy()
        bary = self._bary.detach().cpu().numpy()
        normal_offset = self._normal_offset.detach().cpu().numpy()

        f_dc = self.label_colors[self.refined_labels.squeeze()].detach().cpu().numpy()

        scale = self._scaling.detach().cpu().numpy()
        scale_2 = np.full((scale.shape[0], 1), -np.inf, dtype=np.float32)
        scale = np.concatenate((scale, scale_2), axis=1)

        rotation = self.get_rotation.detach().cpu().numpy()
        relative_rotation = self._rotation.detach().cpu().numpy()

        opacity = self._opacity.detach().cpu().numpy()
        probs = F.softmax(self._logits / self.temperature, dim=-1).detach().cpu().numpy()

        dtype_full = [(attr, 'f4') for attr in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        if self.free_xyz:
            attributes = np.concatenate((xyz, d_xyz, f_dc, opacity, scale, rotation, relative_rotation, bary, normal_offset, probs, self.labels, self.refined_labels), axis=1)
        else:
            attributes = np.concatenate((xyz, f_dc, opacity, scale, rotation, relative_rotation, bary, normal_offset, probs, self.labels, self.refined_labels), axis=1)
            
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_segmented_smplx_mesh(self, path):
        face_colors = SH2RGB(self.label_colors[self.labels.squeeze()].detach().cpu().numpy()) * 255
        vertices = self.vertices.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        segmented_smplx_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        segmented_smplx_mesh.visual.face_colors = face_colors
        segmented_smplx_mesh.export(path)

    def save_segmented_smplx_mesh_refined(self, path):
        refined_face_colors = SH2RGB(self.label_colors[self.refined_labels.squeeze()].detach().cpu().numpy()) * 255
        vertices = self.vertices.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        refined_segmented_smplx_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        refined_segmented_smplx_mesh.visual.face_colors = refined_face_colors
        refined_segmented_smplx_mesh.export(path)

    def load_ply(self, smplx_gaussians, smplx_mesh, label_colors, spatial_lr_scale, canonical_properties, args):
        self.label_colors = torch.tensor(RGB2SH(label_colors / 255.0)).float().cuda()
        
        self.vertices = torch.tensor(smplx_mesh.vertices).float()
        self.faces = torch.tensor(smplx_mesh.faces).long()
        vertex_normals = torch.tensor(smplx_mesh.vertex_normals).float()

        self.posed_triangle_vertices = self.vertices[self.faces].detach().cuda()
        self.posed_vertex_normals = vertex_normals[self.faces].detach().cuda()

        num_gaussians = len(self.faces)

        self.free_xyz = args.free_xyz
        if self.free_xyz:
            self._d_xyz = nn.Parameter(torch.zeros(num_gaussians, 3).float().cuda(), requires_grad=True)

        self._bary = nn.Parameter(torch.zeros(num_gaussians, 3).float().cuda(), requires_grad=True)

        self.unsigned_offset = args.unsigned_offset
        if self.unsigned_offset:
            self._normal_offset = nn.Parameter(torch.zeros(num_gaussians, 1).float().cuda(), requires_grad=True)
        else:
            self._normal_offset = nn.Parameter(torch.full((num_gaussians, 1), -10).float().cuda(), requires_grad=True)
        
        logits = torch.full((num_gaussians, self.label_colors.shape[0]), -5.0, dtype=torch.float32, device="cuda")
        logits[:, 0] = 5.0
        self._logits = nn.Parameter(logits, requires_grad=True)

        self._scaling = nn.Parameter(torch.tensor(np.stack([
            np.asarray(smplx_gaussians["scale_0"]),
            np.asarray(smplx_gaussians["scale_1"])
        ], axis=1)).float().cuda(), requires_grad=True)

        self.posed_base_rotation = nn.Parameter(torch.tensor(np.stack([
            np.asarray(smplx_gaussians["rot_0"]),
            np.asarray(smplx_gaussians["rot_1"]),
            np.asarray(smplx_gaussians["rot_2"]),
            np.asarray(smplx_gaussians["rot_3"])
        ], axis=1)).float().cuda(), requires_grad=False)

        self._rotation = nn.Parameter(torch.tensor(np.concatenate([
            np.ones((num_gaussians, 1)),
            np.zeros((num_gaussians, 3)),
        ], axis=1)).float().cuda(), requires_grad=True)

        self._opacity = nn.Parameter(torch.tensor(
            np.asarray(smplx_gaussians["opacity"])
        ).unsqueeze(1).float().cuda(), requires_grad=False)

        xyz = torch.tensor(np.stack([
            np.asarray(smplx_gaussians["x"]),
            np.asarray(smplx_gaussians["y"]),
            np.asarray(smplx_gaussians["z"])
        ], axis=1)).float().cuda()

        def compute_knn_neighbors(xyz, k=5, chunk_size=20908):
            neighbors = []
            for start in range(0, xyz.shape[0], chunk_size):
                end = min(start + chunk_size, xyz.shape[0])
                x_chunk = xyz[start:end]  # (B, 3)
                dists = (
                    x_chunk.pow(2).sum(dim=1, keepdim=True) +
                    xyz.pow(2).sum(dim=1)[None, :] -
                    2 * x_chunk @ xyz.T
                )  # (B, N)
                topk = torch.topk(dists, k=k, largest=False).indices  # (B, k)
                neighbors.append(topk)
            return torch.cat(neighbors, dim=0)

        self.neighbors = compute_knn_neighbors(xyz, k=5).detach()

        self.init_scaling = self._scaling.clone().detach()

        canonical_vertices = torch.tensor(canonical_properties["vertices"]).float()
        canonical_vertex_normals = torch.tensor(canonical_properties["vertex_normals"]).float()

        self.canonical_triangle_vertices = canonical_vertices[self.faces].detach().cuda()
        self.canonical_vertex_normals = canonical_vertex_normals[self.faces].detach().cuda()

        self.canonical_base_rotation = nn.Parameter(torch.tensor(np.stack([
            np.asarray(canonical_properties["rotations"][:, 0]),
            np.asarray(canonical_properties["rotations"][:, 1]),
            np.asarray(canonical_properties["rotations"][:, 2]),
            np.asarray(canonical_properties["rotations"][:, 3])
        ], axis=1)).float().cuda(), requires_grad=False)

        self.spatial_lr_scale = spatial_lr_scale

    def refine_labels(self, area_threshold=0.001):
        print("Refining labels")

        vertices = self.vertices.detach().cpu().numpy()
        face_indices = self.faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices=vertices, faces=face_indices, process=False)

        probs = F.softmax(self._logits / self.temperature, dim=-1).detach().cpu().numpy()
        labels = np.argmax(probs, axis=-1).flatten()
        refined_labels = labels.copy()

        face_areas = mesh.area_faces
        graph = nx.Graph()
        graph.add_edges_from((int(a), int(b)) for a, b in mesh.face_adjacency)

        def get_connected_components():
            return [
                (list(component), label, face_areas[list(component)].sum(), len(component))
                for label in np.unique(refined_labels)
                for component in nx.connected_components(graph.subgraph(np.where(refined_labels == label)[0]))
            ]

        while True:
            components = get_connected_components()
            small_components = sorted(
                [(faces, label, area, count) for faces, label, area, count in components if area < area_threshold],
                key=lambda x: x[2]
            )
            for faces, old_label, area, count in small_components:
                neighbor_labels = [refined_labels[n] for f in faces for n in graph[f] if n not in faces]
                if neighbor_labels:
                    new_label = Counter(neighbor_labels).most_common(1)[0][0]
                    refined_labels[faces] = new_label
                    print(f"     {count:02d} faces changed from {old_label} to {new_label} (A={area:.5f})")
                    break
            else:
                break

        self.labels = labels[:, None]
        self.refined_labels = refined_labels[:, None]

    def update(self, pose_properties):
        vertices = torch.tensor(pose_properties["vertices"]).float()
        self.faces = torch.tensor(pose_properties["faces"]).long()
        vertex_normals = torch.tensor(pose_properties["vertex_normals"]).float()

        self.posed_triangle_vertices = vertices[self.faces].detach().cuda()
        self.posed_vertex_normals = vertex_normals[self.faces].detach().cuda()

        self.posed_base_rotation = nn.Parameter(torch.tensor(np.stack([
            np.asarray(pose_properties["rotations"][:, 0]),
            np.asarray(pose_properties["rotations"][:, 1]),
            np.asarray(pose_properties["rotations"][:, 2]),
            np.asarray(pose_properties["rotations"][:, 3])
        ], axis=1)).float().cuda(), requires_grad=False)

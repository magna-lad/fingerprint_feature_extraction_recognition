import numpy as np
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay
import tqdm
import random
from sklearn.neighbors import kneighbors_graph

class GraphMinutiae:
    def __init__(self, users_minutiae):
        self.users_minutiae = users_minutiae
        self.fingerprint_graphs = []
        self.graph_metadata = {}

    @staticmethod
    def normalize_minutiae_features(minutiae, orientation_map):
        coords = minutiae[:, :2].astype(np.float32)
        minutiae_angles = minutiae[:, 3]
        
        core_info = GraphMinutiae.find_true_core(orientation_map)
        
        if core_info is not None:
            core_point = np.array(core_info[:2], dtype=np.float32)
            core_orient = core_info[2]
        else:
            core_point = GraphMinutiae.find_core_proxy(minutiae)
            core_orient = 0.0

        centered_coords = coords - core_point

        rotation_angle = -core_orient
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        rot_matrix = np.array([[c, -s], [s, c]])
        
        aligned_coords = centered_coords @ rot_matrix.T
        
        angle_rad = np.deg2rad(minutiae_angles)
        aligned_angle_rad = angle_rad + rotation_angle
        
        angle_sin = np.sin(aligned_angle_rad)
        angle_cos = np.cos(aligned_angle_rad)
        
        type_col = minutiae[:, 2].astype(int)
        type_onehot = np.zeros((len(type_col), 2), dtype=np.float32)
        type_onehot[np.arange(len(type_col)), type_col] = 1.0
        
        dist_to_core = np.linalg.norm(aligned_coords, axis=1).reshape(-1, 1)
        
        angle_to_core_rad = np.arctan2(-aligned_coords[:, 1], -aligned_coords[:, 0])
        core_angle_sin = np.sin(angle_to_core_rad).reshape(-1, 1)
        core_angle_cos = np.cos(angle_to_core_rad).reshape(-1, 1)
        
        return np.column_stack([
            aligned_coords, type_onehot, angle_sin, angle_cos, 
            dist_to_core, core_angle_sin, core_angle_cos, 
            minutiae[:, 2].astype(np.float32)
        ])

    def _build_single_graph(self, minutiae, orientation_map, skeleton,img_path, graph_id):
        """
        UPDATED: Now accepts 'skeleton' argument and stores it in the graph object.
        """
        MINUTIAE_THRESHOLD = 12 
        if minutiae is None or len(minutiae) < MINUTIAE_THRESHOLD:
            return None 
        
        normalized_features = self.normalize_minutiae_features(minutiae, orientation_map)
        coords = normalized_features[:, :2]

        try:
            tri = Delaunay(coords)
        except Exception:
            return None

        edges = set()
        for simplex in tri.simplices:
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[0], simplex[2]))))
        if not edges: return None
        
        edge_list = np.array(list(edges))
        edge_sources, edge_targets = edge_list[:, 0], edge_list[:, 1]
        
        src_coords, tgt_coords = coords[edge_sources], coords[edge_targets]
        distances = np.linalg.norm(src_coords - tgt_coords, axis=1)
        
        src_sin, src_cos = normalized_features[edge_sources, 4], normalized_features[edge_sources, 5]
        tgt_sin, tgt_cos = normalized_features[edge_targets, 4], normalized_features[edge_targets, 5]
        
        dot_product = np.clip((src_cos * tgt_cos) + (src_sin * tgt_sin), -1.0, 1.0)
        relative_angles = np.arccos(dot_product)
        
        deltas = tgt_coords - src_coords
        edge_orientations = np.arctan2(deltas[:, 1], deltas[:, 0])
        
        edge_features = np.vstack([distances, relative_angles, edge_orientations]).T
        
        full_edge_sources = np.concatenate([edge_list[:, 0], edge_list[:, 1]])
        full_edge_targets = np.concatenate([edge_list[:, 1], edge_list[:, 0]])
        edge_index = torch.from_numpy(np.array([full_edge_sources, full_edge_targets])).long()
        edge_attr = torch.from_numpy(np.concatenate([edge_features, edge_features])).float()
        x = torch.from_numpy(normalized_features).float()
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph.graph_id = graph_id
        graph.raw_minutiae = minutiae
        graph.orientation_map = orientation_map
        # --- NEW: Store Skeleton ---
        graph.skeleton = skeleton 
        graph.img_path = img_path
        return graph

    @staticmethod
    def augment_minutiae(minutiae, max_rotation=30, max_translation=25, dropout_prob=0.05, jitter_std=1.5):
        # Augmentation logic unchanged from previous best version
        angle_deg = np.random.uniform(-max_rotation, max_rotation)
        angle_rad = np.radians(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[c, -s], [s, c]])
        
        coords = minutiae[:, :2]
        center = coords.mean(axis=0)
        rotated_coords = (coords - center) @ rotation_matrix.T + center
        
        translation = np.random.uniform(-max_translation, max_translation, size=2)
        final_coords = rotated_coords + translation
        final_angles = (minutiae[:, 3] + angle_deg) % 360
        
        noise = np.random.normal(0, jitter_std, size=final_coords.shape)
        final_coords += noise
        
        augmented = minutiae.copy()
        augmented[:, :2] = final_coords
        augmented[:, 3] = final_angles

        if len(augmented) > 15: 
            mask = np.random.rand(len(augmented)) > dropout_prob
            if mask.sum() >= 5: 
                augmented = augmented[mask]
            
        return augmented

    @staticmethod
    def find_core_proxy(minutiae):
        return minutiae[:, :2].mean(axis=0)

    @staticmethod
    def find_true_core(orientation_map, block_size=16):
        if orientation_map is None: return None
        doubled_orientation = 2 * orientation_map
        rows, cols = doubled_orientation.shape
        cores = []

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                path = [
                    doubled_orientation[r - 1, c - 1], doubled_orientation[r - 1, c],
                    doubled_orientation[r - 1, c + 1], doubled_orientation[r, c + 1],
                    doubled_orientation[r + 1, c + 1], doubled_orientation[r + 1, c],
                    doubled_orientation[r + 1, c - 1], doubled_orientation[r, c - 1],
                ]
                index_sum = 0.0
                for k in range(8):
                    diff = path[(k + 1) % 8] - path[k]
                    if diff > np.pi: diff -= 2 * np.pi
                    elif diff < -np.pi: diff += 2 * np.pi
                    index_sum += diff
                
                if 0.4 < (index_sum / (2 * np.pi)) < 0.6:
                    cores.append((r, c))

        if not cores: return None 
        cores.sort(key=lambda item: item[0], reverse=True)
        best_core_r, best_core_c = cores[0]
        core_orientation = orientation_map[best_core_r, best_core_c]
        scaled_x = best_core_c * block_size + (block_size // 2)
        scaled_y = best_core_r * block_size + (block_size // 2)
        return (scaled_x, scaled_y, core_orientation)

    def graph_maker(self):
        print("Building graphs from all fingerprint minutiae...")
        graphs_with_meta = []
        for uid, udata in tqdm.tqdm(self.users_minutiae.items(), desc="Processing Users"):
            for hand, fingers in udata['fingers'].items():
                for finger_idx, impressions in enumerate(fingers):
                    for impr_idx, impression_data in enumerate(impressions):
                        graph_id = f"{uid}_{hand}_{finger_idx}_{impr_idx}"
                        minutiae = impression_data["minutiae"]
                        orientation_map = impression_data["orientation_map"]
                        # --- NEW: Extract Skeleton ---
                        skeleton = impression_data["skeleton"]
                        img_path = impression_data["img_path"]  
                        
                        graph = self._build_single_graph(minutiae, orientation_map, skeleton,img_path, graph_id)
                        if graph is not None:
                            meta_info = {'graph': graph, 'user_id': uid, 'hand': hand,
                                         'finger_idx': finger_idx, 'impression_idx': impr_idx, 'graph_id': graph_id}
                            graphs_with_meta.append(meta_info)
        self.fingerprint_graphs = graphs_with_meta
        self.graph_metadata = {info['graph_id']: info for info in graphs_with_meta}
        print(f"Successfully built {len(self.fingerprint_graphs)} graphs.")
        return self.fingerprint_graphs

    # ... (Keep create_graph_pairs, get_user_splits, split_pairs_by_user unchanged) ...
    # PASTE THEM HERE from previous code
    def create_graph_pairs(self, num_impostors_per_genuine=3):
        genuine_pairs, impostor_pairs = [], []
        finger_groups = {}
        for info in self.fingerprint_graphs:
            key = (info['user_id'], info['hand'], info['finger_idx'])
            finger_groups.setdefault(key, []).append(info['graph'])
        print("Creating genuine pairs...")
        for key, graphs in tqdm.tqdm(finger_groups.items()):
            if len(graphs) >= 2:
                for i in range(len(graphs)):
                    for j in range(i + 1, len(graphs)):
                        genuine_pairs.append((graphs[i], graphs[j], 1))
        print("Creating impostor pairs...")
        all_graphs = [info['graph'] for info in self.fingerprint_graphs]
        target_num_impostors = num_impostors_per_genuine * len(genuine_pairs)
        while len(impostor_pairs) < target_num_impostors:
            g1, g2 = random.sample(all_graphs, 2)
            uid1 = self.graph_metadata[g1.graph_id]['user_id']
            uid2 = self.graph_metadata[g2.graph_id]['user_id']
            if uid1 != uid2:
                impostor_pairs.append((g1, g2, 0))
        all_pairs = genuine_pairs + impostor_pairs
        random.shuffle(all_pairs)
        return all_pairs

    def get_user_splits(self, train_ratio=0.7, val_ratio=0.15):
        all_user_ids = sorted(list(self.users_minutiae.keys()))
        random.seed(42) 
        random.shuffle(all_user_ids)
        n_users = len(all_user_ids)
        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)
        train_users = set(all_user_ids[:n_train])
        val_users = set(all_user_ids[n_train:n_train + n_val])
        test_users = set(all_user_ids[n_train + n_val:])
        print(f"\nUser Split: {len(train_users)} train, {len(val_users)} validation, {len(test_users)} test.")
        return train_users, val_users, test_users

    def split_pairs_by_user(self, all_pairs, train_users, val_users, test_users):
        train_pairs, val_pairs, test_pairs = [], [], []
        for g1, g2, label in all_pairs:
            uid1 = self.graph_metadata[g1.graph_id]['user_id']
            uid2 = self.graph_metadata[g2.graph_id]['user_id']
            if uid1 in train_users and uid2 in train_users: train_pairs.append((g1, g2, label))
            elif uid1 in val_users and uid2 in val_users: val_pairs.append((g1, g2, label))
            elif uid1 in test_users and uid2 in test_users: test_pairs.append((g1, g2, label))
        print(f"Pair Split: {len(train_pairs)} train, {len(val_pairs)} validation, {len(test_pairs)} test.")
        return train_pairs, val_pairs, test_pairs
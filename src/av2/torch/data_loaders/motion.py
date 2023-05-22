"""PyTorch data-loader for motion forecasting task."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
from rich.progress import track
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import av2._r as rust
from av2.utils.typing import PathType

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MotionDataset(Dataset):  # type: ignore
    def __init__(self, scenario_path: Path, timesteps=110, agent_num=128):
        self.scenario_files = sorted(scenario_path.rglob("*.parquet"))
        self.timesteps = timesteps
        self.agent_num = agent_num

    def __getitem__(self, idx: int):
        scenario_path = self.scenario_files[idx]
        scenario_id = scenario_path.stem.split("_")[-1]
        static_map_path = (
            scenario_path.parents[0] / f"log_map_archive_{scenario_id}.json"
        )

        scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
        static_map = ArgoverseStaticMap.from_json(static_map_path)

        """Load data"""
        # implicit loading from .parquet, see below for explicit data specs
        dict_agent, dict_stc_rg, dict_dyn_rg = {}, {}, {}

        s_xyz = None
        s_bbox_yaw = None
        s_velocity_xy = None
        s_valid = None
        s_id = None
        # Collect data across agent dimension
        for i, track in enumerate(scenario.tracks):
            s_xyz_A = None
            s_bbox_yaw_A = None
            s_velocity_xy_A = None
            s_valid_A = None
            # Collect data across time dimension
            for t, states in enumerate(track.object_states):
                if s_xyz_A is None:
                    s_xyz_A = torch.tensor(states.position, dtype=torch.float).unsqueeze(0)
                    s_bbox_yaw_A = torch.tensor(states.heading, dtype=torch.float).unsqueeze(0)
                    s_velocity_xy_A = torch.tensor(states.velocity, dtype=torch.float).unsqueeze(0)
                    s_valid_A = torch.tensor(states.observed, dtype=torch.bool).unsqueeze(0)
                else:
                    s_xyz_A = torch.cat([s_xyz_A, torch.tensor(states.position)[None, :]], dim=0)
                    s_bbox_yaw_A = torch.cat([s_bbox_yaw_A, torch.tensor(states.heading)[None]], dim=0)
                    s_velocity_xy_A = torch.cat([s_velocity_xy_A, torch.tensor(states.velocity)[None, :]], dim=0)
                    s_valid_A = torch.cat([s_valid_A, torch.tensor(states.observed)[None]], dim=0)
            s_xyz_A = F.pad(s_xyz_A, (0, 0, 0, self.timesteps - s_xyz_A.size(0)))
            s_bbox_yaw_A = F.pad(s_bbox_yaw_A, (0, self.timesteps - s_bbox_yaw_A.size(0)))
            s_velocity_xy_A = F.pad(s_velocity_xy_A, (0, 0, 0, self.timesteps - s_velocity_xy_A.size(0)))
            s_valid_A = F.pad(s_valid_A, (0, self.timesteps - s_valid_A.size(0)))

            if s_xyz is None:
                s_xyz = s_xyz_A.unsqueeze(0)
                s_bbox_yaw = s_bbox_yaw_A.unsqueeze(0)
                s_velocity_xy = s_velocity_xy_A.unsqueeze(0)
                s_valid = s_valid_A.unsqueeze(0)
            else:
                s_xyz = torch.cat((s_xyz, s_xyz_A[None, :, :]), dim=0)
                s_bbox_yaw = torch.cat((s_bbox_yaw, s_bbox_yaw_A[None, :]), dim=0)
                s_velocity_xy = torch.cat((s_velocity_xy, s_velocity_xy_A[None, :, :]), dim=0)
                s_valid = torch.cat((s_valid, s_valid_A[None, :]), dim=0)
        
        s_xyz = F.pad(s_xyz, (0, 1, 0, 0, 0, self.agent_num - s_xyz.size(0)))   # pad 1 for z dimension
        s_bbox_yaw = F.pad(s_bbox_yaw, (0, 0, 0, self.agent_num - s_bbox_yaw.size(0)))
        s_velocity_xy = F.pad(s_velocity_xy, (0, 0, 0, 0, 0, self.agent_num - s_velocity_xy.size(0)))
        s_valid = F.pad(s_valid, (0, 0, 0, self.agent_num - s_valid.size(0)))

        dict_agent['s_xyz'] = s_xyz
        dict_agent['s_bbox_yaw'] = s_bbox_yaw
        dict_agent['s_velocity_xy'] = s_velocity_xy
        dict_agent['s_valid'] = s_valid
        # s_valid correspond to scenario.track.object_states.observed

        # # Agent states #
        # s_bbox_yaw = data['s_bbox_yaw']                                         # [A, T]
        # s_height = data['s_height']                                             # [A, T]
        # s_length = data['s_length']                                             # [A, T]
        # s_speed = data['s_speed']                                               # [A, T]
        # s_timestamp_micros = data['s_timestamp_micros']                         # [A, T]
        # s_valid = data['s_valid']                                               # [A, T]
        # s_vel_yaw = data['s_vel_yaw']                                           # [A, T]
        # s_velocity_xy = data['s_velocity_xy']                                   # [A, T, 2]
        # s_width = data['s_width']                                               # [A, T]
        # s_xyz = data['s_xyz']                                                   # [A, T, 3]
        #
        # # other state features not spanning over time dimension
        # s_difficulty_level = data["s_difficulty_level"]                         # [A]
        # s_id = data["s_id"]                                                     # [A]
        # s_is_sdc = data["s_is_sdc"]                                             # [A]
        # s_objects_of_interest = data["s_objects_of_interest"]                   # [A]
        # s_tracks_to_predict = data["s_tracks_to_predict"]                       # [A]
        # s_type = data["s_type"]                                                 # [A]
        #
        # # Static road graph features loading #
        # if self.load_stc_rg:
        #     sr_dir = data['sr_dir']                                             # [20k, 3]
        #     sr_id = data['sr_id']                                               # [20k]
        #     sr_type = data['sr_type']                                           # [20k]
        #     sr_valid = data['sr_valid']                                         # [20k]
        #     sr_xyz = data['sr_xyz']                                             # [20k, 3]
        #
        #     # optional polylines data
        #     if "pl_xyz" in data.keys():
        #         pl_xyz = data['pl_xyz']                                         # [GS=1000, 20, 3]
        #         pl_type = data['pl_type']                                       # [GS=1000, 20]
        #         pl_valid = data['pl_valid']                                     # [GS=1000]
        #
        # # Dynamic road graph features loading #
        # if self.load_dyn_rg:
        #     dr_id = data['dr_id']                                               # [GD, T]
        #     dr_state = data['dr_state']                                         # [GD, T]
        #     dr_timestamp_micros = data['dr_timestamp_micros']                   # [T]
        #     dr_valid = data['dr_valid']                                         # [GD, T]
        #     dr_xyz = data['dr_xyz']                                             # [GD, T, 3]

        return dict_agent


        # print("scenario", scenario.__dir__())
        # print("scenario_id", scenario.scenario_id)
        # print("timestamps_ns", len(scenario.timestamps_ns))
        # print("tracks", len(scenario.tracks), scenario.tracks[0].__dir__())
        # print("timestep", [object_states.timestep for object_states in scenario.tracks[0].object_states])
        # print("track_ids", [track.track_id for track in scenario.tracks])
        # print("focal_track_id", scenario.focal_track_id)
        # print("city_name", scenario.city_name)
        # print("map_id", scenario.map_id)
        # print("slice_id", scenario.slice_id)
        # # print("AV", [track for track in scenario.tracks if track.track_id == 'AV'])
        # print("position_tensors", position_tensors.shape)

        """Preprocessing"""
        # rotational augmentation for raw data
        if self.aug_mode:
            if 'rand_rotate' in self.aug_mode:
                self._rand_rotate(dict_agent, dict_stc_rg, dict_dyn_rg)

        # note: only light compute should be done here, heavier compute is better stored in the model pass
        # trim timestamp: microsecond in int to second in float
        dict_agent = self._trim_timestamp(dict_agent, old_key_word='s_timestamp_micros', new_key_word='s_t')
        if self.load_dyn_rg:
            dict_dyn_rg = self._trim_timestamp(dict_dyn_rg, old_key_word='dr_timestamp_micros', new_key_word='dr_t')

        dict_emb_agent = self.prepare_agent_emb(dict_agent)
        dict_emb_stc_rg, dict_emb_dyn_rg = {}, {}

        if self.load_stc_rg:
            dict_emb_stc_rg = self.prepare_stc_rg_emb(dict_stc_rg)
        if self.load_dyn_rg:
            dict_emb_dyn_rg = self.prepare_dyn_rg_emb(dict_dyn_rg)

        return dict_emb_agent, dict_emb_stc_rg, dict_emb_dyn_rg, dict_agent, dict_stc_rg, dict_dyn_rg

    def __len__(self):
        return len(self.scenario_files)
        
    @staticmethod
    def _auto_masking(updated_data, original_data, mask_val=-1.0):
        if isinstance(updated_data, torch.Tensor):
            if torch.sum(original_data == mask_val):
                updated_data[original_data == mask_val] = mask_val
        elif isinstance(updated_data, np.ndarray):
            if np.sum(original_data == mask_val):
                updated_data[original_data == mask_val] = mask_val
        else:
            raise NotImplementedError
        return updated_data

    @staticmethod
    def _trim_timestamp(dict_data, old_key_word, new_key_word):
        timestamp_micros = dict_data[old_key_word]
        if isinstance(timestamp_micros, torch.Tensor):
            t_sec = WaymoDataset._auto_masking(timestamp_micros.float() / 1e6, timestamp_micros, mask_val=-1.0)
        elif isinstance(timestamp_micros, np.ndarray):
            t_sec = WaymoDataset._auto_masking(timestamp_micros.astype(np.float32) / 1e6, timestamp_micros, mask_val=-1.0)
        else:
            raise NotImplementedError
        del dict_data[old_key_word]
        dict_data[new_key_word] = t_sec
        return dict_data

    @staticmethod
    def _one_hot_padding_aware(s_type, num_classes, mask_val):
        assert mask_val < 0  # usually -1.0
        flag_mask = s_type == mask_val  # [X, *]
        one_hot = F.one_hot(s_type.clip(min=0).long(), num_classes)  # [X, *, N] <- [X, *]
        one_hot.masked_fill_(flag_mask.unsqueeze(-1), 0.0)  # [X, *, N]
        return one_hot

    def _extract_hidden_padding_mask(self, flag_valid):
        """
        Extract hidden and padding mask.
        @param flag_valid:  [A, T] or [GD, T], valid data flag
        @return:
        """
        """Create hidden mask"""
        # hidden mask: to indicate what information to predict; one means hidden, zero means visible
        hidden_mask = torch.ones_like(flag_valid)  # [A, T]
        hidden_mask[:, :self.past_time_steps] = 0.0  # make the past trajectories visible

        """Create padded mask"""
        # padding mask: to indicate when we have ground-truth data; one means padded, zero means non-padded
        padding_mask = torch.logical_not(flag_valid)  # [A, T], padding is invalid masks

        """Masks fine-tuning"""
        # set padded timesteps to be hidden, so the model is forced to predict padded info
        hidden_mask[padding_mask] = 1.0

        return hidden_mask, padding_mask

    def _extract_padded_time_info(self, vec_t, padding_mask):
        """
        Get interpolated time info so that the model cannot tell if the elements are padded or not.
        For invalid (padded) data, the time info is originally -1, this info must not be revealed to the model.
        @param vec_t:           [A, T] or [GD, T]
        @param padding_mask:    [A, T] or [GD, T]
        @return:
        """
        if padding_mask.sum():
            assert torch.unique(vec_t[padding_mask]).item() == -1.0
        a, t = vec_t.shape
        vec_t_naive = self.s_t_naive[None, :].tile(a, 1)  # [A, T] <- [T]

        # [A, T], replace -1 time info with interpolations
        vec_t_padded = vec_t * (vec_t != -1) + vec_t_naive * (vec_t == -1)
        return vec_t_padded

    def _rand_rotate(self, dict_agent, dict_stc_rg, dict_dyn_rg):
        rand_angle = torch.rand(1) * self.aug_rot_range * 2 - self.aug_rot_range  # consistent rotation per scenario
        c, s = torch.cos(rand_angle), torch.sin(rand_angle)
        mat_rot = torch.stack([c, -s, s, c], dim=-1).view(2, 2)  # [2, 2]

        dicts = (dict_agent, dict_stc_rg, dict_dyn_rg)

        # things to rotate
        # s_bbox_yaw        [A, T]
        # s_vel_yaw         [A, T]
        # s_velocity_xy     [A, T, 2]
        # s_xyz             [A, T, 3]
        # sr_dir            [GS, 3]
        # sr_xyz            [GS, 3]
        # dr_xyz            [GD, T, 3]

        for nm, data in zip(['agent', 'stc_rg', 'dyn_rg'], dicts):
            for key in data.keys():
                if nm == 'agent':
                    no_gt_mask = data['s_valid']  # [A, T]
                elif nm == 'stc_rg':
                    no_gt_mask = data['sr_valid']  # [GS]
                elif nm == 'dyn_rg':
                    no_gt_mask = data['dr_valid']  # [GD, T]
                else:
                    raise NotImplementedError

                if len(data[key].shape) > len(no_gt_mask.shape):
                    _no_gt_mask = no_gt_mask.unsqueeze(-1)
                else:
                    _no_gt_mask = no_gt_mask
                _no_gt_mask = torch.logical_not(_no_gt_mask)

                if 'xy' in key and key != 's_ori_center_xyz':
                    # print(key, data[key].shape, _no_gt_mask.shape)
                    # print(data[key][...,:])

                    data[key][..., :2] = torch.einsum('i j, ... j -> ... i', mat_rot, data[key][..., :2])
                    data[key][..., :2].masked_fill_(_no_gt_mask, -1.0)

                if ('yaw' in key or 'dir' in key) and key != 's_ori_center_yaw':
                    # print(key, data[key].shape, _no_gt_mask.shape)

                    angle = torch.stack((torch.cos(data[key]), torch.sin(data[key])), dim=-1)
                    angle = torch.einsum('i j, ... j -> ... i', mat_rot, angle)
                    data[key] = torch.atan2(angle[..., 1], angle[..., 0])
                    data[key].masked_fill_(_no_gt_mask, -1.0)

    def prepare_agent_emb(self, dict_agent):
        """
        Prepare the agent raw embeddings.
        @param dict_agent: dict of agent data
        @return: time, xyz, other features; hidden mask, padding mask

        Raw agent embedding length = 256 * 4 (xyzt positional encoding) + 14 raw features
        the raw features are:
        s_bbox_yaw = data['s_bbox_yaw']                                         # [A, T],       1
        s_height = data['s_height']                                             # [A, T],       1
        s_length = data['s_length']                                             # [A, T],       1
        s_speed = data['s_speed']                                               # [A, T],       1
        s_vel_yaw = data['s_vel_yaw']                                           # [A, T],       1
        s_velocity_xy = data['s_velocity_xy']                                   # [A, T, 2],    2
        s_width = data['s_width']                                               # [A, T],       1
        s_type = data["s_type"]                                                 # [A],          5, one-hot encoding

        total length so far = 13
        we need to append the binary hidden-mask indicator afterwards to make the total length 14.
        """

        """Load info from dicts of data"""
        s_xyz = dict_agent['s_xyz']                     # [A, T, 3]
        s_t = dict_agent['s_t']                         # [A, T]
        s_valid = dict_agent['s_valid']                 # [A, T]

        s_bbox_yaw = dict_agent['s_bbox_yaw']           # [A, T]
        s_height = dict_agent['s_height']               # [A, T]
        s_length = dict_agent['s_length']               # [A, T]
        s_speed = dict_agent['s_speed']                 # [A, T]
        s_vel_yaw = dict_agent['s_vel_yaw']             # [A, T]
        s_velocity_xy = dict_agent['s_velocity_xy']     # [A, T, 2]
        s_width = dict_agent['s_width']                 # [A, T]
        s_type = dict_agent['s_type']                   # [A]

        """Extract hidden and padding masks"""
        agent_hidden_mask, agent_padding_mask = self._extract_hidden_padding_mask(s_valid)  # [A, T] + [A, T]
        s_t_padded = self._extract_padded_time_info(s_t, agent_padding_mask)  # [A, T]

        """Assemble other features"""
        s_type = WaymoDataset._one_hot_padding_aware(s_type, self.agent_num_types, mask_val=-1.0)  # [A, 5] <- [A]
        s_type = s_type.unsqueeze(-2).repeat(1, s_t.size(-1), 1)  # [A, T, 5] <- [A, 5]

        other_agent_feat_ls = [s_bbox_yaw.unsqueeze(-1), s_height.unsqueeze(-1),
                               s_length.unsqueeze(-1), s_speed.unsqueeze(-1), s_vel_yaw.unsqueeze(-1),
                               s_velocity_xy, s_width.unsqueeze(-1), s_type]  # len: 13
        other_agent_feat = torch.cat(other_agent_feat_ls, dim=-1)  # [A, T, 13]

        # since padding data must be hidden data, we use hidden mask to zero out raw features
        # firstly, zero out everything of hidden/padding data points
        s_xyz = s_xyz.masked_fill(agent_hidden_mask.unsqueeze(-1), 0.0)  # [A, T, 3]
        other_agent_feat.masked_fill_(agent_hidden_mask.unsqueeze(-1), 0.0)  # [A, T, 13]
        # then, concatenate the prediction indicator mask
        other_agent_feat = torch.cat([other_agent_feat, agent_hidden_mask.unsqueeze(-1)], dim=-1)  # [A, T, 14]

        # return: time, xyz, other features; hidden mask, padding mask
        dict_agent_emb = {"time": s_t_padded,
                          "xyz": s_xyz,
                          "other": other_agent_feat,
                          "hidden_mask": agent_hidden_mask,
                          "padding_mask": agent_padding_mask
                          }
        return dict_agent_emb

    def prepare_stc_rg_emb(self, dict_stc_rg):
        """
        Prepare the static RG raw embeddings.
        @param dict_stc_rg: dict of static RG data
        @return: time, xyz, other features; hidden mask, padding mask

        Raw static road graph embedding length = 256 * 3 (xyz positional encoding) + 23 raw features
        Note that the point net structure is firstly applied to each point, so the feature dim is point-wise
        the raw features are:
        sr_dir = data['sr_dir']                                             # [20k, 3], 3
        sr_type = data['sr_type']                                           # [20k],    20, one-hot encoding

        for optional polylines data, follow the above point-wise processing steps
        total length = 23, no need to append the binary indicator for static RG.
        """

        """Load info from dicts of data"""
        sr_dir = dict_stc_rg['sr_dir']                                             # [20k, 3]
        sr_type = dict_stc_rg['sr_type']                                           # [20k]
        sr_valid = dict_stc_rg['sr_valid']                                         # [20k]
        sr_xyz = dict_stc_rg['sr_xyz']                                             # [20k, 3]

        """Extract padding masks"""
        # static RG does not change w.r.t. time, there is therefore no hidden mask
        static_rg_padding_mask = torch.logical_not(sr_valid)  # [20k]

        """Assemble other features"""
        # [20k, 20] <- [20k]
        sr_type = WaymoDataset._one_hot_padding_aware(sr_type, self.stc_rg_num_types, mask_val=-1.0)

        other_stc_rg_feat_ls = [sr_dir, sr_type]  # len: 23
        other_stc_rg_feat = torch.cat(other_stc_rg_feat_ls, dim=-1)  # [20k, 23]

        # return: time, xyz, other features; hidden mask, padding mask
        dict_stc_rg_emb = {"time": torch.as_tensor(torch.nan),
                           "xyz": sr_xyz,
                           "other": other_stc_rg_feat,
                           "hidden_mask": torch.as_tensor(torch.nan),
                           "padding_mask": static_rg_padding_mask
                           }
        return dict_stc_rg_emb

    def prepare_dyn_rg_emb(self, dict_dyn_rg):
        """
        Prepare the dynamic RG raw embeddings.
        @param dict_dyn_rg: dict of dynamic RG data.
        @return: time, xyz, other features; hidden mask, padding mask

        Raw dynamic road graph embedding length = 256 * 4 (xyzt positional encoding) + 10 raw features
        the raw features are:
        dr_state = data['dr_state']                                         # [GD, T], 9, one-hot encoding

        total length so far = 9
        we need to append the binary hidden-mask indicator afterwards to make the total length 10.
        """

        """Load info from dicts of data"""
        dr_state = dict_dyn_rg['dr_state']                                         # [GD, T]
        dr_t = dict_dyn_rg['dr_t']                                                 # [T]
        dr_valid = dict_dyn_rg['dr_valid']                                         # [GD, T]
        dr_xyz = dict_dyn_rg['dr_xyz']                                             # [GD, T, 3]

        """Extract hidden and padding masks"""
        dyn_rg_hidden_mask, dyn_rg_padding_mask = self._extract_hidden_padding_mask(dr_valid)  # [GD, T] + [GD, T]

        if self.mode == 'test':
            pass
        else:
            # note: at training/validation set, for dynamic RG time does not contain -1, its minimum value is 0.0
            assert dr_t.min() >= 0.0

        """Assemble other features"""
        # [GD, T, 9] <- [GD, T]
        dr_state = WaymoDataset._one_hot_padding_aware(dr_state, self.dyn_rg_num_types, mask_val=-1.0)
        other_dyn_rg_feat = dr_state  # [GD, T, 9]

        # since padding data must be hidden data, we use hidden mask to zero out raw features
        # firstly, zero out everything of hidden/padding data points
        dr_xyz = dr_xyz.masked_fill(dyn_rg_hidden_mask.unsqueeze(-1), 0.0)  # [A, T, 3]
        other_dyn_rg_feat.masked_fill_(dyn_rg_hidden_mask.unsqueeze(-1), 0.0)  # [GD, T, 9]
        # then, concatenate the prediction indicator mask
        other_dyn_rg_feat = torch.cat([other_dyn_rg_feat, dyn_rg_hidden_mask.unsqueeze(-1)], dim=-1)  # [GD, T, 10]

        # return: time, xyz, other features; hidden mask, padding mask
        dict_dyn_rg_emb = {"time": dr_t,
                           "xyz": dr_xyz,
                           "other": other_dyn_rg_feat,
                           "hidden_mask": dyn_rg_hidden_mask,
                           "padding_mask": dyn_rg_padding_mask
                           }
        return dict_dyn_rg_emb

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Motion forecast dataloader")
    parser.add_argument('-p', '--path', type=Path, required=True,
                        help="Path of motion-forecast data")
    args = parser.parse_args()
    
    av2_dataset = MotionDataset(args.path)
    dataloader = DataLoader(av2_dataset, batch_size=2, shuffle=False, num_workers=3, pin_memory=False)

    data = next(iter(dataloader))

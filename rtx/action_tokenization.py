import torch

RTX1_ACTION_BOUNDS =           {
                'base_displacement_vector': [-1.0, 1.0],
                'base_displacement_vertical_rotation': [-3.14159, 3.14159],
                'gripper_closedness_action': [-1.0, 1.0],
                'rotation_delta': [-1.5708, 1.5708],
                'terminate_episode': [0.0, 1.0],
                'world_vector': [-1.0, 1.0]
            }

RTX1_ACTION_VOCAB_SIZE = 256

class RTX1ActionTokenizer:
    def __init__(self,  bounds: dict = RTX1_ACTION_BOUNDS, vocab_size: int = RTX1_ACTION_VOCAB_SIZE):
        self.bounds = bounds
        self.vocab_size = vocab_size
    
    def tokenize(self, action: torch.float, lower_bound: float, upper_bound: float) -> torch.int32:
        action = torch.clip(action, lower_bound, upper_bound)
        action = (action - lower_bound) / (upper_bound - lower_bound)
        action = action * (self.vocab_size - 1)
        return action



    def tokenize_dict(self, action: dict) -> torch.Tensor:
        '''Converts a dict of float tensors to a 256-bit tensor.


        Args:
            action (dict): Input action:
            {
                base_displacement_vector, list(Tensor((1,), torch.float64)) sz: 2
                base_displacement_vertical_rotation, list(Tensor((1,), torch.float64)) sz: 1
                gripper_closedness_action, list(Tensor((1,), torch.float64)) sz: 1
                rotation_delta, list(Tensor((1,), torch.float64)) sz: 3
                terminate_episode, list(Tensor((1,), torch.int64)) sz: 3
                world_vector, list(Tensor((1,), torch.float64)) sz: 3
            }
            bounds (dict): Action bounds. If None, default clipping from RT1 paper will be used:
                {
                    'base_displacement_vector': [-1.0, 1.0],
                    'base_displacement_vertical_rotation': [-3.14159, 3.14159],
                    'gripper_closedness_action': [-1.0, 1.0],
                    'rotation_delta': [-1.5708, 1.5708],
                    'terminate_episode': [0, 1.0],
                    'world_vector': [-1.0, 1.0]
                }
            vocab_size (int): Vocabulary size. Default: 256.

        Returns:
            torch.Tensor: 11 dimentional int32 tensor in [0, vocab_size)
        '''

        normalized_action = {}
        for k, v in action.items():
            normalized_action[k] = torch.concatenate(v).squeeze()
            normalized_action[k] = self.tokenize(normalized_action[k], self.bounds[k][0], self.bounds[k][1])

        tokens = torch.zeros(11, dtype=torch.int32)
        tokens[0:2] = normalized_action['base_displacement_vector']
        tokens[2] = normalized_action['base_displacement_vertical_rotation']
        tokens[3] = normalized_action['gripper_closedness_action']
        tokens[4:7] = normalized_action['rotation_delta']
        tokens[7] = normalized_action['terminate_episode'][-1]
        tokens[8:] = normalized_action['world_vector']
        return tokens

    def tokenize_xyzrpyg(self, action: torch.Tensor) -> torch.Tensor:
        tokens = torch.zeros((action.shape[0],action.shape[1],11), dtype=torch.int32)
        tokens[:,:,3] = self.tokenize(action[:,:,6], self.bounds['gripper_closedness_action'][0], self.bounds['gripper_closedness_action'][1])
        tokens[:,:,4:7] = self.tokenize(action[:,:,3:6], self.bounds['rotation_delta'][0], self.bounds['rotation_delta'][1])
        tokens[:,:,8:] = self.tokenize(action[:,:,0:3], self.bounds['world_vector'][0], self.bounds['world_vector'][1])
        return tokens


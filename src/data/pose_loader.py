import os
from pose_format import Pose
from pose_format.torch import TorchPoseBody  # Directly loads into PyTorch tensors

def load_pose_file(file_path):
    """Loads a pre-extracted .pose file and returns tensor data."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, "rb") as buffer:
        # Read the pose file using PyTorch format body wrapper
        pose = Pose.read(buffer, TorchPoseBody)
        
    # pose.body.data contains tensor shape: (Frames, People, Joints, Coordinates)
    tensor_data = pose.body.data
    return tensor_data

if __name__ == "__main__":
    # Test path placeholder
    sample_path = "data/raw_isl/sample.pose"
    data = load_pose_file(sample_path)
    if data is not None:
        print(f"Successfully loaded pose tensor with shape: {data.shape}")
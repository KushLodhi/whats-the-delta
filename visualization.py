import numpy as np
import torch
import matplotlib.pyplot as plt
from model import PhysicsMLP
from physics_sim import run_trajectory

NUM_STEPS = 300   

def load_model(path="physics_mlp.pt"):
    """
    Loads a trained PhysicsMLP from disk and puts it into eval mode.
    """

    model = PhysicsMLP()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def model_rollout(model, initial_state, num_steps=NUM_STEPS):
    """
    Arguments:
        model: trained PhysicsMLP
        initial_state: numpy array [x, y, vx, vy]
        num_steps: how many frames to run for

    Returns:
        numpy array of shape (num_steps, 4)
    """

    predicted_trajectory = [initial_state]
    initial_state_tensor = torch.tensor(initial_state, dtype=torch.float32)

    for _ in range(num_steps-1):
        with torch.no_grad():
            predicted_delta = model(initial_state_tensor)
        initial_state_tensor = initial_state_tensor + predicted_delta
        next_state = initial_state_tensor.numpy().tolist()
        predicted_trajectory.append(next_state)
    
    
    return np.array(predicted_trajectory)



if __name__ == "__main__":
    initial_state = [0.0, 10.0, 0.0, 0.0]
    frames = 300
    actual_trajectory = run_trajectory(initial_state, frames)
    model_from_memory = load_model()
    model_predicted_traj = model_rollout(model_from_memory, initial_state, frames)

    ball_height_actual = actual_trajectory[:, 1]
    ball_height_predicted = model_predicted_traj[:, 1]

    import matplotlib.pyplot as plt
    plt.plot(list(range(frames)), ball_height_actual, label='Actual')
    plt.plot(list(range(frames)), ball_height_predicted, linestyle='--', color='red', label='Predicted')
    plt.xlabel('Frame')
    plt.ylabel('Height')
    plt.legend()
    plt.grid(True)
    plt.show()

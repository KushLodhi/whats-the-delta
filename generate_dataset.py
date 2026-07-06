import numpy as np
from physics_sim import run_trajectory

NUM_TRAJECTORIES = 3000       # number of separate ball drops to simulate
STEPS_PER_TRAJECTORY = 300   # number of frames each trajectory runs for
rng = np.random.default_rng()

def random_initial_state():
    """
    Generates a random starting state for one trajectory
    """

    x = rng.uniform(0,20)
    y = rng.uniform(1.5,20)
    vx = rng.uniform(-2,2)
    vy = rng.uniform(-1,1)

    return np.array([x, y, vx, vy])


def generate_dataset():
    """
    Runs NUM_TRAJECTORIES simulations and collects them into a single dataset.

    Returns:
        numpy array of shape (NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY, 4)
    """
    all_trajectories = []

    #LOOP NUM_TRAJECTORIES TIMES TO GENERATE THAT MANY TRAJECTORIES, EACH WITH STEP_PER_TRAJECTORIES N0. OF FRAMES
    for i in range(NUM_TRAJECTORIES):
        state = random_initial_state()
        nxt_state = run_trajectory(state, STEPS_PER_TRAJECTORY)
        all_trajectories.append(nxt_state)


    return np.array(all_trajectories)


if __name__ == "__main__":
    dataset = generate_dataset()

    print(f"Generated dataset shape: {dataset.shape}")
    # dataset has the shape (NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY, 4)
    # example. (300, 150, 4) — 300 trajectories, 150 frames each, 4 numbers per frame
    print(dataset)

    np.save("trajectories.npy", dataset)
    print("Saved to trajectories.npy")

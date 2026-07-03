import numpy as np

TRAIN_RATIO = 0.8   # fraction of trajectories used for training


def load_data(path="trajectories.npy"):
    """
    Loads the raw dataset saved by generate_dataset.py.
    """
    return np.load(path)


def split_trajectories(data, train_ratio=TRAIN_RATIO):
    """
    Splits the data into training and validation sets, grouped by trajectories. 
    Arguments:
        data: numpy array of shape (N, T, 4) — N trajectories, T frames each
        train_ratio: fraction of trajectories to use for training

    Returns:
        train_data: numpy array of shape (N_train, T, 4)
        val_data:   numpy array of shape (N_val, T, 4)
    """

    num_traj = data.shape[0]        # take the number of individual trajectories
    
    perm = np.random.default_rng()
    new_indices = perm.permutation(num_traj)        # create a shuffled array of indices

    split_point = int(num_traj * train_ratio)       # deciding the spilt point

    # slicing the data into training and validation sets
    train_data = data[new_indices[:split_point]]
    val_data = data[new_indices[split_point:]]

    return train_data, val_data


def make_pairs(trajectories):
    """
    Makes state and delta pairs from the given trajectory data. 'state' and 'delta' are no longer grouped by trajectories.
    Arguments:
        trajectories: numpy array of shape (N, T, 4).

    Returns:
        states: numpy array of shape (N*(T-1), 4) — the "input" half of each pair.
        deltas: numpy array of shape (N*(T-1), 4) — the "target" half of each pair.
    """

    frame_t = trajectories[:, :-1, :]       # drop the last frame in all trajectories
    frame_tplus1 = trajectories[:, 1:, :]       # drop the first frame in all trajectories
    deltas = frame_tplus1 - frame_t        # compute the delta of frame t of a trajectory with frame t+1 of the same trajectory

    # reshaping to remove grouping by trajectories
    states = frame_t.reshape(-1,4)  
    deltas = deltas.reshape(-1,4)

    return states, deltas


if __name__ == "__main__":
    data = load_data()
    print(f"Dataset shape: {data.shape}")

    train_data, val_data = split_trajectories(data)
    print(f"Training trajectories: {train_data.shape[0]}, Validation trajectories: {val_data.shape[0]}")

    train_states, train_deltas = make_pairs(train_data)
    val_states, val_deltas = make_pairs(val_data)

    print(f"Train pairs: {train_states.shape}, Val pairs: {val_states.shape}")

    np.save("train_states.npy", train_states)
    np.save("train_deltas.npy", train_deltas)
    np.save("val_states.npy", val_states)
    np.save("val_deltas.npy", val_deltas)
    print("Saved train & val states and deltas.")

import numpy as np

GRAVITY = -9.8       # acceleration due to gravity, negative = downward
DT = 1 / 60           # time step per frame (60 fps)
GROUND_Y = 0.0        # y-coordinate of the ground
RESTITUTION = 0.7     # bounce energy retained (1.0 = perfectly elastic, 0 = no bounce)
RADIUS = 0.5          # ball radius, to detect ground contact


def step(state):
    """
    Advances the ball's state by one timestep (DT).

    state: numpy array [x, y, vx, vy]
    returns: numpy array [x, y, vx, vy] representing the next state
    """
    
    x, y, vx, vy = state

    # APPLYING GRAVITY TO VERTICAL VELOCITY
    vy += GRAVITY * DT

    # UPDATING POSITION
    x += vx * DT
    y += vy * DT
    
    # CHECKING FOR COLLISION AND FLIPPING THE DIRECTION IF COLLISION IS DETECTED
    if y <= RADIUS:
        y = RADIUS
        vy *= -RESTITUTION


    return np.array([x, y, vx, vy])


def run_trajectory(initial_state, num_steps):
    """
    Runs the simulator forward for num_steps frames, starting from initial_state.

    Returns: numpy array of shape (num_steps, 4). The full history of states, one row per frame.
    """
    trajectory = [initial_state]
    state = initial_state

    for _ in range(num_steps - 1):
        state = step(state)
        trajectory.append(state)

    return np.array(trajectory)


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    initial_state = np.array([2.0, 5.0, 3.687, -5.234])
    trajectory = run_trajectory(initial_state, num_steps=500) #trajectory  has shape (numsteps+1, 4)

    heights = trajectory[:, 1]  # y-coordinate over time

    plt.plot(heights)
    plt.xlabel("Frame")
    plt.ylabel("Height (y)")
    plt.title("Ball height over time — shows a decaying bounce")
    plt.grid(True)
    plt.show()

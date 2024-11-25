import math
import matplotlib.animation
from matplotlib import pyplot as plt
import numpy as np

# Initial model params

# For the ball
BallVelocity = [+0.3, 0]  # Vx, Vy
BallMass = 0.1
StartPosition = [-0.2, 0]  # x, y
mu = 0.0
# For the rod
RodLength = 1
RodMass = 10
RodAngle = 0  # alpha
RodAngleVelocity = 0

# Solving differential equation with Euler method

# Grid
ModelingTime = 10
TimeSteps = ModelingTime*100
dt = ModelingTime / TimeSteps

# Coefficients
c1 = 9.8
c2 = 12 * BallMass * 9.8 / RodMass / RodLength ** 2
LastStep = 0


# let's use hill climbing algorythm for training our model
# # observation function that shows if ball still on the plate
def obs_func(x_pos, angle):
    if x_pos >= RodLength * np.cos(angle) / 2 or x_pos <= -RodLength * np.cos(angle) / 2:
        return False
    return True


# function impacting the current state
def action(alg_weights, state):
    return dt*np.dot(alg_weights, state)


nEpisode = 200
BestTotalReward = 0
TotalRewards = []
BestWeights = []
NoiseScale = 0.1

for episode in range(nEpisode):
    # Initializing
    reward = 0
    noise = NoiseScale*np.random.rand(4)
    weights = np.random.rand(4) + noise
    i = 1

    x = [0] * TimeSteps
    y = [0] * TimeSteps
    alpha = [0] * TimeSteps
    x_d1 = [0] * TimeSteps
    alpha_d1 = [0] * TimeSteps

    x[0] = StartPosition[0]
    y[0] = StartPosition[1]
    alpha[0] = RodAngle
    x_d1[0] = BallVelocity[0]
    alpha_d1[0] = RodAngleVelocity

    while True:

        x[i] = x[i-1] + dt*x_d1[i-1]
        x_d1[i] = x_d1[i-1] + dt*c1*(math.sin(alpha[i-1]) - mu*math.cos(alpha[i-1]))
        alpha[i] = alpha[i-1] + dt*alpha_d1[i-1]
        alpha_d1[i] = alpha_d1[i-1] + dt*c2*x[i-1] - action(weights, [x[i-1], x_d1[i-1], alpha[i-1], alpha_d1[i-1]])

        reward += 1
        TotalRewards.append(reward)

        if not obs_func(x[i], alpha[i]):
            break
        elif i >= TimeSteps - 1:
            break
        i += 1

    if reward > BestTotalReward:
        BestTotalReward = reward
        BestWeights = weights
        NoiseScale = max(NoiseScale / 2, 1e-4)
    else:
        NoiseScale = min(NoiseScale*2, 10)

    if reward == TimeSteps - 1:
        break

    print('Эпизод {}: {}'.format(episode + 1, reward))
print('Best time is ', BestTotalReward)

# Visual part

fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))

line, = ax.plot([-1, 0], [0, 1], color='b', linewidth=1)
point, = plt.plot(StartPosition[0], StartPosition[1], 'ro')

plt.title("Balancing ball model")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.grid()


def update(k):
    line.set_data([-RodLength*np.cos(alpha[k])/2, RodLength*np.cos(alpha[k])/2],
                  [RodLength*np.sin(alpha[k])/2, -RodLength*np.sin(alpha[k])/2])
    point.set_data([x[k]], -x[k]*math.tan(alpha[k]))
    return line, point,


frames = range(TimeSteps - 1)
fig.canvas.draw()
ani = matplotlib.animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=20, repeat=True)
plt.show()

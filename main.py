import numpy as np
import matplotlib.pyplot as plt
from kf import KF
import matplotlib.cm as cm
import matplotlib as mpl
from physicsEquations import physicsPredictor

def variance(arr):
    mean = float(np.sum(arr))/len(arr)
    sqdiff = np.sum([float((x-mean)**2) for x in arr])
    return float(sqdiff)/len(arr)


#variables:
DT = 0.1 #interval of time
NUM_STEPS = 100 #number of updates or time intervals
MEAS_EVERY_STEPS = 1 #how many steps before an update
expected_vx = 0.16002 #needs to be set, if distance was travelled at a constant velocity (.16002 for example_data, .09652 for example_data2 and example_data3)
expected_vy = 0.66294 #needs to be set, if distance was travelled at a constant velocity (.66294 for example_data, .57785 for example_data2 and example_data3)
data_path = "example_data.csv" # change example_data6.csv to your csv

data = np.loadtxt(data_path, delimiter=",", dtype=str) 
t = [float(x[0]) for x in data]
ax = [float(x[1]) for x in data]
ay = [float(x[2]) for x in data]
axvar = variance(ax)
ayvar = variance(ay)
plt.ion()
plt.figure()




#KF:
#velocity should be constant so any acceleration is error
#x direction, units in SI
meas_variancex = axvar
initial_vx = 0 
estimated_vx = expected_vx #depends on your testing
meas_x=0.0
kf = KF(initial_x=meas_x, initial_v=estimated_vx, accel_variance=axvar)
muxs = []
covxs = []
real_xs = []
real_vxs = []

for step in range(NUM_STEPS):    
    covxs.append(kf.cov)
    muxs.append(kf.mean)
    
    kf.predict(dt=DT)

    estimated_vx = ax[step]*DT+estimated_vx
    meas_x = meas_x+DT*estimated_vx
    kf.update(meas_value=meas_x, meas_variance=meas_variancex)

    real_xs.append(expected_vx*step*DT)
    real_vxs.append(expected_vx)
    
#y direction, units in SI
meas_variancey = ayvar
initial_vy = 0 
estimated_vy = expected_vy #depends on your testing
meas_y=0.0
kf2 = KF(initial_x=meas_y, initial_v=estimated_vy, accel_variance=ayvar)
muys = []
covys = []
real_ys = []
real_vys = []

for step in range(NUM_STEPS):
    covys.append(kf2.cov)
    muys.append(kf2.mean)
    
    kf2.predict(dt=DT)

    estimated_vy = ay[step]*DT+estimated_vy
    meas_y = meas_y+DT*estimated_vy
    kf2.update(meas_value=meas_y, meas_variance=meas_variancey)

    real_ys.append(expected_vy*step*DT)
    real_vys.append(expected_vy)



    

#Graphing:
#Graphing predicted vs real and showing 95th percentile for x-axis
plt.subplot(2, 2, 1)
plt.title('Position(x)')
plt.plot([mu[0] for mu in muxs], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(muxs,covxs)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(muxs,covxs)], 'r--')

plt.subplot(2, 2, 2)
plt.title('Velocity(x)')
plt.plot(real_vxs, 'b')
plt.plot([mu[1] for mu in muxs], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(muxs,covxs)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(muxs,covxs)], 'r--')

#Graphing predicted vs real and showing 95th percentile for y-axis
plt.subplot(2, 2, 3)
plt.title('Position(y)')
plt.plot([mu[0] for mu in muys], 'r')
plt.plot(real_ys, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(muys,covys)], 'r--')
plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(muys,covys)], 'r--')

plt.subplot(2, 2, 4)
plt.title('Velocity(y)')
plt.plot(real_vys, 'b')
plt.plot([mu[1] for mu in muys], 'r')
plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(muys,covys)], 'r--')
plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(muys,covys)], 'r--')

#Graphing Bivariate Normal Distribution usign contour maps
def bivariate_normal(x, y, σ_x=1.0, σ_y=1.0, μ_x=0.0, μ_y=0.0, σ_xy=0.0):
    x_μ = x - μ_x
    y_μ = y - μ_y

    ρ = σ_xy / (σ_x * σ_y)
    z = x_μ**2 / σ_x**2 + y_μ**2 / σ_y**2 - 2 * ρ * x_μ * y_μ / (σ_x * σ_y)
    denom = 2 * np.pi * σ_x * σ_y * np.sqrt(1 - ρ**2)
    return np.exp(-z / (2 * (1 - ρ**2))) / denom

def gen_gaussian_plot_vals(μ, C):
    m_x, m_y = float(μ[0]), float(μ[1])
    s_x, s_y = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
    s_xy = C[0, 1]
    return bivariate_normal(X, Y, s_x, s_y, m_x, m_y, s_xy)

# Plot the contour map of the gaussian curve
fig, ax = plt.subplots(figsize=(10, 8))
ax.grid()

x_grid = np.linspace(1.5, 2, 100)
y_grid = np.linspace(6.5, 8.3, 100)
X, Y = np.meshgrid(x_grid, y_grid)

x_pos = [x[0] for x in muxs]
y_pos = [x[0] for x in muys]
Σ = np.cov(np.array([x_pos,y_pos]), bias=True)
Σ = np.matrix(Σ)
x_hat = np.matrix([muxs[-1][0],muys[-1][0]]).T
Z = gen_gaussian_plot_vals(x_hat, Σ)
ax.contourf(X, Y, Z, 6, alpha=0.6, cmap=cm.jet)
cs = ax.contour(X, Y, Z, 6, colors="black")
ax.clabel(cs, inline=1, fontsize=10)

#plotting actual(green), predicted(red), and calculated(using physics and blue) points
plt.plot([real_xs[-1]], [real_ys[-1]], marker="o", markersize=5, markeredgecolor="green",markerfacecolor="green")
plt.plot([x_pos[-1]], [y_pos[-1]], marker="o", markersize=5, markeredgecolor="red",markerfacecolor="red")
pp = physicsPredictor(data_path, expected_vx, expected_vy)
calculated_pos = pp.predict()
plt.plot(calculated_pos[0], calculated_pos[1], marker="o", markersize=5, markeredgecolor="blue",markerfacecolor="blue")

#print all final guesses/value
print("predicted coordinates: ", [x_pos[-1], y_pos[-1]])
print("actual coordinates: ", [real_xs[-1], real_ys[-1]])
print("calculated coordinates: ", calculated_pos[0], calculated_pos[1])

plt.show()
plt.ginput(100)

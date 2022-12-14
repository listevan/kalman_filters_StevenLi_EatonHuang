import numpy as np
import matplotlib.pyplot as plt
from kf import KF
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats import multivariate_normal
from physicsEquations import physicsPredictor


#finding variance from an array
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
t = [float(x[0]) for x in data] #extracting data from csv
ax = [float(x[1]) for x in data]
ay = [float(x[2]) for x in data]
axvar = variance(ax) #finding variance in acceleration (error)
ayvar = variance(ay)



#KF:
#velocity should be constant so any acceleration is error
#x direction, units in SI
meas_variancex = axvar
initial_vx = 0 
estimated_vx = expected_vx #depends on your testing
meas_x=0.0
kf = KF(initial_x=meas_x, initial_v=estimated_vx, accel_variance=axvar)
muxs = [] #mean x and v values
covxs = [] #covariance between x and v
real_xs = [] #actual x, constant velocity
real_vxs = [] #actual v, constant velocity

for step in range(NUM_STEPS):    
    covxs.append(kf.cov)
    muxs.append(kf.mean)

    kf.predict(dt=DT) #predict

    estimated_vx = ax[step]*DT+muxs[-1][1] #finding measurements using error (if none then estimated_vx = expected_vx)
    meas_x = muxs[-1][0]+DT*estimated_vx 
    kf.update(meas_value=meas_x, meas_variance=meas_variancex) #update

    real_xs.append(expected_vx*step*DT) #finding real x using constant velocity
    real_vxs.append(expected_vx)
    
#y direction, units in SI
meas_variancey = ayvar
initial_vy = 0 
estimated_vy = expected_vy #depends on your testing
meas_y=0.0
kf2 = KF(initial_x=meas_y, initial_v=estimated_vy, accel_variance=ayvar)
muys = [] #same as above but for y
covys = [] #same as above but for y
real_ys = [] #same as above but for y
real_vys = [] #same as above but for y

for step in range(NUM_STEPS):
    covys.append(kf2.cov)
    muys.append(kf2.mean)
    
    kf2.predict(dt=DT) #predict

    estimated_vy = ay[step]*DT+muys[-1][1] #same as above but for y
    meas_y = muys[-1][0]+DT*estimated_vy 
    kf2.update(meas_value=meas_y, meas_variance=meas_variancey) #update

    real_ys.append(expected_vy*step*DT) #same as above but for y
    real_vys.append(expected_vy)



    

#Graphing:
#Graphing predicted vs real and showing 95th percentile for x-axis
plt.figure()
plt.subplot(2, 2, 1)
plt.title('Position(x)')
plt.plot([mu[0] for mu in muxs], 'r')
plt.plot(real_xs, 'b')
plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(muxs,covxs)], 'r--') #+-2 standard deviations away -> 95th percentile
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

#Graphing Bivariate Normal Distribution using contour maps
# Plot the contour map of the gaussian curve
plt.figure()
x_grid = np.linspace(1, 2, 100) #might need to change this to fit your data
y_grid = np.linspace(6, 9, 300)
X, Y = np.meshgrid(x_grid, y_grid)
x_pos = [x[0] for x in muxs]
y_pos = [y[0] for y in muys]
Σ = np.array(np.cov(np.array([x_pos,y_pos]), bias=True)+np.matrix([[.01,0],[0,.01]])).tolist() #solving for our covariance matrix
print ("Covariance matrix of x and y: ")
print(Σ)
x_hat = [muxs[-1][0], muys[-1][0]]
pos = np.dstack((X, Y))
Z = multivariate_normal.pdf(pos, x_hat, Σ).reshape(X.shape)

ax = plt.axes(projection='3d')
ax.scatter3D(X, Y, Z, cmap='viridis',linewidth=0);
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.figure()
plt.contourf(X, Y, Z, levels = 10)
plt.colorbar()

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

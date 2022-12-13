import numpy as np


class physicsPredictor:
    def __init__(self, data, expected_vx, expected_vy, DT = 0.1, NUM_STEPS = 100, MEAS_EVERY_STEPS = 1):
        self.expected_v = np.zeros(2)
        self.expected_v[0] = expected_vx
        self.expected_v[1] = expected_vy
        self.data = np.loadtxt(data, delimiter=",", dtype=str)
        self.t = [float(x[0]) for x in self.data]
        self.ax = [float(x[1]) for x in self.data]
        self.ay = [float(x[2]) for x in self.data]
        self.DT = DT
        self.NUM_STEPS = NUM_STEPS
        self.MEAS_EVERY_STEPS = MEAS_EVERY_STEPS
        
    def predict(self):
        #x direction
        calculated_x = 0
        for step in range(self.NUM_STEPS):
            self.expected_v[0]+=self.ax[step]*self.DT #update vx
            calculated_x+=self.expected_v[0]*self.DT #update x

        #y direction
        calculated_y = 0
        for step in range(self.NUM_STEPS):
            self.expected_v[1]+=self.ay[step]*self.DT #update vy
            calculated_y+=self.expected_v[1]*self.DT #update y

        return [[calculated_x],[calculated_y]]


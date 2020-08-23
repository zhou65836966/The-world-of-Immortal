from scipy import stats

import numpy as np
import math

class all_map(object):
    def __init__(self, sigama, energy_value, lengh,sort):
        self.sigama = sigama
        self.energy_value = energy_value
        self.sort = sort
        if lengh % 2 == 0:
            self.mean_value = lengh/2
        else:
            self.mean_value = lengh/2 + 0.5
        self.lengh = lengh
        self.wood_1 = []
        self.wood_2 = []
        self.wood_3 = []
        self.wood = []
        self.fire_1 = []
        self.fire_2 = []
        self.fire_3 = []
        self.fire = []
        self.earth_1 = []
        self.earth = []
        self.metal_1 = []
        self.metal_2 = []
        self.metal_3 = []
        self.metal = []
        self.water_1 = []
        self.water_2 = []
        self.water_3 = []
        self.water = []
        self.exp = []
    def softmax_mat(self,input):

        for i in range(self.lengh):
            for r in range(self.lengh):
                input_exp = input[i, r]
                self.exp.append(input_exp)
        sum_exp = sum(self.exp)
        for i in range(self.lengh):
            for r in range(self.lengh):
                input[i, r] = round(input[i, r] / sum_exp, 10)
        return input

    def softmax_sigama(self,):
        sigama_exp = [math.exp(q) for q in self.sort]
        sum_sigama_exp = sum(sigama_exp)
        softmax = [round(w / sum_sigama_exp, 3) for w in sigama_exp]
        return softmax
    def energy_sort(self,):
        y = np.array(self.softmax_sigama())
        Y_energy = self.energy_value * y
        return Y_energy

    def generate_earth(self, ):
        y = self.energy_sort()[0]
        for i in range(self.lengh):
            prob = stats.norm.pdf(i+1, self.mean_value, self.sigama[0])
            earth_energy = prob
            self.earth_1.append(earth_energy)
        for r in range(self.lengh):
            self.earth.append(self.earth_1)
        earth_mat = np.mat(self.earth)
        earth_T = earth_mat.T
        earth_small = np.minimum(earth_mat, earth_T)
        self.earth_out = self.softmax_mat(earth_small)
        for t in range(self.lengh):
            for u in range(self.lengh):
                self.earth_out[t,u] = self.earth_out[t,u]*y
        return self.earth_out
    def generate_wood(self, ):
        y = self.energy_sort()[1]
        for i in range(self.lengh):
            prob = stats.norm.pdf(i+1, self.lengh, self.sigama[1])
            wood_energy = y*prob
            self.wood_1.append(wood_energy)
        for r in range(self.lengh):
            self.wood.append(self.wood_1)
        for e in range(self.lengh):
            prob_1 = stats.norm.pdf(e+1, self.mean_value, self.sigama[1])
            wood_energy_2 = y*prob_1
            self.wood_2.append(wood_energy_2)
        for o in range(self.lengh):
            self.wood_3.append(self.wood_2)

        wood_mat_1 = np.mat(self.wood)
        wood_mat_2 = np.mat(self.wood_3)
        wood_T = wood_mat_2.T
        wood_small = np.minimum(wood_mat_1, wood_T)
        self.wood_out = self.softmax_mat(wood_small)
        for t in range(self.lengh):
            for u in range(self.lengh):
                self.wood_out[t, u] = self.wood_out[t, u]*y
        return self.wood_out

    def generate_fire(self, ):
        y = self.energy_sort()[2]
        for i in range(self.lengh):
            prob = stats.norm.pdf(i+1, self.lengh, self.sigama[2])
            fire_energy = y*prob
            self.fire_1.append(fire_energy)
        for r in range(self.lengh):
            self.fire.append(self.fire_1)
        for e in range(self.lengh):
            prob = stats.norm.pdf(e+1, self.mean_value, self.sigama[2])
            fire_energy_2 = y*prob
            self.fire_2.append(fire_energy_2)
        for o in range(self.lengh):
            self.fire_3.append(self.fire_2)

        fire_mat_1 = np.mat(self.fire)
        fire_mat_2 = np.mat(self.fire_3)
        fire_T = fire_mat_2.T
        fire_small = np.minimum(fire_mat_1, fire_T)
        fire_small_T = fire_small.T
        self.fire_out = self.softmax_mat(fire_small_T)
        for t in range(self.lengh):
            for u in range(self.lengh):
                self.fire_out[t, u] = self.fire_out[t, u] * y
        return self.fire_out
    def generate_metal(self, ):
        y = self.energy_sort()[3]
        for i in range(self.lengh):
            prob = stats.norm.pdf(self.lengh-i, self.lengh, self.sigama[3])
            metal_energy = y*prob
            self.metal_1.append(metal_energy)
        for r in range(self.lengh):
            self.metal.append(self.metal_1)
        for e in range(self.lengh):
            prob = stats.norm.pdf(e+1, self.mean_value, self.sigama[3])
            metal_energy_2 = y*prob
            self.metal_2.append(metal_energy_2)
        for o in range(self.lengh):
            self.metal_3.append(self.metal_2)

        metal_mat_1 = np.mat(self.metal)
        metal_mat_2 = np.mat(self.metal_3)
        metal_T = metal_mat_2.T
        metal_small = np.minimum(metal_mat_1, metal_T)
        self.metal_out = self.softmax_mat(metal_small)
        for t in range(self.lengh):
            for u in range(self.lengh):
                self.metal_out[t, u] = self.metal_out[t, u] * y
        return self.metal_out
    def generate_water(self, ):
        y = self.energy_sort()[4]
        for i in range(self.lengh):
            prob = stats.norm.pdf(self.lengh-i, self.lengh, self.sigama[4])
            water_energy = y*prob
            self.water_1.append(water_energy)
        for r in range(self.lengh):
            self.water.append(self.water_1)
        for e in range(self.lengh):
            prob = stats.norm.pdf(e+1, self.mean_value, self.sigama[4])
            water_energy_2 = y*prob
            self.water_2.append(water_energy_2)
        for o in range(self.lengh):
            self.water_3.append(self.water_2)

        water_mat_1 = np.mat(self.water)
        water_mat_2 = np.mat(self.water_3)
        water_T = water_mat_2.T
        water_small = np.minimum(water_mat_1, water_T)
        water_small_T = water_small.T
        self.water_out = self.softmax_mat(water_small_T)
        for t in range(self.lengh):
            for u in range(self.lengh):
                self.water_out[t, u] = self.water_out[t, u] * y
        return self.water_out
    def inborn_energy(self,x,y):
        self.generate_earth()
        self.generate_wood()
        self.generate_fire()
        self.generate_metal()
        self.generate_water()
        self.inborn = ['earth:', round(self.earth_out[x, y],2), 'wood:', round(self.wood_out[x, y],2), 'fire:', round(self.fire_out[x, y], 2), 'metal:', round(self.metal_out[x, y],2), 'water:', round(self.water_out[x, y], 2)]
        return self.inborn
    def mat_to_zero(self,):
        del self.wood_1[:]
        del self.wood_2[:]
        del self.wood_3[:]
        del self.wood[:]
        del self.fire_1[:]
        del self.fire_2[:]
        del self.fire_3[:]
        del self.fire[:]
        del self.earth_1[:]
        del self.earth[:]
        del self.metal_1[:]
        del self.metal_2[:]
        del self.metal_3[:]
        del self.metal[:]
        del self.water_1[:]
        del self.water_2[:]
        del self.water_3[:]
        del self.water[:]
        del self.exp[:]






#b = [1,1,1,10,10]
#c = [0.1,0.2,0.3,0.4,0.5]
#aaaaaaa = all_map(b, 1000000, 9, c)
#eareh = aaaaaaa.generate_earth()
#wood = aaaaaaa.generate_wood()
#fire = aaaaaaa.generate_fire()
#metal = aaaaaaa.generate_metal()
#water = aaaaaaa.generate_water()
#eeeee = aaaaaaa.inborn_energy(7,8)
#print(eareh)














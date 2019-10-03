from numpy.random import uniform
from functions import rastrigin, rosenbrock, twonminima
from numpy import array, matmul, add, subtract
from random import random
from statistics import stdev

class Particle_Swarm_Optimization:

    def __init__(self, particle_count, max_iterations, inertia, cognitive, social, feas, func):
        self.particle_count = particle_count
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.particle_positions = list(zip(uniform(low=feas[0][0], high=feas[0][1], size=(particle_count, )),
                                           uniform(low=feas[1][0], high=feas[1][1], size=(particle_count, ))))
        self.particle_velocity = list(zip(uniform(-1.0, 1.0, size=(particle_count, )), uniform(-1.0, 1.0, size=(particle_count, ))))
        self.particle_bests_fitness = []
        self.particle_bests_positions = []
        self.func = func
        self.particles_fitness = []
        self.feas = feas
        
    def clear(self):
        self.particle_positions = list(zip(uniform(low=self.feas[0][0], high=self.feas[0][1], size=(self.particle_count, )),
                                           uniform(low=self.feas[1][0], high=self.feas[1][1], size=(self.particle_count, ))))
        self.particle_velocity = list(zip(uniform(-1.0, 1.0, size=(self.particle_count, )), uniform(-1.0, 1.0, size=(self.particle_count, ))))
        self.particle_bests_fitness = []
        self.particle_bests_positions = []
        self.particles_fitness = []
        
    def initial_fitness(self):
        if self.func == 'rastrigin':
            for each in self.particle_positions:
                self.particles_fitness.append(rastrigin(each[0], each[1]))
                self.particle_bests_fitness.append(rastrigin(each[0], each[1]))
                self.particle_bests_positions.append((each[0], each[1]))
        elif self.func == '2nminima':
            for each in self.particle_positions:
                self.particles_fitness.append(twonminima(each[0], each[1]))
                self.particle_bests_fitness.append(twonminima(each[0], each[1]))
                self.particle_bests_positions.append((each[0], each[1]))
        else: 
            for each in self.particle_positions:
                self.particles_fitness.append(rosenbrock(each[0], each[1]))
                self.particle_bests_fitness.append(rosenbrock(each[0], each[1]))
                self.particle_bests_positions.append((each[0], each[1]))
        
        min_loc = self.particles_fitness.index(min(self.particles_fitness))
    
        return (self.particle_positions[min_loc], self.particles_fitness[min_loc])
        
    def get_fitness(self):
        count = 0
        if self.func == 'rastrigin':
            for each in self.particle_positions:
                self.particles_fitness[count] = rastrigin(each[0], each[1])
                if self.particles_fitness[count] < self.particle_bests_fitness[count]:
                    self.particle_bests_fitness[count] = self.particles_fitness[count]
                    self.particle_bests_positions[count] = (each[0], each[1])
                count = count + 1
        elif self.func == '2nminima':
            for each in self.particle_positions:
                self.particles_fitness[count] = twonminima(each[0], each[1])
                if self.particles_fitness[count] < self.particle_bests_fitness[count]:
                    self.particle_bests_fitness[count] = self.particles_fitness[count]
                    self.particle_bests_positions[count] = (each[0], each[1])
                count = count + 1
        else: 
            for each in self.particle_positions:
                self.particles_fitness[count] = rosenbrock(each[0], each[1])
                if self.particles_fitness[count] < self.particle_bests_fitness[count]:
                    self.particle_bests_fitness[count] = self.particles_fitness[count]
                    self.particle_bests_positions[count] = (each[0], each[1])
                count = count + 1
                
        min_loc = self.particles_fitness.index(min(self.particles_fitness))
        max_loc = self.particles_fitness.index(max(self.particles_fitness))
        mean = sum(self.particles_fitness)/len(self.particles_fitness)
        standard_dev = stdev(self.particles_fitness)
        
        return (self.particle_positions[min_loc], self.particles_fitness[min_loc], self.particles_fitness[max_loc], mean, standard_dev)
    
    def update_position(self, index, swarm_best):
        old_velocity = array([[self.particle_velocity[index][0]], [self.particle_velocity[index][1]]])
        r1 = random()
        r2 = random()
        particle_best = array([[self.particle_bests_positions[index][0]], [self.particle_bests_positions[index][1]]])
        position = array([[self.particle_positions[index][0]], [self.particle_positions[index][1]]])
        swarm_best_m = array([[swarm_best[0]], [swarm_best[1]]])
        new_velocity = add((add((self.inertia * old_velocity), (self.cognitive * r1 * subtract(particle_best, position)))), ((self.social * r2 * subtract(swarm_best_m, position))))
        new_point = add(position, new_velocity)
        self.particle_velocity[index] = (new_velocity[0][0], new_velocity[1][0])
        return (new_point[0][0], new_point[1][0])
        
    def run(self):
        self.clear()
        swarm_tup = self.initial_fitness()
        swarm_best = swarm_tup[1]
        swarm_best_pos = swarm_tup[0]
        k = 0
        min_per_iter = []
        max_per_iter = []
        mean_per_iter = []
        stdev_per_iter = []
        
        while k != self.max_iterations:
            for index in range(self.particle_count):
                self.particle_positions[index] = self.update_position(index, swarm_best_pos)
                fit_tup = self.get_fitness()
                if fit_tup[1] < swarm_best:
                    swarm_best = fit_tup[1]
                    swarm_best_pos = fit_tup[0]
                running_min = swarm_best
            k = k + 1
            min_per_iter.append(fit_tup[1])
            max_per_iter.append(fit_tup[2])
            mean_per_iter.append(fit_tup[3])
            stdev_per_iter.append(fit_tup[4])
            
        return (fit_tup[0], running_min, min_per_iter, max_per_iter, mean_per_iter, stdev_per_iter)

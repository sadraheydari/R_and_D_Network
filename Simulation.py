import numpy as np
import itertools
from tqdm import tqdm


class Simulation:

    def __init__(self, alpha = 10, beta = 1.0, delta = 1.0, theta = 0.5, gamma = 9, phi = 1.3,  n = 3, n_h = 1):
        self.alpha = alpha
        self.beta  = beta
        self.delta = delta
        self.theta = theta
        self.gamma = gamma
        self.phi   = phi
        self.n     = n
        self.n_h   = n_h
        self.n_l   = n - n_h
        self.types = np.array([[1] for _ in range(self.n_h)] + [[self.theta] for _ in range(self.n_l)])
        self.Theta = np.diag(self.types.reshape(-1))
        self.pairwise_stable_graph = None
        self.sim_dict = dict()
        self.convert2number = lambda x: int(''.join(map(str, x)), 2)


    def __create_sim_dict(self):
        number_of_links = (self.n * (self.n - 1)) // 2
        self.sim_dict = {
            key: dict() for key in itertools.product([0, 1], repeat = number_of_links)
        }
        pass
        

    def simulate_key(self, key):
        if key not in self.sim_dict:
            self.sim_dict[key] = dict()
        
        _ = self.calculate_profit(matrix_key=key)
        self.calculate_social_welfare(matrix_key=key)
        pass


    def simulate_market(self, show_progress = False):
        prgs = tqdm if show_progress else (lambda x: x)
        self.__create_sim_dict()
        for key in prgs(self.sim_dict.keys()):
            self.simulate_key(key)
        pass


    def calculate_social_welfare(self, matrix_key):
        res_dict = self.sim_dict[matrix_key]
        Q  = np.sum(res_dict['q'])
        Pi = np.sum(res_dict['pi'])
        W  = (self.beta * (Q**2) / 2) + Pi
        self.sim_dict[matrix_key]['consumer_surplus'] = (self.beta * (Q**2) / 2)
        self.sim_dict[matrix_key]['firm_surplus'] = Pi
        self.sim_dict[matrix_key]['welfare'] = W
        pass
    

    def calculate_optimal_effort(self, matrix_key):
        G = Simulation.create_matrix_from_key(matrix_key, self.n)
        
        b = (self.alpha - self.gamma) * np.ones((self.n, 1))

        degree = (np.ones((1, self.n)) @ G).reshape(-1)
        theta = self.types.reshape(-1)

        A = np.zeros((self.n, self.n))

        for i in range(self.n):
            A[i, i] += ((self.n + 1)**2 * self.phi) / (theta[i] * (self.n - (self.delta * degree[i])))
            A[i, i] -= (self.n + 1) * theta[i]
            for j in range(self.n):
                A[i, j] -= (self.n + 1) * self.delta * G[i, j] * theta[j]
                A[i, j] += (1 + (self.delta * degree[j])) * theta[j]

        e  = np.linalg.solve(A, b)
        self.sim_dict[matrix_key]['e'] = [i for i in e.reshape(-1)]
        return e

    
    def calculate_production_costs(self, matrix_key):
        if 'e' not in self.sim_dict[matrix_key].keys():
            e = self.calculate_optimal_effort(matrix_key)
        else:
            e = np.array(self.sim_dict[matrix_key]['e']).reshape((-1, 1))

        G = Simulation.create_matrix_from_key(matrix_key, self.n)
        c = (self.gamma * np.ones((self.n, 1))) - ((np.identity(self.n) + (self.delta * G)) @ self.Theta @ e)
        self.sim_dict[matrix_key]['c'] = [i for i in c.reshape(-1)]
        
        if np.sum(c < 0) > 0:
            raise Exception('Negative Production cost in matrix\n\t\t' + str(G).replace('\n', '\n\t\t'))
            
        return c


    def calculate_equilibrium_quantity(self, matrix_key):
        if 'c' not in self.sim_dict[matrix_key].keys():
            c = self.calculate_production_costs(matrix_key)
        else:
            c = np.array(self.sim_dict[matrix_key]['c']).reshape((-1, 1))

        q = ((self.alpha * np.ones((self.n, 1))) - c + ((np.ones((self.n, 1)) @ c.T - c @ np.ones((1, self.n))) @ np.ones((self.n, 1)))) / (self.beta * (self.n + 1))
        self.sim_dict[matrix_key]['q'] = [i for i in q.reshape(-1)]

        G = Simulation.create_matrix_from_key(matrix_key, self.n)
        degree = (np.ones((1, self.n)) @ G).reshape(-1)
        q_star = [(self.phi / t) * (self.n + 1) / (self.n - self.delta * d) * e for d, t, e in zip(degree, self.types.reshape(-1), self.sim_dict[matrix_key]['e'])]
        self.sim_dict[matrix_key]['q_star'] =  q_star
        
        return q

    
    def calculate_profit(self, matrix_key):
        if 'q' not in self.sim_dict[matrix_key].keys():
            q = self.calculate_equilibrium_quantity(matrix_key)
        else:
            q = np.array(self.sim_dict[matrix_key]['q']).reshape((-1, 1))
        c = np.array(self.sim_dict[matrix_key]['c']).reshape((-1, 1))


        price = self.alpha - self.beta * np.sum(q)
        profit_m = [(self.beta * (q_i**2)) for q_i in q.reshape(-1)]
        self.sim_dict[matrix_key]['pi_m1'] = profit_m
        self.sim_dict[matrix_key]['pi_m2'] = [i for i in ((price - c) * q).reshape(-1)]
        pi = [(pi_m - (self.phi * (e_i**2))) for pi_m, e_i in zip(profit_m, self.sim_dict[matrix_key]['e'])]
        self.sim_dict[matrix_key]['pi'] = pi
        self.sim_dict[matrix_key]['price'] = price
        return np.array(pi)

   
    def find_goyal_stable_equilibriums(self):
        result = [key for key in self.sim_dict.keys() if self.is_goyal_stable(key)]
        if len(result) == 0:
            print("NO GOYAL STABLE NETWORK FOUND.")
        return result
            

    def is_goyal_stable(self, matrix_key):
        if 'pi' not in self.sim_dict[matrix_key].keys():
            self.simulate_key(matrix_key)
        
        eq = self.sim_dict[matrix_key]
        is_stabe = True
        for t, val in enumerate(matrix_key):
            new_key = matrix_key[:t] + (1-matrix_key[t], ) + matrix_key[t+1:]

            if new_key not in self.sim_dict:
                self.simulate_key(new_key)

            if 'pi' not in self.sim_dict[new_key].keys():
                self.simulate_key(new_key)
            
            new_eq = self.sim_dict[new_key]
            i, j = Simulation.get_coordinates(n = self.n, i = t)

            if val == 1:
                if (eq['pi'][i] < new_eq['pi'][i]) or (eq['pi'][j] < new_eq['pi'][j]):
                    is_stabe = False
                    break
            if val == 0:
                if (eq['pi'][i] < new_eq['pi'][i]) and (eq['pi'][j] < new_eq['pi'][j]):
                    is_stabe = False
                    break
        self.sim_dict[matrix_key]['is_stable'] = is_stabe
        return is_stabe
    

    def is_pairwise_stable_cluster(self):
        cluster_key = KeyGenerator.cluster_key(self.n, self.n_h)
        cluster_matrix = Simulation.create_matrix_from_key(cluster_key, self.n)

        if cluster_key not in self.sim_dict:
            self.simulate_key(cluster_key)
        
        # H-H Deviaton
        hh_deviaton = tuple([0] + [i for i in cluster_key[1:]])
        if hh_deviaton not in self.sim_dict:
            self.simulate_key(hh_deviaton)
        h_profit = self.sim_dict[cluster_key]['pi'][0]
        h_profit_new = self.sim_dict[hh_deviaton]['pi'][0]
        if h_profit < h_profit_new:
            return False
        
        # L-L Deviaton
        ll_deviaton = tuple([i for i in cluster_key[:-1]] + [0])
        if ll_deviaton not in self.sim_dict:
            self.simulate_key(ll_deviaton)
        l_profit = self.sim_dict[cluster_key]['pi'][-1]
        l_profit_new = self.sim_dict[ll_deviaton]['pi'][-1]
        if l_profit < l_profit_new:
            return False

        # H-L Deviaton
        h_idx = 0
        l_idx = self.n - 1
        cluster_matrix[h_idx, l_idx] = 1
        cluster_matrix[l_idx, h_idx] = 1
        h_l_deviaton = KeyGenerator.create_key_from_matrix(cluster_matrix)
        if h_l_deviaton not in self.sim_dict:
            self.simulate_key(h_l_deviaton)
        h_profit = self.sim_dict[cluster_key]['pi'][h_idx]
        l_profit = self.sim_dict[cluster_key]['pi'][l_idx]

        h_profit_new = self.sim_dict[h_l_deviaton]['pi'][h_idx]
        l_profit_new = self.sim_dict[h_l_deviaton]['pi'][l_idx]
        if (h_profit < h_profit_new) and (l_profit < l_profit_new):
            return False

        return True


    def is_pairwise_stable_colmplete(self):
        complete_key = KeyGenerator.complete_key(self.n)
        complete_matrix = Simulation.create_matrix_from_key(complete_key, self.n)

        if complete_key not in self.sim_dict:
            self.simulate_key(complete_key)

        # H-H Deviaton
        hh_deviaton = tuple([0] + [i for i in complete_key[1:]])
        if hh_deviaton not in self.sim_dict:
            self.simulate_key(hh_deviaton)
        h_profit = self.sim_dict[complete_key]['pi'][0]
        h_profit_new = self.sim_dict[hh_deviaton]['pi'][0]
        if h_profit < h_profit_new:
            return False
        
        # L-L Deviaton
        ll_deviaton = tuple([i for i in complete_key[:-1]] + [0])
        if ll_deviaton not in self.sim_dict:
            self.simulate_key(ll_deviaton)
        l_profit = self.sim_dict[complete_key]['pi'][-1]
        l_profit_new = self.sim_dict[ll_deviaton]['pi'][-1]
        if l_profit < l_profit_new:
            return False
        
        # H-L Deviaton
        h_idx = 0
        l_idx = self.n - 1
        complete_matrix[h_idx, l_idx] = 0
        complete_matrix[l_idx, h_idx] = 0   
        h_l_deviaton = KeyGenerator.create_key_from_matrix(complete_matrix)
        if h_l_deviaton not in self.sim_dict:
            self.simulate_key(h_l_deviaton)
        h_profit = self.sim_dict[complete_key]['pi'][h_idx]
        l_profit = self.sim_dict[complete_key]['pi'][l_idx]

        h_profit_new = self.sim_dict[h_l_deviaton]['pi'][h_idx]
        l_profit_new = self.sim_dict[h_l_deviaton]['pi'][l_idx]

        if (h_profit < h_profit_new) or (l_profit < l_profit_new):
            return False
        
        return True


    
    @staticmethod
    def create_matrix_from_key(key, n):
        result = np.zeros((n, n))
        r = 0
        for i in range(n):
            for j in range(i):
                result[i, j] = key[r]
                result[j, i] = key[r]
                r += 1
        return result

    
    @staticmethod
    def get_coordinates(n, i):
        row = int((1 + (1 + 8 * i)**0.5) // 2)
        previous_triangle_number = (row * (row - 1)) // 2
        col = i - previous_triangle_number
        return (row, col)


    def int2bintuple(self, number):
        l = (self.n * (self.n - 1)) // 2
        b = bin(number)[2:]
        return tuple(map(int, ['0' for _ in range(l - len(b))] + [i for i in b]))





class KeyGenerator:

    @staticmethod
    def create_key_from_matrix(matrix):
        key = []
        n = matrix.shape[0]
        for i in range(1, n):
            for j in range(i):
                key.append(matrix[i, j])
        return tuple(map(int, key))


    @staticmethod
    def semi_cluster_key(n, n_h, n_h_c):
        matrix = np.zeros((n, n))
        
        for i in range(n_h):
            for j in range(i):
                matrix[i, j] = 1
                matrix[j, i] = 1
        
        for i in range(n_h, n):
            for j in range(n_h, i):
                matrix[i, j] = 1
                matrix[j, i] = 1
        

        for i in range(n_h_c):
            for j in range(n_h, n):
                matrix[i, j] = 1
                matrix[j, i] = 1
        
        return KeyGenerator.create_key_from_matrix(matrix)
    

    @staticmethod
    def cluster_key(n, n_h):
        return KeyGenerator.semi_cluster_key(n, n_h, n_h_c=0)


    @staticmethod
    def complete_key(n):
        return KeyGenerator.create_key_from_matrix(np.ones((n, n)) - np.identity(n))
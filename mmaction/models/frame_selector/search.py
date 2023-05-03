import torch
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from itertools import combinations, product


def compute_solution(problem, solution):
    if problem['stage'] == 1:
        from torch.nn import CrossEntropyLoss
        criterion = CrossEntropyLoss()

        features = problem['features']
        target   = problem['target']
        model    = problem['model']

        features = features[solution]
        features = torch.FloatTensor(features)#.cuda()

        with torch.no_grad():
            logit = model.cls_head.fc_cls(features.mean(dim=0))
            loss = criterion(logit.unsqueeze(0), target).item()

        return loss 

    elif problem['stage'] == 3:
        features    = problem['features']
        target_feat = problem['target']

        target_feat = target_feat.unsqueeze(0)
        mean_feat = features.mean(dim=0, keepdim=True)
        sim = torch.nn.functional.cosine_similarity(target_feat, mean_feat)

        return 1 - sim


class Guided():

    def __init__(self, problem):
        self.total_frames = problem['features'].shape[0]
        self.n_frames = problem['n_frames']
        self.problem = problem
        self.penalty = np.zeros((self.total_frames), dtype=np.int32)
        self.history = []
        self.num_evals = 0

    @property
    def name(self):
        return 'Guided'

    def set_params(self, params):
        self.solution  = copy.copy(params['solution'])
        self.method    = params['method']
        self.cur_cost  = compute_solution(self.problem, self.solution)
        self.history.append(self.cur_cost)
        self.n_iter    = params['n_iter']
        self.n_epoch   = params['n_epoch']
        self.verbose   = params['verbose']
        self.mu        = params['mu']
        self.patience  = params['patience']


    def solve(self):
        self.solution, self.cur_cost, num_evals = self.LocalSearchSolver()
        self.num_evals += num_evals
        self.history.append(self.cur_cost)
        self.update_penalty()
        self.refresh_params()
        self.last_state = 0
        no_improve_counter = 0
        for epoch in tqdm(range(self.n_epoch),
                                   position=0,
                                   total=self.n_epoch,
                                   disable=not self.verbose):
            tmp_solution, h, num_evals = self.LocalSearchSolver(self.augmented_cost)
            cost = compute_solution(self.problem, tmp_solution)
            self.num_evals += (num_evals + 1)
            self.history.append(cost)
            if self.cur_cost > cost:
                self.solution = tmp_solution
                self.cur_cost = cost
                self.last_state = epoch
                no_improve_counter = 0
            else:
                no_improve_counter += 1
            self.update_penalty()
            self.refresh_params()
            if no_improve_counter > self.patience:
                if self.verbose:
                    print('No better solutions, stoping...')
                break

        if self.verbose:
            print('End cost: {}'.format(self.cur_cost))
        return self.solution


    def update_penalty(self):
        self.utility = np.zeros((self.total_frames), dtype=np.float32)
        for i in range(self.n_frames):
            for j in range(self.total_frames):
                self.utility[self.solution[i]] += \
                        1 / (1 + self.penalty[self.solution[i]])
        maximized = self.utility.max()
        for i in range(self.n_frames):
            if self.utility[self.solution[i]] == maximized:
                self.penalty[self.solution[i]] += 1

    @staticmethod
    def augmented_cost(problem, solution, params):
        cost    = compute_solution(problem, solution)
        penalty = params['penalty']
        mu      = params['mu']
        n       = params['n']
        total_penalty = 0
        for i in range(n):
            total_penalty += penalty[solution[i]]
        _lambda = cost / n
        return cost + mu * _lambda * total_penalty


    def refresh_params(self):
        self.params = dict(solution=self.solution,
                           method = self.method,
                           n_iter = self.n_iter,
                           verbose = False)


    def LocalSearchSolver(self, cost_func=None):
        alg = LocalSearch(self.problem)
        alg.solution = self.solution
        if cost_func:
            cost_params = {'penalty': self.penalty,
                           'mu': self.mu,
                           'n': self.n_frames}
            alg.set_cost_func(cost_func, cost_params)
        alg.set_params(self.params)
        local_search_solve = alg.solve()
        return local_search_solve, alg.cur_cost, alg.num_evals


    def get_history(self):
        # plt.figure(figsize=(10, 4))
        # plt.plot(self.history, label='cost_function')
        # plt.plot(self.last_state+2,
        #          self.history[self.last_state+2],
        #          'o', label='chosen optimum')
        # plt.grid()
        # plt.legend()
        # plt.title(self.name)
        return self.history


class LocalSearch:

    def __init__(self, problem):
        self.total_frames = problem['features'].shape[0]
        self.n_frames = problem['n_frames']
        self.problem = problem
        self.methods = ['first-improvement',
                        'best-improvement',
                        'stochastic-2opt',
                        'first-delta-improvement']
        self.cost_func   = self.simple_cost
        self.cost_params = None
        self.num_evals = 0

    @property
    def name(self):
        return 'LocalSearch'

    @staticmethod
    def simple_cost(problem, solution, params):
        return compute_solution(problem, solution)

    def set_params(self, params):
        self.solution = copy.copy(params['solution'])
        self.method   = params['method']
        self.cur_cost = self.cost_func(self.problem, self.solution, self.cost_params)
        self.n_iter   = params['n_iter']
        self.verbose  = params['verbose']


    def set_cost_func(self, cost_function, param):
        self.cost_func   = cost_function
        self.cost_params = param


    def solve(self):
        if self.method == self.methods[0]:
            return self.first_improvement()
        elif self.method == self.methods[1]:
            return self.best_improvement()
        elif self.method == self.methods[2]:
            return self.stochastic_2opt()
        elif self.method == self.methods[3]:
            return self.first_delta_improvement()
        else:
            raise 'Method must be one of {}'.format(self.methods)


    def first_delta_improvement(self):
        if self.verbose:
            print('Start cost {}'.format(self.cur_cost))

        dont_look  = {x: 0 for x in range(self.total_frames)}
        for i in tqdm_notebook(range(self.n_iter),
                               position=0,
                               total=self.n_iter,
                               disable=not self.verbose):
            flag = True
            for opt in product(np.arange(self.n_frames, dtype=np.int32), np.arange(self.total_frames, dtype=np.int32)):
                if dont_look[opt[1]] >= 19:
                    continue
                diff = self.__delta__(opt[0], opt[1])
                if diff < 0:
                    self.cur_cost += diff
                    self.solution[opt[0]] = opt[1]
                    flag = False
                    # break
                dont_look[opt[1]] += 1
            if flag and self.verbose:
                print('No better solutions, stoping...')
                break
        if self.verbose:
            print('End cost {}'.format(self.cur_cost))

        return self.solution


    def __delta__(self, pos, frame):
        new_solution = copy.deepcopy(self.solution)
        new_solution[pos] = frame
        diff = self.cost_func(self.problem, new_solution, self.cost_params) - self.cur_cost
        self.num_evals += 1
        # if self.cost_params:
        #     # print('Yes')
        #     _lambda = 1.0 #self.cur_cost / (self.n**4)
        #     mu = self.cost_params['mu']
        #     penalty = self.cost_params['penalty']
        #     diff += mu * _lambda * ((penalty[pos][pi[frame]] + penalty[frame][pi[pos]]) - \
        #                             (penalty[pos][pi[pos]] + penalty[frame][pi[frame]]))
        return diff

    def first_improvement(self):

        if self.verbose:
            print('Start cost {}'.format(self.cur_cost))

        comb       = list(combinations(np.arange(self.n, dtype=np.int32), 2))
        dont_look  = {x:np.zeros(self.n, dtype=np.int32) for x in range(self.n)}
        for i in tqdm_notebook(range(self.n_iter),
                      position=0,
                      disable=not self.verbose):

            flag = True
            for opt in comb:
                if (sum(dont_look[opt[0]]) >= 19 or
                    sum(dont_look[opt[1]]) >= 19):
                    continue
                opt = list(opt)
                tmp_solution      = copy.copy(self.solution)
                tmp_solution[opt] = tmp_solution[opt][::-1]
                cost = self.cost_func(self.problem, tmp_solution, self.cost_params)
                self.num_evals += 1
                if cost < self.cur_cost:
                    self.cur_cost = cost
                    self.solution = tmp_solution
                    flag = False
                    break
                dont_look[opt[0]][opt[1]] = 1
                dont_look[opt[1]][opt[0]] = 1
            if flag and self.verbose:
                print('No better solutions, stoping...')
                break

        if self.verbose:
            end_cost = self.cost_func(self.problem, self.solution, self.cost_params)
            print('End cost {}'.format(end_cost))
        return self.solution


    def stochastic_2opt(self):
        if self.verbose:
            print('Start cost {}'.format(self.cur_cost))

        for i in tqdm_notebook(range(self.n_iter), position=0, disable=not self.verbose):
            flag = True
            for j in range(self.n_iter):
                ind_left, ind_right = randint(0, self.n), randint(0, self.n)

                tmp_solution      = copy.copy(self.solution)
                tmp_solution[ind_left:ind_right] = tmp_solution[ind_left:ind_right][::-1]
                cost = self.cost_func(self.problem, tmp_solution, self.cost_params)
                if cost < self.cur_cost:
                    self.cur_cost = cost
                    self.solution = tmp_solution
                    flag = False

            if flag and self.verbose:
                print('No better solutions, stoping...')
                break

        end_cost = self.cost_func(self.problem, self.solution, self.cost_params)
        if self.verbose:
            print('End cost {}'.format(end_cost))
        return self.solution


    def best_improvement(self):
        if self.verbose:
            print('Start cost {}'.format(self.cur_cost))

        dont_look = np.zeros(self.total_frames)
        comb = list(product(np.arange(self.n_frames), np.arange(self.total_frames)))
        for i in tqdm_notebook(range(self.n_iter), position=0, disable=not self.verbose):

            best_opt = None
            for opt in comb:
                opt = list(opt)
                tmp_solution      = copy.copy(self.solution)
                tmp_solution[opt[0]] = opt[1]
                cost = self.cost_func(self.problem, tmp_solution, self.cost_params)
                if cost < self.cur_cost:
                    self.cur_cost = cost
                    best_opt      = opt
            if best_opt:
                self.solution[opt[0]] = opt[1]
            else:
                print('No better solutions, stoping...')
                break
        end_cost = self.cost_func(self.problem, self.solution, self.cost_params)

        if self.verbose:
            print('End cost {}'.format(end_cost))

        return self.solution
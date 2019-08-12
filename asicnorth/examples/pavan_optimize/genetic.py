import numpy as np

class GeneticMinimization():
    def __init__(self, f, xrange, pop_size=50, trn_size=10, mutation=(0.4,0.01), num_pars=2):
        """
        minimize a cost function using a genetic algorithm
        @param f        function to minimize
        @param xrange   2d list where ith entry is (min,max) bounds for xi
        @param pop_size size of the population, default: 50
        @param trn_size number of candidates to select each parent, default: 10
        @param mutation parameters for how mutate children of form (0.4, 0.01)
                        this will randomly mutate 40% of the population by 1%
        @param num_pars number of parents to use for each child, default: 2

        example: g = GeneticMinimization(f, xrange=((-1,1),(-2,2)), pop_size=10, trn_size=5)
                 g.breed()
                 print(g.mean())
                 print(g.std())
                 bestx, besterr = g.best()
        """
        self.f = f #function to minimize
        if hasattr(xrange[0], "__iter__"):
            self.nargs = len(xrange)
        else:
            self.nargs = 1
            xrange = (xrange,)
        self.xrange = tuple((np.min(xrange[i]),np.max(xrange[i])) for i in range(self.nargs))
        self.pop_size = pop_size
        self.trn_size = trn_size
        self.mutation = mutation #(percent to mutate, amount to mutate)
        self.__range = np.array([xrange[i][1]-xrange[i][0] for i in range(self.nargs)])
        pop0 = tuple(xrange[i][0]+np.random.rand(pop_size)*self.__range[i] for i in range(self.nargs))
        self.populations = list()
        self.populations.append(np.matrix(pop0).T)
        self.scores = list()
        self.num_parents = num_pars
        self.gen = 0
    @staticmethod
    def crossover(x):
        xdim = x[0].shape
        mixed = np.array(x[0])
        rands_range = np.ones(xdim)
        for j in range(1, len(x)):
            r = np.multiply(np.random.rand(*xdim), rands_range)
            rands_range -= r
            mixed += np.multiply(r, x[j]-x[0])
        return mixed
    def breed(self):
        pop = self.populations[self.gen]
        new_pop = np.matrix(np.zeros(pop.shape))
        # calculate f(xi) for every xi in population
        scores = self.f(*tuple(pop[:,i].A1 for i in range(self.nargs)))
        self.scores.append(scores)
        # generate new population from best parents
        for i in range(self.pop_size):
            parents = list()
            for par in range(self.num_parents): # choose best candidate from tournament as parent
                candidate_nums = np.random.choice(self.pop_size, self.trn_size, replace=False)
                best_cand = candidate_nums[np.argmin(scores[candidate_nums])]
                parents.append(best_cand)
            child = self.crossover([pop[parent,:] for parent in parents])
            new_pop[i,:] = child
        # mutate the children
        num_mutate = int(self.pop_size*self.mutation[0])
        mutation_index = np.random.choice(self.pop_size, num_mutate, replace=False)
        mutations = np.random.normal(0, self.mutation[1], (num_mutate,self.nargs))
        for j in range(self.nargs):
            mutations[:,j] *= self.__range[j]        
        new_pop[mutation_index,:] += np.array(mutations)
        # save the children
        self.populations.append(new_pop)
        self.gen += 1
    def best(self, gen=None):
        "get the best parameters x from last generation and its cost"
        gen = (gen if gen != None else self.gen-1)
        scores = self.scores[gen]
        i = np.argmin(scores)
        return tuple(self.populations[gen][i,j] for j in range(self.nargs)), scores[i]
    def population(self, gen=None):
        return self.populations[gen if gen!=None else self.gen]
    def population_scores(self, gen=None):
        return self.scores[gen if gen!=None else self.gen-1]
    def mean(self, gen=None):
        return tuple(np.mean(self.population(gen)[:,i]) for i in range(self.nargs))
    def std(self, gen=None):
        return tuple(np.std(self.population(gen)[:,i]) for i in range(self.nargs))

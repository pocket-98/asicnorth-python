import numpy as np

class GradientDescent():
    def __init__(self, f=None, x=None, xrange=None, mass=0.5, df=None, dx=None):
        """
        perform gradient descent with momentum to minimize a cost function
        @param f      function to minimize
        @param x      initial guess or tuple for guess if f takes many args
        @param xrange tuple (min,max) optional bounds for x or tuple of tuples if f takes many args
        @param mass   number between 0 and 1 how likely to keep same direction
        @param df     gradient of function to minimize default: f(x+dx)-f(x)
        @param dx     delta x default: (xrange[1] - xrange[0])/1000 or x/1000
        
        example: g = GradientDescent(f, x=(0,0), xrange=((-1,1),(-2,2)), mass=0.5)
                 g.descend(50)
                 print(g.parameters())
        """
        if type(x) == type(None):
            if type(xrange) == type(None):
                raise Exception("error: must specify at least x or xrange in GradientDescent()")
            else:
                self.nargs = len(xrange)
                self.xrange = tuple((np.min(xrange[i]),np.max(xrange[i])) for i in range(self.nargs))
                x = np.array([(self.xrange[i][1] + self.xrange[i][0])/2.0 for i in range(self.nargs)])
        else:
            if hasattr(x, "__iter__"):
                self.nargs = len(x)
                x = np.array(x, dtype="float")
            else:
                self.nargs = 1
                x = np.array([x], dtype="float")
            if type(xrange) == type(None):
                self.xrange = None
            else:
                self.xrange = tuple((np.min(xrange[i]),np.max(xrange[i])) for i in range(self.nargs))
        self.iterations = list()
        self.iterations.append(x)
        if mass < 0.0 or mass > 1.0:
            raise Exception("error: mass must be between 0.0 and 1.0")
        self.mass = mass
        self.velocity = 0*x
        self.it = 0
        if df:
            self.df = df
        else:
            if not f:
                raise Exception("error: must either specify f or df in GradientDescent()")
            # pick a delta x
            if type(dx) == type(None):
                if type(self.xrange) == type(None):
                    dx = np.array([xi/1000.0 if xi != 0.0 else 0.01 for xi in x])
                else:
                    dx = np.array([(self.xrange[i][1]-self.xrange[i][0])/1000.0 for i in range(self.nargs)])
            # use f(x+dx) - f(x) to calculate gradient
            self.df = lambda x: np.reciprocal(dx) * (np.array([f(*(x+dx[i]*self.uvec(i,self.nargs))) for i in range(self.nargs)]) - f(*x))
    @staticmethod
    def uvec(i, n):
        "unit vector i with dimension n"
        v = np.zeros(n)
        v[i] = 1
        return v
    def descend(self, n=1, alpha=None):
        """
        gradient descent using alpha as scaling factor
        @param n number of iterations to do
        @param alpha function for how to scale gradient per iteration 
        """
        if not alpha:
            alpha = lambda t: 0.05+0.3/t
        for i in range(n):
            x = self.iterations[self.it]
            self.it += 1
            gradf = self.df(x)
            self.velocity *= self.mass
            self.velocity -= gradf
            a = alpha(self.it)
            x += a*self.velocity
            if type(self.xrange) != type(None):
                # stay within range
                for j in range(self.nargs):
                    if x[j] < self.xrange[j][0]:
                        x[j] = self.xrange[j][0]
                    if x[j] > self.xrange[j][1]:
                        x[j] = self.xrange[j][1]
            self.iterations.append(x)
    def parameters(self, it=None):
        "get latest value gradient was calculated at"
        x = self.iterations[it if it else self.it]
        return tuple(x[i] for i in range(self.nargs))

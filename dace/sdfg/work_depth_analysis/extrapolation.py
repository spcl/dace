from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def print_scores(scores):
    for k, v in scores.items():
        print(k.name, v)

class Logistic:

    def __init__(self, name):
        self.x_name = name
        self.name = 'Logistic'

    def f(x, a, b, c):
        return b / (c + np.exp(-a * x))
    
    def fit(self, x, y):
        param, _ = curve_fit(Logistic.f, x, y, maxfev=10000)
        self.a, self.b, self.c = param

    def predict(self, x):
        return Logistic.f(x, self.a, self.b, self.c)
    
    def to_string(self):
        return f'{self.b} / ({self.c} + exp({-self.a} * {self.x_name}))'

class Log:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Log'


    def f(x, a, b):
        return a * np.log(x) + b
    
    def fit(self, x, y):
        param, _ = curve_fit(Log.f, x, y, maxfev=2500)
        self.a, self.b = param

    def predict(self, x):
        return Log.f(x, self.a, self.b)
    
    def to_string(self):
        return f'{self.a} * log({self.x_name}) + {self.b}'

class Plateau:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Plateau'


    def f(x, a, b):
        return (a * x) / (x + b) 
    
    def fit(self, x, y):
        param, _ = curve_fit(Plateau.f, x, y, maxfev=2500)
        self.a, self.b = param

    def predict(self, x):
        return Plateau.f(x, self.a, self.b)
    
    def to_string(self):
        return f'({self.a} * {self.x_name}) / ({self.x_name} + {self.b})'


class Poly:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Poly'


    def f(x, a, b):
        return a * x + b
    
    def fit(self, x, y):
        param, _ = curve_fit(Poly.f, x, y, maxfev=2500)
        self.a, self.b = param

    def predict(self, x):
        return Poly.f(x, self.a, self.b)
    
    def to_string(self):
        return f'{self.a} * {self.x_name} + {self.b}'

class Sqrt:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Sqrt'

    def f(x, a, b):
        return a * np.sqrt(x) + b
    
    def fit(self, x, y):
        param, _ = curve_fit(Sqrt.f, x, y, maxfev=2500)
        self.a, self.b = param

    def predict(self, x):
        return Sqrt.f(x, self.a, self.b)
    
    def to_string(self):
        return f'{self.a} * sqrt({self.x_name}) + {self.b}'  

class Exponential:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Exponential'

    def f(x, a, b):
        return a * np.exp(x) + b
    
    def fit(self, x, y):
        param, _ = curve_fit(Exponential.f, x, y, maxfev=2500)
        self.a, self.b = param

    def predict(self, x):
        return Exponential.f(x, self.a, self.b)
    
    def to_string(self):
        return f'{self.a} * np.exp({self.x_name}) + {self.b}'
    
class Sin:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Sin'

    def f(x, a, b, c, d):
        return a * np.sin(b*x + c) + d
    
    def fit(self, x, y):
        param, _ = curve_fit(Sin.f, x, y, maxfev=2500)
        self.a, self.b, self.c, self.d = param

    def predict(self, x):
        return Sin.f(x, self.a, self.b, self.c, self.d)
    
    def to_string(self):
        return f'{self.a} * sin({self.b}*{self.x_name} + {self.c}) + {self.d}'

class Constant:
    def __init__(self, name):
        self.x_name = name
        self.name = 'Sin'

    def f(x, a):
        return np.ones_like(x) * a
    
    def fit(self, x, y):
        param, _ = curve_fit(Constant.f, x, y, maxfev=2500)
        self.a = param

    def predict(self, x):
        return Constant.f(x, self.a)
    
    def to_string(self):
        return f'{self.a}'



def extrapolate(op_in_map, range_symbol):
    """
    For each key in op_in_map (aka for each SDFG element), we have a list of measured data points y
    for the values in x_values.
    Now we fit a curve and return the best function found via leave-one-out cross validation.
    """

    if len(range_symbol) == 1:
        # only 1 independent variable
        symbol_name = list(range_symbol.keys())[0]
        x = range_symbol[symbol_name].to_list()

        models = [Logistic(symbol_name), Log(symbol_name), Plateau(symbol_name), Poly(symbol_name), Sqrt(symbol_name),
                Exponential(symbol_name), Sin(symbol_name), Constant(symbol_name)]

        for element, y in op_in_map.items():
            all_zero = True
            for q in y:
                if q != 0.0:
                    all_zero = False
                    break
            if all_zero:
                op_in_map[element] = str(0)
                continue
            scores = {}
            for model in models:
                error_sum = 0
                for left_out in range(len(x)):
                    xx = list(x)
                    test_x = xx.pop(left_out)
                    yy = list(y)
                    test_y = yy.pop(left_out)
                    try:
                        model.fit(xx, yy)
                    except RuntimeError:
                        # triggered if no fit was found --> give huge error
                        error_sum += 999999999
                    # predict on left out sample
                    pred = model.predict(test_x)
                    # squared_error = np.square(pred - test_y)
                    # error_sum += squared_error
                    root_error = np.sqrt(np.abs(float(pred - test_y)))
                    error_sum += root_error

                mean_error = error_sum / len(x)
                scores[model] = mean_error

                

            # find model with least error
            min_model = model
            min_error = mean_error
            for model, error in scores.items():
                if error < min_error:
                    min_error = error
                    min_model = model
            
            # fit best model to all points and plot
            min_model.fit(x, y)
            fig, ax = plt.subplots()  # Create a figure containing a single axes.
            ax.scatter(x, y)
            s = 1
            t = x[-1] + 3
            q = np.linspace(s, t, num=(t-s)*5)
            r = min_model.predict(q)
            ax.plot(q, r, label=min_model.to_string())

            fig.tight_layout()
            plt.show()

            op_in_map[element] = min_model.to_string()

    else:
        print('2 independent variables not implemented yet')
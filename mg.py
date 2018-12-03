import numpy as np
import matplotlib.pyplot as plt
import math


class MackeyGlass:
    """
    Generate time-series using the Mackey-Glass equation.
    Equation is numerically integrated by using a fourth-order Runge-Kutta method

    """
    def __init__(self, alpha=0.2, beta=10, gamma=0.1, tau=17):
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.tau = tau
        #self.sample = sample

    def f(self, y_t, y_t_minus_tau):
        return -self.gamma*y_t + self.alpha*y_t_minus_tau/(1 + y_t_minus_tau**self.beta)

    def rk4(self, y_t, y_t_minus_tau, delta_t):
        k1 = delta_t*self.f(y_t,          y_t_minus_tau)
        k2 = delta_t*self.f(y_t+0.5*k1,   y_t_minus_tau)  # + delta_t*0.5
        k3 = delta_t*self.f(y_t+0.5*k2,   y_t_minus_tau) #  + delta_t*0.5
        k4 = delta_t*self.f(y_t+k3,       y_t_minus_tau) # + delta_t
        return y_t + k1/6 + k2/3 + k3/3 + k4/6
    def gen(self, y0=0.5, delta_t=1, n=12000):
        time = 0
        index = 1
        history_length = math.floor(self.tau / delta_t)
        y_history = np.full((history_length, 1), 0.5)
        y_t = y0

        Y = np.zeros((n+1, 1))
        T = np.zeros((n+1, 1))

        for i in range(n+1):
            Y[i] = y_t
            time = time + delta_t
            T[i] = time
            if self.tau == 0:
                y_t_minus_tau = y0
            else:
                y_t_minus_tau = y_history[index]

            y_t_plus_delta = self.rk4(y_t, y_t_minus_tau, delta_t)
            
            print(y_t, y_t_minus_tau, y_t_plus_delta)
            if self.tau != 0:
                y_history[index] = y_t_plus_delta
                index = (index+1) % history_length
            y_t = y_t_plus_delta
        return Y, T
    def plot(self, discard=250*10):
        Y, T = self.gen()
        Y = Y[discard:]
        T = T[discard:]
        ''' plot  '''
        #plt.plot(Y[:-tau], Y[tau:])
        plt.plot(Y[2000-self.tau:2500-self.tau], Y[2000:2500])
        #plt.plot(Y[2000:2500], Y[2000-self.tau:2500-self.tau]) #reverse x,y

        ''' plot labels '''
        plt.title('Mackey-Glass delay differential equation, tau = {}'.format(self.tau))
        plt.xlabel(r'$x(t - \tau)$')
        plt.ylabel(r'$x(t)$')
        plt.show()

mc = MackeyGlass()

mc.plot()
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from sklearn.model_selection import train_test_split


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

    def get_batches(self, x, y, batch_size=20, T=100):
        """ Return a generator for batches """
        n_batches = len(x) // (batch_size * T)
        x, y = x[:n_batches*batch_size*T], y[:n_batches*batch_size*T]
        t_count = 0
        for _ in range(n_batches):
            x_samples = []
            y_samples = []
            for _ in range(batch_size):
                x_samples.append(x[t_count:t_count+T])
                y_samples.append(y[t_count:t_count+T])
            t_count += T
            yield x_samples, y_samples

    def create_de_lstm(self, y_head, NUM_CELL=128):
        # variables
        K = 100  # number of classes
        #std = 0.2 * np.std(y_head)
        #delta_y = 0.04 * np.std(y_head)
        delta_y = np.max(y_head) / K
        print(f"delta_y: {delta_y}")
        # data split
        X_train, X_test = train_test_split(y_head, test_size=0.2, random_state=1)
        y_test = X_test[1:]
        X_test = X_test[:-1]
        X_train, X_val = train_test_split(X_train, test_size=0.2, random_state=1)
        y_train = X_train[1:]
        y_val = X_val[1:]
        X_train = X_train[:-1]
        X_val = X_val[:-1]
        # placeholders
        batch_size = 50
        T = 100
        x = tf.placeholder(tf.float32, [None, None, 1], name='x') # placeholder for inputs
        y = tf.placeholder(tf.float32, [None, None, 1], name='y') # placeholder for outputs
        # model
        cell = tf.nn.rnn_cell.LSTMCell(NUM_CELL, state_is_tuple=True)
        #initial_state = tf.placeholder(tf.float32, ([None, NUM_CELL], [None, NUM_CELL]), name="initial_state")
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
        # softmax output layer
        outputs = tf.reshape(outputs, [-1, NUM_CELL])
        y_ = tf.reshape(y, [-1])

        l_st = tf.layers.dense(outputs, 1, name='prediction')
        l_st = tf.reshape(l_st, [-1])

        # loss and training step
        loss_ = tf.losses.mean_squared_error(labels=y_, predictions=l_st)
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_)

        # training loop
        validation_acc = []
        validation_loss = []

        train_acc = []
        train_loss = []

        epochs = 20

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.initialize_local_variables())
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            iteration = 1
            state = sess.run(initial_state)
            for e in range(epochs):
                # Loop over batches
                for x_, y_ in self.get_batches(X_train, y_train, batch_size, T):
                    # Feed dictionary
                    x_ = np.array(x_)
                    y_ = np.array(y_)
                    x_ = np.reshape(x_, [batch_size, T, 1])
                    y_ = np.reshape(y_, [batch_size, T, 1])
                    feed = {x: x_, y: y_, initial_state: state}
                    loss, _ , new_state = sess.run([loss_, train_step, final_state], 
                                                    feed_dict = feed)
                    train_loss.append(loss)

                    # Print on every 5th iteration
                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, epochs),
                            "Iteration: {:d}".format(iteration),
                            "Train loss: {:6f}".format(np.mean(train_loss)))
                    
                    # Compute validation loss on every 25th iteration
                    if (iteration%25 == 0):
                        val_acc_ = []
                        val_loss_ = []
                        state_v = sess.run(initial_state)
                        for x_v, y_v in self.get_batches(X_val, y_val, batch_size, T):
                            x_v = np.array(x_v)
                            y_v = np.array(y_v)
                            x_v = np.reshape(x_v, [batch_size, T, 1])
                            y_v = np.reshape(y_, [batch_size, T, 1])

                            # Feed
                            feed = {x : x_v, y : y_v, initial_state: state_v}
                            
                            # Loss and new state
                            loss_v, state_v = sess.run([loss_, final_state], feed_dict = feed)
                            
                            val_loss_.append(loss_v)
                        
                        # Print info
                        print("Epoch: {}/{}".format(e, epochs),
                            "Iteration: {:d}".format(iteration),
                            "Validation loss: {:6f}".format(np.mean(val_loss_)),
                            "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                        
                        # Store
                        validation_acc.append(np.mean(val_acc_))
                        validation_loss.append(np.mean(val_loss_))
                    
                    # Iterate 
                    iteration += 1
                    state = new_state

        
            #tf.train.Saver().save(sess,"model/lstm.ckpt")
            #tf.train.Saver().restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_loss = []
            state_t = sess.run(initial_state)
            for x_t, y_t in self.get_batches(X_test, y_test, batch_size, T):
                x_t = np.array(x_t)
                y_t = np.array(y_t)
                x_t = np.reshape(x_t, [batch_size, T, 1])
                y_t = np.reshape(y_, [batch_size, T, 1])
                feed = {x: x_t,
                        y: y_t,
                        initial_state: state_t
                        }
                
                loss_test, test_state = sess.run([loss_, final_state], feed_dict=feed)
                test_loss.append(loss_test)
            print("Train loss: {:6f}".format(np.mean(test_loss)))

    def f(self, y_t, y_t_minus_tau):
        return -self.gamma*y_t + self.alpha*y_t_minus_tau/(1 + y_t_minus_tau**self.beta)

    def rk4(self, y_t, y_t_minus_tau, delta_t):
        k1 = delta_t*self.f(y_t,          y_t_minus_tau)
        k2 = delta_t*self.f(y_t+0.5*k1,   y_t_minus_tau)    # + delta_t*0.5
        k3 = delta_t*self.f(y_t+0.5*k2,   y_t_minus_tau)    # + delta_t*0.5
        k4 = delta_t*self.f(y_t+k3,       y_t_minus_tau)    # + delta_t
        return y_t + k1/6 + k2/3 + k3/3 + k4/6
         
    def gen(self, y0=0.5, delta_t=1, n=160000):
        time = 0
        index = 1
        history_length = math.floor(self.tau / delta_t)
        y_history = np.full(history_length, 0.5)
        y_t = y0
        y_t_ = 0
        Y = np.zeros(n+1)
        X = np.zeros(n+1)
        T = np.zeros(n+1)

        for i in range(n+1):
            Y[i] = y_t
            X[i] = y_t_
            time = time + delta_t
            T[i] = time
            if self.tau == 0:
                y_t_minus_tau = y0
            else:
                y_t_minus_tau = y_history[index]

            y_t_plus_delta = self.rk4(y_t, y_t_minus_tau, delta_t)
            #print(y_t, y_t_minus_tau, y_t_plus_delta, time)
            if self.tau != 0:
                y_history[index] = y_t_plus_delta
                index = (index+1) % history_length
            y_t_ = y_t
            y_t = y_t_plus_delta
        return Y, T, X
    def plot(self, discard=250*10):
        Y, T, X = self.gen()
        Y = Y[discard:]
        T = T[discard:]
        X = X[discard:]
        ''' plot  '''
        #plt.plot(Y[:-tau], Y[tau:])
        plt.plot(Y[2000-self.tau:2500-self.tau], Y[2000:2500])
        #plt.plot(Y[2000:2500], Y[2000-self.tau:2500-self.tau]) #reverse x,y
        #plt.plot(Y[2000:2500], X[2000:2500])
        ''' plot labels '''
        plt.title('Mackey-Glass delay differential equation, tau = {}'.format(self.tau))
        plt.xlabel(r'$x(t - \tau)$')
        plt.ylabel(r'$x(t)$')
        plt.show()

mc = MackeyGlass()
y, t, x = mc.gen()
#mc.plot()
print((np.max(y) - np.min(y))/100, np.std(y), 0.03*np.std(y))
print(np.max(y))
mc.create_de_lstm(y)
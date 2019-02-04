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
        #self.sample = sample

    def prepare_labels(self, y_head, delta_y, K=100):
        labels = np.zeros((y_head.shape[0], K))
        i = 0
        for elem in y_head:
            if elem == 0:
                k = 0
            else:
                k = int(np.ceil(elem / delta_y) - 1)
            labels[i,k] = 1
            i += 1
        return labels

    def get_batches(self, x, y, batch_size = 20):
        """ Return a generator for batches """
        T = 100
        n_batches = len(x) // (batch_size * T)
        x, y = x[:n_batches*batch_size*T], y[:n_batches*batch_size*T]
        t_count = 0
        for _ in range(n_batches):
            x_samples = []
            y_samples = []
            for _ in range(20):
                x_samples.append(x[t_count:t_count+T])
                y_samples.append(y[t_count:t_count+T])
            t_count += T
            yield x_samples, y_samples

        '''
        # Loop over batches and yield
        b = 0
        bb = 0
        while b < len(X) and bb < 21:
            x_ = []
            y_ = []
            for t in range(20):
                x_.append(X[b:b+100])
                y_.append(y[b:b+100])
            b += 100
            bb += 1
            yield x_, y_
        '''


    def create_de_lstm(self, y_head, NUM_CELL=128):
        # variables
        K = 100  # number of classes
        std = 0.2 * np.std(y_head)
        #delta_y = 0.04 * np.std(y_head)
        delta_y = np.max(y_head) / K
        print(f"delta_y: {delta_y}")
        # data split
        prepared_labels = self.prepare_labels(y_head, delta_y, K)
        print(y_head[0], np.argmax(prepared_labels[0]))
        #y_head += np.random.normal(0, std, y_head.shape)
        X_train, X_test, y_train, y_test = train_test_split(y_head, prepared_labels, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
        print(X_train[0], np.argmax(y_train[0]))
        # placeholders
        batch_size = 20
        T = 100
        x = tf.placeholder(tf.float32, [None, None, 1], name='x') # placeholder for inputs
        y = tf.placeholder(tf.uint8, [None, None, K], name='y') # placeholder for outputs
        # model
        cell = tf.nn.rnn_cell.LSTMCell(NUM_CELL, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
        print(outputs.shape)
        # softmax output layer
        outputs = tf.reshape(outputs, [-1, NUM_CELL])
        y_ = tf.reshape(y, [-1, K])
        print(outputs.shape)
        #l_st = tf.layers.dense(outputs, K, name='logits')
        l_st = tf.contrib.layers.linear(outputs, K)
        print(l_st.shape)
        #Hf = tf.reshape(outputs, [-1, NUM_CELL])                   
        #y_logits = tf.contrib.layers.linear(Hf, K)
        y__ = tf.nn.softmax(l_st)
        print(f"y__ shape, y_ shape {y__.shape} {y_.shape}")
        yp = tf.argmax(y__, 1)
        print(f"yp shape: {yp.shape}")
        yp = tf.reshape(yp, [batch_size, -1])
        # outputs (20, ?, 128)
        # Hf shape: (?, 128)
        # l_st shape: (20, ?, 100)
        # y_logits shape : (?, 100)
        # y shape: (?, 100)
        # yp before reshape: (?,)
        # yp after reshape: (20, ?)


        # loss and training step
        loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits = l_st, labels = y_)
        #loss_ = tf.reshape(loss_, [batch_size, -1]) 
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_)

        # Accuracy
        #correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(yp, 1))
        correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y__, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        # training loop
        validation_acc = []
        validation_loss = []

        train_acc = []
        train_loss = []

        epochs = 20

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            iteration = 1
            
            for e in range(epochs):
                # Loop over batches
                print(X_train[0], np.argmax(y_train[0]))
                for x_, y_ in self.get_batches(X_train, y_train, batch_size):
                    # Feed dictionary
                    x_ = np.array(x_)
                    y_ = np.array(y_)
                    #print(x_.shape, y_.shape)
                    x_ = np.reshape(x_, [batch_size, T, 1])
                    #y_ = np.reshape(y_, [-1, K])
                    feed = {x: x_, y : y_}
                    loss, _ , state, acc = sess.run([loss_, train_step, final_state, accuracy], 
                                                    feed_dict = feed)
                    train_acc.append(acc)
                    train_loss.append(loss)
                    
                    # Print at each 5 iters
                    if (iteration % 5 == 0):
                        print("Epoch: {}/{}".format(e, epochs),
                            "Iteration: {:d}".format(iteration),
                            "Train loss: {:6f}".format(np.mean(train_loss)),
                            "Train acc: {:.6f}".format(np.mean(train_acc)))
                    
                    # Compute validation loss at every 25 iterations
                    if (iteration%25 == 0):
                        val_acc_ = []
                        val_loss_ = []
                        for x_v, y_v in self.get_batches(X_val, y_val, batch_size):
                            x_v = np.array(x_v)
                            y_v = np.array(y_v)
                            x_v = np.reshape(x_v, [batch_size, T, 1])
                            #y_v = np.reshape(y_v, [-1, K])
                            # Feed
                            feed = {x : x_v, y : y_v}
                            
                            # Loss
                            loss_v, state_v, acc_v = sess.run([loss_, final_state, accuracy], feed_dict = feed)
                            
                            val_acc_.append(acc_v)
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
        
            #tf.train.Saver().save(sess,"model/lstm.ckpt")
            #tf.train.Saver().restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_acc = []
            for x_t, y_t in self.get_batches(X_test, y_test, batch_size):
                x_t = np.array(x_t)
                y_t = np.array(y_t)
                x_t = np.reshape(x_t, [batch_size, T, 1])
                #y_t = np.reshape(y_t, [-1, K])
                feed = {x: x_t,
                        y: y_t}
                
                batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print("Test accuracy: {:.6f}".format(np.mean(test_acc)))

    def f(self, y_t, y_t_minus_tau):
        return -self.gamma*y_t + self.alpha*y_t_minus_tau/(1 + y_t_minus_tau**self.beta)

    def rk4(self, y_t, y_t_minus_tau, delta_t):
        k1 = delta_t*self.f(y_t,          y_t_minus_tau)
        k2 = delta_t*self.f(y_t+0.5*k1,   y_t_minus_tau)  # + delta_t*0.5
        k3 = delta_t*self.f(y_t+0.5*k2,   y_t_minus_tau) #  + delta_t*0.5
        k4 = delta_t*self.f(y_t+k3,       y_t_minus_tau) # + delta_t
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
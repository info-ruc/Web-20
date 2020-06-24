import tensorflow as tf
import config

class Model():
    def __init__(self):

        self.input_size = config.input_size
        self.cnn_input_size = 5
        self.lstm1_cell_size = config.lstem1_cell_size
        self.lstm2_cell_size = config.lstem2_cell_size
        self.timestep1 = config.timestep1
        self.timestep2 = config.timestep2
        self.cnnwidth = config.cnnwidth
        self.cnnheight = config.cnnheight
        self.forget_biases = 1
        self.cnn_output_size = config.cnn_output_size
        self.cnn_layer = config.cnn_layer
        self.cnn_keep_prob = config.cnn_keep_prob

        self.uselstm1 = config.uselstm1
        self.uselstm2 = config.uselstm2
        self.usecnn = config.usecnn

        self.output_size = config.output_size

        self.lr = config.lr

        self.padding_size = config.padding_size

        self.hidden_size = config.hidden_size
        #self.keep_prob = 0.5

        self.optimizer = config.optimizer

        self.usebn = config.usebn

        self.uselastcell_lstm1 = config.uselastcell_lstm1

        self.uselastcell_lstm2 = config.uselastcell_lstm2


        with tf.name_scope('input'):
            self.lstm_input1 = tf.placeholder(tf.float32,
                                              [None, self.timestep1, self.input_size],
                                              name='lstm_input1')
            self.lstm_input2 = tf.placeholder(tf.float32,
                                              [None, self.timestep2, self.input_size],
                                              name='lstm_input2')
            self.cnn_input = tf.placeholder(tf.float32,
                                            [None,self.cnn_input_size,self.cnnheight, self.cnnwidth],
                                            name='cnn_input')

            self.keep_prob = tf.placeholder(tf.float32)



            self.ys = tf.placeholder(tf.float32,[None,self.output_size],name='output')


        with tf.name_scope('model'):
            if self.uselstm1:
                with tf.variable_scope('lstm_part1'):
                    self.add_lstm1_input_layer(name='lstm1_input')
                with tf.variable_scope('lstm_cell1'):
                    self.add_lstm1_cell()


            if config.uselstm2:
                with tf.variable_scope('lstm_part2'):
                    self.add_lstm2_input_layer(name='lstm2_input')
                with tf.variable_scope('lstm_cell2'):
                    self.add_lstm2_cell()

            if config.usecnn:

                x_image = tf.transpose(self.cnn_input,perm=[0,2,3,1])
                with tf.variable_scope("conv1"):
                    W_conv1 = self.cnn_weight_variable([self.cnnheight, self.cnnwidth, self.cnn_input_size, self.cnn_layer])
                    b_conv1 = self.cnn_bias_variable([self.cnn_layer])
                    h_conv1 = tf.nn.elu(self.conv2d(x_image, W_conv1) + b_conv1)  # 输出20 * 20 * 32
                    h_pool1 = self.max_pool_2x2(h_conv1)  # 输出 5 * 5 * 32

                with tf.variable_scope("cnn_output_layer1"):

                    W_fc1 = self._weight_variable([self.cnnheight//self.padding_size * self.cnnwidth//self.padding_size * self.cnn_layer, self.cnn_output_size])
                    b_fc1 = self._bias_variable([self.cnn_output_size])

                    h_pool2_flat = tf.reshape(h_pool1, [-1, self.cnnheight//self.padding_size * self.cnnwidth//self.padding_size * self.cnn_layer])

                    self.cnn_output1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                    self.cnn_output = tf.nn.dropout(self.cnn_output1, self.keep_prob)
                # with tf.variable_scope("cnn_output_layer2"):
                #     W_fc2 = self._weight_variable(
                #         [self.cnn_output_size,self.cnn_output_size])
                #     b_fc2 = self._bias_variable([self.cnn_output_size])
                #
                #     self.cnn_output = tf.nn.relu(tf.matmul(self.cnn_output1, W_fc2) + b_fc2)
                #     self.cnn_output = tf.nn.dropout(self.cnn_output, self.keep_prob)

        # with tf.name_scope('ensemble_output'):
        #     if self.uselstm1 and self.uselstm2 and self.usecnn:
        #         self.ensemble_output = tf.concat([self.lstm1_cell_outputs,self.lstm2_cell_outputs,self.cnn_output],axis=1)
        #         self.ensemble_size = self.lstm1_output_size + self.lstm2_output_size + self.cnn_output_size
        #
        #     elif self.uselstm1 and self.uselstm2:
        #         self.ensemble_output = tf.concat([self.lstm1_cell_outputs, self.lstm2_cell_outputs],axis=1)
        #         self.ensemble_size = self.lstm1_output_size + self.lstm2_output_size
        #
        #
        #     elif self.uselstm1 and self.usecnn:
        #         self.ensemble_output = tf.concat([self.lstm1_cell_outputs, self.cnn_output],axis=1)
        #         self.ensemble_size = self.lstm1_output_size + self.cnn_output_size
        #
        #     elif self.uselstm2 and self.usecnn:
        #         self.ensemble_output = tf.concat([self.lstm2_cell_outputs, self.cnn_output],axis=1)
        #         self.ensemble_size = self.lstm2_output_size + self.cnn_output_size
        #
        #     elif self.uselstm1:
        #         self.ensemble_output = self.lstm1_cell_outputs
        #         self.ensemble_size = self.lstm1_output_size
        #
        #
        #     elif self.uselstm2:
        #         self.ensemble_output = self.lstm2_cell_outputs
        #         self.ensemble_size = self.lstm2_output_size
        #
        #     else :
        #         self.ensemble_output = self.cnn_output
        #         self.ensemble_size = self.cnn_output_size


        with tf.name_scope('ensemble_output'):
            # if self.uselstm1 and self.uselstm2 and self.usecnn:
            #     self.ensemble_output = tf.concat([self.lstm1_cell_outputs,self.lstm2_cell_outputs,self.cnn_output],axis=1)
            #     self.ensemble_size = self.lstm1_output_size + self.lstm2_output_size + self.cnn_output_size

            if self.uselstm1 and self.uselstm2:
                self.ensemble_output = tf.concat([self.lstm1_cell_outputs, self.lstm2_cell_outputs],axis=1)
                self.ensemble_size = self.lstm1_output_size + self.lstm2_output_size


            # elif self.uselstm1 and self.usecnn:
            #     self.ensemble_output = tf.concat([self.lstm1_cell_outputs, self.cnn_output],axis=1)
            #     self.ensemble_size = self.lstm1_output_size + self.cnn_output_size
            #
            # elif self.uselstm2 and self.usecnn:
            #     self.ensemble_output = tf.concat([self.lstm2_cell_outputs, self.cnn_output],axis=1)
            #     self.ensemble_size = self.lstm2_output_size + self.cnn_output_size
            #
            # elif self.uselstm1:
            #     self.ensemble_output = self.lstm1_cell_outputs
            #     self.ensemble_size = self.lstm1_output_size


            elif self.uselstm2:
                self.ensemble_output = self.lstm2_cell_outputs
                self.ensemble_size = self.lstm2_output_size

            else :
                self.ensemble_output = self.cnn_output
                self.ensemble_size = self.cnn_output_size


        with tf.variable_scope('out_hidden'):
            if self.usebn:
                self.add_bn_ensemble_hidden_layer()
            else:
                self.add_ensemble_hidden_layer()
        with tf.variable_scope('output'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            #self.train_op = tf.train.GradientDescentOptimizer(LR)
            if self.optimizer == 'adam':
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
            else:
                self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)

    def compute_cost(self):

        losses = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.reshape(self.pred, [-1], name='reshape_pred'),
                                                      tf.reshape(self.ys, [-1], name='reshape_target'))))

       # losses = tf.reduce_mean(tf.abs(tf.reshape(self.pred, [-1], name='reshape_pred') - tf.reshape(self.ys, [-1], name='reshape_target')))

        with tf.name_scope('average_cost'):
            # self.cost = tf.div(
            #     tf.reduce_sum(losses, name='losses_sum'),
            #     self.batch_size,
            #     name='average_cost')
            self.cost = losses
            tf.summary.scalar('cost', self.cost)
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.reshape(self.pred, [-1], name='reshape_pred'),
                                                      tf.reshape(self.ys, [-1], name='reshape_target'))))


    def add_ensemble_hidden_layer(self):
        # shape = (batch * steps, cell_size)

        with tf.name_scope("out"):
            Ws_out = self._weight_variable([self.ensemble_size, self.hidden_size])
            bs_out = self._bias_variable([self.hidden_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            #self.ensemble_output = tf.contrib.layers.batch_norm(self.ensemble_output)
            self.hidden = tf.nn.relu(tf.matmul(self.ensemble_output, Ws_out) + bs_out)
            self.hidden = tf.nn.dropout(self.hidden,self.keep_prob)




    def add_bn_ensemble_hidden_layer(self):
        with tf.name_scope("out"):
            Ws_out = self._weight_variable([self.ensemble_size, self.hidden_size])
            bs_out = self._bias_variable([self.hidden_size, ])
        with tf.name_scope('Wx_plus_b'):
            h1 = tf.matmul(self.ensemble_output, Ws_out)
            batch_mean, batch_var = tf.nn.moments(h1, [0])
            scale2 = tf.Variable(tf.ones([self.hidden_size]))
            beta2 = tf.Variable(tf.zeros([self.hidden_size]))
            self.hidden = tf.nn.batch_normalization(h1, batch_mean, batch_var, beta2, scale2,
                                                          variance_epsilon=1e-3)
            self.hidden = tf.nn.dropout(self.hidden, self.keep_prob)

    def add_output_layer(self):
        # shape = (batch * steps, cell_size)

        with tf.name_scope("out"):
            if self.usecnn:
                Ws_out = self._weight_variable([self.hidden_size + self.cnn_output_size, self.output_size])
                bs_out = self._bias_variable([self.output_size, ])
            else:
                Ws_out = self._weight_variable([self.hidden_size, self.output_size])
                bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            if self.usecnn:
                self.hidden = tf.concat([self.hidden, self.cnn_output],axis=1)
                #self.hidden = tf.contrib.layers.batch_norm(self.hidden)
            self.pred = tf.matmul(self.hidden, Ws_out) + bs_out

    def add_lstm1_input_layer(self,name):
        l_in_x = tf.reshape(self.lstm_input1, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.lstm1_cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.lstm1_cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.lstm1_in_y = tf.reshape(l_in_y, [-1, self.timestep1, self.lstm1_cell_size], name='2_3D')
        #self.lstm1_in_y = tf.reshape(self.lstm1_in_y[:, -1, :], [-1, self.lstm1_cell_size], name='2_2D')
        self.lstm1_in_y = tf.nn.dropout(self.lstm1_in_y,self.keep_prob)


    def add_lstm1_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm1_cell_size, forget_bias=self.forget_biases, state_is_tuple=True)
        # with tf.name_scope('initial_state'):
        #     self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.lstm1_cell_outputs, self.lstm1_cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.lstm1_in_y, dtype=tf.float32, time_major=False)

        if self.uselastcell_lstm1:
            self.lstm1_cell_outputs = tf.reshape(self.lstm1_cell_outputs[:, -1, :], [-1, self.lstm1_cell_size], name='2_2D')
            self.lstm1_output_size = self.lstm1_cell_size
        else:
            self.lstm1_cell_outputs = tf.reshape(self.lstm1_cell_outputs, [-1, self.timestep1 * self.lstm1_cell_size],
                                                 name='2_2D')
            self.lstm1_output_size = self.timestep1 * self.lstm1_cell_size

    def add_lstm2_input_layer(self, name):
        l_in_x = tf.reshape(self.lstm_input2, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.lstm2_cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.lstm2_cell_size, ])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.lstm2_in_y = tf.reshape(l_in_y, [-1, self.timestep2, self.lstm2_cell_size], name='2_3D')
        #self.lstm2_in_y = tf.reshape(self.lstm2_in_y [:, -1, :], [-1, self.lstm2_cell_size], name='2_2D')
        self.lstm2_in_y = tf.nn.dropout(self.lstm2_in_y,self.keep_prob)

    def add_lstm2_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm2_cell_size, forget_bias=self.forget_biases,
                                                 state_is_tuple=True)
        # with tf.name_scope('initial_state'):
        #     self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.lstm2_cell_outputs, self.lstm2_cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.lstm2_in_y, dtype=tf.float32, time_major=False)

        if self.uselastcell_lstm2:
            self.lstm2_cell_outputs = tf.reshape(self.lstm2_cell_outputs[:, -1, :], [-1, self.lstm2_cell_size], name='2_2D')
            self.lstm2_output_size = self.lstm2_cell_size
        else:
            self.lstm2_cell_outputs = tf.reshape(self.lstm2_cell_outputs, [-1, self.timestep2 * self.lstm2_cell_size],
                                                 name='2_2D')
            self.lstm2_output_size = self.timestep2 * self.lstm2_cell_size


    def cnn_weight_variable(self,shape):
        inital = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(inital)

    def cnn_bias_variable(self,shape):
        inital = tf.constant(0.1, shape=shape)
        return tf.Variable(inital)

    def conv2d(self,x, W):
        # strides是步长，四维的列表 [1,x_movement,y_movement,1]
        # must have strides[0]=strides[3]=1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, self.padding_size, self.padding_size, 1], strides=[1, self.padding_size, self.padding_size, 1], padding='SAME')

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

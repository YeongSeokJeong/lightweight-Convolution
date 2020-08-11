import tensorflow as tf

class lightweight_Convolution(tf.keras.layers.Layer):
    def __init__(self, d_model, filter_size, head_num):
        super(lightweight_Convolution, self).__init__()
        initializer = tf.keras.initializers.Zeros()
        self.cnn_layers = [Conv2D(1,(1, filter_size), padding = 'same', activation = 'relu', kernel_initializer = initializer) for _ in range(head_num)]
        self.d_model = d_model
        self.filter_size = filter_size
        self.head_num = head_num
        
    def reshape(self, inp_tensor):
        batch_size = tf.shape(inp_tensor)[0]
        seq_len = tf.shape(inp_tensor)[1]
        inp_tensor = tf.reshape(inp_tensor, shape = (batch_size, self.head_num, seq_len, self.d_model//self.head_num, 1))
        return inp_tensor
    
    def softmax_weight_norm(self, cnn_layer):
        try:
            if not cnn_layer.get_weights():
                return cnn_layer
            origin_weight, origin_bias = cnn_layer.get_weights()
            origin_shape = origin_weight.shape
            origin_weight = tf.reshape(origin_weight, [-1,self.filter_size])
            
            softmax_norm_weight = tf.nn.softmax(origin_weight)
            softmax_norm_weight = tf.reshape(softmax_norm_weight, origin_shape)
            cnn_layer.set_weights([softmax_norm_weight, origin_bias])
        except RuntimeError as x:
            print(x)
        finally:
            return cnn_layer
        return cnn_layer
    
    def call(self, inp_tensor):
        batch_size = tf.shape(inp_tensor)[0]
        inp_tensor = self.reshape(inp_tensor)
        oup_tensor = []
        for i in range(self.head_num):
            self.cnn_layers[i] = self.softmax_weight_norm(self.cnn_layers[i])
            step_head = self.cnn_layers[i](inp_tensor[::,i])
            oup_tensor.append(step_head)
        oup_tensor = tf.reshape(oup_tensor, shape = (batch_size, -1,self.d_model))
        return oup_tensor
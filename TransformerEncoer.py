from keras.layers import Layer, Reshape, Permute, LayerNormalization, Dropout, Dense, LSTM
import tensorflow as tf
import numpy as np

class PE_original(Layer):
    def __init__(self, emd_dim: int, max_seqL=256, scale_v=10000):
        super(PE_original, self).__init__()
        self.emd_dim = emd_dim
        self.max_seqL = max_seqL
        self.scale_v = scale_v

    def build(self, input_shape):
        # 创建位置编码矩阵
        encoding = np.zeros((self.max_seqL, self.emd_dim))
        pos = np.arange(0, self.max_seqL).reshape(-1, 1)
        _2i = np.arange(0, self.emd_dim, step=2)

        encoding[:, 0::2] = np.sin(pos / (self.scale_v ** (_2i / self.emd_dim)))
        encoding[:, 1::2] = np.cos(pos / (self.scale_v ** (_2i / self.emd_dim)))

        # 注册为不可训练权重
        self.encoding = self.add_weight(
            name='positional_encoding',
            shape=(self.max_seqL, self.emd_dim),
            initializer=tf.constant_initializer(encoding),
            trainable=False
        )
        super(PE_original, self).build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.encoding[:seq_len, :]


class PE_LSTM(Layer):
    """
    一个MxN的矩阵，可计算每个位置的位置情况
    """
    
    def __init__(self, emd_dim: int, seqL):
        self.emd_dim = emd_dim
        self.seqLL = seqL
        super(PE_LSTM, self).__init__()

    def build(self, input_shape):
        self.lstm = LSTM(self.emd_dim, activation='relu', return_sequences=True, name='lstm_hidden')
        super(PE_LSTM, self).build(input_shape)

    def call(self, x):
        lstm_hidden = self.lstm(x)
        return lstm_hidden


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, seql, is_resnet=False, mask=None):
        self.is_resnet = is_resnet
        self.mask = mask
        self.num_heads = num_heads
        self.d_model = d_model
        self.seql = seql
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads
        super(MultiHeadAttention, self).__init__()

    def build(self, input_shape):
        self.WQ = Dense(self.d_model, name='Q')
        self.WK = Dense(self.d_model, name='K')
        self.WV = Dense(self.d_model, name='V')
        self.dense = Dense(self.d_model)
        self.split = Reshape((self.seql, self.num_heads, self.depth))
        self.permute = Permute([2, 1, 3])
        self.concat = Reshape((self.seql, self.d_model))
        super(MultiHeadAttention, self).build(input_shape)
    
    def split_heads(self, x):
        x = self.split(x)
        x = self.permute(x)
        return x
    
    def call(self, x, mask=None):
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # 缩放点积
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.depth, tf.float32)  # 缩放的维度
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if self.mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        res = tf.transpose(output, perm=[0, 2, 1, 3])
        out = self.concat(res)
        if self.is_resnet:
            return x + out, attention_weights
        else:
            return out, attention_weights


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, seql, dropout_rate, mask=None, name=None):
        self.d_model = d_model
        self.num_head = num_heads
        self.seql = seql
        self.mask = mask
        self.dropout_rate = dropout_rate
        super(EncoderLayer, self).__init__(name=name)

    def build(self, input_shape):
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_head, 
                                     seql=self.seql, mask=self.mask)
        self.fnn = Dense(self.d_model, activation='relu')
        self.drop1 = Dropout(self.dropout_rate)
        self.drop2 = Dropout(self.dropout_rate)
        super(EncoderLayer, self).build(input_shape)
    
    def call(self, x, mask=None):
        attn, attention_weights = self.mha(x)
        attn = self.drop1(attn)
        n1 = self.layernorm1(x + attn)
        fnn = self.fnn(n1)
        fnn = self.drop2(fnn)
        n2 = self.layernorm2(fnn + n1)
        return n2, attention_weights


class Encoder(Layer):
    def __init__(self, d_model, num_heads, seql, dropout_rate, N_block, name=None, pe_type='LSTM'):
        self.d_model = d_model
        self.num_head = num_heads
        self.seql = seql
        self.N = N_block
        self.dropout_rate = dropout_rate
        self.pe_type = pe_type
        self.layers_list = []
        super(Encoder, self).__init__(name=name)

    def build(self, input_shape):
        if self.pe_type == 'LSTM':
            self.pe = PE_LSTM(self.d_model, self.seql)
        elif self.pe_type == 'original':
            self.pe = PE_original(self.d_model, self.seql)
        else:
            raise ValueError("pe_type must be 'LSTM' or 'original'")
            
        for idx in range(self.N):
            self.layers_list.append(EncoderLayer(d_model=self.d_model, num_heads=self.num_head, 
                                                seql=self.seql, dropout_rate=self.dropout_rate, 
                                                name=f'EncoderLayer_{idx}'))
        super(Encoder, self).build(input_shape)
    
    def call(self, x, mask=None):
        pe_embedding = self.pe(x)
        x = x + pe_embedding
        mha_results_list = []
        attn_results_list = []
        for mha_layer in self.layers_list:
            x, attention_weights = mha_layer(x)
            mha_results_list.append(x)
            attn_results_list.append(attention_weights)
        return x, pe_embedding, mha_results_list, attn_results_list
    
    def compute_output_shape(self, input_shape):
        # 返回四个输出的形状
        batch_size, seq_len, d_model = input_shape
        # 第一个输出: 编码后的特征
        output1_shape = (batch_size, seq_len, d_model)
        # 第二个输出: 位置编码
        output2_shape = (batch_size, seq_len, d_model)
        # 第三个输出: 每层的输出列表
        output3_shape = [self.N, batch_size, seq_len, d_model]
        # 第四个输出: 每层的注意力权重列表
        output4_shape = [self.N, batch_size, self.num_head, seq_len, seq_len]
        return [output1_shape, output2_shape, output3_shape, output4_shape]
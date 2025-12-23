import os
os.environ["KERAS_BACKEND"] = "jax" 
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras
from keras import ops
from keras import layers
import jax
from jax import numpy as jnp
import numpy as np

from functools import partial
from itertools import product
from einops import rearrange, repeat

def sin(x): # npu hack
    INV_PI_HALF = 0.6366197723675813
    PI_HALF = 1.5707963267948966
    S_c1, S_c2 = -0.166666667,  0.008333333
    C_c1, C_c2 = -0.500000000,  0.041666667
    q = ops.round(x * INV_PI_HALF)
    r = x - q * PI_HALF
    r2 = r * r
    m_cos = q - 2.0 * ops.floor(q * 0.5)
    rem4 = q - 4.0 * ops.floor(q * 0.25)
    m_neg = ops.floor(rem4 * 0.5)
    c1 = S_c1 + m_cos * (C_c1 - S_c1)
    c2 = S_c2 + m_cos * (C_c2 - S_c2)
    head = r + m_cos * (1.0 - r)
    sign = 1.0 - 2.0 * m_neg
    poly = 1.0 + r2 * (c1 + r2 * c2)
    return sign * head * poly

def cos(x):
    PI_HALF = 1.57079632679
    return sin(x + PI_HALF)

etable_min = -20
etable_max = 20
etable_values = ops.array(np.exp2(np.arange(etable_min, etable_max)).astype(np.float32))

def exp(x): # npu hack
    offset = -etable_min 
    LN2 = 0.69314718056     
    INV_LN2 = 1.44269504089 
    k = ops.cast(x * INV_LN2+0.5, 'int')
    r = x - k * LN2 
    c1 = 1.0
    c2 = 0.5            
    c3 = 0.166666667    
    c4 = 0.041666667    
    c5 = 0.008333333    
    poly = 1 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))))
    k = ops.clip(k, etable_min, etable_max-1)
    idx = k + offset
    scale = etable_values[idx]
    return poly * scale

def cpc(x):
    r, i = x[...,0], x[...,1]
    return r+1j*i

def cpd(x):
    return ops.stack([ops.real(x), ops.imag(x)], axis=-1)

def cp(x):
    return x+0j

class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def build(self, input_shape):
        super().build(input_shape)
        self.weight = self.add_weight(shape=(input_shape[-1],),
                                        dtype='float32',
                                        trainable=True)
    
    def call(self, x, z=None):
        x = ops.cast(x, 'float32')
        if z is not None:
            x = x*layers.Activation('relu6')(z)
        return x*ops.rsqrt(ops.mean((x*x), -1, True)+self.eps)*self.weight

def segsum(x):
    T = x.shape[-1]
    x = repeat(x, '... d -> ... d e', e=T)
    mask = np.tril(np.ones((T, T), dtype='float32'), -1)
    x *= mask
    x_segsum = ops.cumsum(x, axis=-2)
    mask = 1-np.tril(np.ones((T, T), dtype='float32'))
    #x_segsum -= ops.numpy.nan_to_num(ops.array(np.inf)*mask)
    x_segsum -= ops.exp(ops.array(1000)*mask)-1
    return x_segsum

sample = []

def ssd(x, A, B, C, chunk=64, init_stat=None):
    x, A, B, C = [rearrange(i, "b (c l) ... -> b c l ...", l=chunk) for i in (x, A, B, C)]
    A = rearrange(A, "b c l h -> b h c l")
    A_cum = ops.cumsum(A, axis=-1)
    L = jnp.exp(segsum(A))
    # print(C.dtype, B.dtype, L.dtype, x.dtype)
    Y_diag = jnp.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", ops.tile(C, (1, 1, 1, L.shape[1], 1)), ops.tile(B, (1, 1, 1, L.shape[1], 1)), L, x)
    
    decay_states = jnp.exp(A_cum[...,-1:]-A_cum)
    states = jnp.einsum("bclhn, bhcl, bclhp -> bchpn", ops.tile(B, (1, 1, 1, L.shape[1], 1)), decay_states, x)
    
    if init_stat is None:
        init_stat = ops.zeros_like(states[:, :1])
    states = ops.concatenate([init_stat, states], axis=1)
    decay_chunk = jnp.exp(segsum(ops.pad(A_cum[...,-1], ((0, 0), (0, 0), (1, 0)))))
    new_states = jnp.einsum('bhzc, bchpn -> bzhpn', decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1:]
    
    state_decay_out = jnp.exp(A_cum)
    Y_off = jnp.einsum("bclhn, bchpn, bhcl -> bclhp", ops.tile(C, (1, 1, 1, L.shape[1], 1)), states, state_decay_out)
    
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state

class SSMrPPG(layers.Layer):
    def __init__(self, expand=2, d_state=16, k_conv=4, chunk_size=64, headdim=64, dtype='float32', **kv):
        super().__init__(**kv)
        self._dtype = dtype
        self.expand = expand
        self.d_state = d_state
        self.k_conv = k_conv
        self.chunk_size = chunk_size
        self.headdim = headdim
        
    def build(self, input_shape):
        super().build(input_shape)
        self.d_inner = self.expand * input_shape[-1]
        self.nheads = self.d_inner // self.headdim
        d_model = input_shape[-1]
        d_in_proj = 3 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = layers.Dense(d_in_proj, use_bias=False, dtype=self._dtype)
        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = layers.DepthwiseConv2D((1, self.k_conv), padding='valid', activation='relu6', dtype=self._dtype) # npu hack
        self.A_log = self.add_weight(
            shape=(self.nheads,2),
            dtype='float32', 
            trainable=True
        )
        self.dt_bias = self.add_weight(shape=(self.nheads,), dtype='float32', trainable=True)
        self.D = self.add_weight(shape=(self.nheads*2, 1), dtype='float32', trainable=True)
        self.init_conv_state = self.add_weight(shape=(1, self.k_conv, self.d_inner + 2 * self.d_state), dtype='float32', trainable=True)
        self.init_ssm_state = self.add_weight(shape=(1, 1, self.nheads, self.headdim, self.d_state, 2), dtype='float32', trainable=True)
        self.norm = RMSNorm()
        self.out_proj = layers.Dense(d_model, use_bias=False, dtype=self._dtype)
        self.chunk_size = min(self.chunk_size, input_shape[1])
        self.call(ops.zeros(input_shape))
    
    def step_chunk(self, x, state, dt=1/30, **kv): # For GPU/TPU Computation Efficient Training
        if state is None:
            state = self.init_state(x.shape)
        state = cpc(ops.stack(state[0], axis=-1))[:,None], state[1]
        x = ops.cast(x, self._dtype)
        
        Alog = ops.stack([-ops.exp(self.A_log[:,0]), self.A_log[:,1]], axis=-1)
        A = cpc(Alog)
        zxbcdt = self.in_proj(x)
        z, xBC, _ = ops.split(zxbcdt, [2*self.d_inner, 3 * self.d_inner + 2 * self.d_state], axis=-1)
        dt = cp(ops.clip(ops.exp(_ + self.dt_bias), 0, 6)*dt*30)
        conv_state = ops.concatenate([state[1], xBC], axis=1)
        
        xBC = self.conv1d(conv_state[:,None])[:, 0, -xBC.shape[1]:]
        x, B, C = ops.split(xBC, [self.d_inner, self.d_state + self.d_inner], axis=-1)
        x, B, C = cp(x), cp(B), cp(C)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        y, new_state = ssd(x * dt[..., None], A * dt, B[..., None, :], C[..., None, :], self.chunk_size, init_stat=state[0])
        attn = y
        y = rearrange(cpd(y), 'b l h p c -> b l (h c) p')
        x = rearrange(cpd(x), 'b l h p c -> b l (h c) p')
        y += ops.cast(x, 'float32') * self.D  
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)
        return y, [cpd(new_state)[:,0], conv_state[:, -self.k_conv:]], attn

    def call(self, x, **kv):
        x, _, attn = self.step_chunk(x, None)
        return x
    
    def init_state(self, input_shape):
        if not self.built:
            self.build((input_shape[0], self.chunk_size, input_shape[-1]))
        state = self.init_ssm_state, self.init_conv_state
        state = ops.repeat(state[0],input_shape[0],axis=0), ops.repeat(state[1],input_shape[0],axis=0)
        state = (state[0][:,0,...,0], state[0][:,0,...,1]), state[1]
        return state

    def step(self, x, state, dt=1/30, **kv):               # For Embedding Memory Efficient Inference
        state_real, state_imag = state[0][0], state[0][1]  # Manually handle complex numbers
        zxbcdt = self.in_proj(ops.squeeze(x, axis=1))
        z, xBC, dt_param = ops.split(zxbcdt, [2*self.d_inner, 3 * self.d_inner + 2 * self.d_state], axis=-1)
        z_real, z_imag = ops.split(z, [self.d_inner], axis=-1)
        conv_state = ops.concatenate([state[1][:, 1:], ops.expand_dims(xBC, axis=1)], axis=1)
        xBC = self.conv1d(ops.expand_dims(conv_state, axis=1))[:, 0, -1]
        x, B, C = ops.split(xBC, [self.d_inner, self.d_inner + self.d_state], axis=-1)
        
        A_real = -ops.exp(self.A_log[:,0]) # Fused exp kernel
        A_imag = self.A_log[:,1]
        dt = ops.clip(exp(dt_param + self.dt_bias), 0, 6) * dt*30
        dt_A_real = dt * A_real
        dt_A_imag = dt * A_imag
        exp_dt_A_real = exp(dt_A_real)
        cos_dt_A_imag = cos(dt_A_imag)
        sin_dt_A_imag = sin(dt_A_imag)
        
        dA_real = exp_dt_A_real * cos_dt_A_imag
        dA_imag = exp_dt_A_real * sin_dt_A_imag

        x = ops.reshape(x, (x.shape[0], -1, self.headdim))
        
        dt_exp = ops.expand_dims(dt, axis=(-1, -2))
        B_exp = ops.expand_dims(B, axis=(1, 2))
        x_exp = ops.expand_dims(x, axis=-1)
        
        dBx = dt_exp * B_exp * x_exp

        dA_real_broadcast = ops.expand_dims(dA_real, axis=(-1, -2))
        dA_imag_broadcast = ops.expand_dims(dA_imag, axis=(-1, -2))

        state_dA_real = state_real * dA_real_broadcast - state_imag * dA_imag_broadcast
        state_dA_imag = state_real * dA_imag_broadcast + state_imag * dA_real_broadcast

        ssm_state_real = state_dA_real + dBx
        ssm_state_imag = state_dA_imag

        C_expanded = ops.expand_dims(ops.expand_dims(C, axis=1), axis=1) 
        y_real = ops.sum(ssm_state_real * C_expanded, axis=-1)
        y_imag = ops.sum(ssm_state_imag * C_expanded, axis=-1)
        
        y_combined = ops.stack([y_real, y_imag], axis=-1)
        
        y = ops.reshape(ops.transpose(y_combined, (0, 1, 3, 2)), (y_combined.shape[0], -1, y_combined.shape[2]))
        
        x_reshaped = ops.reshape(x, x.shape + (1,))
        x_combined = ops.concatenate([x_reshaped, ops.zeros_like(x_reshaped)], axis=-1)
        
        x_rearranged = ops.reshape(ops.transpose(x_combined, (0, 1, 3, 2)), (x_combined.shape[0], -1, x_combined.shape[2]))
        y += self.D * ops.cast(x_rearranged, 'float32')
        y = ops.reshape(y, (y.shape[0], -1))
        z_reshaped = ops.reshape(z, (z.shape[0], self.d_inner // self.headdim, -1))
        
        z_rearranged = ops.reshape(z_reshaped, (z_reshaped.shape[0], -1))
        y = self.norm(y, z_rearranged)
        y = self.out_proj(y)
        return ops.expand_dims(y, axis=1), [(ssm_state_real, ssm_state_imag), conv_state]

class TNM(layers.Layer):
    def __init__(self, axis=1, eps=1e-5, max_scale=6, **kw):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.max_scale = max_scale
        
    def build(self, input_shape):
        super().build(input_shape)
        shape = ops.mean(ops.zeros(input_shape), (0, self.axis), True).shape
        self.lbd = self.add_weight(shape=shape[-1:], trainable=True)
        self.gamma = self.add_weight(shape=shape[-1:], trainable=True)
        self.beta = self.add_weight(shape=shape[-1:], trainable=True)
        self.xmean = self.add_weight(shape=shape, trainable=True)
        self.x2mean = self.add_weight(initializer=keras.initializers.Constant(value=-3),
                                      shape=shape, trainable=True)

    def init_state(self, input_shape):
        return {'xmean':ops.concatenate([ops.array(self.xmean)]*input_shape[0], axis=0),
                'x2mean':ops.exp(ops.concatenate([ops.array(self.x2mean)]*input_shape[0], axis=0))}

    def call(self, x, dt=1/30, training=None):
        x, _ = self.step_chunk(x, self.init_state(x.shape))
        return x
    
    def _step_chunk(self, x, state=None, dt=1/30, training=None):
        if state is None:
            state = self.init_state(x.shape)
        xs = ops.moveaxis(x, self.axis, 0)
        decay = 1.0 - 0.6931 / ops.exp(self.lbd) * dt
        
        def _op(e1, e2):
            a1, b1 = e1
            a2, b2 = e2
            return a2 * a1, a2 * b1 + b2
            
        def scan_ema(inputs, state):
            coeffs = jnp.full_like(inputs, decay)
            p, outputs = jax.lax.associative_scan(_op, (coeffs, inputs), axis=0) # parallel scan
            return outputs + state*p
            
        xmean = scan_ema(xs * (1 - decay), ops.moveaxis(state['xmean'], self.axis, 0))
        x2mean = scan_ema(ops.square(xs - xmean) * (1 - decay), ops.moveaxis(state['x2mean'], self.axis, 0))
        istd = ops.rsqrt(x2mean + self.eps)
        y = (xs - xmean) * istd * self.gamma + self.beta
        return ops.moveaxis(y, 0, self.axis), {'xmean': ops.moveaxis(xmean[-1:], 0, self.axis), 
                                              'x2mean': ops.moveaxis(x2mean[-1:], 0, self.axis)}

    def step_chunk(self, x, state=None, dt=1/30, training=None):
        if state is None:
            state = self.init_state(x.shape)
        xs = ops.moveaxis(x, self.axis, 0)
        def step(x, state, dt=1/30, training=None):
            decay = 1.0 - 0.6931 / ops.exp(self.lbd) * dt
            xmean = (1-decay)*x + decay*state['xmean']
            x2mean = (1-decay)*ops.square(x-xmean) + decay*state['x2mean']
            istd = ops.rsqrt(x2mean+self.eps)
            return (x-xmean)*istd, {'xmean':xmean, 'x2mean':x2mean}
        def scan_fn(state, x_t):
            x_t = ops.expand_dims(x_t, self.axis)
            out, new_state = jax.remat(step)(x_t, state, dt)
            return new_state, out
        # Chunk scan, XLA compiler will unroll it.
        # It's amazing. The XLA compiler fused operations for every chunk and reduced HBM I/O by several times.
        state, y = jax.lax.scan(scan_fn, state, xs, unroll=16) # unroll !!!
        y = ops.squeeze(y, axis=self.axis + 1)
        y = ops.moveaxis(y, 0, self.axis)
        return y*self.gamma+self.beta, state
    
    def step(self, x, state, dt=1/30, training=None):
        decay = 1.0 - 0.6931 / ops.exp(self.lbd) * dt # Fused exp kernel
        xmean = (1-decay)*x + decay*state['xmean']
        x2mean = (1-decay)*ops.square(x-xmean) + decay*state['x2mean']
        istd = ops.rsqrt(x2mean+self.eps)
        return (x-xmean)*istd*self.gamma+self.beta, {'xmean':xmean, 'x2mean':x2mean}

class SSConv(layers.Layer):
    def __init__(self, filters=64, kernel=(3, 3), chunk=64, **kv):
        super().__init__(**kv) 
        self.k = filters
        self.chunk = chunk
        self.s = kernel
    
    def build(self, input_shape):
        super().build(input_shape)
        self.conv1 = keras.Sequential([layers.Conv2D(self.k, self.s, padding='same', dtype='mixed_float16'), 
                                       layers.Dense(self.k*2, activation='relu6')])
        self.proj = layers.Dense(self.k, use_bias=False, dtype='float32')
        self.reduce = lambda x:ops.mean(x, (1, 2))
        self.conv2 = layers.Conv2D(self.k, self.s, padding='same', activation='relu6', dtype='mixed_float16')
        self.ssm = SSMrPPG(chunk_size=self.chunk, d_state=128, headdim=self.k//8, expand=2)
        self.drop = layers.GaussianDropout(0.)
        #self.drop = lambda x,**y:x
        #self.fuse = layers.Dense(self.k, use_bias=False)
        self.tn = TNM(frames=self.chunk, axis=1)
        self.call(ops.zeros(input_shape))
    
    def step_chunk(self, x0, state, **kv):
        b = x0.shape[0]
        x0, tn_state = self.tn.step_chunk(x0)
        x0 = rearrange(x0, 'b l h w c -> (b l) h w c')
        x1 = self.conv1(x0)
        x2 = self.reduce(x1)
        x2 = rearrange(x2, '(b l) c -> b l c', b=b)
        x2, ssm_state, attn = self.ssm.step_chunk(x2, state=state['ssm'], **kv)
        x2 = self.proj(x2)
        x0 += rearrange(x2, 'b l c -> (b l) c')[:,None,None]
        x1 = self.conv2(x0)
        x1 = rearrange(x1, '(b l) h w c -> b l h w c', b=b)
        return x1, {'tn':tn_state, 'ssm':ssm_state}

    def call(self, x0, training=None):
        b = x0.shape[0]
        x0 = self.tn(x0, training=training)
        x0 = rearrange(x0, 'b l h w c -> (b l) h w c')
        x1 = self.conv1(x0)
        x2 = self.reduce(x1)
        x2 = self.drop(x2, training=training)
        x2 = rearrange(x2, '(b l) c -> b l c', b=b)
        x2 = self.ssm(x2)
        self.ssd_features = x2
        x2 = self.proj(x2)
        x0 += rearrange(x2, 'b l c -> (b l) c')[:,None,None]
        x1 = self.conv2(x0)
        self.out_features = x1
        #attns.append(np.array(attn))
        x1 = rearrange(x1, '(b l) h w c -> b l h w c', b=b)
        return x1
    
    def init_state(self, input_shape):
        return {'ssm':self.ssm.init_state((input_shape[0], self.chunk, self.k)),
                'tn':self.tn.init_state(input_shape)}
    
    def step(self, x0, state, **kv):
        b = x0.shape[0]
        x0, tn_state = self.tn.step(x0, state['tn'], **kv)
        x0 = rearrange(x0, 'b l h w c -> (b l) h w c')
        x1 = self.conv1(x0)
        x2 = self.reduce(x1)
        x2 = rearrange(x2, '(b l) c -> b l c', b=b)
        x2, ssm_state = self.ssm.step(x2, state['ssm'], **kv)
        x2 = self.proj(x2)
        x0 += rearrange(x2, 'b l c -> (b l) c')[:,None,None]
        x1 = self.conv2(x0)
        x1 = rearrange(x1, '(b l) h w c -> b l h w c', b=b)
        return x1, {'tn':tn_state, 'ssm':ssm_state}

class SSCBlock(layers.Layer):
    def __init__(self, n, filters=32, chunk=64, kernel=(3, 3), downsample=True, **kv):
        super().__init__(**kv)
        self.k = filters
        self.ssc = keras.Sequential([SSConv(filters, chunk=chunk, kernel=kernel) for _ in range(n)])
        self.in_proj = layers.Dense(filters, use_bias=False, dtype='float32')
        self.ds = layers.Conv2D(filters, (2, 2), (2, 2), padding='same', dtype='float32') if downsample else lambda x:x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(ops.zeros(input_shape))

    def call(self, x, training=None):
        x1 = self.in_proj(x)
        x = self.ssc(x1, training=training)
        x += x1
        b = x.shape[0]
        x = rearrange(x, 'b l h w c -> (b l) h w c')
        x = self.ds(x)
        x = rearrange(x, '(b l) h w c -> b l h w c', b=b)
        return x

    def step_chunk(self, x, state, **kv):
        x = x1 = self.in_proj(x)
        new_state = []
        for n, layer in enumerate(self.ssc.layers):
            x, s = layer.step_chunk(x, state[n], **kv)
            new_state.append(s)
        x += x1
        b = x.shape[0]
        x = rearrange(x, 'b l h w c -> (b l) h w c')
        x = self.ds(x)
        x = rearrange(x, '(b l) h w c -> b l h w c', b=b)
        return x, new_state
    
    def init_state(self, input_shape):
        #print(input_shape)
        return tuple([i.init_state((*input_shape[:-1], self.k)) for i in self.ssc.layers])
    
    def step(self, x, state, **kv):
        x = x1 = self.in_proj(x)
        new_state = []
        for n, layer in enumerate(self.ssc.layers):
            x, s = layer.step(x, state[n], **kv)
            new_state.append(s)
        x += x1
        b = x.shape[0]
        x = rearrange(x, 'b l h w c -> (b l) h w c')
        x = self.ds(x)
        x = rearrange(x, '(b l) h w c -> b l h w c', b=b)
        return x, new_state

class FacePhys(keras.Model):

    def __init__(self, n_layers=[2, 2, 2, 2], filters=[32]*4, chunk_size=64, **kv):
        super().__init__(**kv)
        self.network = keras.Sequential([SSCBlock(l, f, chunk_size) for l, f in zip(n_layers, filters)])
        self.out_proj = keras.Sequential([layers.Dense(32, use_bias=False, dtype='float32')])
        self.head = keras.Sequential([
            SSMrPPG(chunk_size=chunk_size, headdim=8),
            SSMrPPG(chunk_size=chunk_size, headdim=8),
            layers.Dense(1, dtype='float32'),
        ])
        self.n_layers = n_layers
        self.filters = filters

    def build(self, input_shape):
        super().build(input_shape)
        self.call(ops.zeros(input_shape))

    def init_state(self, input_shape):
        b, l, h, w, c = input_shape
        r = []
        for ly, f, layer in zip(self.n_layers, self.filters, self.network.layers):
            r.append(layer.init_state((b, l, h, w, f)))
            h, w = (h+1)//2, (w+1)//2
        for layer in self.head.layers:
            if isinstance(layer, (SSMrPPG, TNM)):
                r.append(layer.init_state((b, l, 32)))
        return r
    
    def call(self, x, training=None,**kv):
        x = self.network(x, training=training)
        x = ops.mean(x, (2, 3))
        x = self.out_proj(x, training=training)
        x = self.head(x, training=training)
        return x[...,0]

    def step_chunk(self, x, state, **kv):
        new_state = []
        for n, layer in enumerate(self.network.layers):
            x, s = layer.step_chunk(x, state[n], **kv)
            new_state.append(s)
        x = ops.mean(x, (2, 3))
        x = self.out_proj(x)
        for m, layer in enumerate(self.head.layers):
            if isinstance(layer, (TNM,)):
                x, s = layer.step_chunk(x, state[n+m+1], **kv)
                new_state.append(s)
            elif isinstance(layer, (SSMrPPG,)):
                x, s, _ = layer.step_chunk(x, state[n+m+1], **kv)
                new_state.append(s)
            else:
                x = layer(x)
        return x[...,0], new_state
        
    def step(self, x, state, **kv):
        new_state = []
        for n, layer in enumerate(self.network.layers):
            x, s = layer.step(x, state[n], **kv)
            new_state.append(s)
        x = ops.mean(x, (2, 3))
        x = self.out_proj(x)
        for m, layer in enumerate(self.head.layers):
            if isinstance(layer, (SSMrPPG, TNM)):
                x, s = layer.step(x, state[n+m+1], **kv)
                new_state.append(s)
            else:
                x = layer(x)
        return x[...,0], new_state

# model = FacePhys([2]*4, [32]*4)
# model.build((1, 1, 36, 36, 3));
# model.summary()
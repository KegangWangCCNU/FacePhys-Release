import matplotlib.pyplot as plt
import numpy as np

def save_state(file, state):
    import gzip, json
    with gzip.open(file, 'w') as f:
        f.write(json.dumps({k:np.round(v, 1).tolist() for k, v in state.items()}).encode())

def load_state(file):
    import gzip, json
    with gzip.open(file, 'r') as f:
        return {k:np.array(v, dtype='float32') for k,v in json.loads(f.read().decode()).items()}

def inference(input, model, state=None, fps=30):
    import time
    if not state:
        state = model.init_state
    input = np.array(input, 'float32')
    y, t, dt = [], time.time(), np.array(1/fps, 'float32')
    for i in range(input.shape[-4]):
        r, state = model(input[i,:,:,:], state, dt=dt)
        y.append(r)
    y = np.array(y)
    print(f"Inference FPS: {input.shape[-4]/(time.time()-t):.1f}")
    return y, state

def load_model_onnx(path):
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.log_severity_level = 3
    sess = ort.InferenceSession(path, sess_options=sess_options)
    def run(img, state, dt=1/30):
        result = sess.run(None, {"input": img[None, None], "dt": [dt], **state})
        bvp, new_state = result[0][0, 0], result[1:]
        return bvp, dict(zip(state, new_state))
    run.init_state = {input.name:np.zeros(input.shape, dtype='float32') for input in sess.get_inputs()[2:]}
    return run 

def load_model_litert(path):
    #import tflite_runtime.interpreter as tflite
    import ai_edge_litert.interpreter as tflite 
    interpreter = tflite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    output_index = [i['index'] for i in interpreter.get_output_details()]
    def run(img, state, dt=1/30):
        interpreter.set_tensor(0, img[None, None].astype('float32'))
        interpreter.set_tensor(1, np.array([dt], dtype='float32'))
        for i, s in enumerate(state.values()):
            interpreter.set_tensor(i+2, s)
        interpreter.invoke()
        for s, i in zip(state.values(), output_index[1:]):
            s[:] = interpreter.get_tensor(i)
        return interpreter.get_tensor(output_index[0])[0,0], state
    run.init_state = {t['name']:np.zeros(t['shape'], dtype=np.float32) for t in interpreter.get_input_details()[2:]}
    return run 

def load_model_jax(path):
    from FacePhys_model import FacePhys, jax
    model = FacePhys([2]*4, [32]*4)
    model.build((1, 1, 36, 36, 3));
    model.load_weights(path)
    @jax.jit
    def run(x, state, dt=1/30):
        r, state = model.step(x[None, None], state, dt=dt)
        return r[0, 0], state
    run.init_state = model.init_state((1, 1, 36, 36, 3))
    _, s = run(np.zeros((36, 36, 3), 'float32'), run.init_state, dt=np.array(1/30, 'float32'))
    run(np.zeros((36, 36, 3), 'float32'), s, dt=np.array(1/30, 'float32'))
    return run  

def onnx2trt(onnx_file, target):
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GB Memory
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = list(tensor.shape)
        static_shape = []
        for dim in shape:
            if dim == -1 or dim is None:
                static_shape.append(1)
            else:
                static_shape.append(dim)
        final_shape = tuple(static_shape)
        profile.set_shape(name, final_shape, final_shape, final_shape)
    config.add_optimization_profile(profile)
    config.builder_optimization_level = 5
    serialized_engine = builder.build_serialized_network(network, config)
    with open(target, "wb") as f:
        f.write(serialized_engine)

def load_model_trt(path):
    import tensorrt as trt
    import pycuda.autoinit
    import pycuda.driver as cuda
    TRT_CONFIG = {
        'INPUT_IMG': 'input',
        'INPUT_DT': 'dt',
        'OUTPUT_RESULT': 'Identity:0',
        'STATE_PAIRS': [
            (f'Identity_{i}:0', f'state_in_{i-1}') for i in range(1, 47)
        ]
    }
    
    logger = trt.Logger(trt.Logger.ERROR)
    with open(path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    meta = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dims = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        vol = trt.volume(dims)
        if vol < 0: vol = 1
        host = cuda.pagelocked_empty(vol, dtype)
        dev_alloc = cuda.mem_alloc(host.nbytes)
        dev_ptr = int(dev_alloc)
        context.set_tensor_address(name, dev_ptr)
        meta[name] = {
            'h': host, 
            'd': dev_ptr, 
            'allocation': dev_alloc,
            'shape': dims, 
            'dtype': dtype, 
            'nbytes': host.nbytes
        }
    state_update_list = []
    for out_name, in_name in TRT_CONFIG['STATE_PAIRS']:
        state_update_list.append((meta[in_name]['d'], meta[out_name]['d'], meta[in_name]['nbytes']))

    img_meta = meta[TRT_CONFIG['INPUT_IMG']]
    dt_meta = meta[TRT_CONFIG['INPUT_DT']]
    res_meta = meta[TRT_CONFIG['OUTPUT_RESULT']]

    init_state_dict = {}
    for _, in_name in TRT_CONFIG['STATE_PAIRS']:
        m = meta[in_name]
        init_state_dict[in_name] = np.zeros(m['shape'], dtype=m['dtype'])
        
    def run(img, state, dt=1.0/30.0):
        np.copyto(img_meta['h'], img.ravel())
        np.copyto(dt_meta['h'], np.array([dt], dtype=dt_meta['dtype']))
        
        cuda.memcpy_htod_async(img_meta['d'], img_meta['h'], stream)
        cuda.memcpy_htod_async(dt_meta['d'], dt_meta['h'], stream)
        first_val = next(iter(state.values()))
        is_gpu_ptr = isinstance(first_val, int)
        if not is_gpu_ptr:
            for in_name, arr in state.items():
                cuda.memcpy_htod_async(meta[in_name]['d'], np.ascontiguousarray(arr), stream)
        context.execute_async_v3(stream_handle=stream.handle)
        for dst, src, size in state_update_list:
            cuda.memcpy_dtod_async(dst, src, size, stream)
        future_result = cuda.pagelocked_empty(1, dtype=np.float32)
        cuda.memcpy_dtoh_async(future_result, res_meta['d'], stream)
        new_state_gpu = {}
        for _, in_name in TRT_CONFIG['STATE_PAIRS']:
            new_state_gpu[in_name] = meta[in_name]['d']
        return future_result, new_state_gpu
    run.sync = stream.synchronize
    run.init_state = init_state_dict
    
    return run

def load_model_onnx_cuda(path):
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    sess = ort.InferenceSession(path, sess_options=sess_options, providers=['CUDAExecutionProvider'])
    io_binding = sess.io_binding()
    input_nodes = sess.get_inputs()
    output_nodes = sess.get_outputs()
    state_input_names = [n.name for n in input_nodes[2:]]
    state_output_names = [n.name for n in output_nodes[1:]]
    gpu_state_registry = {}
    for node in input_nodes[2:]:
        gpu_state_registry[node.name] = ort.OrtValue.ortvalue_from_shape_and_type(
            node.shape, np.float32, 'cuda', 0
        )
    def run(img, state, dt=1/30):
        io_binding.bind_cpu_input("input", img[None, None].astype(np.float32))
        io_binding.bind_cpu_input("dt", np.array([dt], dtype='float32'))
        first_val = next(iter(state.values()))
        if isinstance(first_val, np.ndarray):
            for name in state_input_names:
                io_binding.bind_cpu_input(name, state[name].astype(np.float32))
        else:
            for name in state_input_names:
                io_binding.bind_ortvalue_input(name, state[name])
        io_binding.bind_output(output_nodes[0].name, 'cpu')
        for i, out_name in enumerate(state_output_names):
            in_name = state_input_names[i]
            io_binding.bind_ortvalue_output(out_name, gpu_state_registry[in_name])
        sess.run_with_iobinding(io_binding)
        outputs = io_binding.get_outputs()
        bvp = outputs[0]
        return bvp, gpu_state_registry
    init_state = {name: np.zeros(n.shape, dtype=np.float32) 
                  for name, n in zip(state_input_names, input_nodes[2:])}
    run.init_state = init_state
    return run
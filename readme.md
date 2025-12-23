## FacePhys Model Publication 

FacePhys is a lightweight rPPG algorithm that uses State Space Models (SSMs) to model heart states. With a memory usage of only 4MB, it is ideal for on-device inference. 

## FacePhys Inference 

### Inference Performance

FacePhys features a custom State Space Model (SSM) relying on frequent small tensor ops and dimension reshaping. While the model is lightweight, **the overhead from numerous fragmented operators limits inference speed.**

We tested this on multiple engines with a **batch size of 1** to simulate real-time scenarios.

#### 1. GPU Performance
**Device:** NVIDIA Tesla V100 SXM2 (32GB)

| Runtime | Inference FPS | Backend |
| :--- | ---: | :---: |
| **JAX JIT** | **1002** | CUDA |
| **TensorRT** | 778 | CUDA |
| **ONNX-Runtime** | 101 | CUDA |
| **TensorFlow JIT** | 84 | CUDA |
| **Torch Baseline** | 11 | CUDA |
| **Torch Compile** | 7 | CUDA |

> **Note:** PyTorch full graph compilation failed due to graph breaks.

#### 2. CPU Performance
**Device:** Dual Intel Xeon Gold 6138

| Runtime | Inference FPS | Backend |
| :--- | ---: | :---: |
| **LiteRT** | **337** | CPU |
| **ONNX-Runtime** | 180 | CPU |
| **JAX JIT** | 167 | CPU |
| **TensorFlow JIT** | 96 | CPU |
| **Torch Compile** | 8 | CPU |
| **Torch Baseline** | 7 | CPU |

#### Deployment Strategy

**The Bottleneck: Kernel Launch Overhead**
As shown in the GPU benchmarks, standard runtimes struggle with the SSM's fragmented operators. While GPUs possess high compute power, the latency from launching thousands of tiny kernels creates a bottleneck. **JAX-XLA and TensorRT** overcome this by performing **kernel fusion**—compiling the graph into fewer, monolithic kernels—drastically reducing launch overhead.

**Why LiteRT?**
Since rPPG targets **on-device usage** rather than cloud deployment, we prioritized CPU efficiency over raw GPU throughput. Unlike `mamba-ssm`, we avoided writing custom CUDA kernels to maintain portability. Instead, we utilized the [LiteRT](https://github.com/google-ai-edge/LiteRT) cross-platform runtime, which provided the best balance of performance and portability on CPU backends.  

### Example 

```python
from inference import * 
import matplotlib.pyplot as plt 

ipt   = np.load('weights/input.npy') # (1, 1600, 36, 36, 3)
state = load_state('weights/state.gz')

model = load_model_onnx('weights/model.onnx')
wave, final_state = inference(ipt[0], model, state)

plt.plot(wave)
```

For more examples, see `examples.ipynb`.
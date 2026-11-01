# pamin-runtime

Compute for Pamin modules that run models locally, written and measured once. Used by [PaminMemory](https://github.com/pamin-labs/pamin-memory).

| Module | Holds |
| --- | --- |
| `kernels` | The dot product dispatched at run time (AVX2/FMA on x86-64, NEON on Linux aarch64, Accelerate on Apple silicon, a portable eight-lane loop elsewhere) and IEEE binary16 conversion, widened with F16C or the aarch64 fp16 instructions |
| `device` | The ONNX Runtime execution providers a machine can run (CUDA, Core ML, DirectML, plugin and Windows ML catalog NPUs) with their startup guards, the CPU provider, the CUDA inventory and what a session was actually assigned |
| `budget` | Threads per forward pass, and the session options every model is built with |
| `prepared` | The key that ties a prepared, memory-mapped model copy to the runtime build, the CPU's matrix features and the optimization level that wrote it |

Nothing here knows about a model, a module or a transport.

```rust
use pamin_runtime::{budget, device};

let threads = budget::threads(None)?; // one per core unless the caller says otherwise
for (device, target) in device::accelerators(&[])? {
    // Load the model on `target`; on failure try the next, then `device::cpu()`.
    let builder = budget::options(threads, target)?;
}
```

Status: 0.1, extracted from PaminMemory's `pamin-index`. Every kernel and device path is tested in CI on its own hardware: x86-64 Linux, Linux aarch64, Apple silicon and Windows.

Licensed under Apache-2.0.

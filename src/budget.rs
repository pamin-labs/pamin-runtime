//! How much of the machine one forward pass is allowed to use.
//!
//! ONNX Runtime's default is intra-op threads equal to
//! `available_parallelism()`: one pass takes every core.
//!
//! That is the right default for a single query and the wrong one for a server.
//! Measured on four cores with nothing shared -- one model per worker, no lock
//! anywhere -- throughput *falls* as callers are added, 182 embeddings a second
//! at one worker to 54 at eight, because the second caller finds the first
//! caller's threads rather than an idle core. Splitting the cores between
//! callers instead of between the layers of one pass is the other way to divide
//! them, and which one wins is a property of the machine rather than of this
//! code.
//!
//! So it is the caller's setting ([`threads`]), and its default is what the
//! runtime already did.
//!
//! Whether an idle thread in that pool spins is not a setting: it blocks. See
//! [`options`].

use std::num::NonZeroUsize;

use ort::session::{Session, builder::SessionBuilder};

use crate::device::{Target, cpu, gpu_providers};
use crate::{Error, Result};

/// How many threads inference may keep busy: `asked`, or one per core.
pub fn threads(asked: Option<NonZeroUsize>) -> std::io::Result<usize> {
    Ok(match asked {
        Some(threads) => threads.get(),
        None => std::thread::available_parallelism()?.get(),
    })
}

/// What a session asks of ONNX Runtime before it has a model to load:
/// `target`'s providers, failing on any that will not register, the
/// [`LEVEL`](crate::prepared::LEVEL) optimizations, `threads` intra-op threads,
/// and the flags DirectML and Core ML need.
///
/// One thing here is not ONNX Runtime's default, and does not change a score:
/// intra-op threads block when they run out of work instead of spinning.
/// ONNX Runtime lets them spin by default, which saves a wake-up when the next
/// piece of work comes at once and holds a core for as long as it does not --
/// and the embedder and the reranker each own a pool of one thread per core,
/// so on a machine with anything else to do the spinning threads take cores
/// from the threads that have work, the other model's included.
///
/// Measured through `pamin serve` and `pamin search` at the defaults, one
/// hundred queries (sixty from the own corpus, forty from an XQuAD-R subset),
/// a fresh server per arm so no query is a cache hit, three rounds in rotated
/// order, on four cores shared with other work: with two busy loops beside
/// the server a search took 0.754 of what it did spinning (geometric mean of
/// per-query ratios; faster on 86 of 100, sign-flip `p = 0.0001`), and at the
/// machine's own load -- 2.5 to 13 -- 1.033 (`p = 0.068`), eight concurrent
/// callers getting 1.17 times the throughput. Rankings, query and passage
/// embeddings and `accurate` scores were bit-identical. The rule, written
/// before the run, is in pamin-memory's `docs/adr/0001-tech-selection.md`.
pub fn options(threads: usize, target: impl Into<Target>) -> Result<SessionBuilder> {
    let target = target.into();
    let providers = match &target {
        Target::Providers(providers) => providers.clone(),
        Target::Npu { .. } => gpu_providers()
            .into_iter()
            .map(|(_, ep)| ep.fail_silently())
            .chain([cpu()])
            .collect(),
    };
    let unready =
        |error: &dyn std::fmt::Display| Error::Runtime(format!("preparing a session: {error}"));
    let builder = Session::builder().map_err(|error| unready(&error))?;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let builder = if providers
        .iter()
        .any(|provider| provider.downcast_ref::<ort::ep::CoreML>().is_some())
    {
        // Gemm fusion transposes large constant weights into inline MIL
        // tensors. Keep the original initializer blobs for CoreML instead.
        builder
            .with_disabled_optimizers("MatMulAddFusion")
            .map_err(|error| unready(&error))?
    } else {
        builder
    };
    #[cfg(target_os = "windows")]
    let builder = if providers
        .iter()
        .any(|provider| provider.downcast_ref::<ort::ep::DirectML>().is_some())
    {
        // DirectML requires these before registration. Registration alone
        // does not validate them; model loading otherwise falls back to CPU.
        builder
            .with_memory_pattern(false)
            .map_err(|error| unready(&error))?
            .with_parallel_execution(false)
            .map_err(|error| unready(&error))?
    } else {
        builder
    };
    let builder = match target {
        Target::Providers(_) => builder,
        Target::Npu { provider, id } => {
            let environment = ort::environment::Environment::current().map_err(|e| unready(&e))?;
            let devices: Vec<_> = environment
                .devices()
                .filter(|device| {
                    device.hardware_device().ty() == ort::memory::DeviceType::NPU
                        && device.hardware_device().id() == id
                        && device.ep().is_ok_and(|name| name == provider)
                })
                .collect();
            if devices.is_empty() {
                return Err(Error::Runtime(format!("NPU {provider}/{id} unavailable")));
            }
            builder
                .with_devices(devices, None)
                .map_err(|error| Error::Incompatible(format!("registering a provider: {error}")))?
        }
    };
    // A provider that cannot be registered -- its library or a library it
    // needs does not load, or it finds no device -- stays that way for the
    // life of the process. Calling it transient revalidated the CPU plan
    // every RUNTIME_RETRY: both models were reloaded beside the resident
    // ones, about 300 MB more at the peak and a stalled search, every five
    // minutes on every x86-64 Linux machine without CUDA.
    builder
        .with_execution_providers(providers)
        .map_err(|error| Error::Incompatible(format!("registering a provider: {error}")))?
        .with_config_entry("session.record_ep_graph_assignment_info", "1")
        .map_err(|error| unready(&error))?
        .with_optimization_level(crate::prepared::LEVEL)
        .map_err(|error| unready(&error))?
        .with_intra_threads(threads)
        .map_err(|error| unready(&error))?
        .with_intra_op_spinning(false)
        .map_err(|error| unready(&error))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn threads() -> usize {
        super::threads(None).expect("the default budget")
    }

    #[test]
    fn a_budget_is_what_was_asked_or_one_per_core() {
        assert_eq!(super::threads(NonZeroUsize::new(3)).unwrap(), 3);
        assert_eq!(
            threads(),
            std::thread::available_parallelism().unwrap().get()
        );
    }

    /// CUDA that cannot be registered here -- no device, or its libraries
    /// do not load -- is incompatible, not transient, so a caller does not
    /// keep retrying it.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn a_provider_that_cannot_register_is_incompatible() {
        let cuda = ort::ep::CUDA::default().build().error_on_failure();
        match super::options(1, vec![cuda]) {
            Err(Error::Incompatible(message)) => {
                assert!(message.contains("registering a provider"), "{message}");
            }
            Err(other) => panic!("expected Incompatible, got {other}"),
            // CUDA registers here, so there is no failure to classify.
            Ok(_) => {}
        }
    }

    /// Every session's intra-op threads block rather than spin; see
    /// [`options`]. Read back from ONNX Runtime rather than from our own
    /// call, because an entry that never reached the options is exactly the
    /// failure this guards: the runtime's default is to spin.
    #[test]
    fn idle_inference_threads_block_rather_than_spin() {
        use ort::AsPointer;

        let builder = options(threads(), vec![cpu()]).expect("session options");
        let mut value = [0 as std::ffi::c_char; 8];
        let mut size = value.len();
        // SAFETY: the options pointer is live for the borrow of `builder`, the
        // key is NUL-terminated, and `size` is the length of `value`, which
        // the runtime writes no further than.
        let status = unsafe {
            (ort::api().GetSessionConfigEntry)(
                builder.ptr(),
                c"session.intra_op.allow_spinning".as_ptr(),
                value.as_mut_ptr(),
                &mut size,
            )
        };
        assert!(
            status.0.is_null(),
            "no spinning entry reached the options, so ONNX Runtime spins"
        );
        // SAFETY: on success the runtime wrote a NUL-terminated string of
        // `size` bytes, NUL included, into `value`.
        let set = unsafe { std::ffi::CStr::from_ptr(value.as_ptr()) };
        assert_eq!(set, c"0", "intra-op spinning is not off");
    }
}

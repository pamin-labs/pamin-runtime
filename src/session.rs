//! ONNX Runtime sessions over a model's file.
//!
//! Every session is built with [`crate::budget::options`], and every one
//! loaded from a file goes through [`session`] or [`cpu_session`], so the
//! clean mappings it created are counted as its own (see
//! [`crate::resident::loading`]) and the providers it was actually assigned
//! are logged. [`SharedCpuSession`] is the one session several threads may
//! run at once.

use std::path::{Path, PathBuf};

use std::borrow::Cow;
use std::ffi::CString;
use std::ptr::NonNull;

use ort::session::{Session, SessionInputValue, SessionOutputs, builder::SessionBuilder};
use ort::value::DynValue;

use crate::budget::options;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::device::coreml;
use crate::device::{Target, assigned_providers, cpu};
use crate::{Error, Result};

/// An ONNX Runtime session over `model`, built the way `fastembed` 6.1 builds
/// one.
///
/// PaminMemory's models were loaded by `fastembed` until they needed a
/// tokenizer it would not let them share, and what its builder chose decides
/// the scores: the execution providers in order, ONNX Runtime's layout
/// optimizations, and [`threads`](crate::budget::threads) or one per core. So
/// those are what this chooses, in the order it chose them. This also applies
/// DirectML's required execution flags and excludes MatMulAddFusion on CoreML
/// to keep transposed constant weights out of the serialized graph.
///
/// `model` is asked for only once the providers have registered, because
/// asking for it can mean downloading it, and each device has an export of
/// its own. Found the other way round, a CPU-only Linux machine fetched the
/// `accurate` reranker's 1,136 MB half-precision export before learning that
/// CUDA would not register -- which it does in about a millisecond -- and
/// then kept the file, never loading it, beside the int8 export it runs.
///
/// `model`'s own error is returned as it is, so a failed download reads as one.
pub fn session<E: From<Error>>(
    threads: usize,
    target: impl Into<Target>,
    model: impl FnOnce() -> std::result::Result<PathBuf, E>,
) -> std::result::Result<Session, E> {
    let target = target.into();
    let expected = match &target {
        Target::Npu { provider, .. } => Some(provider.clone()),
        _ => None,
    };
    let mut builder = options(threads, target)?;
    let model = model()?;
    let session = commit(&mut builder, &model)?;
    if let Some(provider) = expected {
        let assigned = assigned_providers(&session)?;
        if assigned.get(&provider).copied().unwrap_or(0) == 0 {
            return Err(
                Error::Runtime(format!("NPU {provider} was assigned no model nodes")).into(),
            );
        }
    }
    report_model(&session, &model);
    Ok(session)
}

/// Preparation probes use the product options, but are not scoring sessions.
pub fn probe(threads: usize, model: &Path) -> Result<Session> {
    commit(&mut options(threads, vec![cpu()])?, model)
}

fn commit(builder: &mut SessionBuilder, model: &Path) -> Result<Session> {
    Ok(commit_owning(builder, model)?.0)
}

/// [`commit`], and the clean mappings of `model`'s directory the session
/// created; see [`crate::resident::loading`]. Every session loaded from a file
/// goes through here, so no other load's mappings can be counted as its own.
fn commit_owning(
    builder: &mut SessionBuilder,
    model: &Path,
) -> Result<(Session, Vec<(usize, usize)>)> {
    crate::resident::loading(model.parent().unwrap_or(model), || {
        builder
            .commit_from_file(model)
            .map_err(|error| Error::Runtime(format!("loading {}: {error}", model.display())))
    })
}

/// Preparation probes do not report themselves as scoring sessions.
fn report_model(session: &Session, model: &Path) {
    tracing::info!(model_graph = %serde_json::to_string(&model.to_string_lossy()).expect("serialize a model path"), "loaded ONNX graph");
    match assigned_providers(session) {
        Ok(nodes) => tracing::info!(
            model_graph = %serde_json::to_string(&model.to_string_lossy()).expect("serialize a model path"),
            assigned_nodes = %serde_json::to_string(&nodes).expect("serialize provider counts"),
            "ONNX graph execution-provider assignment"
        ),
        Err(error) => {
            tracing::warn!(model_graph = %serde_json::to_string(&model.to_string_lossy()).expect("serialize a model path"), %error, "could not inspect execution-provider assignment")
        }
    }
}

/// Refuse an unavailable CoreML provider before fetching its export. The
/// cache-aware session registers again once the content-derived path is known.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn coreml_source<E: From<Error>>(
    threads: usize,
    model: impl FnOnce() -> std::result::Result<PathBuf, E>,
) -> std::result::Result<PathBuf, E> {
    drop(options(threads, vec![coreml().build().error_on_failure()])?);
    model()
}

/// Static shapes prevent CoreML from silently rejecting unbounded ANE regions.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn fixed_coreml<E: From<Error>>(
    threads: usize,
    model: impl FnOnce() -> std::result::Result<PathBuf, E>,
    rows: usize,
    tokens: usize,
) -> std::result::Result<(Session, PathBuf), E> {
    let path = coreml_source(threads, model)?;
    Ok((fixed_coreml_at(threads, &path, rows, tokens)?, path))
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn fixed_coreml_at(threads: usize, path: &Path, rows: usize, tokens: usize) -> Result<Session> {
    let cache = coreml_cache(path, rows, tokens)?;
    std::fs::create_dir_all(cache.parent().expect("cache has a parent"))?;
    // Keep the lock outside the directory being rebuilt. Unlinking a held
    // lock would let another process lock a different inode concurrently.
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(cache.with_extension("lock"))?;
    lock.lock()?;
    recover_coreml_cache(&cache)?;
    let reused = cache.join("ready").is_file();
    if !reused {
        crate::activity::prepared();
        reset_coreml_cache(&cache)?;
    }
    let load = || -> Result<Session> {
        let provider = coreml()
            .with_model_cache_dir(cache.to_string_lossy())
            .build()
            .error_on_failure();
        let error =
            |e: &dyn std::fmt::Display| Error::Runtime(format!("static CoreML session: {e}"));
        let mut builder = options(threads, vec![provider])?
            .with_dimension_override("batch_size", rows as i64)
            .map_err(|e| error(&e))?
            .with_dimension_override("sequence_length", tokens as i64)
            .map_err(|e| error(&e))?;
        builder.commit_from_file(path).map_err(|e| error(&e))
    };
    let session = match load() {
        Ok(session) => session,
        Err(error) if reused => {
            tracing::warn!(%error, "cached CoreML package failed to load; rebuilding once");
            rebuild_coreml_cache(&cache, load)?
        }
        Err(error) => return Err(error),
    };
    // A crash before every partition loads leaves no marker. Its files must
    // be rebuilt under the same lock rather than mistaken for a valid cache.
    publish_coreml_cache(&cache)?;
    report_model(&session, path);
    Ok(session)
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn recover_coreml_cache(cache: &Path) -> Result<()> {
    let previous = cache.with_extension("previous");
    if previous.exists() {
        if cache.join("ready").is_file() {
            std::fs::remove_dir_all(previous)?;
        } else {
            if cache.exists() {
                std::fs::remove_dir_all(cache)?;
            }
            std::fs::rename(previous, cache)?;
        }
    }
    Ok(())
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn publish_coreml_cache(cache: &Path) -> Result<()> {
    let partial = cache.join("ready.partial");
    std::fs::File::create(&partial)?.sync_all()?;
    std::fs::rename(partial, cache.join("ready"))?;
    Ok(())
}

#[cfg(any(all(target_os = "macos", target_arch = "aarch64"), test))]
fn rebuild_coreml_cache<T>(cache: &Path, load: impl FnOnce() -> Result<T>) -> Result<T> {
    crate::activity::prepared();
    let previous = cache.with_extension("previous");
    std::fs::rename(cache, &previous)?;
    std::fs::create_dir_all(cache)?;
    match load().and_then(|model| {
        publish_coreml_cache(cache)?;
        Ok(model)
    }) {
        Ok(model) => {
            // Publication succeeded. Failed retirement can be retried on the
            // next load; it must not discard the valid replacement.
            if let Err(error) = std::fs::remove_dir_all(previous) {
                tracing::warn!(%error, "could not retire previous CoreML cache");
            }
            Ok(model)
        }
        Err(error) => {
            recover_coreml_cache(cache)?;
            Err(error)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn reset_coreml_cache(cache: &Path) -> Result<()> {
    if cache.exists() {
        std::fs::remove_dir_all(cache)?;
    }
    std::fs::create_dir_all(cache)?;
    Ok(())
}

/// The prepared parent already identifies the external weights by content.
/// Also key the graph, runtime and overrides: ORT's URL key omits all three.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn coreml_cache(model: &std::path::Path, rows: usize, tokens: usize) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    let parent = model
        .parent()
        .ok_or_else(|| Error::Runtime("prepared CoreML model has no parent directory".into()))?;
    let graph = crate::prepared::source_digest(model)?;
    let runtime = format!("{:x}", Sha256::digest(ort::info().as_bytes()));
    // Bump this policy version when the compile options above change.
    Ok(parent
        .join("coreml-all-v2")
        .join(runtime)
        .join(format!("{graph}-{rows}-{tokens}")))
}

/// A CPU session over `model` with `threads` intra-op threads, otherwise
/// built as [`session`] builds one, and the clean mappings of `model`'s
/// directory it created, which the holder of the session may release; see
/// [`crate::resident::loading`].
pub fn cpu_session(model: &Path, threads: usize) -> Result<(Session, Vec<(usize, usize)>)> {
    let mut builder = options(threads, vec![cpu()])?
        .with_intra_threads(threads)
        .map_err(|error| Error::Runtime(format!("preparing a session: {error}")))?;
    let (session, mapped) = commit_owning(&mut builder, model)?;
    report_model(&session, model);
    Ok((session, mapped))
}

/// A pass's outputs, owned: by name, in the order the graph lists them.
pub type Outputs = Vec<(String, DynValue)>;

fn failed(error: &dyn std::fmt::Display) -> Error {
    Error::Runtime(format!("running the model: {error}"))
}

/// The one session [`run_shared`] runs from several threads at once: loaded
/// on ONNX Runtime's CPU execution provider alone, with every node of the
/// graph assigned to it.
///
/// Concurrent `Run` is safe on some execution providers and not on others,
/// so this type is the only way onto the concurrent path, and it admits the
/// CPU alone. ONNX Runtime documents `Run` as callable from several threads
/// on one session (`InferenceSession::Run` in
/// `onnxruntime/core/session/inference_session.h`), and an execution provider
/// that cannot take that says so through `ConcurrentRunSupported`, for which
/// ONNX Runtime then serializes `Run` itself (XNNPACK, WebGPU, WebNN, JS in
/// 1.28). DirectML does not declare it and documents that only one thread may
/// call `Run` at a time; CUDA and TensorRT have concurrency reports against
/// them. None of those may reach this type: [`load`](Self::load) builds the
/// session from the CPU provider alone and refuses it unless ONNX Runtime
/// assigned every node to that provider.
pub struct SharedCpuSession(Session);

impl SharedCpuSession {
    /// The session, and the clean mappings of `model`'s directory it
    /// created; see [`crate::resident::loading`].
    pub fn load(model: &Path) -> Result<(Self, Vec<(usize, usize)>)> {
        let (session, mapped) = cpu_session(model, 1)?;
        cpu_alone(&assigned_providers(&session)?)?;
        Ok((Self(session), mapped))
    }
}

/// Whether `assigned`, a session's node count by execution provider, puts
/// every node on the CPU provider. An empty assignment is refused too: it
/// means ONNX Runtime recorded none, not that there was nothing to assign.
fn cpu_alone(assigned: &std::collections::BTreeMap<String, usize>) -> Result<()> {
    if assigned.is_empty() {
        return Err(failed(&"a shared session recorded no provider assignment"));
    }
    match assigned
        .iter()
        .find(|(provider, nodes)| *provider != "CPUExecutionProvider" && **nodes > 0)
    {
        Some((provider, _)) => Err(failed(&format!(
            "a shared session assigned nodes to {provider}, not the CPU alone"
        ))),
        None => Ok(()),
    }
}

/// One forward pass over `feed` on `session`, which other threads may be
/// running at the same moment, as [`Session::run`] would make it.
///
/// `ort` runs a session only through `&mut`: its author does not trust ONNX
/// Runtime's `Run` from several threads, naming "certain EP allocators or
/// statistics trackers" in `ort`'s source, and recommends a session per
/// thread. Here the session is a [`SharedCpuSession`] -- the CPU provider
/// alone, its arena off, no profiling -- of one intra-op thread, so that every
/// pass runs on the thread that asked for it. So this is `Run` itself, through
/// ONNX Runtime's C API, called on a shared session -- the call the `ort`
/// crate makes, without the exclusive borrow it puts in front of it.
pub fn run_shared(
    session: &SharedCpuSession,
    feed: &[(Cow<'static, str>, SessionInputValue<'static>)],
) -> Result<Outputs> {
    use ort::AsPointer;
    let session = &session.0;
    let names = |names: &mut dyn Iterator<Item = &str>| -> Result<Vec<CString>> {
        names
            .map(|name| CString::new(name).map_err(|error| failed(&error)))
            .collect()
    };
    let inputs = names(&mut feed.iter().map(|(name, _)| name.as_ref()))?;
    let outputs = names(&mut session.outputs().iter().map(|output| output.name()))?;
    let input_names: Vec<_> = inputs.iter().map(|name| name.as_ptr()).collect();
    let output_names: Vec<_> = outputs.iter().map(|name| name.as_ptr()).collect();
    let values: Vec<_> = feed.iter().map(|(_, value)| value.ptr()).collect();
    let mut fetched = vec![std::ptr::null_mut(); outputs.len()];
    // SAFETY: the session is live for the call, and ONNX Runtime allows `Run`
    // on it from several threads at once on the CPU provider, the only one a
    // `SharedCpuSession` admits (see above). Every name is a
    // NUL-terminated string and every input a live value, all outliving the
    // call; `fetched` has a slot for each output name, which `Run` fills.
    let status = unsafe {
        (ort::api().Run)(
            session.ptr().cast_mut(),
            std::ptr::null(),
            input_names.as_ptr(),
            values.as_ptr(),
            values.len(),
            output_names.as_ptr(),
            output_names.len(),
            fetched.as_mut_ptr(),
        )
    };
    // SAFETY: the status comes straight from ONNX Runtime and is read once.
    unsafe { ort::Error::result_from_status(status) }.map_err(|error| failed(&error))?;
    let owner = session.inner();
    outputs
        .into_iter()
        .zip(fetched)
        .map(|(name, value)| {
            let value = NonNull::new(value).ok_or_else(|| failed(&"an output came back empty"))?;
            // SAFETY: `Run` succeeded, so `value` is an output it allocated
            // and this pass now owns; `owner` keeps the session alive for as
            // long as the value is.
            let value = unsafe { DynValue::from_ptr(value, Some(std::sync::Arc::clone(&owner))) };
            Ok((name.to_string_lossy().into_owned(), value))
        })
        .collect()
}

/// `outputs` as [`run_shared`] returns them.
pub fn owned(outputs: SessionOutputs<'_>) -> Outputs {
    outputs
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn threads() -> usize {
        crate::budget::threads(None).expect("the default budget")
    }

    /// A device that will not register is found before its model is asked for,
    /// since asking can mean downloading an export this machine cannot run.
    ///
    /// Whether an accelerator registers is a property of the machine, so each
    /// is first registered on a builder of its own; only one that fails there
    /// says anything about the order. On a machine with no GPU -- every one
    /// this suite has run on -- that is each of them.
    #[test]
    fn a_device_that_will_not_register_is_never_asked_for_its_model() {
        for (device, provider) in
            crate::device::accelerators(&[]).expect("no plugin library is named")
        {
            let registers = options(threads(), provider.clone()).is_ok();
            if registers {
                continue;
            }
            let asked = std::cell::Cell::new(false);
            let loaded = session(threads(), provider, || {
                asked.set(true);
                Err(Error::Runtime("no model here".into()))
            });
            assert!(
                loaded.is_err(),
                "{} registered the second time",
                device.name()
            );
            assert!(
                !asked.get(),
                "{}'s model was asked for before the device refused to register",
                device.name()
            );
        }
    }

    #[test]
    fn absent_plugin_npu_fails_before_model_fetch() {
        let fetched = std::cell::Cell::new(false);
        let result = session::<Error>(
            threads(),
            Target::Npu {
                provider: "absent-test-provider".into(),
                id: u32::MAX,
            },
            || {
                fetched.set(true);
                Ok("unused.onnx".into())
            },
        );
        assert!(result.is_err());
        assert!(!fetched.get());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn coreml_cache_isolates_graph_content_and_static_shapes() {
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("model.onnx");
        std::fs::write(&model, b"first graph").unwrap();
        let first = coreml_cache(&model, 4, 64).unwrap();
        assert_eq!(first, coreml_cache(&model, 4, 64).unwrap());
        assert_ne!(first, coreml_cache(&model, 4, 128).unwrap());
        assert_ne!(first, coreml_cache(&model, 2, 64).unwrap());
        std::fs::write(&model, b"updated graph").unwrap();
        assert_ne!(first, coreml_cache(&model, 4, 64).unwrap());
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "compiles two tiny native CoreML models"]
    fn coreml_compiled_cache_preserves_shapes_and_updated_weights()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temporary = tempfile::tempdir()?;
        let model = temporary.path().join("model.onnx");
        // Same dynamic MatMul/Add/Relu graph; diagonal weights are 2 or 3.
        // Loading 64, then 128, then 64 also checks compiled cache reuse.
        let models: [(&[u8], f32); 2] = [
            (include_bytes!("../tests/fixtures/coreml-cache-2.onnx"), 2.0),
            (include_bytes!("../tests/fixtures/coreml-cache-3.onnx"), 3.0),
        ];
        for (bytes, scale) in models {
            std::fs::write(&model, bytes)?;
            for tokens in [64, 128, 64] {
                let (mut session, _) =
                    fixed_coreml(threads(), || Ok::<_, Error>(model.clone()), 4, tokens)?;
                let input = ort::value::Tensor::from_array((
                    [4, tokens, 8],
                    vec![0.25f32; 4 * tokens * 8],
                ))?;
                let output = session.run(ort::inputs!["x" => input])?;
                let (shape, values) = output["y"].try_extract_tensor::<f32>()?;
                assert_eq!(shape.as_ref(), [4, tokens as i64, 8]);
                assert!(values.iter().all(|value| *value == scale * 0.25 + 0.125));
                let cache = coreml_cache(&model, 4, tokens)?;
                assert!(
                    std::fs::read_dir(cache)?.count() > 1,
                    "CoreML wrote no cache"
                );
            }
        }
        Ok(())
    }

    /// MatMul/Add fusion transposes a large FFN weight into an inline MIL
    /// constant. Check the serialized native graph, not our option value.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "compiles a small native CoreML model"]
    fn coreml_keeps_matrix_weights_out_of_the_graph()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        use crate::onnx::*;
        use ort::ep::coreml::{ComputeUnits, ModelFormat};

        fn tensor(name: &str, dimensions: &[u64], values: &[f32]) -> Vec<u8> {
            let mut tensor = Vec::new();
            for &dimension in dimensions {
                put_varint_field(&mut tensor, TENSOR_DIMS, dimension);
            }
            put_varint_field(&mut tensor, TENSOR_DATA_TYPE, 1);
            put_bytes(&mut tensor, TENSOR_NAME, name.as_bytes());
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            put_bytes(&mut tensor, TENSOR_RAW_DATA, &bytes);
            tensor
        }
        fn info(name: &str, dimensions: &[u64]) -> Vec<u8> {
            let mut shape = Vec::new();
            for &dimension in dimensions {
                let mut value = Vec::new();
                put_varint_field(&mut value, 1, dimension);
                put_bytes(&mut shape, 1, &value);
            }
            let mut tensor = Vec::new();
            put_varint_field(&mut tensor, TENSOR_TYPE_ELEM_TYPE, 1);
            put_bytes(&mut tensor, 2, &shape);
            let mut ty = Vec::new();
            put_bytes(&mut ty, TYPE_TENSOR, &tensor);
            let mut info = Vec::new();
            put_bytes(&mut info, VALUE_INFO_NAME, name.as_bytes());
            put_bytes(&mut info, VALUE_INFO_TYPE, &ty);
            info
        }
        fn node(op: &str, inputs: &[&str], output: &str) -> Vec<u8> {
            let mut node = Vec::new();
            for input in inputs {
                put_bytes(&mut node, NODE_INPUT, input.as_bytes());
            }
            put_bytes(&mut node, NODE_OUTPUT, output.as_bytes());
            put_bytes(&mut node, NODE_NAME, op.as_bytes());
            put_bytes(&mut node, NODE_OP_TYPE, op.as_bytes());
            node
        }
        let mut graph = Vec::new();
        put_bytes(&mut graph, 2, b"FFN matrix and bias");
        put_bytes(
            &mut graph,
            GRAPH_NODE,
            &node("MatMul", &["x", "weight"], "product"),
        );
        put_bytes(
            &mut graph,
            GRAPH_NODE,
            &node("Add", &["product", "bias"], "y"),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("weight", &[1024, 4096], &vec![1.0 / 4096.0; 1024 * 4096]),
        );
        put_bytes(
            &mut graph,
            GRAPH_INITIALIZER,
            &tensor("bias", &[4096], &vec![0.125; 4096]),
        );
        put_bytes(&mut graph, GRAPH_INPUT, &info("x", &[1, 32, 1024]));
        put_bytes(
            &mut graph,
            GRAPH_VALUE_INFO,
            &info("product", &[1, 32, 4096]),
        );
        put_bytes(&mut graph, GRAPH_OUTPUT, &info("y", &[1, 32, 4096]));
        let mut model = Vec::new();
        put_varint_field(&mut model, 1, 8);
        let mut opset = Vec::new();
        put_varint_field(&mut opset, 2, 17);
        put_bytes(&mut model, MODEL_OPSET_IMPORT, &opset);
        put_bytes(&mut model, MODEL_GRAPH, &graph);

        let cache = tempfile::tempdir()?;
        let provider = ort::ep::CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_compute_units(ComputeUnits::CPUAndGPU)
            .with_model_cache_dir(cache.path().display())
            .build()
            .error_on_failure();
        let mut session = options(threads(), vec![provider])?.commit_from_memory(&model)?;
        let output = session.run(ort::inputs!["x" => ort::value::Tensor::from_array(([1,32,1024],vec![0.25f32;32*1024]))?])?;
        let (_, values) = output["y"].try_extract_tensor::<f32>()?;
        assert_eq!(values.len(), 32 * 4096);
        assert!(
            values.iter().all(|v| (*v - 0.1875).abs() < 1e-5),
            "matrix or bias result changed"
        );

        let mut directories = vec![cache.path().to_path_buf()];
        let mut graphs = Vec::new();
        let mut native_matmul = false;
        while let Some(directory) = directories.pop() {
            for entry in std::fs::read_dir(directory)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    directories.push(entry.path());
                } else if entry.file_name() == "model.mlmodel" {
                    let bytes = std::fs::read(entry.path())?;
                    native_matmul |= bytes.windows(6).any(|word| word == b"matmul");
                    graphs.push(bytes.len() as u64);
                }
            }
        }
        println!("native CoreML graph sizes: {graphs:?}");
        assert!(
            graphs.iter().any(|&bytes| bytes > 0),
            "no native CoreML graph was produced; the premise fell back to CPU"
        );
        assert!(
            native_matmul,
            "the matrix operation was not lowered into the native CoreML graph"
        );
        assert!(
            graphs.iter().all(|&bytes| bytes < 1024 * 1024),
            "matrix weights were serialized inline instead of into the weights file: {graphs:?}"
        );
        Ok(())
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "compiles a tiny native CoreML model and corrupts only its temporary cache"]
    fn coreml_rebuilds_interrupted_and_damaged_published_caches()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temporary = tempfile::tempdir()?;
        let model = temporary.path().join("model.onnx");
        std::fs::write(
            &model,
            include_bytes!("../tests/fixtures/coreml-cache-2.onnx"),
        )?;
        for published in [false, true] {
            let (session, _) = fixed_coreml(threads(), || Ok::<_, Error>(model.clone()), 4, 64)?;
            drop(session);
            let cache = coreml_cache(&model, 4, 64)?;
            assert!(cache.join("ready").is_file());
            let mut damaged = 0;
            for directory in std::fs::read_dir(&cache)? {
                let directory = directory?.path();
                if !directory.is_dir() {
                    continue;
                }
                for partition in std::fs::read_dir(directory)? {
                    let package = partition?.path().join("model");
                    if package.join("Manifest.json").is_file() {
                        std::fs::remove_file(package.join("Manifest.json"))?;
                        std::fs::remove_dir_all(package.join("compiled_model.mlmodelc"))?;
                        damaged += 1;
                    }
                }
            }
            assert!(damaged > 0, "did not reproduce the missing manifest");
            if !published {
                std::fs::remove_file(cache.join("ready"))?;
            }
            let (mut session, _) =
                fixed_coreml(threads(), || Ok::<_, Error>(model.clone()), 4, 64)?;
            let output = session.run(ort::inputs!["x" => ort::value::Tensor::from_array((
                [4, 64, 8], vec![0.25f32; 4 * 64 * 8],
            ))?])?;
            let (_, values) = output["y"].try_extract_tensor::<f32>()?;
            assert!(values.iter().all(|value| *value == 0.625));
            assert!(cache.join("ready").is_file());
        }
        Ok(())
    }

    #[test]
    fn failed_cache_retry_preserves_the_published_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let cache = temporary.path().join("cache");
        std::fs::create_dir(&cache).unwrap();
        std::fs::write(cache.join("ready"), b"published").unwrap();
        let result: Result<()> = rebuild_coreml_cache(&cache, || {
            std::fs::write(cache.join("partial"), b"failed replacement")?;
            Err(Error::Runtime("transient provider failure".into()))
        });
        assert!(result.is_err());
        assert_eq!(std::fs::read(cache.join("ready")).unwrap(), b"published");
        assert!(!cache.join("partial").exists());
        assert!(!cache.with_extension("previous").exists());
        std::fs::rename(&cache, cache.with_extension("previous")).unwrap();
        std::fs::create_dir(&cache).unwrap();
        std::fs::write(cache.join("partial"), b"interrupted retry").unwrap();
        recover_coreml_cache(&cache).unwrap();
        assert_eq!(std::fs::read(cache.join("ready")).unwrap(), b"published");
        rebuild_coreml_cache(&cache, || {
            std::fs::write(cache.join("replacement"), b"loaded")?;
            Ok(())
        })
        .unwrap();
        assert!(!cache.with_extension("previous").exists());
        assert_eq!(std::fs::read(cache.join("replacement")).unwrap(), b"loaded");
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    #[ignore = "requires the native CoreML execution provider"]
    fn modern_coreml_assigns_normalization_that_legacy_format_cannot() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("normalization.onnx");
        std::fs::write(
            &path,
            include_bytes!("../tests/fixtures/coreml-normalization.onnx"),
        )?;
        let mut modern = session(threads(), vec![coreml().build().error_on_failure()], || {
            Ok::<_, Error>(path.clone())
        })?;
        let assigned = assigned_providers(&modern)?;
        assert!(
            assigned
                .get("CoreMLExecutionProvider")
                .is_some_and(|nodes| *nodes > 0)
        );
        let legacy = session(
            threads(),
            vec![
                ort::ep::CoreML::default()
                    .with_model_format(ort::ep::coreml::ModelFormat::NeuralNetwork)
                    .build()
                    .error_on_failure(),
            ],
            || Ok::<_, Error>(path.clone()),
        )?;
        let old = assigned_providers(&legacy)?;
        assert_eq!(old.get("CoreMLExecutionProvider").copied().unwrap_or(0), 0);
        assert!(
            old.get("CPUExecutionProvider")
                .is_some_and(|nodes| *nodes > 0)
        );
        let input = ort::value::Tensor::from_array(([1, 4], vec![1.0f32, 2.0, 3.0, 4.0]))
            .map_err(|error| Error::Runtime(error.to_string()))?;
        let output = modern
            .run(ort::inputs!["x" => input])
            .map_err(|error| Error::Runtime(error.to_string()))?;
        let (_, values) = output["y"]
            .try_extract_tensor::<f32>()
            .map_err(|error| Error::Runtime(error.to_string()))?;
        let expected = [-1.5f32, -0.5, 0.5, 1.5].map(|x| x / 1.25001f32.sqrt());
        assert_eq!(values.len(), expected.len());
        assert!(
            values
                .iter()
                .zip(expected)
                .all(|(value, expected)| (*value - expected).abs() < 1e-3)
        );
        Ok(())
    }

    #[test]
    fn preparation_probes_do_not_claim_to_be_product_model_loads() {
        // Tracing callsite interest is process-global. Other parallel session
        // tests use these same callsites without this thread-local collector.
        // Isolate the capture, while retaining real model loads and assertions.
        const CHILD: &str = "MODEL_LOAD_CAPTURE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "inference::tests::preparation_probes_do_not_claim_to_be_product_model_loads",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .status()
                .unwrap();
            assert!(
                status.success(),
                "isolated model provenance regression failed"
            );
            return;
        }
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct Capture(Arc<Mutex<Vec<u8>>>);
        impl Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let capture = Capture(Arc::default());
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .without_time()
            .with_ansi(false)
            .with_writer(move || writer.clone())
            .finish();
        let temporary = tempfile::tempdir().unwrap();
        let model = temporary.path().join("attention.onnx.partial");
        std::fs::write(
            &model,
            include_bytes!("../tests/fixtures/coreml-cache-2.onnx"),
        )
        .unwrap();
        tracing::subscriber::with_default(subscriber, || {
            let _probe = probe(threads(), &model).unwrap();
            assert!(
                capture.0.lock().unwrap().is_empty(),
                "probe logged a product load"
            );
            let _product =
                session(threads(), vec![cpu()], || Ok::<_, Error>(model.clone())).unwrap();
        });
        let bytes = capture.0.lock().unwrap();
        let log = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(log.matches("loaded ONNX graph").count(), 1);
        assert!(log.contains("attention.onnx.partial"));
    }

    #[test]
    fn sessions_record_actual_provider_assignment() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        std::fs::write(
            &path,
            include_bytes!("../tests/fixtures/coreml-cache-2.onnx"),
        )
        .unwrap();
        let session = session(threads(), vec![cpu()], || Ok::<_, Error>(path.clone())).unwrap();
        let assigned = assigned_providers(&session).unwrap();
        assert!(
            assigned
                .get("CPUExecutionProvider")
                .is_some_and(|nodes| *nodes > 0)
        );
        assert_eq!(
            assigned.len(),
            1,
            "a CPU-only session reported another provider"
        );
        let unrecorded = Session::builder()
            .unwrap()
            .with_execution_providers(vec![cpu()])
            .unwrap()
            .commit_from_file(path)
            .unwrap();
        assert!(
            assigned_providers(&unrecorded).is_err(),
            "assignment inspection worked without enabling its collection premise"
        );
    }

    /// Only a session whose every node is on the CPU provider may be run from
    /// several threads at once.
    #[test]
    fn only_the_cpu_alone_reaches_the_shared_session() {
        let assigned = |entries: &[(&str, usize)]| {
            entries
                .iter()
                .map(|(provider, nodes)| (provider.to_string(), *nodes))
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        assert!(cpu_alone(&assigned(&[("CPUExecutionProvider", 900)])).is_ok());
        assert!(cpu_alone(&assigned(&[])).is_err(), "nothing recorded");
        for other in [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "TensorrtExecutionProvider",
            "XnnpackExecutionProvider",
        ] {
            assert!(
                cpu_alone(&assigned(&[("CPUExecutionProvider", 10), (other, 1)])).is_err(),
                "{other} reached the shared session"
            );
        }
    }

    /// A real graph loaded for sharing records its nodes on the CPU provider,
    /// so the guard above admits it rather than refusing every model.
    #[test]
    fn a_cpu_graph_loads_as_a_shared_session() {
        let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/coreml-normalization.onnx");
        SharedCpuSession::load(&model).expect("a CPU graph is admitted");
    }
}

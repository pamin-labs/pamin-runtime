//! Which devices a session may run on, and what it was actually given.
//!
//! Each platform's ONNX Runtime carries the accelerator that platform has, with
//! no build flag: CUDA on x86-64 Linux, Core ML on Apple silicon, DirectML on
//! Windows, plus any NPU a plugin library or the Windows ML catalog brings.
//! [`accelerators`] lists the ones to try before the [`cpu`], best first; a
//! caller loads a model on each in turn and keeps the first that works.

use std::path::{Path, PathBuf};

use ort::ep::ExecutionProviderDispatch;
use ort::session::Session;

use crate::{Error, Result};

/// What a session is asked to run on: execution providers in order, or one
/// plugin NPU that device discovery found. Model code owns its exports; this
/// module owns physical device selection and fallback.
#[derive(Clone)]
pub enum Target {
    Providers(Vec<ExecutionProviderDispatch>),
    Npu { provider: String, id: u32 },
}

impl From<Vec<ExecutionProviderDispatch>> for Target {
    fn from(providers: Vec<ExecutionProviderDispatch>) -> Self {
        Self::Providers(providers)
    }
}

/// The execution provider selected for a model's forward passes.
///
/// Recorded on every loaded model rather than inferred, because the answer is
/// not what the platform says: every x86-64 Linux build can use CUDA, and one
/// on a machine with no GPU, or with the wrong CUDA, runs on the CPU -- and a
/// score cannot be read without knowing which. A model may run a different
/// export on each, so the two need not produce bit-identical scores and are
/// not interchangeable in a measurement. A registered accelerator may still delegate unsupported
/// nodes to CPU; CoreML itself may use CPU, GPU or the Neural Engine. Inspect
/// the runtime's node profile before calling this a GPU measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda,
    CoreMl,
    DirectMl,
    Npu,
}

impl Device {
    /// The name a record or a log line spells the device with.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
            Self::Npu => "npu",
        }
    }
}

/// The CPU, with ONNX Runtime's memory arena off.
///
/// The arena keeps every buffer a pass ever needed and hands it back to the
/// next pass instead of to the system, so a model's resident size settles at
/// its largest batch and stays there. Measured on the `accurate` reranker, six
/// passes of sixteen passages at 256 tokens: 1,682 MB resident with the arena
/// against 1,077 MB without -- 605 MB of activations held for a batch shape
/// that recurs only when the next search runs -- with scores bit-identical and
/// no slower pass. Through the whole search path on MIRACL (the `MEMORY` arm),
/// a hundred `accurate` searches hold 5,914 MB anonymous against 6,208 MB --
/// less than the six fixed passes, because a real search scores fewer and
/// shorter pairs. The embedder saves nothing measurable there, since a query
/// is one short text; it takes the same setting because its largest batch is
/// a bulk write's, which that arm does not exercise.
///
/// The arena holds activations. The weights are the other half, and loaded
/// from the file the hub serves they are copied onto the heap: +664 MB
/// anonymous for the `accurate` reranker's 570 MB export in a bare session.
/// So a CPU session loads a prepared copy whose weights ONNX Runtime maps
/// from disk instead, +11 MB in the same session -- see [`crate::prepared`].
pub fn cpu() -> ExecutionProviderDispatch {
    ort::ep::CPU::default().with_arena_allocator(false).build()
}

/// Counts assigned nodes after basic optimizations, not executed kernel time.
/// CoreML dispatch does not reveal its internal CPU/GPU/ANE placement.
pub fn assigned_providers(session: &Session) -> Result<std::collections::BTreeMap<String, usize>> {
    use ort::AsPointer;
    let checked = |status| {
        // SAFETY: each status comes directly from ORT; this consumes it once
        // and the error wrapper releases it, including on an early return.
        unsafe { ort::Error::result_from_status(status) }
            .map_err(|error| Error::Runtime(format!("reading provider assignment: {error}")))
    };
    let mut subgraphs = std::ptr::null();
    let mut count = 0;
    // SAFETY: the session and both output variables remain live. Returned
    // arrays and entries are borrowed from this session, never freed here.
    checked(unsafe {
        (ort::api().Session_GetEpGraphAssignmentInfo)(session.ptr(), &mut subgraphs, &mut count)
    })?;
    let mut providers = std::collections::BTreeMap::new();
    if count == 0 {
        return Ok(providers);
    }
    if subgraphs.is_null() {
        return Err(Error::Runtime(
            "provider assignment returned a null array".into(),
        ));
    }
    // SAFETY: ORT returned `count` entries, valid for the session borrow.
    for &subgraph in unsafe { std::slice::from_raw_parts(subgraphs, count) } {
        if subgraph.is_null() {
            return Err(Error::Runtime(
                "provider assignment returned a null subgraph".into(),
            ));
        }
        let mut name = std::ptr::null();
        let mut nodes = std::ptr::null();
        let mut node_count = 0;
        // SAFETY: this is a live session-owned subgraph and valid outputs.
        checked(unsafe { (ort::api().EpAssignedSubgraph_GetEpName)(subgraph, &mut name) })?;
        checked(unsafe {
            (ort::api().EpAssignedSubgraph_GetNodes)(subgraph, &mut nodes, &mut node_count)
        })?;
        if name.is_null() {
            return Err(Error::Runtime(
                "provider assignment returned a null name".into(),
            ));
        }
        // SAFETY: ORT returns a session-owned NUL-terminated provider name.
        let name = unsafe { std::ffi::CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        *providers.entry(name).or_default() += node_count;
    }
    Ok(providers)
}

/// Use the same modern format for every CoreML model. The legacy NeuralNetwork
/// format cannot accept the standard normalization operators used by modern
/// transformer graphs. ALL permits the framework's CPU/GPU/ANE combination.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn coreml() -> ort::ep::CoreML {
    ort::ep::CoreML::default()
        .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
        .with_compute_units(ort::ep::coreml::ComputeUnits::All)
}

/// The accelerators to try before the CPU, best first.
///
/// Whatever this platform's runtime carries, with no build flag: CUDA on
/// x86-64 Linux, Core ML on Apple silicon, DirectML on Windows, nothing
/// elsewhere. A machine without the device -- or, for CUDA, without the
/// driver, CUDA 13 and cuDNN 9 -- fails to register it and runs on the CPU,
/// so the available provider is tried before the CPU. Successful registration
/// does not establish how much of the graph that provider accelerates.
/// A caller that must stay on the CPU -- for a measurement that must be
/// comparable with a CPU one, or on a machine whose GPU belongs to something
/// else -- does not ask.
///
/// Each is set to fail loudly on registration rather than fall through to the
/// CPU inside ONNX Runtime, which is its default. A silent fallback would load
/// the GPU's fp16 export onto the CPU -- slower than the int8 one it was
/// chosen over -- and report nothing; failing here lets the caller load the
/// CPU's own export instead and record that it did.
///
/// Refuses a caller naming a plugin library that was refused for bringing a
/// provider another library already brought; see [`claim`].
///
/// `ep_libraries` are absolute paths of NPU plugin execution-provider
/// libraries to register; only their providers and the built-in ones are
/// listed.
pub fn accelerators(ep_libraries: &[PathBuf]) -> Result<Vec<(Device, Target)>> {
    let mut result = Vec::new();
    if let Ok(environment) = ort::environment::Environment::current() {
        let plugins = register_plugins(&environment, ep_libraries)?;
        #[cfg(target_os = "windows")]
        crate::winml::register(&environment);
        for device in environment.devices() {
            let hardware = device.hardware_device();
            if hardware.ty() == ort::memory::DeviceType::NPU
                && let Ok(provider) = device.ep()
                && provider != "CoreMLExecutionProvider"
                && requested(provider, &plugins, ep_libraries)
            {
                let target = Target::Npu {
                    provider: provider.into(),
                    id: hardware.id(),
                };
                result.push((Device::Npu, target));
            }
        }
    }
    result.extend(
        gpu_providers()
            .into_iter()
            .map(|(device, ep)| (device, vec![ep.error_on_failure()].into())),
    );
    Ok(result)
}

/// Retain registered libraries for the environment/session lifetime. Paths
/// are explicit because vendor plugin installation differs by platform.
///
/// Each library is registered the first time any caller names it, and kept
/// by path: a caller naming a library no earlier caller did gets it
/// registered, rather than finding the set fixed by whoever came first. A
/// path that is relative, or failed to register, is recorded too, so it is
/// warned about once rather than on every call.
///
/// The registry is the process's, so a library one caller named stays
/// registered for every later one. Returns which library each plugin
/// execution provider came from -- the providers that gained devices when
/// that library was registered -- so [`requested`] can keep a caller to the
/// libraries it named. A library bringing a provider another one already
/// brought is unregistered and refused; see [`claim`].
fn register_plugins(
    environment: &std::sync::Arc<ort::environment::Environment>,
    libraries: &[PathBuf],
) -> Result<std::collections::BTreeMap<String, PathBuf>> {
    #[derive(Default)]
    struct Registry {
        libraries: std::collections::BTreeMap<PathBuf, Option<ort::ep::ExecutionProviderLibrary>>,
        providers: std::collections::BTreeMap<String, PathBuf>,
        /// Why each refused library was refused.
        refused: std::collections::BTreeMap<PathBuf, String>,
    }
    static REGISTERED: std::sync::Mutex<Option<Registry>> = std::sync::Mutex::new(None);
    let mut registered = REGISTERED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let registry = registered.get_or_insert_with(Registry::default);
    // Devices per provider, so a library bringing more devices of a provider
    // already listed is seen bringing that provider.
    let providers = || -> std::collections::BTreeMap<String, usize> {
        let mut devices = std::collections::BTreeMap::new();
        for device in environment.devices() {
            if let Ok(provider) = device.ep() {
                *devices.entry(provider.to_owned()).or_default() += 1;
            }
        }
        devices
    };
    for path in libraries {
        if registry.libraries.contains_key(path) {
            continue;
        }
        let library = if !path.is_absolute() {
            tracing::warn!(path = %path.display(), "execution-provider library must be absolute");
            None
        } else {
            let before = providers();
            // Named by the order of registration, which no two share.
            let name = format!("pamin-plugin-{}", registry.libraries.len());
            match environment.register_ep_library(name, path) {
                Ok(library) => {
                    let brought: Vec<String> = providers()
                        .into_iter()
                        .filter(|(provider, count)| {
                            before.get(provider).is_none_or(|was| count > was)
                        })
                        .map(|(provider, _)| provider)
                        .collect();
                    match claim(&mut registry.providers, &before, &brought, path) {
                        Ok(()) => Some(library),
                        Err(refusal) => {
                            if let Err(error) = library.unregister() {
                                tracing::warn!(path = %path.display(), %error, "could not unregister a refused execution-provider library");
                            }
                            registry.refused.insert(path.clone(), refusal);
                            None
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(path = %path.display(), %error, "execution-provider library unavailable");
                    None
                }
            }
        };
        registry.libraries.insert(path.clone(), library);
    }
    if let Some(refusal) = libraries.iter().find_map(|path| registry.refused.get(path)) {
        return Err(Error::PluginConflict(refusal.clone()));
    }
    Ok(registry.providers.clone())
}

/// Records that the library at `path` brought the plugin providers
/// `brought`, or refuses it if another library already brought one of them,
/// or the runtime had one of them built in (`before` holds the providers
/// listed before the library was registered).
///
/// ONNX Runtime plans a device by its provider's name, and a device does not
/// say which library it came from. Two sources of one provider name cannot be
/// told apart: a caller naming the library would plan on the other's devices,
/// or a caller not naming it would be filtered off both and fall back without
/// a word. So the library is refused, naming what already provides it.
fn claim(
    providers: &mut std::collections::BTreeMap<String, PathBuf>,
    before: &std::collections::BTreeMap<String, usize>,
    brought: &[String],
    path: &Path,
) -> std::result::Result<(), String> {
    let built_in = brought
        .iter()
        .find(|provider| before.contains_key(*provider) && !providers.contains_key(*provider));
    if let Some(provider) = built_in {
        return Err(format!(
            "the execution-provider library {} provides {provider}, which this ONNX \
             Runtime already provides; a provider is chosen by name, so the library's \
             devices could not be told from the built-in ones: use them from separate \
             processes",
            path.display()
        ));
    }
    let taken = brought.iter().find_map(|provider| {
        providers
            .get(provider)
            .filter(|owner| owner.as_path() != path)
            .map(|owner| (provider, owner))
    });
    if let Some((provider, owner)) = taken {
        return Err(format!(
            "the execution-provider library {} provides {provider}, which {} already \
             provides in this process; a provider is chosen by name, so only one of \
             them can be used: name one, or use them from separate processes",
            path.display(),
            owner.display()
        ));
    }
    for provider in brought {
        providers.insert(provider.clone(), path.to_path_buf());
    }
    Ok(())
}

/// Whether a device of `provider` is one this caller may plan on: a provider
/// no plugin library brought (a built-in one), or one from a library in
/// `asked`. `plugins` is which library each plugin provider came from, as
/// [`register_plugins`] returns it.
fn requested(
    provider: &str,
    plugins: &std::collections::BTreeMap<String, PathBuf>,
    asked: &[PathBuf],
) -> bool {
    plugins
        .get(provider)
        .is_none_or(|library| asked.contains(library))
}

pub(crate) fn gpu_providers() -> Vec<(Device, ExecutionProviderDispatch)> {
    let providers = vec![
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        (Device::CoreMl, coreml().build()),
        #[cfg(target_os = "windows")]
        (Device::DirectMl, ort::ep::DirectML::default().build()),
    ];
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    let providers = {
        let mut providers = providers;
        if nvidia_driver(|node| node.exists()) || nvidia_gpu(pci_vendors()) {
            providers.push((Device::Cuda, ort::ep::CUDA::default().build()));
        }
        providers
    };
    providers
}

/// Whether this host has an NVIDIA driver for CUDA to run on: a device node
/// the driver creates, `/dev/nvidiactl`, or `/dev/dxg` where WSL 2 passes a
/// GPU through; or `/proc/driver/nvidia`, there once the kernel module is
/// loaded, for a headless host where the first CUDA process creates the nodes
/// through `nvidia-modprobe`. Without a driver, registering the CUDA provider
/// can only fail, and it fails slowly: about 3 s on every server start on a
/// CPU-only host, since a plan naming CUDA persists only with a GPU inventory
/// read (q101).
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn nvidia_driver(exists: impl Fn(&Path) -> bool) -> bool {
    ["/dev/nvidiactl", "/dev/dxg", "/proc/driver/nvidia"]
        .into_iter()
        .any(|node| exists(Path::new(node)))
}

/// Whether one of `vendors`, PCI vendor ids as sysfs spells them, is NVIDIA's:
/// a GPU whose module is not loaded yet, which `nvidia-modprobe` also loads
/// for the first CUDA process.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn nvidia_gpu(vendors: impl IntoIterator<Item = String>) -> bool {
    vendors.into_iter().any(|vendor| vendor.trim() == "0x10de")
}

/// The vendor id of every PCI device sysfs lists.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn pci_vendors() -> impl Iterator<Item = String> {
    std::fs::read_dir("/sys/bus/pci/devices")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|device| std::fs::read_to_string(device.path().join("vendor")).ok())
}

/// The variables that choose which CUDA devices a session sees, and in
/// what order: part of every persistent plan's identity, and of any record
/// that has to say which GPU a CUDA run used.
pub const CUDA_SELECTION: [&str; 3] = [
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
];

/// The machine's NVIDIA GPUs -- index, UUID, PCI bus, name and driver, as
/// `nvidia-smi` lists them -- the inventory a CUDA plan's identity holds;
/// `None` where it cannot be read. With [`CUDA_SELECTION`], it says which
/// physical GPU a CUDA session ran on.
pub fn cuda_devices() -> Option<String> {
    #[cfg(unix)]
    {
        let mut command = std::process::Command::new("nvidia-smi");
        command.args([
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
            "--format=csv,noheader",
        ]);
        cuda_inventory(&bounded_inventory(
            command,
            std::time::Duration::from_secs(2),
        )?)
    }
    #[cfg(not(unix))]
    None
}

/// At most one inventory child and cleanup worker may exist in this process.
/// A wedged driver keeps the permit until the killed child is actually reaped;
/// further probes return no identity without adding children or threads.
#[cfg(unix)]
struct InventoryPermit(std::sync::Arc<std::sync::atomic::AtomicBool>);

#[cfg(unix)]
impl InventoryPermit {
    fn acquire(busy: std::sync::Arc<std::sync::atomic::AtomicBool>) -> Option<Self> {
        busy.compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
        .ok()?;
        Some(Self(busy))
    }
}

#[cfg(unix)]
impl Drop for InventoryPermit {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

#[cfg(unix)]
fn bounded_inventory(
    command: std::process::Command,
    timeout: std::time::Duration,
) -> Option<Vec<u8>> {
    static BUSY: std::sync::OnceLock<std::sync::Arc<std::sync::atomic::AtomicBool>> =
        std::sync::OnceLock::new();
    let permit = InventoryPermit::acquire(BUSY.get_or_init(Default::default).clone())?;
    inventory_with_permit(command, timeout, permit)
}

#[cfg(unix)]
fn inventory_with_permit(
    mut command: std::process::Command,
    timeout: std::time::Duration,
    permit: InventoryPermit,
) -> Option<Vec<u8>> {
    use std::io::Read;
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    use std::process::Stdio;
    // The standard library owns nonblocking configuration and descriptor
    // transfer. A socket pair avoids handwritten fcntl FFI.
    let (mut output, writer) = UnixStream::pair().ok()?;
    output.set_nonblocking(true).ok()?;
    command
        .stdout(Stdio::from(OwnedFd::from(writer)))
        .stderr(Stdio::null());
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("cuda-inventory".into())
        .spawn(move || {
            let Some(mut child) = command.spawn().ok() else {
                // No child exists: release admission before waking the caller.
                drop(permit);
                let _ = send.send(None);
                return;
            };
            let deadline = std::time::Instant::now() + timeout;
            let result = (|| {
                let mut bytes = Vec::new();
                let mut buffer = [0; 4096];
                loop {
                    let exited = child.try_wait().ok()?;
                    loop {
                        match output.read(&mut buffer) {
                            Ok(0) => break,
                            Ok(read) => {
                                if bytes.len() + read > 65536 {
                                    return None;
                                }
                                bytes.extend_from_slice(&buffer[..read]);
                            }
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => break,
                            Err(_) => return None,
                        }
                        if std::time::Instant::now() >= deadline {
                            return None;
                        }
                    }
                    if let Some(status) = exited {
                        return status.success().then_some(bytes);
                    }
                    if std::time::Instant::now() >= deadline {
                        return None;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            })();
            if result.is_none() {
                let _ = child.kill();
                // This wait can outlast the caller's deadline but retains the sole
                // permit. No subsequent inventory can create another reaper.
                let _ = child.wait();
            }
            drop(permit);
            let _ = send.send(result);
        })
        .ok()?;
    receive
        .recv_timeout(timeout + std::time::Duration::from_millis(100))
        .ok()?
}

#[cfg(any(unix, test))]
fn cuda_inventory(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?.trim();
    if text.is_empty()
        || text.lines().any(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            fields.len() != 5
                || fields.iter().any(|field| field.is_empty())
                || !fields[1].starts_with("GPU-")
        })
    {
        return None;
    }
    Some(text.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CUDA is a candidate only where there is an NVIDIA driver or GPU: a
    /// device node, natively or through WSL 2, the loaded kernel module, or an
    /// NVIDIA PCI device.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn cuda_is_planned_only_with_an_nvidia_driver() {
        use std::path::Path;
        let only = |present: &'static str| move |node: &Path| node == Path::new(present);
        assert!(super::nvidia_driver(only("/dev/nvidiactl")));
        assert!(super::nvidia_driver(only("/dev/dxg")));
        assert!(super::nvidia_driver(only("/proc/driver/nvidia")));
        assert!(!super::nvidia_driver(only("/dev/dri/renderD128")));
        assert!(!super::nvidia_driver(|_: &Path| false));
        let vendors = |ids: &[&str]| ids.iter().map(|id| format!("{id}\n")).collect::<Vec<_>>();
        assert!(super::nvidia_gpu(vendors(&["0x8086", "0x10de"])));
        assert!(!super::nvidia_gpu(vendors(&["0x8086", "0x1af4", "0x1002"])));
        assert!(!super::nvidia_gpu(vendors(&[])));
    }

    /// A caller plans on the plugins it named and the built-in providers, and
    /// not on a plugin only another caller named.
    #[test]
    fn a_caller_sees_only_the_plugins_it_named() {
        use std::path::PathBuf;
        let a = PathBuf::from("/opt/a/libnpu_ep.so");
        let b = PathBuf::from("/opt/b/libnpu_ep.so");
        let plugins = [
            ("ANpuExecutionProvider".to_string(), a.clone()),
            ("BNpuExecutionProvider".to_string(), b.clone()),
        ]
        .into();
        let requested = |provider, asked: &[PathBuf]| super::requested(provider, &plugins, asked);
        assert!(requested("ANpuExecutionProvider", std::slice::from_ref(&a)));
        assert!(!requested(
            "BNpuExecutionProvider",
            std::slice::from_ref(&a)
        ));
        assert!(!requested("ANpuExecutionProvider", &[]));
        assert!(requested("ANpuExecutionProvider", &[b.clone(), a.clone()]));
        assert!(requested("BNpuExecutionProvider", &[b, a]));
        // A provider no plugin brought is built in, and every caller has it.
        assert!(requested("QNNExecutionProvider", &[]));
    }

    /// A second library bringing a provider the first already brought is
    /// refused, naming both, and leaves the provider the first's.
    #[test]
    fn a_library_bringing_a_taken_provider_is_refused() {
        use std::path::PathBuf;
        let a = PathBuf::from("/opt/a/libnpu_ep.so");
        let b = PathBuf::from("/opt/b/libnpu_ep.so");
        let npu = ["NpuExecutionProvider".to_string()];
        let mut providers = std::collections::BTreeMap::new();
        let none = std::collections::BTreeMap::new();
        super::claim(&mut providers, &none, &npu, &a).unwrap();
        // Listed before `b` registered, as `a`'s.
        let listed = std::collections::BTreeMap::from([(npu[0].clone(), 1)]);
        let refusal = super::claim(&mut providers, &listed, &npu, &b).unwrap_err();
        for named in [
            "/opt/a/libnpu_ep.so",
            "/opt/b/libnpu_ep.so",
            "NpuExecutionProvider",
        ] {
            assert!(refusal.contains(named), "{refusal}");
        }
        assert_eq!(providers["NpuExecutionProvider"], a);
        // Another provider is the second library's own.
        super::claim(
            &mut providers,
            &listed,
            &["OtherExecutionProvider".to_string()],
            &b,
        )
        .unwrap();
        assert_eq!(providers["OtherExecutionProvider"], b);
    }

    /// A library bringing more devices of a provider the runtime has built in
    /// is refused, and the provider stays the runtime's.
    #[test]
    fn a_library_bringing_a_built_in_provider_is_refused() {
        let path = std::path::PathBuf::from("/opt/a/libdml_ep.so");
        let dml = ["DmlExecutionProvider".to_string()];
        let mut providers = std::collections::BTreeMap::new();
        let built_in = std::collections::BTreeMap::from([(dml[0].clone(), 1)]);
        let refusal = super::claim(&mut providers, &built_in, &dml, &path).unwrap_err();
        assert!(refusal.contains("/opt/a/libdml_ep.so"), "{refusal}");
        assert!(refusal.contains("already provides"), "{refusal}");
        assert!(providers.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn inventory_spawn_failure_returns_without_the_inventory_timeout() {
        let root = tempfile::tempdir().unwrap();
        let command = std::process::Command::new(root.path().join("missing-nvidia-smi"));
        let busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let permit = InventoryPermit::acquire(busy.clone()).unwrap();
        let start = std::time::Instant::now();
        assert!(
            inventory_with_permit(command, std::time::Duration::from_secs(3), permit).is_none()
        );
        assert!(start.elapsed() < std::time::Duration::from_secs(1));
        assert!(
            InventoryPermit::acquire(busy).is_some(),
            "spawn failure retained inventory admission"
        );
    }

    #[cfg(unix)]
    #[test]
    fn inventory_probe_bounds_stalls_and_output_and_reaps_children() {
        use std::time::Duration;
        let mut success = std::process::Command::new("sh");
        success.args(["-c", "printf '0, GPU-test, pci, name, driver\\n'"]);
        let bytes = bounded_inventory(success, Duration::from_secs(1)).unwrap();
        assert!(cuda_inventory(&bytes).is_some());
        let mut failure = std::process::Command::new("sh");
        failure.args(["-c", "exit 1"]);
        assert!(bounded_inventory(failure, Duration::from_secs(1)).is_none());
        let mut stalled = std::process::Command::new("sh");
        stalled.args(["-c", "exec sleep 30"]);
        let start = std::time::Instant::now();
        assert!(bounded_inventory(stalled, Duration::from_millis(50)).is_none());
        assert!(start.elapsed() < Duration::from_secs(2));
        let mut noisy = std::process::Command::new("sh");
        noisy.args(["-c", "while :; do printf 'too much inventory'; done"]);
        assert!(bounded_inventory(noisy, Duration::from_secs(1)).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn inventory_admission_remains_held_until_cleanup_releases_it() {
        use std::sync::{Arc, atomic::AtomicBool};
        let busy = Arc::new(AtomicBool::new(false));
        let permit = InventoryPermit::acquire(busy.clone()).unwrap();
        let (ready, started) = std::sync::mpsc::channel();
        let (release, wait) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _permit = permit;
            ready.send(()).unwrap();
            wait.recv().unwrap();
        });
        started.recv().unwrap();
        for _ in 0..100 {
            assert!(InventoryPermit::acquire(busy.clone()).is_none());
        }
        release.send(()).unwrap();
        worker.join().unwrap();
        assert!(InventoryPermit::acquire(busy).is_some());
    }

    #[cfg(unix)]
    #[test]
    fn stalled_inventory_child_has_exited_before_admission_is_released() {
        use std::time::Duration;
        let root = tempfile::tempdir().unwrap();
        let pid_file = root.path().join("child.pid");
        let mut command = std::process::Command::new("sh");
        command.args(["-c", "echo $$ > \"$1\"; exec sleep 30", "inventory"]);
        command.arg(&pid_file);
        let busy = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let permit = InventoryPermit::acquire(busy.clone()).unwrap();
        assert!(inventory_with_permit(command, Duration::from_millis(100), permit).is_none());
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let _released = loop {
            if let Some(permit) = InventoryPermit::acquire(busy.clone()) {
                break permit;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "cleanup did not release admission"
            );
            std::thread::yield_now();
        };
        let pid = std::fs::read_to_string(pid_file).unwrap();
        // kill -0 distinguishes an exited/reaped process from a still-live or
        // zombie child; it neither signals nor changes the child.
        assert!(
            !std::process::Command::new("kill")
                .args(["-0", pid.trim()])
                .stderr(std::process::Stdio::null())
                .status()
                .unwrap()
                .success()
        );
        assert!(InventoryPermit::acquire(busy).is_none());
    }

    #[test]
    fn cuda_inventory_requires_device_ids_and_distinguishes_devices() {
        let first = cuda_inventory(b"0, GPU-a, 0000:01:00.0, A100, 580.0\n").unwrap();
        let second = cuda_inventory(b"0, GPU-b, 0000:02:00.0, A100, 580.0\n").unwrap();
        assert_ne!(first, second);
        assert!(cuda_inventory(b"").is_none());
        assert!(cuda_inventory(b"0, N/A, 0000:01:00.0, A100, 580.0").is_none());
        assert!(cuda_inventory(b"NVIDIA-SMI has failed").is_none());
    }
}

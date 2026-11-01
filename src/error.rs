//! Runtime errors.

/// What can go wrong preparing a session or reading a device.
///
/// The variants keep apart what a caller treats differently: a provider that
/// will not register on this machine stays that way for the life of the
/// process, while any other failure may pass.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A provider that cannot be registered here: its library, or one it
    /// needs, does not load, or it finds no device.
    #[error("{0}")]
    Incompatible(String),
    /// A provider whose outputs do not match the CPU's reference outputs.
    #[error("{0}")]
    Numerical(String),
    /// Anything else ONNX Runtime refused.
    #[error("{0}")]
    Runtime(String),
    /// A plugin execution-provider library that cannot be told apart from
    /// one already registered; see `device::accelerators`.
    #[error("{0}")]
    PluginConflict(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// Result alias for runtime operations.
pub type Result<T> = std::result::Result<T, Error>;

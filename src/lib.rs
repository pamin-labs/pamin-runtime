//! Compute for Pamin modules that run models locally.
//!
//! - [`kernels`]: vector math dispatched by instruction set at run time.
//! - [`device`]: which ONNX Runtime execution providers a machine can run, and
//!   what a session was actually given.
//! - [`prepared`]: what keys a prepared, memory-mapped copy of a model to the
//!   runtime and CPU that wrote it.
//!
//! Nothing here knows about a model, a module or a transport.

pub mod device;
mod error;
pub mod kernels;
pub mod prepared;
#[cfg(target_os = "windows")]
mod winml;

pub use error::{Error, Result};

//! Compute for Pamin modules that run models locally.
//!
//! - [`kernels`]: vector math dispatched by instruction set at run time.
//! - [`device`]: which ONNX Runtime execution providers a machine can run, and
//!   what a session was actually given.
//! - [`budget`]: how much of the machine one forward pass may use, and the
//!   session options every model is built with.
//! - [`activity`]: what model loads in this process fetched and prepared.
//! - [`resident`]: handing a model's mapped weights back to the page cache
//!   while it is idle.
//! - [`prepared`]: what keys a prepared, memory-mapped copy of a model to the
//!   runtime and CPU that wrote it.
//!
//! Nothing here knows about a model, a module or a transport.

pub mod activity;
pub mod budget;
pub mod device;
mod error;
pub mod kernels;
pub mod prepared;
pub mod resident;
#[cfg(target_os = "windows")]
mod winml;

pub use error::{Error, Result};

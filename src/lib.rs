//! Compute for Pamin modules that run models locally.
//!
//! - [`kernels`]: vector math dispatched by instruction set at run time.
//! - [`prepared`]: what keys a prepared, memory-mapped copy of a model to the
//!   runtime and CPU that wrote it.
//!
//! Nothing here knows about a model, a module or a transport.

mod error;
pub mod kernels;
pub mod prepared;

pub use error::{Error, Result};

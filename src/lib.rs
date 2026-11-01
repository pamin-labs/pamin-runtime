//! Compute for Pamin modules that run models locally.
//!
//! - [`kernels`]: vector math dispatched by instruction set at run time.
//!
//! Nothing here knows about a model, a module or a transport.

mod error;
pub mod kernels;

pub use error::{Error, Result};

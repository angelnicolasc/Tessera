//! Strongly-typed errors raised by the Tessera core. Public callers should match on these
//! variants rather than relying on `anyhow::Error` string contents.

use thiserror::Error;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, TesseraError>;

/// Errors produced by the block manager and its collaborators.
#[derive(Debug, Error)]
pub enum TesseraError {
    /// The free pool was exhausted at allocation time.
    #[error("out of MLA blocks (used={used}, total={total})")]
    OutOfBlocks {
        /// Blocks currently held by requests at the time of the failure.
        used: u32,
        /// Total blocks the manager was configured to own.
        total: u32,
    },

    /// A user-supplied [`crate::config::MlaBlockConfig`] failed validation. The message names
    /// the specific field and the constraint that was violated.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A block id was referenced that the manager has no record of. This usually indicates a
    /// double-free or a logic bug in the caller, not a transient condition.
    #[error("unknown block id: {0}")]
    UnknownBlock(u32),

    /// The underlying device backend (CPU mock or CUDA) failed an allocation, copy or readback.
    #[error("device backend error: {0}")]
    Backend(#[source] anyhow::Error),
}

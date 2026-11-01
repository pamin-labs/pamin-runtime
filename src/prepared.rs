//! What keys a prepared copy of a model to the machine that wrote it.
//!
//! A prepared copy is the model's graph written back out by ONNX Runtime after
//! optimization, with every initializer -- the packed ones included -- in an
//! external data file that a session maps rather than copies onto the heap.
//! Writing, finding, holding and collecting copies is the caller's; this
//! module owns what decides whether a copy written here is valid there.
//!
//! A graph optimized with layout transformations is specific to the
//! instruction set it was optimized on, and ONNX Runtime documents that it
//! may need that instruction set to run. And the runtime finds a packed
//! weight on disk by a hash of the packed bytes, so a copy written on a CPU
//! that packs differently -- AMX against AVX-512 VNNI against AVX2 -- loads
//! and scores correctly but finds none of them and packs onto the heap again,
//! silently. So [`key`] covers the runtime build, the architecture and its
//! matrix-relevant [`features`], and the optimization level, and a machine
//! that differs in any of them writes its own copy.

use std::fs::File;
use std::path::Path;

use ort::session::builder::GraphOptimizationLevel;
use sha2::{Digest, Sha256};

/// The optimization level a copy is written at.
///
/// It has to be the level the copy is loaded at, so it is the one
/// [`crate::budget::options`] builds every session at: `Level3`, ONNX
/// Runtime's layout level. Loading runs the optimizers again over the
/// optimized graph. A change of level is exactly what a check that a copy
/// scores bit-identically would have to be re-run for, so the level is part
/// of the key.
pub const LEVEL: GraphOptimizationLevel = GraphOptimizationLevel::Level3;

/// The directory name a copy of the source `described` is kept under, and the
/// full description it was derived from, for the log.
///
/// `described` says what the source is; this adds everything that decides
/// whether a copy written here is valid there: the runtime's build, the
/// architecture and its matrix features, and the optimization level. Hashed
/// because the build string alone is longer than some filesystems allow a
/// name to be. The spelling is part of every existing key: changing it
/// orphans every copy on every machine.
pub fn key(described: &str) -> (String, String) {
    let described = format!(
        "{described}; runtime 1.{} {}; {} {}; {LEVEL:?}",
        ort::MINOR_VERSION,
        ort::info(),
        std::env::consts::ARCH,
        features().join(","),
    );
    let digest = Sha256::digest(described.as_bytes());
    let key = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    (key, described)
}

/// Content identity without keeping the complete model on the heap.
pub fn source_digest(source: &Path) -> std::io::Result<String> {
    let mut digest = Sha256::new();
    std::io::copy(&mut File::open(source)?, &mut digest)?;
    Ok(format!("{:x}", digest.finalize()))
}

/// The CPU features that decide how the matrix kernels pack a weight.
///
/// Named rather than read as raw CPUID, because raw leaves also carry
/// microcode mitigation bits that change with an update and would orphan every
/// copy for nothing. AMX is read from CPUID directly: `std` does not yet
/// detect it on a stable toolchain. That reads what the CPU has, not whether
/// the kernel lets this process use it, which is the one case where two
/// machines could share a key and pack differently -- and the cost of that is
/// the heap packing described above, not a wrong score.
#[cfg(target_arch = "x86_64")]
pub fn features() -> Vec<&'static str> {
    use std::arch::is_x86_feature_detected as has;
    let mut found: Vec<&'static str> = [
        ("avx", has!("avx")),
        ("avx2", has!("avx2")),
        ("fma", has!("fma")),
        ("f16c", has!("f16c")),
        ("avx512f", has!("avx512f")),
        ("avx512bw", has!("avx512bw")),
        ("avx512dq", has!("avx512dq")),
        ("avx512vl", has!("avx512vl")),
        ("avx512vnni", has!("avx512vnni")),
        ("avx512bf16", has!("avx512bf16")),
        ("avx512fp16", has!("avx512fp16")),
        ("avxvnni", has!("avxvnni")),
        ("avxvnniint8", has!("avxvnniint8")),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect();
    if std::arch::x86_64::__cpuid(0).eax >= 7 {
        let extended = std::arch::x86_64::__cpuid_count(7, 0).edx;
        for (bit, name) in [(22, "amx-bf16"), (24, "amx-tile"), (25, "amx-int8")] {
            if extended >> bit & 1 == 1 {
                found.push(name);
            }
        }
    }
    found
}

#[cfg(target_arch = "aarch64")]
pub fn features() -> Vec<&'static str> {
    use std::arch::is_aarch64_feature_detected as has;
    [
        ("neon", has!("neon")),
        ("fp16", has!("fp16")),
        ("dotprod", has!("dotprod")),
        ("i8mm", has!("i8mm")),
        ("bf16", has!("bf16")),
        ("sve", has!("sve")),
        ("sve2", has!("sve2")),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn features() -> Vec<&'static str> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key names the source, the runtime, the CPU and the level, and a
    /// different source is a different key.
    #[test]
    fn a_key_names_what_decides_a_copy() {
        let (key, described) = key("source model.onnx 4 bytes modified 1");
        assert_eq!(key.len(), 32);
        assert!(key.bytes().all(|byte| byte.is_ascii_hexdigit()));
        for part in [
            "source model.onnx 4 bytes modified 1; runtime 1.",
            std::env::consts::ARCH,
            "Level3",
        ] {
            assert!(described.contains(part), "{described}");
        }
        assert_ne!(key, super::key("source model.onnx 5 bytes modified 1").0);
    }

    #[test]
    fn a_source_digest_is_its_sha256() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), b"abc").unwrap();
        assert_eq!(
            source_digest(file.path()).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

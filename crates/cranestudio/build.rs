//! `--features sycl` only: bake the out-of-tree SYCL kernel libraries' directories
//! into the binary as an rpath so it runs straight from `target/release/`.
//!
//! Crane's build scripts emit these rpaths with `rustc-link-arg`, which does not
//! reach a *dependent* binary's link — and `cranestudio` is the binary, not
//! `crane-serve` — so without this the binary fails at startup with
//! `libcrane_gdn_sycl.so: cannot open shared object file`. This mirrors Crane's
//! `build_sycl_rpath.rs` (crane-serve's build script) at the pinned revision.
//! The rpath points into the build tree, so such a binary is not relocatable.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    if std::env::var_os("CARGO_FEATURE_SYCL").is_some() {
        emit_sycl_rpath();
    }
}

fn emit_sycl_rpath() {
    let Ok(out_dir) = std::env::var("OUT_DIR") else {
        return;
    };
    // .../build/<pkg>-<hash>/out -> .../build, where the dependencies' build
    // scripts (already run by now) left their `out` directories.
    let Some(build_root) = Path::new(&out_dir).parent().and_then(Path::parent) else {
        return;
    };
    let Ok(entries) = std::fs::read_dir(build_root) else {
        return;
    };

    // Newest first, so a stale OUT_DIR from an earlier build never shadows the
    // library this build just produced.
    let mut found: Vec<(SystemTime, PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let dir = entry.path().join("out");
        for lib in ["libcandle_sycl.so", "libcrane_gdn_sycl.so"] {
            if let Ok(meta) = std::fs::metadata(dir.join(lib))
                && !found.iter().any(|(_, p)| *p == dir)
            {
                found.push((meta.modified().unwrap_or(UNIX_EPOCH), dir.clone()));
            }
        }
    }
    found.sort_by_key(|(when, _)| std::cmp::Reverse(*when));

    // Old-style DT_RPATH, not DT_RUNPATH: RUNPATH applies only to the object that
    // carries it, while RPATH is also searched for the dependencies of the
    // `dlopen`ed SYCL adapters, which need it.
    println!("cargo::rustc-link-arg-bins=-Wl,--disable-new-dtags");
    for (_, dir) in found {
        println!("cargo::rustc-link-arg-bins=-Wl,-rpath,{}", dir.display());
        println!("cargo::rerun-if-changed={}", dir.display());
    }
    for dir in oneapi_runtime_dirs() {
        println!("cargo::rustc-link-arg-bins=-Wl,-rpath,{dir}");
    }
}

fn oneapi_runtime_dirs() -> Vec<String> {
    let root = std::env::var("ONEAPI_ROOT").ok().or_else(|| {
        Path::new("/opt/intel/oneapi")
            .exists()
            .then(|| "/opt/intel/oneapi".into())
    });
    let Some(root) = root else {
        return Vec::new();
    };
    [
        "compiler/latest/lib",
        "mkl/latest/lib",
        "mkl/latest/lib/intel64",
        "tcm/latest/lib",
        "umf/latest/lib",
        "tbb/latest/lib/intel64/gcc4.8",
    ]
    .iter()
    .map(|sub| format!("{root}/{sub}"))
    .filter(|p| Path::new(p).exists())
    .collect()
}

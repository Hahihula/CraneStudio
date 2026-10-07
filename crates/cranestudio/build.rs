//! `--features sycl` only: make the binary find the SYCL kernel libraries that
//! Crane's build scripts produce (`libcrane_gdn_sycl.so`, `libcandle_sycl.so`).
//!
//! Crane builds them into its own `OUT_DIR`s, and a dependency's `rustc-link-arg`
//! does not reach this binary's link, so without help the binary dies at startup
//! with `libcrane_gdn_sycl.so: cannot open shared object file`. Scanning the build
//! tree for them from here does not work either: Cargo does not order this build
//! script after `crane-core`'s, so on a clean build they don't exist yet.
//!
//! Instead the binary looks beside itself, in `sycl-libs/` (as `cuda-libs/` does
//! for CUDA), and whoever packages the build copies the libraries there — see
//! `build:linux-sycl` in `.gitlab-ci.yml`. That also keeps the result relocatable.

use std::path::Path;

fn main() {
    if std::env::var_os("CARGO_FEATURE_SYCL").is_none() {
        return;
    }
    // Old-style DT_RPATH, not DT_RUNPATH: RUNPATH applies only to the object that
    // carries it, while RPATH is also searched for the dependencies of the
    // `dlopen`ed SYCL adapters, which need it.
    println!("cargo::rustc-link-arg-bins=-Wl,--disable-new-dtags");
    println!("cargo::rustc-link-arg-bins=-Wl,-rpath,$ORIGIN/sycl-libs");
    for dir in oneapi_runtime_dirs() {
        println!("cargo::rustc-link-arg-bins=-Wl,-rpath,{dir}");
    }
}

/// oneAPI runtime directories on the build machine, as Crane's own binaries bake in.
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

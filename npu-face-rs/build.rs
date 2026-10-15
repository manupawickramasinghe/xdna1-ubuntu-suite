fn main() {
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"]).expect("onnx.proto");

    let xrt = std::env::var("XRT_ROOT").unwrap_or_else(|_| "/opt/xilinx/xrt".into());
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file("shim/xmm_shim.cpp")
        .include("shim")
        .include(format!("{xrt}/include"))
        .flag_if_supported("-Wno-unused-parameter")
        .compile("xmm_shim");

    println!("cargo:rustc-link-search=native={xrt}/lib");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{xrt}/lib");
    println!("cargo:rerun-if-changed=shim/xmm_shim.cpp");
    println!("cargo:rerun-if-changed=shim/xdna_runner.hpp");
    println!("cargo:rerun-if-changed=proto/onnx.proto");
}

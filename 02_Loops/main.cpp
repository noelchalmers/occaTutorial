#include <iostream>
#include <random>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Loops in OKL kernels"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device mode. Can be Serial, OpenMP, OpenCL, CUDA, HIP, or SYCL (default: Serial)")
      .withArg()
      .withDefaultValue("Serial")
    )
    .addOption(
      occa::cli::option('M', "dimM",
                        "Matrix Rows")
      .withArg()
      .withDefaultValue("1000")
    )
    .addOption(
      occa::cli::option('N', "dimN",
                        "Matrix Columns")
      .withArg()
      .withDefaultValue("1000")
    )
    .addOption(
      occa::cli::option('K', "dimK",
                        "Matrix contraction dimension")
      .withArg()
      .withDefaultValue("1000")
    );

  occa::json args = parser.parseArgs(argc, argv);
  return args;
}

int main(int argc, const char **argv) {

  // Parse arguments to json
  occa::json args = parseArgs(argc, argv);

  std::string mode;
  if (args["options/device"]=="Serial") {
    mode = "{mode: 'Serial'}";
  } else if (args["options/device"]=="OpenMP") {
    mode = "{mode: 'OpenMP'}";
  } else if (args["options/device"]=="OpenCL") {
    mode = "{mode: 'OpenCL', platform_id: 0, device_id: 0}";
  } else if (args["options/device"]=="CUDA") {
    mode = "{mode: 'CUDA', device_id: 0}";
  } else if (args["options/device"]=="HIP") {
    mode = "{mode: 'HIP', device_id: 0}";
  } else if (args["options/device"]=="SYCL") {
    mode = "{mode: 'SYCL', device_id: 0}";
  }
  // Create & setup occa::device
  occa::device device(mode);

  // Create some matrices in host memory
  const int M = std::stoi(args["options/dimM"]);
  const int N = std::stoi(args["options/dimN"]);
  const int K = std::stoi(args["options/dimK"]);
  const int LDA = M;
  const int LDB = K;
  const int LDC = M;
  std::vector<float> A(K * LDA);
  std::vector<float> B(N * LDB);
  std::vector<float> C(N * LDC);

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  for (int k = 0; k < K; ++k) {
    for (int m = 0; m < M; ++m) {
      A[m + k * LDA]  = dist(gen);
    }
  }
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < K; ++k) {
      B[k + n * LDB]  = dist(gen);
    }
  }

  /*
  Data can be compied into device memory on creation
  The copy to device is synchronous, w.r.t. the host
  */
  occa::memory o_A = device.malloc<float>(K * LDA, A.data());
  occa::memory o_B = device.malloc<float>(N * LDB, B.data());

  occa::memory o_C = device.malloc<float>(N * LDC);

  /*
  Detailed options for kernel compilation can be controlled by
  passing an occa::json list of options.

  Things like compilation flags and compile-time definitions
  can be specified this way
  */
  occa::json properties;
  properties["defines"].asObject();

  // Compile-time constants can be added as defines
  properties["defines/N_TILE_SIZE"] = 16;
  properties["defines/M_TILE_SIZE"] = 16;

  // Specify compiler flags based on the device backend
  if(device.mode()=="Serial") {
    properties["compiler_flags"] += "-O3 ";
    properties["compiler_flags"] += "-g "; //debugging
  } else if(device.mode()=="CUDA"){ // add backend compiler optimization for CUDA
    properties["compiler_flags"] += " -O3 ";
    properties["compiler_flags"] += "--prec-div=false ";
    properties["compiler_flags"] += "--prec-sqrt=false ";
    properties["compiler_flags"] += "--use_fast_math ";
    properties["compiler_flags"] += "--fmad=true ";
  } else if(device.mode()=="OpenCL"){ // add backend compiler optimization for OPENCL
    properties["compiler_flags"] += " -cl-std=CL2.0 ";
    properties["compiler_flags"] += " -cl-strict-aliasing ";
    properties["compiler_flags"] += " -cl-mad-enable ";
    properties["compiler_flags"] += " -cl-no-signed-zeros ";
    properties["compiler_flags"] += " -cl-unsafe-math-optimizations ";
    properties["compiler_flags"] += " -cl-fast-relaxed-math ";
  } else if(device.mode()=="HIP"){ // add backend compiler optimization for HIP
    properties["compiler_flags"] += " -O3 ";
    properties["compiler_flags"] += " -funsafe-math-optimizations ";
    properties["compiler_flags"] += " -ffast-math ";
  }

  occa::kernel matrixMultiply = device.buildKernel(
                                    OCCA_BUILD_DIR "/02_Loops/matrixMultiply.okl",
                                    "matrixMultiply",
                                    properties
                                   );

  // Launch kernel
  matrixMultiply(M, N, K,
                 o_A, LDA,
                 o_B, LDB,
                 o_C, LDC);

  /*Compute reference matrix product*/
  std::vector<float> Cref(N * LDC);
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      float c = 0.0;
      for (int k = 0; k < K; ++k) {
        c = A[m + k * LDA] * B[k + n * LDB];
      }
      Cref[m + n * LDC] = c;
    }
  }

  // Copy result to the host
  o_C.copyTo(C.data());

  // Check correctness
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      if (!occa::areBitwiseEqual(C[m + n * LDC],
                                 Cref[m + n * LDC])) {
        std::cout << "FAILED" << std::endl;
        throw 1;
      }
    }
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}

#include <iostream>
#include <random>

#include <hip/hip_runtime.h>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

// HIP error check
#define HIP_CHECK(command)                                    \
{                                                             \
  hipError_t stat = (command);                                \
  if(stat != hipSuccess)                                      \
  {                                                           \
    std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
    " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
    exit(-1);                                                 \
  }                                                           \
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Using inline okl kernels"
    )
    .addOption(
      occa::cli::option('n', "entries",
                        "Vector length")
      .withArg()
      .withDefaultValue("1000")
    );

  occa::json args = parser.parseArgs(argc, argv);
  return args;
}

int main(int argc, const char **argv) {

  // Parse arguments to json
  occa::json args = parseArgs(argc, argv);

  // Create & setup occa::device
  std::string mode("{mode: 'HIP', device_id: 0}");
  occa::device device(mode);

  if (device.mode()!="HIP") {
    std::cout << "Example requires HIP backend" << std::endl;
    exit(-1);
  }

  // Create some vectors in host memory
  const int entries = std::stoi(args["options/entries"]);
  std::vector<float> a(entries);
  std::vector<float> b(entries);
  std::vector<float> ab(entries);

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  for (int i = 0; i < entries; ++i) {
    a[i]  = dist(gen);
    b[i]  = dist(gen);
  }

  // Make some device arrays with native HIP API
  float *h_a, *h_b, *h_ab;
  HIP_CHECK(hipMalloc(&h_a,  entries * sizeof(float)));
  HIP_CHECK(hipMalloc(&h_b,  entries * sizeof(float)));
  HIP_CHECK(hipMalloc(&h_ab, entries * sizeof(float)));

  /*
  Wrap native allocations with occa::memory

  OCCA will not automatically free wrapped memory
  */
  occa::memory o_a  = device.wrapMemory<float>(h_a,  entries);
  occa::memory o_b  = device.wrapMemory<float>(h_b,  entries);
  occa::memory o_ab = device.wrapMemory<float>(h_ab, entries);

  // Send data to device
  o_a.copyFrom(a.data());
  o_b.copyFrom(b.data());

  /*
  Tell OCCA to treat the source file as native to the backend,
  not a OKL file that should be translated
  */
  occa::json properties;
  properties["okl/enabled"] = false;

  occa::kernel addVectors = device.buildKernel(
                                    OCCA_BUILD_DIR "/06_Native_Interop/addVectors.cpp",
                                    "addVectors",
                                    properties
                                   );

  /*
  Since this is a native HIP kernel, we must inform OCCA
  of the launch dimensions
  */
  const int blockSize = 256;
  occa::dim grid((entries+blockSize-1)/blockSize);
  occa::dim block(blockSize);
  addVectors.setRunDims(grid, block);

  addVectors(entries, o_a, o_b, o_ab);

  // Copy result to the host
  o_ab.copyTo(ab.data());

  // Check correctness
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(ab[i], a[i] + b[i])) {
      std::cout << "FAILED" << std::endl;
      throw 1;
    }
  }
  std::cout << "PASSED!" << std::endl;
  return 0;

  HIP_CHECK(hipFree(h_a));
  HIP_CHECK(hipFree(h_b));
  HIP_CHECK(hipFree(h_ab));
}

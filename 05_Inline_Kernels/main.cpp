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
      "Using inline okl kernels"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device mode. Can be Serial, OpenMP, OpenCL, CUDA, HIP, or SYCL (default: Serial)")
      .withArg()
      .withDefaultValue("Serial")
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

  occa::memory o_a = device.malloc<float>(entries, a.data());
  occa::memory o_b = device.malloc<float>(entries, b.data());
  occa::memory o_ab = device.malloc<float>(entries);

  /*
  To make an inline JIT kernel, we first need to describe the arguments to 
  be passed. OCCA will determine the occa::device being used based on the 
  occa::memory buffers passed. 
  */
  occa::scope scope(
    {/*Arguments*/
      {"entries", entries},
      {"a", o_a},
      {"b", o_b},
      {"ab", o_ab}
    }, 
    {/*Properties for kernel compilation*/
      // Define TILE_SIZE at compile-time
      {"defines/TILE_SIZE", 256}
    });

  /*
  The kernel source itself is wrapped in the OCCA_JIT define. The kernel is
  immedaitely JIT compiled and launched, passing the arguments described in the 
  occa::scope. 
  */
  OCCA_JIT(scope, (
    for (int i = 0; i < entries; ++i; @tile(TILE_SIZE, @outer, @inner)) {
      ab[i] = a[i] + b[i];
    }
  ));

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
}

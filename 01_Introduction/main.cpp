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
      "Introduction to OCCA"
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

  /*
  OCCA's core functionality is tied to an occa::device
  The device abstracts the seperate backend APIs into a single
  interface. Memory, streams, synchronization, and JIT compilation
  of kernels are all acessed through the device.
  */
  occa::device device;

  /*
  The device is uninitialized at creation. At runtime, the user
  sets up the device by selecting a backend. If the selected backend
  is not available (i.e. not found at OCCA's build time or no compatible
  device is found) the backend falls back to 'Serial'
  */
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

  device.setup(mode);

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

  /*
  Device memory is wrapped in an occa::memory object.
  Device memory allocation is done via an occa::device method
  and will allocate using the backend for that device.
  occa::memory wrappers use a reference counting mechanism
  so manually free'ing is not required.

  Copying an occa::memory is *shallow* copy. Both occa::memory
  objects will point to the same physical device memory internally.
  If a full copy is desired, an occa::memory::clone() method can be used
  */
  occa::memory o_a = device.malloc<float>(entries);
  occa::memory o_b = device.malloc<float>(entries);
  occa::memory o_ab = device.malloc<float>(entries);

  /*
  Data can be copied between occa::memory objects and regular host
  pointers with the copyTo/copyFrom methods. These copies are synchronous
  w.r.t. the host, meaning all data is safe to use after the copy returns.
  */
  o_a.copyFrom(a.data());
  o_b.copyFrom(b.data());

  /*
  Device kernels are abstracted into occa::kernel functors. They are
  built at runtime for the selected backend in the occa::device. By default
  the source file is assumed to be written in the OCCA Kernel Language (OKL).
  This language is a slightly-decorated C, which OCCA will translate into
  compatible kernel source for the device backend. The translated kernel will
  then be compiled, cached, and dynamically loaded.

  Some useful info from the kernel construction can be printed to console when
  OCCA_VERBOSE=1.
  */
  occa::kernel addVectors = device.buildKernel(
                                    OCCA_BUILD_DIR "/01_Introduction/addVectors.okl",
                                    "addVectors"
                                   );

  /*
  Once built, kernels are launched as a functor.

  When allowed by the backend, kernel launches are *asynchronous* w.r.t. the host.
  Notably, kernels will be synchronous in Serial and OpenMP modes, but async
  otherwise.
  */
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
}

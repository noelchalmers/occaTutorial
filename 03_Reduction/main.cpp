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
      "Reduction OKL kernel"
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
      .withDefaultValue("100000")
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

  const int maxBlocks = 512;
  const int blockSize = 256;

  // Create some matrices in host memory
  const int entries = std::stoi(args["options/entries"]);
  std::vector<double> x(entries);

  std::mt19937 gen(0);
  std::uniform_real_distribution<double> dist(-1, 1);

  for (int i = 0; i < entries; ++i) {
    x[i] = dist(gen);
  }


  occa::memory o_x = device.malloc<double>(entries, x.data());

  // scratch space for reduction
  occa::memory o_scratch = device.malloc<double>(maxBlocks);

  occa::memory o_sum = device.malloc<double>(1);

  /*
  Pinned host buffer for reduction
  Using pinned host memory for Host<->Device transfers can significantly
  increase bandwidth of transfer. Pinned host memory is also required
  to do asynchronous transfers
  */
  occa::memory h_sum = device.malloc<double>(1, occa::json("host", true));

  occa::json properties;
  properties["defines"].asObject();

  properties["defines/MAX_BLOCKS"] = maxBlocks;
  properties["defines/BLOCK_SIZE"] = blockSize;


  occa::kernel sumKernel = device.buildKernel(
                                    OCCA_BUILD_DIR "/03_Reduction/sum.okl",
                                    "sum",
                                    properties
                                   );

  // Queue kernel
  const int Nblocks = (entries < maxBlocks) ? entries : maxBlocks;
  sumKernel(entries, Nblocks, o_x, o_scratch, o_sum);

  // Queue copy of result back to host
  h_sum.copyFrom(o_sum,
                 /*Nbytes*/sizeof(double),
                 /*Offset*/0,
                 /*Async*/ occa::json("async", true));

  /*Compute reference sum*/
  double sumRef = 0.0;
  for (int i = 0; i < entries; ++i) {
    sumRef += x[i];
  }

  /*
  Since both the kernel and the copy were queue
  asynchronously, we have to wait for the operations
  to complete.
  */
  device.finish();

  // Get the sum out of the host-pinned location
  const double sum = *(static_cast<double*>(h_sum.ptr()));

  // Check correctness
  if (std::abs(sum-sumRef) > 1.0E-5) {
    std::cout << "FAILED" << std::endl;
    throw 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}

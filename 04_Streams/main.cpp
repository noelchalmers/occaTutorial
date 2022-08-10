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
      "Managing Streams"
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

  // Create some vectors in host memory
  const int entries = std::stoi(args["options/entries"]);
  std::vector<float> x(entries);
  std::vector<float> y(entries);

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1, 1);

  for (int i = 0; i < entries; ++i) {
    x[i] = dist(gen);
    y[i] = dist(gen);
  }


  occa::memory o_x = device.malloc<float>(entries, x.data());
  occa::memory o_y = device.malloc<float>(entries, y.data());

  // Device buffers for output vectors 
  occa::memory o_z = device.malloc<float>(entries);
  occa::memory o_p = device.malloc<float>(entries);

  /*
  Pinned host buffers for host copies
  Using pinned host memory for Host<->Device transfers can significantly
  increase bandwidth of transfer. Pinned host memory is also required
  to do asynchronous transfers
  */
  occa::memory h_z = device.malloc<float>(entries, occa::json("host", true));
  occa::memory h_p = device.malloc<float>(entries, occa::json("host", true));

  // Build kernels 
  occa::kernel addVectors  = device.buildKernel(
                                    OCCA_BUILD_DIR "/04_Streams/kernels.okl",
                                    "addVectors");
  occa::kernel multVectors = device.buildKernel(
                                    OCCA_BUILD_DIR "/04_Streams/kernels.okl",
                                    "multVectors");


  /*
  Queues of device work can be managed through occa::streams. 
  Streams are ordering mechanisms for queueing work to a device. 
  Commands queued into a stream are guarenteed to complete in order.
  Commands queued into different streams have no guarenteed ordering
  w.r.t. one another, and may or may not overlap in execution.
  
  For GPU modes such as CUDA/HIP, occa::streams wrap an underlying
  cudaStream_t or hipStream_t, with the usual stream ordering semantics.
  
  For Serial and OpenMP modes, kernel calls and memcpies are all 
  synchronous, and no overlap is possible. Use of multiple streams,
  therefore, has no effect for these modes.
  */

  /*
  Upon setup, each occa::device will create an underlying occa::stream.
  For CUDA/HIP modes, this stream is non-NULL.

  The occa::device tracks a currently selected stream in which to 
  queue all work. The currently selected stream can be accessed directly
  */
  occa::stream stream1 = device.getStream();

  // New streams can be created on a device
  occa::stream stream2 = device.createStream();


  /*
  Start some work by setting the device to use stream2, queuing a 
  kernel, and copying the result back, all async.
  */
  device.setStream(stream2);

  addVectors(entries, o_x, o_y, o_z);
  
  h_z.copyFrom(o_z,
               /*Nbytes*/entries*sizeof(float),
               /*Offset*/0,
               /*Async*/ occa::json("async", true));

  // Now switch to stream1 and queue more work
  device.setStream(stream1);

  multVectors(entries, o_x, o_y, o_p);
  
  h_p.copyFrom(o_p,
               /*Nbytes*/entries*sizeof(float),
               /*Offset*/0,
               /*Async*/ occa::json("async", true));  

  /*
  To use the results, we must wait for the work to finish

  Note that device::finish() flushes only the device's current stream,
  not the entire device. I.e. in CUDA/HIP mode, device::finish()
  wraps cudaStreamSynchronize and hipStreamSynchronize, respectively
  */
  device.setStream(stream2);
  device.finish();

  // h_z is now safe to use
  float* z = static_cast<float*>(h_z.ptr());

  // Check correctness
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(z[i], x[i] + y[i])) {
      std::cout << "FAILED" << std::endl;
      throw 1;
    }
  }

  // Now wait for stream1
  device.setStream(stream1);
  device.finish();

  // h_p is now safe to use
  float* p = static_cast<float*>(h_p.ptr());

  // Check correctness
  for (int i = 0; i < entries; ++i) {
    if (!occa::areBitwiseEqual(p[i], x[i] * y[i])) {
      std::cout << "FAILED" << std::endl;
      throw 1;
    }
  }  

  std::cout << "PASSED!" << std::endl;
  return 0;
}

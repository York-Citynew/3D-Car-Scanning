# 3D Car Reconstruction with Instant-NGP

This project provides two pipeline implementations for **3D car reconstruction using NVIDIA Instant-NGP**, designed for different camera configurations.

## Pipeline Variants

| Files                                  | Capture Setup        | Description                                                             |
| -------------------------------------- | -------------------- | ----------------------------------------------------------------------- |
| `pipeline.py` and `pipeline.bat`       | Single moving camera | Processes a video captured by a single camera moving around the object. |
| `ourPipeline.py` and `ourPipeline.bat` | Multiple cameras     | Designed for a multi-camera capture setup.                              |

Each Python script has a corresponding Windows batch file (`.bat`) to facilitate execution.

## Getting Started

### Option 1: Using a Prebuilt Instant-NGP Package

For a quick setup, download the appropriate Windows release from the official [NVIDIA Instant-NGP repository](https://github.com/NVlabs/instant-ngp). Select a build compatible with your NVIDIA GPU.

1. **Download and extract Instant-NGP.** Follow the installation instructions in the official repository.
2. **Copy the Python scripts** into the `scripts` directory of the extracted Instant-NGP package.
3. **Copy the batch files** into the root directory of the package, alongside `instant-ngp.exe`.
4. **Run the appropriate batch file** for your camera configuration.

The resulting directory structure should resemble the following:

```text
instant-ngp/
├── instant-ngp.exe
├── pipeline.bat
├── ourPipeline.bat
└── scripts/
    ├── pipeline.py
    └── ourPipeline.py
```

Choose the pipeline that matches your capture setup.

**Important:** The standard prebuilt Windows releases may not include the Python bindings required by these scripts. If a script imports `pyngp`, use a source-built Instant-NGP installation with compatible Python bindings, as described below.

### Option 2: Building Instant-NGP from Source

For development, debugging, and further experiments, **building Instant-NGP from source is recommended**. This provides access to the developer Python bindings and allows you to modify the underlying implementation when necessary.

Refer to the official [Instant-NGP build instructions](https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux) for the complete prerequisites and setup procedure.

On Windows, the basic compilation commands are:

```bash
git clone --recursive https://github.com/NVlabs/instant-ngp
cd instant-ngp

cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```

Follow the official instructions to configure the required compiler, CUDA toolkit, Python environment, and other dependencies.

Once you have a compatible Instant-NGP installation, place the Python scripts in its `scripts` directory and the corresponding batch files in its root directory, as described above.

## Notes

* Use `pipeline.py` and `pipeline.bat` for the single-moving-camera workflow.
* Use `ourPipeline.py` and `ourPipeline.bat` for the multi-camera workflow.
* Ensure that the input data, file paths, and configuration settings match your capture setup.
* If you encounter a `ModuleNotFoundError` for `pyngp`, consult the official build instructions and verify that the Python bindings are available to the Python environment used to run the pipeline.

## Credits

This project uses [NVIDIA Instant-NGP](https://github.com/NVlabs/instant-ngp), an implementation of neural graphics primitives for fast neural rendering and related applications.

For details about the original method, see:

> Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. *Instant Neural Graphics Primitives with a Multiresolution Hash Encoding.* ACM Transactions on Graphics, 2022.

* [Official repository](https://github.com/NVlabs/instant-ngp)
* [Research paper](https://doi.org/10.1145/3528223.3530127)

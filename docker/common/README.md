# Shared wheel delivery

`Dockerfile` has four stages. `wheel` verifies and exports the selected artifact; `runtime` installs it in the approved PyTorch base; `development` adds public headers and the test application; `wheelhouse` exports only the wheel and its checksum file from a scratch image.

`wheel.py` is the single installer used by every recipe. It accepts an exact filename and SHA256, verifies the copied bytes, rejects undeclared wheelhouse contents and runs pip with `--no-deps`. It checks that the intended framework distribution exists in the base. The later qualification profile verifies that framework's actual imported version, revision and execution behavior.

Development composition additionally requires CMake 3.21 or newer, C++ and HIP compilers, and Ninja or Make in the approved base. These requirements are checked during the build. Framework execution requirements still depend on the selected wheel's prebuilt kernels.

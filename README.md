# nc-rd-facemask
Facemask C++

## Install OpenCV

- On macOS
+ Dlib: https://medium.com/beesightsoft/install-dlib-on-macos-946dc53f7d7e
+ OpenCV full: https://medium.com/beesightsoft/macos-mojave-10-14-3-setup-environment-for-opencv4-0-1-c-develop-fcae955d6b33

- On Windows
+ Dlib: https://medium.com/beesightsoft/build-dlib-on-windows-534209e8340a
+ OpenCV full: https://medium.com/beesightsoft/build-opencv-opencv-contrib-on-windows-2e3b1ca96955
+ Or download prebuilt package: https://drive.google.com/open?id=1GA9u8lUlid5HTMYu70qUucKX5m3AgdkO (download release version, extract and put opencv and dlib to `D:/_Program Files/`)

- Config on CLion: https://medium.com/beesightsoft/config-cmake-opencv-for-clion-a1ee72c03f4

# Build
```
mkdir build
cd build

# For windows 64
cmake -DCMAKE_GENERATOR_PLATFORM=x64 -DCMAKE_CL_64=1 -Ax64 ..
# Other
cmake ..

cmake --build . --config Release
```

# For CLion
- Build configuration -> Edit -> Select CMake Application target -> Working directory point to current project path.
- Note for windows64:
Remember setting arch support is amd64 (File -> Settings -> Build, Execution, Deployment -> Toolchains -> Visual Studio, Architecture: amd64)
- Create new Release profile (CLion -> Preferences... -> Build, Execution, Deployment -> CMake, Click + -> New Release profile, move to on top as default)
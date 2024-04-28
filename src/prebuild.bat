@echo off
setlocal enabledelayedexpansion

:: Define the base directory to start searching from
set BASE_DIR=C:\

:: Find the OpenBLAS library directory
for /f "delims=" %%a in ('dir %BASE_DIR% /s /b ^| findstr "libopenblas.*"') do (
    set LIB_DIR=%%~dpa
    goto :found_lib
)
echo Error: Could not find OpenBLAS library directory
exit /b 1
:found_lib

:: Find the OpenBLAS include directory
for /f "delims=" %%a in ('dir %BASE_DIR% /s /b /ad ^| findstr /i "openblas"') do (
    set INCLUDE_DIR=%%a
    goto :found_include
)
echo Error: Could not find OpenBLAS include directory
exit /b 1
:found_include

:: Set the install prefix (customize this as needed)
set INSTALL_PREFIX=%BASE_DIR%

:: Define the version (customize this as needed)
set VERSION=0.3.21

:: Create the .pc file from the template
set "input_file=openblas.pc.in"
set "output_file=openblas.pc"

if exist %input_file% (
    del %output_file%
    for /f "delims=" %%a in (%input_file%) do (
        set "line=%%a"
        set "line=!line:@CMAKE_INSTALL_PREFIX@=%INSTALL_PREFIX%!"
        set "line=!line:@OPENBLAS_LIBRARY_DIR@=%LIB_DIR%!"
        set "line=!line:@OPENBLAS_INCLUDE_DIR@=%INCLUDE_DIR%!"
        set "line=!line:0.3.21=%VERSION%!"
        echo !line! >> %output_file%
    )
) else (
    echo Error: Template file '%input_file%' not found
    exit /b 1
)

@echo off

setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================================
REM OUR MULTI-CAMERA VEHICLE RECONSTRUCTION PIPELINE
REM ============================================================================

cd /d D:\INGP

set "ROOT=D:\INGP"

set "SCRIPT=%ROOT%\scripts\ourPipeline.py"

set "TESTDATA=%ROOT%\testdata"

set "PYTHON=python"

REM ============================================================================
REM BASIC CHECKS
REM ============================================================================

if not exist "%SCRIPT%" (
    echo.
    echo ERROR: Python pipeline not found:
    echo %SCRIPT%
    echo.
    pause
    exit /b 1
)

if not exist "%TESTDATA%\." (
    echo.
    echo ERROR: Testdata directory not found:
    echo %TESTDATA%
    echo.
    pause
    exit /b 1
)

if not exist "%ROOT%\instant-ngp.exe" (
    echo.
    echo ERROR: instant-ngp.exe not found:
    echo %ROOT%\instant-ngp.exe
    echo.
    pause
    exit /b 1
)

REM ============================================================================
REM COLMAP FROM PATH
REM ============================================================================

set "COLMAP="

for /f "delims=" %%I in ('where colmap 2^>nul') do (
    if not defined COLMAP set "COLMAP=%%I"
)

if not defined COLMAP (
    echo.

    echo ERROR: COLMAP was not found on PATH.

    echo.

    echo Run this command manually to verify:

    echo   where colmap

    echo.

    pause

    exit /b 1
)

if not exist "%COLMAP%" (
    echo.

    echo ERROR: COLMAP was found on PATH but the file does not exist:

    echo %COLMAP%

    echo.

    pause

    exit /b 1
)

echo.

echo COLMAP:

echo   %COLMAP%

REM ============================================================================
REM PROJECT SELECTION
REM ============================================================================

:ASK_PROJECT

echo.
echo ============================================================================
echo                          AVAILABLE PROJECTS
echo ============================================================================
echo.

set /a COUNT=0

for /d %%D in ("%TESTDATA%\*") do (
    set /a COUNT+=1
    set "PROJECT_!COUNT!=%%~nxD"
    echo   !COUNT! = %%~nxD
)

echo.

if %COUNT%==0 (
    echo ERROR: No project folders found in:
    echo %TESTDATA%
    echo.
    pause
    exit /b 1
)

echo Enter the project number or exact folder name.
echo.

set "PROJECT_INPUT="

set /p "PROJECT_INPUT=Project: "

if not defined PROJECT_INPUT (
    echo.
    echo ERROR: Project selection is required.
    goto ASK_PROJECT
)

echo(%PROJECT_INPUT%| findstr /r /x /c:"[1-9][0-9]*" >nul

if not errorlevel 1 (
    if %PROJECT_INPUT% LEQ %COUNT% (
        set "PROJECT_NAME=!PROJECT_%PROJECT_INPUT%!"
        goto PROJECT_SELECTED
    )

    echo.
    echo ERROR: Invalid project number.
    goto ASK_PROJECT
)

if exist "%TESTDATA%\%PROJECT_INPUT%\." (
    set "PROJECT_NAME=%PROJECT_INPUT%"
    goto PROJECT_SELECTED
)

echo.
echo ERROR: Project folder not found:
echo %PROJECT_INPUT%

goto ASK_PROJECT


:PROJECT_SELECTED

echo.
echo Selected project:
echo   %PROJECT_NAME%

REM ============================================================================
REM SAM 3 MODEL PATH
REM ============================================================================

:ASK_SAM3

echo.
echo ============================================================================
echo                              SAM 3 MODEL
echo ============================================================================
echo.

set "SAM3_INPUT="

set /p "SAM3_INPUT=Path to SAM 3 model [default: D:\models\Segmentation\sam3.pt]: "

if not defined SAM3_INPUT (
    set "SAM3_INPUT=D:\models\Segmentation\sam3.pt"
)

if not exist "%SAM3_INPUT%" (
    echo.
    echo ERROR: SAM 3 model not found:
    echo %SAM3_INPUT%
    echo.
    goto ASK_SAM3
)

REM ============================================================================
REM FRAME COUNT
REM
REM Empty input = approximately 2 FPS.
REM ============================================================================

:ASK_FRAMES

echo.
echo ============================================================================
echo                             FRAME COUNT
echo ============================================================================
echo.

echo Enter the number of frames to extract from EACH camera video.
echo Leave empty to use approximately 2 FPS.
echo.

set "FRAME_INPUT="

set /p "FRAME_INPUT=Frame count [default: 2 FPS]: "

if not defined FRAME_INPUT (
    set "FRAME_ARG="
    goto ASK_PROMPT
)

echo(%FRAME_INPUT%| findstr /r /x /c:"[1-9][0-9]*" >nul

if errorlevel 1 (
    echo.
    echo ERROR: Frame count must be a positive integer.
    goto ASK_FRAMES
)

set "FRAME_ARG=--frames %FRAME_INPUT%"

REM ============================================================================
REM SAM 3 PROMPT
REM ============================================================================

:ASK_PROMPT

echo.
echo ============================================================================
echo                              SAM 3 PROMPT
echo ============================================================================
echo.

echo Example:
echo   toy truck in green, yellow and black
echo.

set "SAM_PROMPT="

set /p "SAM_PROMPT=SAM 3 prompt: "

if not defined SAM_PROMPT (
    echo.
    echo ERROR: SAM 3 prompt is required.
    goto ASK_PROMPT
)

REM ============================================================================
REM FEATURE EXTRACTOR
REM ============================================================================

:ASK_FEATURE

echo.
echo ============================================================================
echo                          FEATURE EXTRACTOR
echo ============================================================================
echo.

echo   1 = ALIKED-N32
echo   2 = SIFT
echo.

echo Leave empty for ALIKED-N32.
echo.

set "FEATURE_INPUT="

set /p "FEATURE_INPUT=Feature extractor [default: 1]: "

if not defined FEATURE_INPUT (
    set "FEATURE=aliked"
    goto ASK_MATCHER
)

if "%FEATURE_INPUT%"=="1" (
    set "FEATURE=aliked"
    goto ASK_MATCHER
)

if "%FEATURE_INPUT%"=="2" (
    set "FEATURE=sift"
    goto ASK_MATCHER
)

echo.
echo ERROR: Enter 1, 2, or leave empty.

goto ASK_FEATURE

REM ============================================================================
REM FEATURE MATCHER
REM ============================================================================

:ASK_MATCHER

echo.
echo ============================================================================
echo                           FEATURE MATCHER
echo ============================================================================
echo.

echo   1 = LightGlue
echo   2 = Brute-force
echo.

echo Leave empty for LightGlue.
echo.

set "MATCHER_INPUT="

set /p "MATCHER_INPUT=Feature matcher [default: 1]: "

if not defined MATCHER_INPUT (
    set "MATCHER=lightglue"
    goto START
)

if "%MATCHER_INPUT%"=="1" (
    set "MATCHER=lightglue"
    goto START
)

if "%MATCHER_INPUT%"=="2" (
    set "MATCHER=bruteforce"
    goto START
)

echo.
echo ERROR: Enter 1, 2, or leave empty.

goto ASK_MATCHER

REM ============================================================================
REM START
REM ============================================================================

:START

echo.
echo ============================================================================
echo                          STARTING PIPELINE
echo ============================================================================
echo.

echo Project:
echo   %TESTDATA%\%PROJECT_NAME%

echo.

echo COLMAP:
echo   %COLMAP%

echo.

echo SAM 3 model:
echo   %SAM3_INPUT%

echo.

if defined FRAME_INPUT (
    echo Frames:
    echo   %FRAME_INPUT% per camera
) else (
    echo Frames:
    echo   approximately 2 FPS
)

echo.

echo SAM 3 prompt:
echo   %SAM_PROMPT%

echo.

echo Feature extractor:

if "%FEATURE%"=="aliked" (
    echo   ALIKED-N32
) else (
    echo   SIFT
)

echo.

echo Feature matcher:

if "%MATCHER%"=="lightglue" (
    echo   LightGlue
) else (
    echo   Brute-force
)

echo.

REM ============================================================================
REM RUN PYTHON CORE
REM ============================================================================

%PYTHON% "%SCRIPT%" ^
    --project "%PROJECT_NAME%" ^
    --prompt "%SAM_PROMPT%" ^
    --feature "%FEATURE%" ^
    --matcher "%MATCHER%" ^
    --sam3 "%SAM3_INPUT%" ^
    --colmap "%COLMAP%" ^
    %FRAME_ARG%

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo                            PIPELINE FAILED
    echo ============================================================================
    echo.
    pause
    exit /b 1
)

echo.

echo ============================================================================
echo                           PIPELINE FINISHED
echo ============================================================================
echo.

pause

exit /b 0
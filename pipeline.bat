@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d D:\INGP

set "ROOT=D:\INGP"
set "TESTDATA=%ROOT%\testdata"
set "SCRIPT=%ROOT%\scripts\pipeline.py"
set "PYTHON=python"

echo.
echo ============================================================
echo                    INSTANT-NGP PIPELINE
echo ============================================================
echo.

if not exist "%SCRIPT%" (
    echo ERROR: Python script not found:
    echo %SCRIPT%
    pause
    exit /b 1
)

if not exist "%TESTDATA%" (
    echo ERROR: testdata folder not found:
    echo %TESTDATA%
    pause
    exit /b 1
)


:PROJECT_INPUT

echo Available projects:
echo.

for /d %%D in ("%TESTDATA%\*") do (
    echo   %%~nxD
)

echo.
set "PROJECT_NAME="
set /p "PROJECT_NAME=Project folder from testdata: "

if not defined PROJECT_NAME (
    echo.
    echo ERROR: Project folder is required.
    echo.
    goto :PROJECT_INPUT
)

set "PROJECT=%TESTDATA%\%PROJECT_NAME%"

if not exist "%PROJECT%\" (
    echo.
    echo ERROR: Project folder does not exist:
    echo %PROJECT%
    echo.
    goto :PROJECT_INPUT
)


set "VIDEO="
set /a VIDEO_COUNT=0

for %%F in ("%PROJECT%\*.mp4") do (
    set /a VIDEO_COUNT+=1
    set "VIDEO=%%~fF"
)

if !VIDEO_COUNT! EQU 0 (
    echo.
    echo ERROR: No MP4 video found in:
    echo %PROJECT%
    echo.
    goto :PROJECT_INPUT
)

if !VIDEO_COUNT! GTR 1 (
    echo.
    echo ERROR: More than one MP4 video found in:
    echo %PROJECT%
    echo.
    goto :PROJECT_INPUT
)


echo.
echo Video:
echo !VIDEO!
echo.

set "INFO_FILE=%TEMP%\instant_ngp_video_info_%RANDOM%.txt"

"%PYTHON%" "%SCRIPT%" --video-info "!VIDEO!" > "!INFO_FILE!"

if errorlevel 1 (
    echo.
    echo ERROR: Could not read video information.
    del /q "!INFO_FILE!" >nul 2>&1
    pause
    exit /b 1
)

type "!INFO_FILE!"

set "DEFAULT_FRAMES="

for /f "tokens=6" %%A in ('findstr /c:"Default 1 FPS frame count:" "!INFO_FILE!"') do (
    set "DEFAULT_FRAMES=%%A"
)

del /q "!INFO_FILE!" >nul 2>&1

if not defined DEFAULT_FRAMES (
    echo.
    echo ERROR: Could not determine default frame count.
    pause
    exit /b 1
)


:FRAME_INPUT

echo.
set "FRAME_COUNT="
set /p "FRAME_COUNT=Number of frames to extract [default: !DEFAULT_FRAMES!]: "

if not defined FRAME_COUNT (
    set "FRAME_COUNT=!DEFAULT_FRAMES!"
)

set "INVALID_FRAME_COUNT="

for /f "delims=0123456789" %%A in ("!FRAME_COUNT!") do (
    set "INVALID_FRAME_COUNT=1"
)

if defined INVALID_FRAME_COUNT (
    echo.
    echo ERROR: Frame count must be a positive integer.
    goto :FRAME_INPUT
)

if !FRAME_COUNT! LSS 1 (
    echo.
    echo ERROR: Frame count must be at least 1.
    goto :FRAME_INPUT
)

echo.
echo Using !FRAME_COUNT! frames.


:SAM_PROMPT_INPUT

echo.
set "SAM_PROMPT="
set /p "SAM_PROMPT=SAM3 prompt: "

if not defined SAM_PROMPT (
    echo.
    echo ERROR: SAM3 prompt is required.
    echo.
    goto :SAM_PROMPT_INPUT
)


:SAM3_INPUT

echo.
set "SAM3_PATH="
set /p "SAM3_PATH=Path to sam3.pt [default: D:\models\segmentation\sam3.pt]: "

if not defined SAM3_PATH (
    set "SAM3_PATH=D:\models\segmentation\sam3.pt"
)

if not exist "!SAM3_PATH!" (
    echo.
    echo ERROR: sam3.pt was not found:
    echo !SAM3_PATH!
    echo.
    goto :SAM3_INPUT
)


:COLMAP_INPUT

echo.
where COLMAP.bat >nul 2>&1

if errorlevel 1 (
    set "COLMAP_PATH=D:\Program Files\colmap-x64-windows-cuda\COLMAP.bat"
) else (
    set "COLMAP_PATH="
    for /f "delims=" %%A in ('where COLMAP.bat') do (
        if not defined COLMAP_PATH set "COLMAP_PATH=%%A"
    )
)

if not defined COLMAP_PATH (
    set "COLMAP_PATH=D:\Program Files\colmap-x64-windows-cuda\COLMAP.bat"
)

if not exist "!COLMAP_PATH!" (
    echo.
    echo ERROR: COLMAP.bat was not found.
    echo Expected:
    echo !COLMAP_PATH!
    echo.
    echo Make sure the COLMAP folder is in the Windows PATH.
    pause
    exit /b 1
)

echo COLMAP:
echo !COLMAP_PATH!


:PIPELINE_MODE_INPUT

echo.
echo ============================================================
echo Pipeline mode
echo ============================================================
echo 1 = Whole RGBA pipeline
echo 2 = Original RGB for COLMAP + RGBA for Instant-NGP
echo.

set "PIPELINE_MODE="
set /p "PIPELINE_MODE=Pipeline mode [default: 1]: "

if not defined PIPELINE_MODE (
    set "PIPELINE_MODE=1"
)

if "!PIPELINE_MODE!"=="1" goto :PIPELINE_MODE_VALID
if "!PIPELINE_MODE!"=="2" goto :PIPELINE_MODE_VALID

echo.
echo ERROR: Enter 1 or 2.
echo.
goto :PIPELINE_MODE_INPUT


:PIPELINE_MODE_VALID


:FEATURE_INPUT

echo.
echo ============================================================
echo Feature extraction
echo ============================================================
echo 1 = ALIKED-N32
echo 2 = SIFT
echo.

set "FEATURE_TYPE="
set /p "FEATURE_TYPE=Feature extraction [default: 1]: "

if not defined FEATURE_TYPE (
    set "FEATURE_TYPE=1"
)

if "!FEATURE_TYPE!"=="1" goto :FEATURE_VALID
if "!FEATURE_TYPE!"=="2" goto :FEATURE_VALID

echo.
echo ERROR: Enter 1 or 2.
echo.
goto :FEATURE_INPUT


:FEATURE_VALID


:MATCH_INPUT

echo.
echo ============================================================
echo Feature matching
echo ============================================================
echo 1 = LightGlue
echo 2 = BruteForce
echo.

set "MATCH_TYPE="
set /p "MATCH_TYPE=Feature matching [default: 1]: "

if not defined MATCH_TYPE (
    set "MATCH_TYPE=1"
)

if "!MATCH_TYPE!"=="1" goto :MATCH_VALID
if "!MATCH_TYPE!"=="2" goto :MATCH_VALID

echo.
echo ERROR: Enter 1 or 2.
echo.
goto :MATCH_INPUT


:MATCH_VALID

set "GUIDED=0"

if "!MATCH_TYPE!"=="1" goto :MATCH_DONE

:GUIDED_INPUT

echo.
echo Guided matching is available for BruteForce.
echo Y = enabled
echo N = disabled
echo.

set "GUIDED_INPUT="
set /p "GUIDED_INPUT=Guided matching [default: N]: "

if not defined GUIDED_INPUT (
    set "GUIDED_INPUT=N"
)

if /I "!GUIDED_INPUT!"=="Y" (
    set "GUIDED=1"
    goto :MATCH_DONE
)

if /I "!GUIDED_INPUT!"=="N" (
    set "GUIDED=0"
    goto :MATCH_DONE
)

echo.
echo ERROR: Enter Y or N.
goto :GUIDED_INPUT


:MATCH_DONE


:MATCH_MODE_INPUT

echo.
echo ============================================================
echo Matching strategy
echo ============================================================
echo 1 = Sequential
echo 2 = Exhaustive
echo.

set "MATCH_MODE="
set /p "MATCH_MODE=Matching strategy [default: 1]: "

if not defined MATCH_MODE (
    set "MATCH_MODE=1"
)

if "!MATCH_MODE!"=="1" goto :MATCH_MODE_VALID
if "!MATCH_MODE!"=="2" goto :MATCH_MODE_VALID

echo.
echo ERROR: Enter 1 or 2.
echo.
goto :MATCH_MODE_INPUT


:MATCH_MODE_VALID

set "OVERLAP=10"
set "QUADRATIC=1"

if "!MATCH_MODE!"=="2" goto :CONFIGURATION


:OVERLAP_INPUT

echo.
set "OVERLAP="
set /p "OVERLAP=Sequential overlap [default: 10]: "

if not defined OVERLAP (
    set "OVERLAP=10"
)

set "INVALID_OVERLAP="

for /f "delims=0123456789" %%A in ("!OVERLAP!") do (
    set "INVALID_OVERLAP=1"
)

if defined INVALID_OVERLAP (
    echo.
    echo ERROR: Overlap must be a positive integer.
    goto :OVERLAP_INPUT
)

if !OVERLAP! LSS 1 (
    echo.
    echo ERROR: Overlap must be at least 1.
    goto :OVERLAP_INPUT
)


:QUADRATIC_INPUT

echo.
echo Quadratic overlap:
echo Y = enabled
echo N = disabled
echo.

set "QUADRATIC_INPUT="
set /p "QUADRATIC_INPUT=Quadratic overlap [default: Y]: "

if not defined QUADRATIC_INPUT (
    set "QUADRATIC_INPUT=Y"
)

if /I "!QUADRATIC_INPUT!"=="Y" goto :QUADRATIC_Y
if /I "!QUADRATIC_INPUT!"=="N" goto :QUADRATIC_N

echo.
echo ERROR: Enter Y or N.
goto :QUADRATIC_INPUT


:QUADRATIC_Y

set "QUADRATIC=1"
goto :CONFIGURATION


:QUADRATIC_N

set "QUADRATIC=0"
goto :CONFIGURATION


:CONFIGURATION

echo.
echo ============================================================
echo Configuration
echo ============================================================
echo Project:              !PROJECT_NAME!
echo Video:                !VIDEO!
echo Frames:               !FRAME_COUNT!
echo SAM3 prompt:          !SAM_PROMPT!
echo SAM3 model:           !SAM3_PATH!
echo COLMAP:               !COLMAP_PATH!

if "!PIPELINE_MODE!"=="1" (
    echo Pipeline:          Whole RGBA
) else (
    echo Pipeline:          Original COLMAP + RGBA Instant-NGP
)

if "!FEATURE_TYPE!"=="1" (
    echo Features:          ALIKED-N32
) else (
    echo Features:          SIFT
)

if "!MATCH_TYPE!"=="1" (
    echo Matching:          LightGlue
) else (
    echo Matching:          BruteForce
    if "!GUIDED!"=="1" (
        echo Guided:         Enabled
    ) else (
        echo Guided:         Disabled
    )
)

if "!MATCH_MODE!"=="1" (
    echo Strategy:          Sequential
    echo Overlap:           !OVERLAP!
    if "!QUADRATIC!"=="1" (
        echo Quadratic:      Enabled
    ) else (
        echo Quadratic:      Disabled
    )
) else (
    echo Strategy:          Exhaustive
)

echo ============================================================
echo.

set "START="
set /p "START=Press Enter to start, or type N to cancel: "

if /I "!START!"=="N" (
    echo Cancelled.
    pause
    exit /b 0
)


echo.
echo ============================================================
echo                    RUNNING PIPELINE
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" ^
    --project "!PROJECT!" ^
    --video "!VIDEO!" ^
    --frames "!FRAME_COUNT!" ^
    --prompt "!SAM_PROMPT!" ^
    --sam3 "!SAM3_PATH!" ^
    --colmap "!COLMAP_PATH!" ^
    --pipeline-mode "!PIPELINE_MODE!" ^
    --feature-type "!FEATURE_TYPE!" ^
    --match-type "!MATCH_TYPE!" ^
    --guided "!GUIDED!" ^
    --match-mode "!MATCH_MODE!" ^
    --overlap "!OVERLAP!" ^
    --quadratic "!QUADRATIC!"

if errorlevel 1 (
    echo.
    echo ============================================================
    echo                    PIPELINE FAILED
    echo ============================================================
    pause
    exit /b 1
)


echo.
echo ============================================================
echo                 PIPELINE FINISHED
echo ============================================================
echo.


set "NGP=%ROOT%\instant-ngp.exe"

if not exist "%NGP%" (
    set "NGP=%ROOT%\testbed.exe"
)

if not exist "%NGP%" (
    echo ERROR: Could not find instant-ngp.exe or testbed.exe in:
    echo %ROOT%
    pause
    exit /b 1
)


echo Starting Instant-NGP...
echo Scene:
echo !PROJECT!
echo.

start "" "%NGP%" --scene "!PROJECT!"

endlocal
@echo off
REM ============================================================================
REM Pipeline Completo de Análisis de Sincronización EEG - Windows Version
REM ============================================================================

setlocal EnableDelayedExpansion

set WORKERS=12
set CONDITIONS=DMT EC EO
set SCRIPT_DIR=%~dp0

echo ========================================================================
echo INICIANDO PIPELINE COMPLETO
echo ========================================================================
echo Workers: %WORKERS%
echo Condiciones: %CONDITIONS%
echo Working directory: %SCRIPT_DIR%
echo ========================================================================

cd /d "%SCRIPT_DIR%"

call conda activate dmt_fz
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo activar el ambiente dmt_fz
    pause
    exit /b 1
)

echo [PASO 1] Procesamiento EEG
python fwd.py --jobs 0 --workers %WORKERS% --conditions %CONDITIONS%
if %errorlevel% neq 0 goto error

echo [PASO 2a] Consolidación de fases
python save_load_pickle.py --conditions %CONDITIONS%
if %errorlevel% neq 0 goto error

echo [PASO 2b] Filtrado por redes
python multi2pool2.py
if %errorlevel% neq 0 goto error

echo [PASO 2c] Análisis temporal
python calculate_syncro.py --workers %WORKERS% --conditions %CONDITIONS%
if %errorlevel% neq 0 goto error

echo [PASO 2d] Order parameter
python generate_order.py --workers %WORKERS% --conditions %CONDITIONS%
if %errorlevel% neq 0 goto error

echo [PASO 3] Generación de gráficos
python plot_order.py --workers %WORKERS%
if %errorlevel% neq 0 goto error

echo [PASO 4] Correlaciones
python pearson.py
if %errorlevel% neq 0 goto error

set /p CLUSTER="Ejecutar clustering (2-4 horas)? [y/N]: "
if /i "%CLUSTER%"=="y" (
    echo [PASO 5] Clustering
    python clustering.py
    if %errorlevel% neq 0 goto error
)

echo ========================================================================
echo PIPELINE COMPLETADO EXITOSAMENTE
echo ========================================================================
pause
exit /b 0

:error
echo [ERROR] Pipeline interrumpido
pause
exit /b 1

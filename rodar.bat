@echo off
rem PotteryCV: prepara tudo na primeira vez e abre o programa (dois cliques).
rem 1) uv (Python + pacotes, sem admin)  2) modelos do MediaPipe  3) .venv + requirements  4) main.py
setlocal
cd /d "%~dp0"
title PotteryCV

where uv >nul 2>nul
if errorlevel 1 set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    echo [1/3] Instalando o uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 goto erro
)

if not exist models mkdir models
set "MP=https://storage.googleapis.com/mediapipe-models"
if not exist models\hand_landmarker.task (
    echo [2/3] Baixando hand_landmarker.task...
    curl -fL -o models\hand_landmarker.task %MP%/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    if errorlevel 1 goto erro
)
if not exist models\face_landmarker.task (
    echo [2/3] Baixando face_landmarker.task...
    curl -fL -o models\face_landmarker.task %MP%/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
    if errorlevel 1 goto erro
)

if not exist .venv\Scripts\python.exe (
    echo [3/3] Criando o ambiente Python...
    uv venv --python 3.12 .venv
    if errorlevel 1 goto erro
)
echo [3/3] Conferindo as bibliotecas...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto erro

echo Abrindo o PotteryCV...
.venv\Scripts\python.exe main.py %*
if errorlevel 1 goto erro
exit /b 0

:erro
echo.
echo Algo deu errado: veja a mensagem acima.
pause
exit /b 1

@echo off
rem PotteryCV portatil (dois cliques).
rem Primeira vez: baixa um Python portatil para python\ e instala as bibliotecas nele (precisa de internet).
rem Depois: a pasta inteira roda em qualquer Windows 64 bits sem instalar nada. Copie a pasta toda.
setlocal
cd /d "%~dp0"
title PotteryCV
set "PY=%~dp0python\python.exe"

rem pronto = Python portatil existe e as bibliotecas instaladas batem com o requirements.txt
if exist "%PY%" if exist python\requirements.txt fc /b requirements.txt python\requirements.txt >nul 2>nul && goto modelos

call :uv || goto erro
if not exist "%PY%" (
    echo [1/3] Baixando o Python portatil...
    uv python install 3.12 --install-dir "%~dp0python_tmp" --no-bin --no-registry || goto erro
    for /d %%d in ("%~dp0python_tmp\cpython-3.12.*") do move "%%d" "%~dp0python" >nul
    rd /s /q "%~dp0python_tmp" 2>nul
)
if not exist "%PY%" goto erro
echo [2/3] Instalando as bibliotecas no Python portatil...
uv pip install --python "%PY%" --break-system-packages -r requirements.txt || goto erro
copy /y requirements.txt python\requirements.txt >nul

:modelos
if not exist models mkdir models
set "MP=https://storage.googleapis.com/mediapipe-models"
if not exist models\hand_landmarker.task (
    echo [3/3] Baixando hand_landmarker.task...
    curl -fL -o models\hand_landmarker.task %MP%/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task || goto erro
)
if not exist models\face_landmarker.task (
    echo [3/3] Baixando face_landmarker.task...
    curl -fL -o models\face_landmarker.task %MP%/face_landmarker/face_landmarker/float16/latest/face_landmarker.task || goto erro
)

echo Abrindo o PotteryCV...
"%PY%" main.py %*
if errorlevel 1 goto erro
exit /b 0

:uv
rem uv: baixa o Python e as bibliotecas (sem admin). So e usado na preparacao.
where uv >nul 2>nul && exit /b 0
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul && exit /b 0
echo Instalando o uv...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
where uv >nul 2>nul
exit /b %errorlevel%

:erro
echo.
echo Algo deu errado: veja a mensagem acima.
pause
exit /b 1

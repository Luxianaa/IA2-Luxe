@echo off
REM Doble clic para abrir Hilo en el navegador (http://localhost:8501)
cd /d "%~dp0"
title Hilo - Merci Collection
if not exist ".venv\Scripts\python.exe" (
    echo No se encontro el entorno .venv. Crealo con:
    echo   py -3.11 -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
REM Ollama (sirve Qwen 2.5 localmente): si no esta corriendo, se inicia minimizado
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if exist "%OLLAMA%" (
    "%OLLAMA%" list >nul 2>&1 || start "Ollama" /min "%OLLAMA%" serve
) else (
    echo Aviso: Ollama no esta instalado; Hilo funcionara solo con plantillas.
)
REM Si ya hay un Hilo abierto, el navegador mostraria esa version vieja (sin los cambios)
netstat -ano | findstr /r /c:":8501 .*LISTENING" >nul
if not errorlevel 1 (
    echo.
    echo ATENCION: Hilo ya esta abierto en otra ventana negra.
    echo Cierra esa ventana ^(o todas las de Hilo^) y vuelve a abrir este archivo,
    echo asi se cargan los ultimos cambios.
    echo.
    pause
    exit /b 1
)
echo Iniciando Hilo... el navegador se abrira en unos segundos.
echo (No cierres esta ventana mientras uses la app)
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause

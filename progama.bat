@echo off
setlocal

echo === Ir a la carpeta del proyecto ===
cd /d "C:\Users\dimas\OneDrive\Escritorio\Streamlit practicas"
echo Carpeta actual:
cd

echo.
echo === Verificar venv ===
if not exist "venv\Scripts\python.exe" (
  echo ERROR: No existe venv\Scripts\python.exe
  echo Crea el venv con:  python -m venv venv
  pause
  exit /b 1
)

echo.
echo === Ejecutando Streamlit con el Python del venv ===
"venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.address localhost

echo.
echo === Si llegaste aqui, Streamlit se cerro o fallo ===
pause
endlocal

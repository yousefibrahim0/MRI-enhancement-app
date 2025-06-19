@echo off
echo ======================================
echo 🚀 MRI Enhancement App Starting...
echo ======================================
echo.

echo ✅ Installing all required packages...
pip install -r requirements.txt

echo.
echo ✅ Packages installed successfully!
echo.

echo ======================================
echo ✅ Starting server...
echo ======================================
start cmd /k "python -m uvicorn main:app --host 127.0.0.2 --port 8000 --reload"

timeout /t 4 >nul

echo.
echo 🌐 Opening Chrome Browser...
start chrome http://127.0.0.2:8000/

echo.
echo ✅ Server is running at http://127.0.0.2:8000/
echo ======================================

pause

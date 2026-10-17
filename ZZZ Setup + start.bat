@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV=venv"
set "VENV_PY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\.installed"
set "LOGDIR=logs"
set "SETUP_LOG=%LOGDIR%\setup.log"
set "REQ=requirements.txt"

REM ═══════════════ 1. Поиск системного Python ═══════════════
set "SYS_PY="
where python >nul 2>nul && set "SYS_PY=python"
if not defined SYS_PY (
    where py >nul 2>nul && set "SYS_PY=py -3"
)
if not defined SYS_PY (
    echo [ОШИБКА] Python не найден в PATH.
    echo Установите Python 3.10+ и повторите.
    pause
    exit /b 1
)

REM ═══════════════ 2. Нужна ли установка? ═══════════════
if exist "%VENV_PY%" if exist "%MARKER%" goto :run

echo.
echo ================================================================
echo  ПЕРВИЧНАЯ УСТАНОВКА ОКРУЖЕНИЯ
echo ================================================================
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

if exist "%VENV%" (
    echo [SETUP] Найден неполный venv — удаляю...
    rmdir /s /q "%VENV%" 2>nul
)

echo [SETUP] Создаю виртуальное окружение...
echo [%date% %time%] === Создание venv === >> "%SETUP_LOG%"
%SYS_PY% -m venv "%VENV%" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 goto :fail

echo [SETUP] Обновляю pip...
echo [%date% %time%] === Обновление pip === >> "%SETUP_LOG%"
"%VENV_PY%" -m pip install --upgrade pip >> "%SETUP_LOG%" 2>&1
if errorlevel 1 goto :fail

echo [SETUP] Устанавливаю зависимости из %REQ%...
echo [%date% %time%] === pip install -r %REQ% === >> "%SETUP_LOG%"
"%VENV_PY%" -m pip install -r "%REQ%" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 goto :fail

echo [SETUP] Проверяю импорты (aiogram, openai, httpx, dotenv)...
"%VENV_PY%" -c "import aiogram, openai, httpx, dotenv" >> "%SETUP_LOG%" 2>&1
if errorlevel 1 (
    echo [FAIL] Проверка импортов не прошла. >> "%SETUP_LOG%"
    goto :fail
)

echo ok> "%MARKER%"
echo [%date% %time%] === Установка завершена успешно === >> "%SETUP_LOG%"

echo.
echo [SETUP] Готово. Лог установки: %SETUP_LOG%
echo.

REM ═══════════════ 3. Проверка .env ═══════════════
:run
if not exist ".env" (
    echo.
    echo [ВНИМАНИЕ] Файл .env не найден.
    echo Создайте .env со строкой:  TELEGRAM_TOKEN=твой_токен
    echo.
    pause
)

REM ═══════════════ 4. Запуск бота ═══════════════
echo [RUN] Запуск бота...
"%VENV_PY%" run.py

set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" if not "%EXITCODE%"=="130" (
    echo.
    echo [Бот упал] Код выхода: %EXITCODE%
    pause
)

endlocal
exit /b 0

REM ═══════════════ Обработка ошибок установки ═══════════════
:fail
echo.
echo [ОШИБКА] Установка не удалась.
echo Подробности смотрите в: %SETUP_LOG%
pause
exit /b 1
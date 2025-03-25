@echo off
chcp 65001 > nul
set PYTHONUNBUFFERED=1

echo.
echo === Запуск скриптов (вывод только в консоль) ===
echo.

:: Первый скрипт
echo [%time%] Запуск: kja\kja_tdgraph_tests.py
python "kja\kja_tdgraph_tests.py"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ОШИБКА] Скрипт kja_tdgraph_tests.py завершился с кодом %ERRORLEVEL%
    pause
    exit /b
)
echo [%time%] Успешно завершён: kja_tdgraph_tests.py

:: Второй скрипт
echo.
echo [%time%] Запуск: perm\perm_tdgraph_tests.py
python "perm\perm_tdgraph_tests.py"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ОШИБКА] Скрипт perm_tdgraph_tests.py завершился с кодом %ERRORLEVEL%
    pause
    exit /b
)
echo [%time%] Успешно завершён: perm_tdgraph_tests.py

echo.
echo === Все скрипты выполнены! ===
pause
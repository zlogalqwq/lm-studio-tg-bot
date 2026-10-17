import os, sys

# ── Отключаем QuickEdit: клик по окну не должен блокировать stdout ──
def _disable_quickedit() -> None:
    """QuickEdit включает режим выделения при клике в консоль.
    Пока выделение активно, Windows блокирует все записи в консоль —
    процесс выглядит зависшим, пока не нажмёшь Enter."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32

        STD_INPUT_HANDLE = -10
        ENABLE_QUICK_EDIT_MODE = 0x0040
        ENABLE_EXTENDED_FLAGS = 0x0080

        handle = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if handle in (0, -1, None):
            return

        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return

        # Снимаем QUICK_EDIT, остальные флаги сохраняем.
        # ENABLE_EXTENDED_FLAGS должен быть установлен, иначе
        # изменение QUICK_EDIT игнорируется.
        new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
        if new_mode != mode.value:
            kernel32.SetConsoleMode(handle, new_mode)
    except Exception:
        pass


_disable_quickedit()  # ← самое первое действие

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging

from bot.main import run

log = logging.getLogger("run")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n⏹ Бот остановлен (Ctrl+C).", file=sys.stderr)
        sys.exit(130)
    except asyncio.CancelledError:
        print("\n⏹ Бот остановлен (cancelled).", file=sys.stderr)
        sys.exit(130)
    except Exception:
        log.exception("Bot crashed")
        sys.exit(1)
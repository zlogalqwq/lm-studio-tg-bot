import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging

from bot.main import run

log = logging.getLogger("run")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        # Обычно ловит сам aiogram (handle_signals=True), но на всякий случай.
        print("\n⏹ Бот остановлен (Ctrl+C).", file=sys.stderr)
        sys.exit(130)  # 128 + SIGINT — стандартный код
    except asyncio.CancelledError:
        print("\n⏹ Бот остановлен (cancelled).", file=sys.stderr)
        sys.exit(130)
    except Exception:
        log.exception("Bot crashed")
        sys.exit(1)
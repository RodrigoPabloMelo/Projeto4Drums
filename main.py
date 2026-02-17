import asyncio
import platform
import logging
from playground.virtual_drums import VirtualDrums
from config.config import CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main async function for Pyodide compatibility."""
    app = VirtualDrums()
    try:
        app.setup()
        while True:
            app.update_loop()
            await asyncio.sleep(0.1 / CONFIG['fps'])
    except SystemExit:
        pass
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        app.cleanup()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
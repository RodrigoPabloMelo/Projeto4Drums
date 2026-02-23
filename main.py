import asyncio
import platform
import logging
from playground.virtual_drums import VirtualDrums
from config.config import CONFIG

# Configura logging basico para acompanhar eventos e erros.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Logger do modulo atual.
logger = logging.getLogger(__name__)

async def main():
    """Funcao principal assincrona (compatibilidade com Pyodide)."""
    # Cria a aplicacao principal.
    app = VirtualDrums()
    try:
        # Inicializa camera, audio e recursos do MediaPipe.
        app.setup()
        # Loop principal: processa frames e aguarda entre iteracoes.
        while True:
            app.update_loop()
            await asyncio.sleep(0.1 / CONFIG['fps'])
    except SystemExit:
        # Saida solicitada pelo usuario.
        pass
    except Exception as e:
        # Registra qualquer erro inesperado.
        logger.error(f"Unexpected error: {e}")
    finally:
        # Garante liberacao de recursos.
        app.cleanup()

if platform.system() == "Emscripten":
    # Em Pyodide/Emscripten, agenda a tarefa sem bloquear.
    asyncio.ensure_future(main())
else:
    # Em execucao local, roda o loop principal diretamente.
    if __name__ == "__main__":
        asyncio.run(main())

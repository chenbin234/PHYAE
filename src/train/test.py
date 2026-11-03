# Then in each module:
import logging
logger = logging.getLogger(__name__)  # or logging.getLogger('app.module_name')

def some_function():
    logger.warning("Function called")
    logger.info("Function finished")
    logger.error("Function failed")

some_function()
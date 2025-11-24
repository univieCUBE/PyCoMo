import multiprocessing
import os
import pickle
from platform import system
from tempfile import mkstemp
from multiprocessing import get_context, Pool
from cobra.util.process_pool import ProcessPool, _init_win_worker
from typing import Any, Callable, Optional, Tuple, Type
from pycomo.helper.logger import get_logger_name
import logging

logger = logging.getLogger(get_logger_name())
logger.debug('Process Pool Logger initialized.')


__all__ = ("SpawnProcessPool",)

class SpawnProcessPool(ProcessPool):
    """ProcessPool that explicitly uses 'spawn' start method for better cross-platform compatibility."""
    
    def __init__(
        self,
        processes: Optional[int] = None,
        initializer: Optional[Callable] = None,
        initargs: Tuple = (),
        maxtasksperchild: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize a process pool using the 'spawn' start method.
        
        This implementation ensures consistent behavior across different platforms
        by always using the 'spawn' start method, rather than relying on
        platform-specific defaults.
        """
        logger.debug("Init SpawnProcessPool")
        Pool.__init__(self, **kwargs)
        
        # Handle Windows-specific initialization same as parent
        self._filename = None
        if initializer is not None and system() == "Windows":
            descriptor, self._filename = mkstemp(suffix=".pkl")
            with os.fdopen(descriptor, mode="wb") as handle:
                pickle.dump((initializer,) + initargs, handle)
            initializer = _init_win_worker
            initargs = (self._filename,)
            
        # Create pool using spawn context instead of default multiprocessing.Pool
        self._pool = get_context("spawn").Pool(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild,
        )
        logger.debug("Finished init SpawnProcessPool")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the process pool. 
        This method is adjusted to circumvent problems with pool.close in Python 3.12."""
        logger.debug("ProcessPool starting exit context")
        try:
            logger.debug("ProcessPool try to close")
            self._pool.terminate()
            logger.debug("ProcessPool try to join")
            self._pool.join()
            logger.debug("ProcessPool joined")
        finally:
            logger.debug("ProcessPool try cleanup")
            self._clean_up()
        logger.debug("ProcessPool try super exit")
        result = self._pool.__exit__(exc_type, exc_val, exc_tb)
        logger.debug("ProcessPool super exit done")
        return result
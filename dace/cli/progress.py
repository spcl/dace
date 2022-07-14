# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Progress bar command line interface. """

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import time
from typing import Generator, Optional, TypeVar

from dace import config

T = TypeVar('T')


def optional_progressbar(iter: Generator[T, None, None],
                         title: Optional[str] = None,
                         n: Optional[int] = None,
                         progress: Optional[bool] = None,
                         time_threshold: float = 5.0) -> Generator[T, None, None]:
    """
    Creates a progress bar for lengthy processes, depending on the time spent iterating over the generator.

    :param iter: The original generator to iterate over.
    :param title: An optional title for the progress bar.
    :param n: An optional length (number of elements in the generator).
    :param progress: A boolean stating whether to always show progress (True), never (False), or depending on the
                     DaCe configuration and time spent (see ``time_threshold``).
    :param time_threshold: Time (in seconds) specifying how long to wait before showing the progress bar.
    """
    # tqdm is unavailable, use original generator
    if tqdm is None:
        yield from iter
        return
    # Config override
    if progress is None and not config.Config.get_bool('progress'):
        yield from iter
        return

    # If length was not given, try to determine from generator (if, e.g., list)
    if n is None:
        try:
            n = len(iter)
        except (TypeError, AttributeError):
            n = None

    # Collect starting data
    if progress is True:
        pbar = tqdm(total=n, desc=title)
    else:
        pbar = None

    start = time.time()
    for counter, elem in enumerate(iter):
        if pbar is None and (time.time() - start) > time_threshold:
            pbar = tqdm(total=n, desc=title, initial=counter)

        yield elem

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()


class OptionalProgressBar:
    """
    Creates a progress bar for lengthy processes, depending on the time spent between successive calls to ``next()``.
    """

    def __init__(self,
                 n: int,
                 title: Optional[str] = None,
                 progress: Optional[bool] = None,
                 time_threshold: float = 5.0) -> None:
        # Config override
        if tqdm is None or (progress is None and not config.Config.get_bool('progress')):
            self.skip = True
        else:
            self.skip = progress is False

        self.pbar = None
        if self.skip:
            return

        self.title = title
        self.n = n
        self.threshold = time_threshold

        if progress is True:
            self.pbar = tqdm(total=n, desc=title)

        self.counter = 0
        self.start_time = time.time()

    def restart(self):
        """
        Restarts the timer and closes any existing progress bar.
        """
        self.done()
        self.counter = 0
        self.start_time = time.time()

    def next(self):
        """
        Advances the progress bar. If visible, shows progress, otherwise updates in the background.
        If the time threshold has passed and the progress bar should appear, this method creates it.
        """
        if self.skip:
            return

        self.counter += 1
        if self.pbar is None and (time.time() - self.start_time) > self.threshold:
            self.pbar = tqdm(total=self.n, desc=self.title, initial=self.counter)
        elif self.pbar is not None:
            self.pbar.update(1)

    def done(self):
        """
        Closes the progress bar, if it was opened.
        """
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        self.counter = 0

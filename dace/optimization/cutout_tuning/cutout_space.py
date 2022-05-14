# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Optional

from dace import SDFG, InstrumentationType


class CutoutSpace(ABC):
    """
    
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Name of the space.

        :return: the name of the space.
        """
        pass

    @abstractmethod
    def cutouts(self) -> Generator[SDFG, None, None]:
        """
        Independent (!) parts of the SDFG to be optimized in isolation.
        A cutout may comprise multiple states though.
        
        :return: a generator of cutouts.
        """
        pass

    @abstractmethod
    def configurations(self, cutout: SDFG) -> Generator[Any, None, None]:
        """
        The configuration space of a cutout to be searched.

        :param cutout: the cutout.
        :return: a generator of configs.
        """
        pass

    @abstractmethod
    def apply_config(self, cutout: SDFG, config: Any, make_copy=True) -> Optional[SDFG]:
        """
        Applies the configuration to the cutout.

        :param sdfg: the SDFG to which the config is applied.
        :param config: the config.
        :param make_copy: apply on a copy of the SDFG or on the actual SDFG. 
        :return: the modified SDFG.
        """
        pass

    @abstractmethod
    def translate_config(self, cutout: SDFG, sdfg: SDFG, config: Any) -> Any:
        pass

    @abstractmethod
    def encode_config(self, config: Any) -> str:
        """
        Returns a string representation of the configuration.

        :param config: the config as an object.
        :return: the config as a string.
        """
        pass

    @abstractmethod
    def decode_config(self, config: str) -> Any:
        """
        Parses the config object from its string representation.

        :param config: the config as a string.
        :return: the config as an object.
        """
        pass

    '''
    @abstractmethod
    def extract_patterns(self, sdfg: SDFG, cutout: SDFG, config: str) -> Tuple[Dict[str, Any]]:
        """
        Translate the configuration found in a cutout into (possibly multiple) general patterns that can be found in other SDFGs and cutouts.

        :param sdfg: the SDFG that the cutout was extracted from.
        :param cutout: the cutout.
        :param config: the config in string representation.
        :return: a collection of patterns
        """
        pass
    '''

    def optimize(self, sdfg: SDFG, dreport, instrumentation_type: InstrumentationType) -> Dict[Any, str]:
        """
        Searches for the best configuration of each cutout and applies those configurations to the SDFG subsequently.

        :param sdfg: the SDFG to be optimized.
        :param dreport: the data report to be used for measurements.
        :param instrumentation_type: the instrumentation type for comparing results.
        :return:
        """
        database = {}
        return database

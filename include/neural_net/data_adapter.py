from abc import ABC, abstractmethod
from typing import Generic

from .defs import RawInputType, InputType


class DataAdapter(Generic[RawInputType, InputType]):
    """Base class for data adapters.

    A DataAdapter performs several operations on the data pipeline. At this
    point, only feature extraction is supported, but the class may be extended
    in the future to support output postprocessing, a fit method, and state
    persistence through files (analogous to Normalizer).

    """
    
    def extract_feats(self, raw_input: RawInputType) -> InputType:
        """Transform a single raw input into a processed input. This method        
        plugs into the NeuralNet pipeline immediately before normalization:

        raw_input -> extract_feats -> normalize -> forward -> unnormalize        

        Args:
            raw_input: A single raw input of type RawInputType.

        Returns:
            A processed input of type InputType.
        """
        
        # By default, this method is the identity function. 
        return raw_input

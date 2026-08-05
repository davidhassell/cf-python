import cfdm

from . import CorrelationModel,     Quantization, mixin


class UncertaintyAncillary(mixin.PropertiesData, cfdm.UncertaintyAncillary):
    """TODOU"""
    
    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._CorrelationModel = CorrelationModel
        instance._Quantization = Quantization
        return instance

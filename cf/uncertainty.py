import cfdm

from . import ProbabilityDistribution,     Quantization, mixin


class Uncertainty(mixin.PropertiesData, cfdm.Uncertainty):
    """TODOU"""
    
    def __new__(cls, *args, **kwargs):
        """Store component classes."""
        instance = super().__new__(cls)
        instance._ProbabilityDistribution = ProbabilityDistribution
        instance._Quantization = Quantization
        return instance

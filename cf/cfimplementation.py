import cfdm

from . import (
    AuxiliaryCoordinate,
    Bounds,
    CellConnectivity,
    CellMeasure,
    CellMethod,
    CoordinateConversion,
    CoordinateReference,
    Count,
    Datum,
    DimensionCoordinate,
    Domain,
    DomainAncillary,
    DomainAxis,
    DomainTopology,
    Field,
    FieldAncillary,
    Index,
    InteriorRing,
    InterpolationParameter,
    List,
    NodeCountProperties,
    PartNodeCountProperties,
    Quantization,
    TiePointIndex,
)
from .data import Data
from .data.array import (
    AggregatedArray,
    BoundsFromNodesArray,
    CellConnectivityArray,
    GatheredArray,
    PointTopologyArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
    SubsampledArray,
    XnetcdfArray,
)
from .functions import CF


class CFImplementation(cfdm.CFDMImplementation):
    """A container for the CF data model implementation for `cf`.

    .. versionadded:: 3.0.0

    """

    def nc_set_dataset_chunksizes(self, data, sizes, override=False):
        """Set the data dataset chunksizes.

        .. versionadded:: 3.16.2

        :Parameters:

            data: `Data`
                The data.

            sizes: sequence of `int`
                The new dataset chunk sizes.

            override: `bool`, optional
                If True then set the dataset chunks sizes even if some
                have already been specified. If False, the default,
                then only set the dataset chunks sizes if some none
                have already been specified.

        :Returns:

            `None`

        """
        if override or not data.nc_dataset_chunksizes():
            data.nc_set_dataset_chunksizes(sizes)

    def set_construct(self, parent, construct, axes=None, copy=True, **kwargs):
        """Insert a construct into a field or domain.

        Does not attempt to conform cell method nor coordinate
        reference constructs, as that has been handled by `cfdm`.

        .. versionadded:: 3.14.0

        :Parameters:

            parent: `Field` or `Domain`
               On what to set the construct

            construct:
                The construct to set.

            axes: `tuple` or `None`, optional
                The construct domain axes, if applicable.

            copy: `bool`, optional
                Whether or not to set a copy of *construct*.

            kwargs: optional
                Additional parameters to the `set_construct` method of
                *parent*.

        :Returns:

            `str`
                The construct identifier.

        """
        kwargs.setdefault("conform", False)
        return super().set_construct(
            parent, construct, axes=axes, copy=copy, **kwargs
        )


_implementation = CFImplementation(
    cf_version=CF(),
    AggregatedArray=AggregatedArray,
    AuxiliaryCoordinate=AuxiliaryCoordinate,
    CellConnectivity=CellConnectivity,
    CellMeasure=CellMeasure,
    CellMethod=CellMethod,
    CoordinateReference=CoordinateReference,
    DimensionCoordinate=DimensionCoordinate,
    Domain=Domain,
    DomainAncillary=DomainAncillary,
    DomainAxis=DomainAxis,
    DomainTopology=DomainTopology,
    Field=Field,
    FieldAncillary=FieldAncillary,
    Bounds=Bounds,
    InteriorRing=InteriorRing,
    InterpolationParameter=InterpolationParameter,
    CoordinateConversion=CoordinateConversion,
    Datum=Datum,
    List=List,
    Index=Index,
    Count=Count,
    NodeCountProperties=NodeCountProperties,
    PartNodeCountProperties=PartNodeCountProperties,
    Data=Data,
    BoundsFromNodesArray=BoundsFromNodesArray,
    CellConnectivityArray=CellConnectivityArray,
    GatheredArray=GatheredArray,
    PointTopologyArray=PointTopologyArray,
    Quantization=Quantization,
    RaggedContiguousArray=RaggedContiguousArray,
    RaggedIndexedArray=RaggedIndexedArray,
    RaggedIndexedContiguousArray=RaggedIndexedContiguousArray,
    SubsampledArray=SubsampledArray,
    TiePointIndex=TiePointIndex,
    XnetcdfArray=XnetcdfArray,
)


def implementation():
    """Return a container for the CF data model implementation.

    .. versionadded:: 3.1.0

    .. seealso:: `cf.example_field`, `cf.read`, `cf.write`

    :Returns:

        `CFImplementation`
            A container for the CF data model implementation.

    """
    return _implementation.copy()

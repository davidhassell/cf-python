class RaggedArrayMixin:
    """Mixin TODODASKDOCS class for a container of an array.

    .. versionadded:: TODODASKVER

    """

    def to_dask_array(self, chunks="auto"):
        """Create a dask array TODODASKDOCS.

        .. versionadded:: TODODASKVER

        :Parameters:

            chunks: `int`, `tuple`, `dict` or `str`, optional
                Specify the chunking of the returned dask array.

                Any value accepted by the *chunks* parameter of the
                `dask.array.from_array` function is allowed.

                The chunk sizes implied by *chunks* for a dimension that
                has been fragemented are ignored and replaced with values
                that are implied by that dimensions fragment sizes.

        :Returns:

            `dask.array.Array`

        """
        from functools import partial

        import dask.array as da
        from dask import config
        from dask.array.core import getter, normalize_chunks
        from dask.base import tokenize

        name = (f"{self.__class__.__name__}-{tokenize(self)}",)

        dtype = self.dtype

        context = partial(config.set, scheduler="synchronous")

        compressed_dimensions = self.compressed_dimensions()
        conformed_data = self.conformed_data()
        compressed_data = conformed_data["data"]

        # Get the (cfdm) subarray class
        Subarray = self.get_Subarray()

        # Set the chunk sizes for the dask array
        chunks = self.subarray_shapes(chunks)
        chunks = normalize_chunks(
            self.subarray_shapes(chunks),
            shape=self.shape,
            dtype=dtype,
        )

        dsk = {}
        for u_indices, u_shape, c_indices, chunk_location in zip(
            *self.subarrays(chunks)
        ):
            subarray = Subarray(
                data=compressed_data,
                indices=c_indices,
                shape=u_shape,
                compressed_dimensions=compressed_dimensions,
                context_manager=context,
            )

            dsk[name + chunk_location] = (
                getter,
                subarray,
                Ellipsis,
                False,
                False,
            )

        # Return the dask array
        return da.Array(dsk, name[0], chunks=chunks, dtype=dtype)
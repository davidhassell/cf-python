class Group:

    def __init__(self, group, group_span=None, group_contiguous=None, coord=None, size=None, n=-1):
        self.group = group
        self.group_span = group_span
        self.group_contiguous = group_contiguous

        self.coord = coord
        try:
            self.bounds = coord.get_bounds(None)
        except AttributeError:
            self.bounds = None
            
        if coord is not None:
            size = coord.size
            
        if isinstance(group, np.ndarray):
            if group.dtype.kind != "i":
                raise ValueError(
                    f"Can't group by numpy array of type {group.dtype.name}"
                )

            if classification.shape != (size,):
                raise ValueError(
                    "Can't group by numpy array with incorrect "
                    f"shape: {group.shape}"
                )

            classification = group.copy()
        else:
            classification = self.initialise_classification(size, n)

        self.classification = classification
        self.ignore_n = -1
        
    @classmethod
    def group_by_integer(cls, group):
        size = self.size

        classification = []
        extend = classification.extend
        for n in range(size // group):
            extend((n,) * group)
            
        d = size -  len(classification)
        if d:
            if self.group_span is not False and self.group_span is not None:
                extend((-1,) * d)
            else:
                n +=1                
                extend((n,) * d)
                
        self.classification = np.array(classification)
        self.n = n + 1
    
    @classmethod
    def initialise_classification(cls, size, n=-1):
        return np.full((size,), n, int)
        self.n = n + 1
    
    @classmethod
    def indices(cls, classification):
        where = np.where

        ids = np.unique(classification)
        if ids[0] < 0:
            ids = ids[ids >= 0]

        ids = ids.tolist()
        for i in ids:
            index = where(classification == i)[0]
            yield index

    @classmethod
    def tyu(cls, coord, group_by, time_interval):
        """Returns bounding values and limits for a general collapse.
    
        :Parameters:
    
            coord: `DimensionCoordinate`
                The dimension coordinate construct associated with
                the collapse.
    
            group_by: `str`
                As for the *group_by* parameter of the `collapse` method.
    
            time_interval: `bool`
                If True then then return a tuple of date-time
                objects. If False return a tuple of `Data` objects.
    
        :Returns:
    
            `tuple`
                A tuple of 4 `Data` objects or, if *time_interval*
                is True, a tuple of 4 date-time objects.

        """
        coord = self.coord
        group_by = self.group_by
        
        bounds = coord.get_bounds(None)
        
        if bounds is None and group_by == "bounds":
            raise ValueError(
                f"{coord.identity()!r} coordinate bounds "
                f"are required with group_by={group_by!r}"
            )

        if (bounds is None and group_by is None) or group_by == "coords":
            self.group_by = 'coords'
            if coord.increasing:
                lower = coord.data[0]
                upper = coord.data[-1]
            else:
                lower = coord.data[-1]
                upper = coord.data[0]
                
            lower_limit = lower
            upper_limit = upper

        elif bounds is not None:
            self.group_by = 'bounds'
            lower_bounds = coord.lower_bounds
            upper_bounds = coord.upper_bounds
            lower = lower_bounds[0]
            upper = upper_bounds[0]
            lower_limit = lower_bounds[-1]
            upper_limit = upper_bounds[-1]
                        
        if time_interval:
            units = coord.Units
            if units.isreftime:
                lower = lower.datetime_array[0]
                upper = upper.datetime_array[0]
                lower_limit = lower_limit.datetime_array[0]
                upper_limit = upper_limit.datetime_array[0]
            elif not units.istime:
                raise ValueError(
                    f"Can't group by TimeDuration "
                    f"when coordinates have units {coord.Units!r}"
                )
            
        return (lower, upper, lower_limit, upper_limit,    group_by)

    @classmethod
    def by_timeduration(cls,
        classification,
        n,
        coord,
        interval,
        lower,
        upper,
        lower_limit,
        upper_limit,
        group_by,
        extra_condition=None,
    ):
        """Prepares for a collapse where the group is a
        TimeDuration.

        :Parameters:

            classification: `numpy.ndarray`

            n: `int`

            coord: `DimensionCoordinate`

            interval: `TimeDuration`

            lower: date-time object

            upper: date-time object

            lower_limit: `datetime`

            upper_limit: `datetime`

            group_by: `str`

            extra_condition: `Query`, optional

        :Returns:

            (`numpy.ndarray`, `int`)

        """
        if self.lower is None:
            self.tyu(time_interval=True)

        if self.classification is None:
            self.initialise_classification(-1)
            
        group_by_coords = self.group_by == "coords"

        if coord.increasing:
            # Increasing dimension coordinate
            lower, upper = interval.bounds(lower)
            while lower <= upper_limit:
                lower, upper = interval.interval(lower)
                classification, n, lower, upper = cls.ddddd(
                    classification,
                    n,
                    lower,
                    upper,
                    True,
                    coord,
                    group_by_coords,
                    extra_condition,
                )
        else:
            # Decreasing dimension coordinate
            lower, upper = interval.bounds(upper)
            while upper >= lower_limit:
                lower, upper = interval.interval(upper, end=True)
                classification, n, lower, upper = cls.ddddd(
                    classification,
                    n,
                    lower,
                    upper,
                    False,
                    coord,
                    group_by_coords,
                    extra_condition,
                )

        return classification, n

    @classmethod
    def ddddd(cls,
        classification,
        n,
        lower,
        upper,
        increasing,
        coord,
        group_by_coords,
        extra_condition,
              ignore_n
    ):
        """Returns configuration for a general collapse.

        :Parameter:

            extra_condition: `Query`

        :Returns:

            `numpy.ndarray`, `int`, date-time, date-time

        """
        if self.group_by == "bounds":
            q = ge(self.lower, attr="lower_bounds") & le(
                self.upper, attr="upper_bounds"
            )
        else:
            q = ge(self.lower) & lt(self.upper)

        if extra_condition:
            q &= extra_condition

        index = q.evaluate(self.coord).array

        #new
        ignore_group, ignore_n = cls.ggggg(
            classification, ignore_n,
            coord, index,
            group_span, group_contiguous
        )
            
        if not ignore_group:
            self.classification[index] = n
            n += 1

        if increasing:
            lower = upper
        else:
            upper = lower

        self.lower = lower
        self.upper = upper
            
#        return classification, n, lower, upper, ignore_n

    @classmethod
    def by_data(cls,
        classification,
        n,
        coord,
        interval,
        lower,
        upper,
        lower_limit,
        upper_limit,
        group_by,
        extra_condition=None,
    ):
        """Prepares for a collapse where the group is a data
        interval.

        :Returns:

            `numpy.ndarray`, `int`

        """
        group_by_coords = group_by == "coords"

        if coord.increasing:
            # Increasing dimension coordinate
            lower = lower.squeeze()
            while lower <= upper_limit:
                upper = lower + interval
                classification, n, lower, upper = cls.ddddd(
                    classification,
                    n,
                    lower,
                    upper,
                    True,
                    coord,
                    group_by_coords,
                    extra_condition,
                )
        else:
            # Decreasing dimension coordinate
            upper = upper.squeeze()
            while upper >= lower_limit:
                lower = upper - interval
                classification, n, lower, upper = cls.ddddd(
                    classification,
                    n,
                    lower,
                    upper,
                    False,
                    coord,
                    group_by_coords,
                    extra_condition,
                )

        return classification, n

    @class_method
    def by_queries(cls,
        classification,
        n,
        coord,
                   queries,
        parameter,
        extra_condition=None,
        within=False,
    ):
        """Processes a group selection.

        :Parameters:

            classification: `numpy.ndarray`

            n: `int`

            coord: `DimensionCoordinate`

            queries: sequence of `Query`

            parameter: `str`
                The name of the `cf.Field.collapse` parameter which
                defined *selection*. This is used in error messages.

                *Parameter example:*
                  ``parameter='within_years'``

            extra_condition: `Query`, optional

        :Returns:

            `numpy.ndarray`, `int`

        """
        # Create an iterator for stepping through each Query in
        # the selection sequence
        try:
            queries = iter(queries)
        except TypeError:
            raise ValueError(
                "Can't collapse: Bad parameter value: "
                f"{parameter}={selection!r}"
            )

        n = self.n
        classification =  self.classification
      
        for condition in queries:
            if not isinstance(condition, Query):
                raise ValueError(
                    f"Can't collapse: {parameter} sequence contains a "
                    f"non-{Query.__name__} object: {condition!r}"
                )

            if extra_condition is not None:
                condition &= extra_condition

            index = condition.evaluate(coord).array

            classification[index] = n
            n += 1

        self.n = n
        self.classification = classification
#        return classification, n

    @classmethod
    def discern_groups(cls, classification):
        """Processes a group classification.

        .. seealso:: `discrern_runs_within`

        :Parameters:

            classification: `numpy.ndarray`

        :Returns:

            `numpy.ndarray`

        **Examples**

        >>> classification = np.array(
        ...   [0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1]
        ... )
        >>> print(classification)
        [ 0  0  0 -1 -1 -1 -1 -1  0  0  0 -1 -1 -1 -1]
        >>> print(_discern_runs(classification))
        [ 0  0  0 -1 -1 -1 -1 -1  1  1  1 -2 -2 -2 -2]

        """
        n = 0
        m = -1

        x = np.where(np.diff(classification))[0] + 1
        if not x.size:
            if classification[0] >= 0:
                value = n
            else:
                value = m

            classification[:] = value
            return classification

        x = x.tolist()
        
        if classification[0] >= 0:
            classification[0 : x[0]] = n
            n += 1
        else:
            classification[0 : x[0]] = m
            m -= 1

        for i, j in zip(x[:-1], x[1:]):
            if classification[i] >= 0:
                value = n
                n += 1
            else:
                value = m
                m -= 1

            classification[i:j] = value

        if classification[x[-1]] >= 0:
            classification[x[-1] :] = n
        else:
            classification[x[-1] :] = m

        if group_span is not False and group_span is not None or group_contiguous:
            ignore_n = -1
            for index in tuple(cls.indices(classification)):
                _, ignore_n = cls.ggggg(
                    classification, ignore_n,
                    coord, index,
                    group_span, group_contiguous
                )
            
        return classification

    
    @classmethod
    def ggggg(cls, classification,ignore_n,  coord, index, group_span, group_contiguous):
        """TODO"""
        group_span = self.group_span
        group_contiguous = self.group_contiguous

        ignore_group = False

        if group_span is not False and group_span is not None:
            bounds = bounds[index]
            lb = bounds[0, 0].get_data(_fill_value=False)
            ub = bounds[-1, 1].get_data(_fill_value=False)
                
            if coord.T:
                lb = lb.datetime_array.item()
                ub = ub.datetime_array.item()
            
            if not coord.increasing:
                lb, ub = ub, lb
                
            if group_span + lb != ub:
                self.classification[index] = self.ignore_n
                self.ignore_n -= 1
                return
                
        # Ignore a non-contiguous group
        if (
                not ignore_group 
                and self.group_contiguous
                and bounds is not None
                and not bounds.contiguous(
                    overlap=(self.group_contiguous == 2)
                )
        ):
            self.classification[index] = self.ignore_n
            self.ignore_n -= 1
            return

        # Still here?
        self.classification[index] = self.n
        self.n += 1
                
    @classmethod
    def discern_runs_within(cls, classification, coord):
        """Processes group classification for a 'within' collapse.

        """
        size = classification.size
        if size <= 1:
            return classification

        n = classification.max() + 1

        start = 0
        for i, c in enumerate(classification[: size - 1].tolist()):
            if c < 0:
                continue

            if not coord[i : i + 2].contiguous(overlap=False):
                classification[start : i + 1] = n
                start = i + 1
                n += 1

        return classification

    @classmethod
    def time_interval_over(cls,
        classification,
        n,
        coord,
        interval,
        lower,
        upper,
        lower_limit,
        upper_limit,
        group_by,
        extra_condition=None,
    ):
        """Prepares for a collapse over some TimeDuration.

        :Parameters:

            classification: `numpy.ndarray`

            n: `int`

            coord: `DimensionCoordinate`

            interval: `TimeDuration`

            lower: date-time

            upper: date-time

            lower_limit: date-time

            upper_limit: date-time

            group_by: `str`

            extra_condition: `Query`, optional

        :Returns:

            (`numpy.ndarray`, `int`)

        """
        group_by_coords = group_by == "coords"

        if coord.increasing:
            # Increasing dimension coordinate
            # lower, upper = interval.bounds(lower)
            upper = interval.interval(upper)[1]
            while lower <= upper_limit:
                lower, upper = interval.interval(lower)
                classification, n, lower, upper = cls.ddddd(
                    classification,
                    n,
                    lower,
                    upper,
                    True,
                    coord,
                    group_by_coords,
                    extra_condition,
                )
        else:
            # Decreasing dimension coordinate
            # lower, upper = interval.bounds(upper)
            lower = interval.interval(upper, end=True)[0]
            while upper >= lower_limit:
                lower, upper = interval.interval(upper, end=True)
                classification, n, lower, upper = cls.ddddd(
                    classification,
                    n,
                    lower,
                    upper,
                    False,
                    coord,
                    group_by_coords,
                    extra_condition,
                )

        return classification, n


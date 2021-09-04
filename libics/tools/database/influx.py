from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS, SYNCHRONOUS
import numpy as np

from libics.env import logging
from libics.core.data.sequences import DataSequence


###############################################################################


class InfluxDB(object):

    """
    InfluxDB v2 client.

    Parameters
    ----------
    host : `str`
        Database server IP.
    port : `int`
        Database server port.
    org : `str`
        InfluxDB organisation.
    token : `str`
        InfluxDB token. Needs read/write access.
    timeout : `str`
        Query timeout in seconds (s).
    asynchr : `bool`
        Whether to communicate synchronous or asynchronous.

    Attributes
    ----------
    buckets : `list(str)`
        List of bucket names in the database.
    measurements : `dict(str->list(str))`
        List of measurement values for each bucket.
    tags : `dict(str->list(str))`
        List of tag keys for each bucket.
        Excludes default keys like `_field, _value, etc.`
    fields : `dict(str->list(str))`
        List of field keys for each bucket.
    """

    LOGGER = logging.get_logger("libics.tools.database.influx.InfluxDB")

    def __init__(self, **kwargs):
        # Parameters
        self.host = "localhost"
        self.port = 8086
        self.org = None
        self.token = None
        self.timeout = 10
        self.asynchr = False
        for k, v in kwargs.items():
            if k[0] != "_":
                setattr(self, k, v)
        # InfluxDB
        self._client = None
        self._wapi = None
        self._qapi = None
        self._bapi = None
        self._buckets = None
        self._measurements = {}
        self._tags = {}
        self._fields = {}
        self._setup_influxdb()

    # +++++++++++++++++++++++++++++
    # Properties
    # +++++++++++++++++++++++++++++

    def __str__(self):
        s = [f"Server: {self.url}\nOrganisation: {self.org}"]
        for bucket in self.buckets:
            s.append(f"Bucket: {bucket}")
            s.append(f" → Measurements: {self.measurements[bucket]}")
            s.append(f" → Tags: {self.tags[bucket]}")
            s.append(f" → Fields: {self.fields[bucket]}")
        return "\n".join(s)

    @property
    def url(self):
        return f"http://{self.host}:{self.port:d}"

    @property
    def buckets(self):
        if self._buckets is None:
            self.update_buckets()
        return self._buckets

    @property
    def measurements(self):
        for bucket in self.buckets:
            if bucket not in self._measurements:
                self.update_measurements(bucket)
        return self._measurements

    @property
    def tags(self):
        for bucket in self.buckets:
            if bucket not in self._tags:
                self.update_tags(bucket)
        return self._tags

    @property
    def fields(self):
        for bucket in self.buckets:
            if bucket not in self._fields:
                self.update_fields(bucket)
        return self._fields

    # +++++++++++++++++++++++++++++
    # InfluxDB
    # +++++++++++++++++++++++++++++

    def _setup_influxdb(self):
        """Sets up the InfluxDB v2 client."""
        if self.token is None or self.org is None:
            raise ValueError("InfluxDB parameters not set")
        self._client = InfluxDBClient(
            self.url, self.token, org=self.org, timeout=self.timeout*1e3
        )
        self._wapi = self._client.write_api(
            write_options=ASYNCHRONOUS if self.asynchr else SYNCHRONOUS
        )
        self._qapi = self._client.query_api()
        self._bapi = self._client.buckets_api()
        self.update_buckets()

    def _write(self, bucket, record, **kwargs):
        """
        Writes to the database.

        Parameters
        ----------
        bucket : `str`
            InfluxDB bucket name.
        record : `str` or `Point`
            Flux write command or InfluxDBClient point.
        **kwargs
            Keyword arguments passed to the write API.
        """
        self.LOGGER.debug(
            record.to_line_protocol() if isinstance(record, Point) else record
        )
        return self._wapi.write(bucket, record=record, **kwargs)

    def _query(self, query, **kwargs):
        """
        Queries the database.

        Parameters
        ----------
        query : `str`
            Flux query command.
        **kwargs
            Keyword arguments passed to the query API.

        Returns
        -------
        result : `pd.DataFrame`
            InfluxDB tables as pandas data frame.
        """
        self.LOGGER.debug(query)
        return self._qapi.query_data_frame(query, **kwargs)

    def _read_buckets(self):
        """Queries the database for bucket names."""
        _buckets = self._bapi.find_buckets().buckets
        return sorted([_bucket.name for _bucket in _buckets])

    def _get_query_bucket(self, bucket):
        """Flux query string: choose bucket."""
        if bucket not in self.buckets:
            self.update_buckets()
            if bucket not in self.buckets:
                raise ValueError(f"invalid bucket: {bucket}")
        return f'from(bucket: "{bucket}")'

    def _get_query_range(self, start, stop):
        """Flux query string: choose time range."""
        if start is None:
            raise ValueError("start time not specified")
        q = f' |> range(start: {start}'
        if stop is None:
            q += ')'
        else:
            q += f', stop: {stop})'
        return q

    def _get_query_filter(self, operation="and", **tags):
        """
        Flux query string: filter on tag keys.

        Parameters
        ----------
        operation : `str`
            `"and", "or"`: Whether to apply logical and/or to the filter.
        **tags : `Any`
            Keys are tag key names; performs filtering for values.
            Can be used to filter i.a. `_measurement` and `_field`.
        """
        if operation not in ["and", "or"]:
            raise ValueError(f"invalid operation: {operation}")
        _qfilter = []
        for k, v in tags.items():
            if v is None:
                continue
            elif isinstance(v, str) or np.isscalar(v):
                v = [v]
            for vv in v:
                if isinstance(vv, str):
                    vv = f'"{vv}"'
                _qfilter.append(f'(r.{k} == {vv})')
        if len(_qfilter) == 0:
            q = ""
        else:
            q = f' |> filter(fn: (r) => {f" {operation} ".join(_qfilter)})'
        return q

    def _get_query_aggregate_window(self, window, function, rmv_nan):
        """Flux query string: window and aggregate Influx tables."""
        if window is None or function is None:
            q = ""
        else:
            create_empty = "false" if rmv_nan else "true"
            q = (f' |> aggregateWindow(every: {window}, '
                 f'fn: {function}, createEmpty: {create_empty})')
        return q

    # +++++++++++++++++++++++++++++
    # API
    # +++++++++++++++++++++++++++++

    def update_buckets(self):
        """Queries the database to update the buckets property."""
        self._buckets = self._read_buckets()

    def update_measurements(self, bucket):
        """Queries the database to update the measurements property."""
        q = f"""
        import "influxdata/influxdb/schema"
        schema.measurements(bucket: "{bucket}")
        """
        ds = self._query(q)
        self._measurements[bucket] = sorted(ds["_value"].tolist())

    def update_tags(self, bucket):
        """Queries the database to update the tags property."""
        q = f"""
        import "influxdata/influxdb/schema"
        schema.tagKeys(bucket: "{bucket}")
        """
        ds = self._query(q)
        tags = set(ds["_value"].tolist())
        for key in ["_start", "_stop", "_measurement", "_field"]:
            tags.discard(key)
        self._tags[bucket] = sorted(list(tags))

    def update_fields(self, bucket):
        """Queries the database to update the fields property."""
        q = f"""
        import "influxdata/influxdb/schema"
        schema.fieldKeys(bucket: "{bucket}")
        """
        ds = self._query(q)
        self._fields[bucket] = sorted(ds["_value"].tolist())

    def read_measurement_values(self, bucket, measurement):
        """Queries the database and gets all measurement values."""
        q = f"""
        import "influxdata/influxdb/schema"
        schema.measurementFieldKeys(
            bucket: "{bucket}",
            measurement: "{measurement}"
        )
        """
        ds = self._query(q)
        return sorted(ds["_value"].tolist())

    def write_point(
        self, bucket, measurement, tags=None, fields=None, time=None
    ):
        """
        Writes a point to the database.

        Parameters
        ----------
        bucket : `str`
            Bucket name.
        measurement : `str`
            Measurement value.
        tags : `dict(str->str)` or `None`
            Dictionary containing tag keys and values.
        fields : `dict(str->Any)` or `None`
            Dictionary containing field keys and values.
        time : `int` or `None`
            Timestamp in nanoseconds (ns).
            If `None`, uses the current time.
        """
        if bucket not in self.buckets:
            self.update_buckets()
            if bucket not in self.buckets:
                raise ValueError(f"invalid bucket: {bucket}")
        point = Point(measurement)
        if tags is not None:
            for k, v in tags.items():
                point = point.tag(k, v)
        if fields is not None:
            for k, v in fields.items():
                point = point.field(k, v)
        if time is not None:
            point = point.time(time)
        q = point.to_line_protocol()
        return self._write(bucket, q)

    def read_points(
        self, bucket, start="-1d", stop=None,
        window=None, function=None, rmv_nan=True,
        measurement=None, field=None, **tags
    ):
        """
        Queries the database.

        Parameters
        ----------
        bucket : `str`
            Bucket name.
        start, stop : `str`
            Extracted time range.
        window : `str`
            Time window per extracted point.
            E.g.: `"1m", "2h", ...`
        function : `str`
            Function applied to aggregated points.
            E.g.: `"mean", "median", "last", ...`.
        rmv_nan : `bool`
            Whether to remove windows containing no data.
        measurement : `list(str)`
            Filter for measurement values.
        field : `list(str)`
            Filter for field keys.
            Note that all extracted keys must have the same data type.
        **tags : `str`
            Filter for tag keys and values.

        Returns
        -------
        ds : `DataSequence`
            Data sequence with the following columns:
            `"time", "measurement", "field", "value", <tags>`.
        """
        # Construct query
        q = self._get_query_bucket(bucket)
        q += self._get_query_range(start, stop)
        q += self._get_query_filter(operation="or", _measurement=measurement)
        q += self._get_query_filter(operation="and", **tags)
        q += self._get_query_filter(operation="or", _field=field)
        q += self._get_query_aggregate_window(window, function, rmv_nan)
        q += " |> yield()"
        # Execute query
        ds = DataSequence(self._query(q))
        if len(ds) > 0:
            ds = ds.drop(columns=["result", "table", "_start", "_stop"])
            ds = ds.rename(columns={
                "_time": "time", "_measurement": "measurement",
                "_field": "field", "_value": "value"
            })
        return ds

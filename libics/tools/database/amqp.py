import collections
import json
import os
import pika
import random
import string
import time
from typing import Any, List
import uuid

from libics.env import logging, DIR_LIBICS
from libics.core import io
from libics.core.util import misc, path
from libics.core.io.iomisc import NumpyJsonEncoder


###############################################################################


class AmqpRemoteError(RuntimeError):
    pass


class AmqpLocalError(RuntimeError):
    pass


class AmqpReplyTimeoutError(RuntimeError):
    pass


###############################################################################


class AmqpConnection:

    """
    RPC base class via AMQP message queues.

    Parameters
    ----------
    config : `str` or `dict`
        Configuration file path or dictionary specifying below parameters.
        Duplicate parameters overwrite the `config` parameters.
    host : `str`
        Message broker server IP.
    port : `int`
        Message broker server port.
    credentials : `dict(str->str)`
        Message broker access credentials with items: `"username", "password"`.
    blocking : `bool`
        Whether to establish a blocking connection.
    """

    LOGGER = logging.get_logger("libics.tools.database.amqp.AmqpBase")
    CONFIG_PATH = os.path.join(DIR_LIBICS, "tools", "database", "amqp")

    @staticmethod
    def discover_configs(config_path=None):
        """
        Returns a list of all file names within the default configuration path.
        """
        if config_path is None:
            config_path = AmqpConnection.CONFIG_PATH
        return path.get_folder_contents(config_path).files

    @staticmethod
    def find_configs(config_path=None):
        AmqpConnection.LOGGER.warning(
            "find_configs DEPRECATED: use alias `discover_configs` instead"
        )
        return AmqpConnection.discover_configs(config_path=config_path)

    def __init__(self, config=None, **kwargs):
        # Parameters
        self.host = "localhost"
        self.port = 5672
        self.credentials = None
        self.blocking = False
        # Parse parameters
        if config is not None:
            _kwargs = kwargs
            if isinstance(config, str):
                if not os.path.exists(config):
                    config = os.path.join(self.CONFIG_PATH, config)
                if not os.path.exists(config):
                    raise ValueError(f"Invalid `config` ({str(config)})")
                kwargs = io.load(config)
            elif isinstance(config, collections.Mapping):
                kwargs = config
            else:
                raise ValueError("Invalid `config`")
            kwargs.update(_kwargs)
        for k, v in kwargs.items():
            if k[0] != "_":
                setattr(self, k, v)
        # pika
        self._connection = None
        self._channel = None
        self._queues = []
        self._is_running = False

    @property
    def credentials(self):
        return self._credentials

    @credentials.setter
    def credentials(self, val):
        if val is None:
            val = pika.PlainCredentials("guest", "guest")
        self._credentials = misc.assume_construct_obj(
            val, pika.PlainCredentials
        )

    def get_url(self, connect_status=True):
        if self.host is None:
            self.LOGGER.error("get_url failed because no host was set")
            return "ERR_NO_HOST"
        s = f"amqp://{self.host}"
        if self.port is not None:
            s = s + f":{self.port:d}"
        if self.credentials is not None:
            s = self.credentials.username + "@" + s
        if connect_status:
            c = f" [{'connected' if self.is_connected() else 'disconnected'}]"
            s += c
        return s

    def __str__(self):
        return f"AmqpConnection({self.get_url()})"

    def __repr__(self):
        return (
            f"<'{self.__class__.__name__}' at {hex(id(self))}> "
            + self.get_url()
        )

    def handle_error(self, *_, err_raise=logging.ERROR, err_msg=""):
        if isinstance(err_raise, Exception):
            raise err_raise(err_msg)
        else:
            self.LOGGER.log(level=err_raise, msg=err_msg)

    # +++++++++++++++++++++++++++++++++++++++
    # Setup and connection
    # +++++++++++++++++++++++++++++++++++++++

    def connect(self, callback=None):
        """Sets up an AMQP connection and channel."""
        conn_params = pika.ConnectionParameters(
            host=self.host, port=self.port, credentials=self.credentials
        )
        if self.blocking:
            self._connection = pika.BlockingConnection(parameters=conn_params)
            self._connect_channel(callback=callback)
        else:
            self._connection = pika.SelectConnection(
                parameters=conn_params,
                on_open_callback=lambda *_, **__: self._connect_channel(
                    *_, callback=callback, **__
                ),
                on_open_error_callback=lambda *args: self.handle_error(
                    *args, err_raise=RuntimeError,
                    err_msg=f"could not connect to {self.host}:{self.port:d}"
                )
            )

    def _connect_channel(self, *_, callback=None, **__):
        self._channel = self._connection.channel()
        if callback is not None:
            callback(self.is_connected())

    def close(self):
        """Closes the AMQP connection."""
        if self.is_connected():
            self._connection.close()
            self.join()
        self._channel = None
        self._connection = None

    def is_connected(self):
        """Returns whether the AMQP connection is established."""
        is_connected = (
            self._channel is not None
            and self._connection is not None
            and self._connection.is_open
        )
        return is_connected

    def run(self):
        """Starts the AMQP I/O loop."""
        self._is_running = True
        self._channel.start_consuming()

    def stop(self):
        """
        Stops the AMQP I/O loop.

        Note that this function is non-blocking;
        if applicable, call :py:meth:`join` afterwards.
        """
        if self.is_running():
            self._channel.stop_consuming()
        self._is_running = False

    def join(self):
        """Waits for the I/O loop to stop."""
        if not self.blocking:
            self._connection.ioloop.start()

    def is_running(self):
        """Returns whether the AMQP I/O loop is running."""
        return self._is_running

    def __getattr__(self, name):
        """Calls the AMQP channel methods."""
        return getattr(self._channel, name)


###############################################################################


def api_method(reply=False, timeout=1):
    """
    Decorator registering a method as API method.

    Parameters
    ----------
    ret : `bool`
        Whether to expect a reply value.
    timeout : `float`
        API reply timeout in seconds (s).
    """
    class api_func_generator:

        def __init__(self, func):
            self.func = func

        # Called after owner (class of method) creation
        def __set_name__(self, owner, name):
            owner.API_METHODS = owner.API_METHODS | {name}
            if reply:
                owner.API_REPLIES = owner.API_REPLIES | {name}
                owner.API_TIMEOUTS = owner.API_TIMEOUTS.copy()
                owner.API_TIMEOUTS[name] = timeout
            setattr(owner, name, self.func)
    return api_func_generator


class AmqpApiBase:

    """
    AMQP RPC API base class.

    Parameters
    ----------
    amqp_client : `AmqpConnection`
        AMQP client.
    instance_id : `str`
        Unique name of the API instance.

    Notes
    -----
    Usage:

    * Subclass this base class and set the class attribute :py:attr:`API_ID`
      to a unique API-identifying string.
    * Subclasses to a base API, which should use the same exchange as the
      parent API can be distinguished by setting the class attribute
      :py:attr:`API_SUB_ID`.
    * Set an appropriate :py:attr:`API_VERSION`.
    * The methods that should be accessible via the RPC API should be
      decorated with :py:func:`api_method`.
    * Using :py:class:`AmqpRpcFactory`, these API classes can be turned
      into an RPC server or client class.
    """

    LOGGER = logging.get_logger("libics.tools.database.amqp.AmqpApiBase")
    API_ID = "TEST"
    API_SUB_ID = "default"
    API_VERSION = "0.0.0"
    API_METHODS = set()     # set(str): Names of API methods
    API_REPLIES = set()     # set(str): Names of API methods requiring reply
    API_TIMEOUTS = dict()   # dict(str->float): Timeout in seconds (s)

    def __init__(self, instance_id="default", _random_id=None):
        # IDs
        self.instance_id = instance_id
        if _random_id is None:
            _random_id = "rnd_" + "".join(
                random.choices(string.ascii_lowercase, k=24)
            )
        self._random_id = _random_id
        # Callbacks
        self.cbs_on_entry = {}
        self.cbs_on_exit = {}

    def register_cb_on_entry(self, method_name, func):
        """
        Adds a callback function called before executing the API method.

        Parameters
        ----------
        method_name : `str`
            API method name.
        func : `callable`
            Callback function. Must accept the same arguments as the API method
            and may only raise `AmqpLocalError`.
        """
        if method_name not in self.API_METHODS:
            raise ValueError(f"invalid API method name: {method_name}")
        if not callable(func):
            raise ValueError(f"invalid callback function: {str(func)}")
        if method_name in self.cbs_on_entry:
            self.LOGGER.warning(
                f"Overwriting callback on entry for API method: {method_name}"
            )
        self.cbs_on_entry[method_name] = func

    def register_cb_on_exit(self, method_name, func):
        """
        Adds a callback function called after executing the API method.

        Parameters
        ----------
        method_name : `str`
            API method name.
        func : `callable`
            Callback function. Must accept the return value of the API method
            as arguments and may only raise `AmqpLocalError`.
        """
        if method_name not in self.API_METHODS:
            raise ValueError(f"invalid API method name: {method_name}")
        if not callable(func):
            raise ValueError(f"invalid callback function: {str(func)}")
        if method_name in self.cbs_on_exit:
            self.LOGGER.warning(
                f"Overwriting callback on exit for API method: {method_name}"
            )
        self.cbs_on_exit[method_name] = func

    @classmethod
    def get_exchange_id(cls):
        return cls.API_ID

    def get_queue_id(self):
        return ".".join([
            self.API_ID, self.API_SUB_ID, self.instance_id, self._random_id
        ])

    def get_routing_key(self):
        return ".".join([
            self.API_ID, self.API_SUB_ID, self.instance_id, "*"
        ])

    def _get_str(self):
        return (
            f" → API ID: {self.API_ID}.{self.API_SUB_ID}\n"
            f" → API version: {self.API_VERSION}\n"
            f" → API methods: {str(self.API_METHODS)}\n"
        )

    def __str__(self):
        return f"API: {self.__class__.__name__}\n" + self._get_str()

    def __repr__(self):
        return (
            f"<'{self.__class__.__name__}' at {hex(id(self))}>\n"
            + self._get_str()
        )

    # +++++++++++++++++++++++++++++++++++++++++
    # RPC API
    # +++++++++++++++++++++++++++++++++++++++++

    @api_method(reply=True)
    def ping(self) -> bool:
        """Returns `True`."""
        return True

    @api_method(reply=True)
    def ping_args(self, *args, **kwargs) -> Any:
        """Returns any passed parameters."""
        if len(args) == 0:
            if len(kwargs) == 0:
                return
            else:
                return kwargs
        elif len(args) == 1:
            args = args[0]
        if len(kwargs) == 0:
            return args
        else:
            return args, kwargs

    @api_method(reply=True)
    def get_api(self) -> List[str]:
        """Gets a list of all API method names."""
        return list(self.API_METHODS)

    @api_method(reply=True)
    def help(self, func_id: str) -> str:
        """Gets the docstring of the API method."""
        try:
            doc = getattr(self, func_id).__doc__
        except AttributeError:
            raise AmqpLocalError(f"{str(func_id)} is not a valid API method")
        doc = "" if doc is None else doc
        return doc


class AmqpRpcBase:

    """
    AMQP RPC base object.

    Notes
    -----
    AMQP specifications:

    * Messages (RPC calls) are sent to an AMQP topic exchange with the
      exchange name matching `API_ID`, by convention this should be all caps.
    * The AMQP routing keys have the form `API_ID.API_SUB_ID.instance_id.*`.
    * The AMQP queue instances have an additional random string attached,
      i.e., `API_ID.SUBAPI_ID.instance_id.random_id`. The random
      string is used to allow for multiple code instances to subscribe to
      the same API instance (e.g. one code instance executes the commands,
      another code instance logs the API traffic).

    Message specifications:

    ```
    {
        "__meta__": {
            "__rpc_version": "x.y.z",
            "__api_version": "x.y.z"
        },
        "func_id": "my_unique_function_identifier",
        "func_args": [
            arg0, arg1, ...
        ],
        "func_kwargs": {
            "kw0": kwarg0, "kw1": kwarg1, ...
        }
    }
    ```
    """

    RPC_VERSION = "0.0.0"
    API = None

    def __init__(self):
        self._api = None
        self._amqp_conn = None

    def setup_amqp(self, amqp_connection=None, blocking=True, **kwargs):
        """
        Sets up the AMQP connection object.

        If passing an `amqp_connection`, a separate connection with its
        parameters is created and is overwritten by `kwargs`.
        """
        _kwargs = {}
        if amqp_connection is not None:
            _kwargs["host"] = amqp_connection.host
            _kwargs["port"] = amqp_connection.port
            if amqp_connection.credentials:
                _kwargs["credentials"] = {
                    "username": amqp_connection.credentials.username,
                    "password": amqp_connection.credentials.password
                }
        _kwargs.update(kwargs)
        self._amqp_conn = AmqpConnection(blocking=blocking, **_kwargs)

    def setup_api(self, *args, api_object=None, **kwargs):
        """Sets up the API object."""
        if api_object is not None:
            self._api = api_object
        else:
            self._api = self.API(*args, **kwargs)

    def connect_amqp(self):
        """Establishes an AMQP connection."""
        if not self._amqp_conn.is_connected():
            self._amqp_conn.connect()

    def close_amqp(self):
        """Closes an AMQP connection."""
        self._amqp_conn.close()

    def __getattr__(self, name):
        if self._api is None:
            raise AttributeError("API has not been set up")
        return getattr(self._api, name)

    @classmethod
    def serialize_request(cls, func_id, *args, **kwargs):
        _meta = {
            "__rpc_version": cls.RPC_VERSION,
            "__api_version": cls.API.API_VERSION
        }
        _d = {
            "__meta__": _meta,
            "func_id": func_id,
            "func_args": args,
            "func_kwargs": kwargs
        }
        _msg = json.dumps(_d, cls=NumpyJsonEncoder)
        return _msg

    @classmethod
    def deserialize_request(cls, _msg):
        try:
            _d = json.loads(_msg)
            func_id = _d["func_id"]
            if "func_args" in _d:
                func_args = tuple(_d["func_args"])
            else:
                func_args = ()
            if "func_kwargs" in _d:
                func_kwargs = dict(_d["func_kwargs"])
            else:
                func_kwargs = {}
        except (json.decoder.JSONDecodeError, TypeError):
            raise RuntimeError(f"invalid JSON message: {str(_msg)}")
        except KeyError:
            raise RuntimeError(f"invalid RPC format: {str(_msg)}")
        return func_id, func_args, func_kwargs

    @classmethod
    def serialize_reply(cls, *args, err=False):
        _d = {"reply": args}
        if err:
            _d["error"] = str(err)
        try:
            _msg = json.dumps(_d, cls=NumpyJsonEncoder)
        except TypeError as e:
            _msg = json.dumps({"error": f"ENCODING_ERROR: {str(e)}"})
        return _msg

    @classmethod
    def deserialize_reply(cls, _msg):
        try:
            _d = json.loads(_msg)
        except json.decoder.JSONDecodeError:
            raise RuntimeError(f"invalid reply: {str(_msg)}")
        if "error" in _d:
            raise AmqpRemoteError(str(_d['error']))
        try:
            ret = _d["reply"]
            if len(ret) == 1:
                ret = ret[0]
        except (KeyError, TypeError):
            raise RuntimeError(f"invalid RPC reply format: {_msg}")
        return ret

    def local_dispatcher(self, channel, method, properties, body):
        """Local dispatcher used by RPC server."""
        # Initialize variables
        ret = None
        func_errs = []
        # Process message
        try:
            func_id, func_args, func_kwargs = self.deserialize_request(body)
            try:
                func = getattr(self._api, func_id)
            except AttributeError:
                self.LOGGER.error(f"local method `{func_id}` unavailable")
                channel.basic_ack(delivery_tag=method.delivery_tag)
                return
        except RuntimeError as e:
            self.LOGGER.error(f"local dispatch error: {str(e)}")
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return
        # Execute callback on entry
        if func_id in self._api.cbs_on_entry:
            try:
                self._api.cbs_on_entry[func_id](*func_args, **func_kwargs)
            except RuntimeError as e:
                self.LOGGER.error(
                    f"error during callback on entry for `{func_id}`: {str(e)}"
                )
                func_errs.append(f"CB_ENTRY_ERROR: {str(e)}")
                # For server-side debugging
                import traceback
                traceback.print_exc()
        # Execute function
        try:
            ret = func(*func_args, **func_kwargs)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except (RuntimeError, TypeError, ValueError) as e:
            self.LOGGER.error(
                f"error executing local method `{func_id}`: {str(e)}"
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)
            ret = None
            func_errs.append(f"FUNCTION_ERROR: {str(e)}")
            # For server-side debugging
            import traceback
            traceback.print_exc()
        # Execute callback on exit
        if func_id in self._api.cbs_on_exit:
            try:
                self._api.cbs_on_exit[func_id](ret)
            except RuntimeError as e:
                self.LOGGER.error(
                    f"error during callback on exit for `{func_id}`: {str(e)}"
                )
                func_errs.append(f"CB_EXIT_ERROR: {str(e)}")
                # For server-side debugging
                import traceback
                traceback.print_exc()
        # Reply
        if properties.reply_to:
            if len(func_errs) == 0:
                func_err = None
            else:
                func_err = "; ".join(func_errs)
            _msg = self.serialize_reply(ret, err=func_err)
            channel.basic_publish(
                self.get_exchange_id(),
                properties.reply_to,
                _msg
            )
        elif func_id in self._api.API_REPLIES:
            self.LOGGER.warning(
                f"no reply queue received for func_id: `{func_id}`"
            )

    def remote_dispatcher(self, func_id, *args, **kwargs):
        """Remote dispatcher used by RPC client."""
        # Create JSON message
        _msg = self.serialize_request(func_id, *args, **kwargs)
        self.LOGGER.debug(f"remote_dispatcher: {_msg}")
        # If request only
        if func_id not in self._api.API_REPLIES:
            try:
                self._amqp_conn.basic_publish(
                    self.get_exchange_id(), self.get_routing_key(), _msg
                )
            except pika.exceptions.StreamLostError:
                self.LOGGER.warning("Lost AMQP connection, reconnecting")
                self.connect_amqp()
                self._amqp_conn.basic_publish(
                    self.get_exchange_id(), self.get_routing_key(), _msg
                )
        # If request and reply
        else:
            _cor_id = str(uuid.uuid4())
            try:
                _reply_queue = self._amqp_conn.queue_declare(
                    "", exclusive=True
                )
            except pika.exceptions.StreamLostError:
                self.LOGGER.warning("Lost AMQP connection, reconnecting")
                self.connect_amqp()
                _reply_queue = self._amqp_conn.queue_declare(
                    "", exclusive=True
                )
            self._amqp_conn.queue_bind(
                _reply_queue.method.queue,
                self.get_exchange_id()
            )
            _props = pika.BasicProperties(
                reply_to=_reply_queue.method.queue,
                correlation_id=_cor_id
            )
            # Sent request
            self._amqp_conn.basic_publish(
                self.get_exchange_id(), self.get_routing_key(), _msg,
                properties=_props
            )
            # Wait for reply
            _timeout = 1
            if func_id in self._api.API_TIMEOUTS:
                _timeout = self._api.API_TIMEOUTS[func_id]
            _sleep_time = min(0.1, _timeout / 50)
            _start_time = time.time()
            while time.time() - _start_time < _timeout:
                r_method, r_props, r_body = self._amqp_conn.basic_get(
                    _reply_queue.method.queue
                )
                if r_method is not None:
                    self._amqp_conn.basic_ack(r_method.delivery_tag)
                    self._amqp_conn.queue_delete(_reply_queue.method.queue)
                    return self.deserialize_reply(r_body)
                time.sleep(_sleep_time)
            self._amqp_conn.queue_delete(_reply_queue.method.queue)
            raise AmqpReplyTimeoutError(f"func_id: `{func_id}`")


def _remote_method_factory(name):
    def _remote_method(obj, *args, **kwargs):
        return obj.remote_dispatcher(name, *args, **kwargs)
    return _remote_method


class AmqpRpcFactory:

    """
    Factory class that turns API classes into RPC servers/clients.

    Methods
    -------
    make_rpc_server
        Generates an `AmqpRpcServer` class from an API class.
    make_rpc_client
        Generates an `AmqpRpcClient` class from an API class.
    """

    @classmethod
    def make_rpc_server(cls, Api: AmqpApiBase, cls_name: str):
        """
        Creates an AMQP RPC server class.

        Accepts API method calls via an AMQP broker from a remote client.
        """
        # Subclass RPC base class
        class AmqpRpcServer(AmqpRpcBase):
            LOGGER = logging.get_logger(
                "libics.tools.database.amqp.AmqpRpcServer"
            )
            API = Api

            def connect_amqp(self):
                super().connect_amqp()
                self._amqp_conn.exchange_declare(
                    self.get_exchange_id(), exchange_type="topic"
                )
                self._amqp_conn.queue_declare(self.get_queue_id())
                self._amqp_conn.queue_purge(self.get_queue_id())
                self._amqp_conn.queue_bind(
                    self.get_queue_id(), self.get_exchange_id(),
                    routing_key=self.get_routing_key()
                )
                self._amqp_conn.basic_consume(
                    self.get_queue_id(), self.local_dispatcher,
                    exclusive=True
                )

            def close_amqp(self):
                self._amqp_conn.queue_delete(self.get_queue_id())
                super().close_amqp()

            def run(self):
                return self._amqp_conn.run()

            def stop(self):
                return self._amqp_conn.stop()

            def join(self):
                return self._amqp_conn.join()

        # Set docs
        AmqpRpcServer.__name__ = cls_name
        AmqpRpcServer.__doc__ = f"AMQP RPC server for {Api.__name__}"
        if Api.__doc__:
            AmqpRpcServer.__doc__ += "\n\n" + Api.__doc__
        return AmqpRpcServer

    @classmethod
    def make_rpc_client(cls, Api: AmqpApiBase, cls_name: str):
        """
        Creates an AMQP RPC client class.

        API method calls are sent via an AMQP broker to a remote server.
        """
        # Subclass RPC base class
        class AmqpRpcClient(AmqpRpcBase):
            LOGGER = logging.get_logger(
                "libics.tools.database.amqp.AmqpRpcClient"
            )
            API = Api

            def connect_amqp(self):
                super().connect_amqp()
                self._amqp_conn.exchange_declare(
                    self.get_exchange_id(), exchange_type="topic"
                )

        # Create RPC methods
        for func_id in list(Api.API_METHODS):
            _remote_method = _remote_method_factory(func_id)
            _remote_method.__doc__ = getattr(Api, func_id).__doc__
            setattr(AmqpRpcClient, func_id, _remote_method)
        # Set docs
        AmqpRpcClient.__name__ = cls_name
        AmqpRpcClient.__doc__ = f"AMQP RPC client for {Api.__name__}"
        if Api.__doc__:
            AmqpRpcClient.__doc__ += "\n\n" + Api.__doc__
        return AmqpRpcClient

    @classmethod
    def make_dynamic_rpc_client(
        cls, api_id, api_sub_id="default", instance_id="default",
        cls_name=None
    ):
        """
        Creates an AMQP RPC client class.

        API method calls are sent via an AMQP broker to a remote server.

        Parameters
        ----------
        api_id, api_sub_id, instance_id : `str`
            API ID, API sub ID and instance ID determining the AMQP
            exchange from which to obtain the API specifications.
        """
        class AmqpDynamicApi(AmqpApiBase):
            API_ID = api_id
            API_SUB_ID = api_sub_id

        class AmqpDynamicRpcClient(AmqpRpcBase):
            """
            Dynamically generated AMQP RPC client.

            Uses a running RPC server to obtain the API specifications and
            to automatically construct a client for this API locally.

            Usage:

            * Construct object.
            * Set up the AMQP connection and connect to the message broker.
            * Set up the API by passing the IDs.
            """

            LOGGER = logging.get_logger(
                "libics.tools.database.amqp.AmqpDynamicRpcClient"
            )
            API = AmqpDynamicApi

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._fetched_api_docs = {}
                self._fetched_api_signatures = {}

            def _fetch_api(self):
                func_ids = self.remote_dispatcher("get_api")
                self._api.API_METHODS = set(func_ids)

            def _fetch_docstrings(self):
                for func_id in self._api.API_METHODS:
                    self._fetched_api_docs[func_id] = (
                        self.remote_dispatcher("help", func_id)
                    )

            def _fetch_signatures(self):
                for func_id in self._api.API_METHODS:
                    self._fetched_api_signatures[func_id] = (
                        self.remote_dispatcher("get_api_signature", func_id)
                    )

            def _construct_api(self):
                for func_id in list(self._api.API_METHODS):
                    _remote_method = _remote_method_factory(func_id)
                    if func_id in self._fetched_api_docs:
                        _remote_method.__doc__ = self._fetched_api_docs[func_id]
                    setattr(AmqpDynamicRpcClient, func_id, _remote_method)

            def setup_api(
                self, instance_id=instance_id,
                fetch_docstrings=False, fetch_signatures=False
            ):
                """
                Constructs this client using the remote API.

                Parameters
                ----------
                fetch_docstrings, fetch_signatures : `bool`
                    Whether to fetch the API method docstring/type signature.
                    Note that this might be slow.
                """
                # Construct API object
                self._api = self.API(instance_id=instance_id)
                # Establish AMQP exchange
                self._amqp_conn.exchange_declare(
                    self.get_exchange_id(), exchange_type="topic"
                )
                # Fetch API from remote
                self._fetch_api()
                if fetch_docstrings:
                    self._fetch_docstrings()
                if fetch_signatures:
                    self._fetch_signatures()
                self._construct_api()

        if cls_name:
            AmqpDynamicRpcClient.__name__ = cls_name
        return AmqpDynamicRpcClient


###############################################################################


if __name__ == "__main__":

    import sys
    run_server = (len(sys.argv) > 1 and sys.argv[1].lower() == "server")

    print(f"Testing {'server' if run_server else 'client'}")

    class TestApi(AmqpApiBase):

        LOGGER = logging.get_logger("TestApi")
        API_ID = "TEST"
        API_VERSION = "0.0.0"

        @api_method()
        def test_server(self, *args, **kwargs):
            """Tests the server response."""
            print("test_server(")
            if len(args) > 0:
                print("   ", ", ".join([
                    str(arg) for arg in args
                ]))
            if len(kwargs) > 0:
                print("   ", ", ".join([
                    f"{str(k)}={str(v)}" for k, v in kwargs.items()
                ]))
            print(")")

    TestServer = AmqpRpcFactory.make_rpc_server(
        TestApi, "TestServer"
    )
    TestClient = AmqpRpcFactory.make_rpc_client(
        TestApi, "TestClient"
    )

    test_conn_param = dict(
        host="localhost",
        port=5672,
        credentials=dict(username="guest", password="guest")
    )

    # Test server code
    if run_server:
        test_server = TestServer()
        print("Setting up...")
        test_server.setup_api()
        test_server.setup_amqp(**test_conn_param)
        print("Connecting...")
        test_server.connect_amqp()
        try:
            print("\nRunning...\n")
            test_server.run()
        except KeyboardInterrupt:
            print("Closing...")
            test_server.stop()
            test_server.close_amqp()

    # Test client code
    else:
        test_client = TestClient()
        # test_client.LOGGER.setLevel(logging.DEBUG)
        print("Setting up...")
        test_client.setup_api()
        test_client.setup_amqp(**test_conn_param)
        print("Connecting...")
        test_client.connect_amqp()
        print("Calling `ping()`")
        print("Return:", test_client.ping())
        print("Calling `test_server()`")
        print("Return:", test_client.test_server())
        print("Calling `test_server(123, 456, asdf='ghjk')`")
        print("Return:", test_client.test_server(123, 456, asdf="ghjk"))
        print("Calling `get_api()`")
        print("Return:", test_client.get_api())
        print("Calling `help('ping_args')`")
        print("Return:", test_client.help("ping_args"))
        print("Closing...")
        test_client.close_amqp()

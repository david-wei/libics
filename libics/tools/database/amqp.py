import collections
import json
import numpy as np
import os
import pika
import random
import string

from libics.env import logging, DIR_LIBICS
from libics.core import io
from libics.core.util import misc, path


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
    def find_configs(config_path=None):
        """
        Returns a list of all file names within the default configuration path.
        """
        if config_path is None:
            config_path = AmqpConnection.CONFIG_PATH
        return path.get_folder_contents(config_path).files

    def __init__(self, config=None, **kwargs):
        # Parameters
        self.host = "localhost"
        self.port = 15672
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
        self._is_connected = False
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
        self._is_connected = True
        if callback is not None:
            callback(self.is_connected())

    def close(self):
        """Closes the AMQP connection."""
        if self.is_connected():
            self._connection.close()
            self.join()
        self._is_connected = False
        self._channel = None
        self._connection = None

    def is_connected(self):
        """Returns whether the AMQP connection is established."""
        return np.all([
            self._is_connected,
            self._channel is not None,
            self._connection is not None
        ])

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


def api_method(reply=False):
    """
    Decorator registering a method as API method.

    Parameters
    ----------
    ret : `bool`
        Whether to expect a reply value.
    """
    class api_func_generator:

        def __init__(self, func):
            self.func = func

        # Called after owner (class of method) creation
        def __set_name__(self, owner, name):
            owner.API_METHODS.add(name)
            if reply:
                owner.API_REPLIES.add(name)
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
    API_METHODS = set()
    API_REPLIES = set()

    def __init__(self, instance_id="default", _random_id=None):
        self.instance_id = instance_id
        if _random_id is None:
            _random_id = "rnd_" + "".join(
                random.choices(string.ascii_lowercase, k=24)
            )
        self._random_id = _random_id

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

    @api_method(reply=True)
    def get_api_methods(self):
        """Gets a list of all API method names."""
        return self.API_METHODS


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
        },
        "callback": {
            "queue": "callback_queue",
            "func_id": "callback_func_id"
        }
    }
    ```
    """

    RPC_VERSION = "0.0.0"
    API = None

    def __init__(self):
        self._api = None
        self._amqp_conn = None

    def setup_amqp(self, blocking=True, **kwargs):
        """Sets up the AMQP connection object."""
        self._amqp_conn = AmqpConnection(blocking=blocking, **kwargs)

    def setup_api(self, *args, **kwargs):
        """Sets up the API object."""
        self._api = self.API(*args, **kwargs)

    def connect_amqp(self):
        """Establishes an AMQP connection."""
        if not self._amqp_conn.is_connected():
            self._amqp_conn.connect()

    def close_amqp(self):
        """Closes an AMQP connection."""
        self._amqp_conn.close()

    def __getattr__(self, name):
        return getattr(self._api, name)

    @classmethod
    def serialize(cls, func_id, *args, **kwargs):
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
        _msg = json.dumps(_d)
        return _msg

    @classmethod
    def deserialize(cls, _msg):
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
            # TODO: handle callback
        except (json.decoder.JSONDecodeError, TypeError):
            raise RuntimeError(f"invalid JSON message: {str(_msg)}")
        except KeyError:
            raise RuntimeError(f"invalid RPC format: {str(_msg)}")
        return func_id, func_args, func_kwargs

    def local_dispatcher(self, channel, method, properties, body):
        """Local dispatcher used by RPC server."""
        try:
            func_id, func_args, func_kwargs = self.deserialize(body)
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
        try:
            _ = func(*func_args, **func_kwargs)
            channel.basic_ack(delivery_tag=method.delivery_tag)
            # TODO: handle return, e.g. perform RPC callback etc.
        except (RuntimeError, TypeError) as e:
            self.LOGGER.error(
                f"error executing local method `{func_id}`: {str(e)}"
            )
            channel.basic_ack(delivery_tag=method.delivery_tag)
            return

    def remote_dispatcher(self, func_id, *args, **kwargs):
        """Remote dispatcher used by RPC client."""
        _msg = self.serialize(func_id, *args, **kwargs)
        self.LOGGER.debug(f"remote_dispatcher: {_msg}")
        self._amqp_conn.basic_publish(
            self.get_exchange_id(), self.get_routing_key(), _msg
        )
        # TODO: handle callback


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
    def make_rpc_server(cls, Api, cls_name):
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
    def make_rpc_client(cls, Api, cls_name):
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
        for func_id in Api.API_METHODS:
            def _remote_method(self, *args, **kwargs):
                return self.remote_dispatcher(func_id, *args, **kwargs)
            _remote_method.__doc__ = getattr(Api, func_id).__doc__
            setattr(AmqpRpcClient, func_id, _remote_method)
        # Set docs
        AmqpRpcClient.__name__ = cls_name
        AmqpRpcClient.__doc__ = f"AMQP RPC client for {Api.__name__}"
        if Api.__doc__:
            AmqpRpcClient.__doc__ += "\n\n" + Api.__doc__
        return AmqpRpcClient


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
        test_client.LOGGER.setLevel(logging.DEBUG)
        print("Setting up...")
        test_client.setup_api()
        test_client.setup_amqp(**test_conn_param)
        print("Connecting...")
        test_client.connect_amqp()
        print("Calling `test_server`")
        test_client.test_server()
        print("Calling `test_server(123, 456, asdf='ghjk')`")
        test_client.test_server(123, 456, asdf="ghjk")
        print("Closing...")
        test_client.close_amqp()

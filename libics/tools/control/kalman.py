import numpy as np


###############################################################################


class KalmanFilter:

    """
    Implements a scalar or 1D Kalman filter (TODO: n-dim).

    Parameters
    ----------
    initial_estimate, initial_covariance : `float` or `Array[float]`
        Initial state estimate, and its covariance, respectively.
    process_model, process_covariance : `float` or `Array[float]`
        Mapping from previous to subsequent state (between iterations),
        and its covariance, respectively
    observation_model, observation_covariance : `float` or `Array[float]`
        Mapping from state to observation space,
        and the covariance in observation space, respectively.
    control_model : `float` or `Array[float]`
        Mapping from control to state space.

    Attributes
    ----------
    state_observations : `np.ndarray(float)`
        Sequence of observations with dimensions: `[iterations, {data}]`.
    state_estimates : `np.ndarray(float)`
        Sequence of state estimates with dimensions: `[iterations, {data}]`.
    state_covariances : `np.ndarray(float)`
        Sequence of state covariances with dimensions:
        `[iterations, {data}, {data}]`.
    """

    def __init__(self, initial_estimate=0, initial_covariance=1, **kwargs):
        self._process_model = 1
        self._process_covariance = 1
        self._observation_model = 1
        self._observation_covariance = 1
        self._control_model = 1
        self._state_observations = [initial_estimate]
        self._state_estimates = [initial_estimate]
        self._state_covariances = [initial_covariance]
        # Parse parameters
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def process_model(self):
        return self._process_model

    @process_model.setter
    def process_model(self, val):
        self._process_model = np.array(val)

    @property
    def observation_model(self):
        return self._observation_model

    @observation_model.setter
    def observation_model(self, val):
        self._observation_model = np.array(val)

    @property
    def control_model(self):
        return self._control_model

    @control_model.setter
    def control_model(self, val):
        self._control_model = np.array(val)

    @property
    def process_covariance(self):
        return self._process_covariance

    @process_covariance.setter
    def process_covariance(self, val):
        self._process_covariance = np.array(val)

    @property
    def observation_covariance(self):
        return self._observation_covariance

    @observation_covariance.setter
    def observation_covariance(self, val):
        self._observation_covariance = np.array(val)

    @property
    def state_observations(self):
        return np.array(self._state_observations)

    @property
    def state_estimates(self):
        return np.array(self._state_estimates)

    @property
    def state_covariances(self):
        return np.array(self._state_covariances)

    def add_observation(self, observation, control=None):
        """
        Adds an observation iteration.

        Parameters
        ----------
        observation : `float` or `Array[float]`
            Data in observation space in this iteration.
        control : `None` or `float` or `Array[float]`
            Control value applied prior to the observation in this iteration.
            If `None`, no control is applied.

        Returns
        -------
        state_estimate : `float` or `Array[float]`
            Posterior state estimate.
        """
        observation = np.array(observation)
        if control is not None:
            control = np.array(control)
        # Observation
        self._state_observations.append(observation)
        # Prediction
        _state_estimate_prior = np.dot(
            self._process_model, self._state_estimates[-1]
        )
        if control is not None:
            _state_estimate_prior = (
                _state_estimate_prior + np.dot(self._control_model, control)
            )
        _state_covariance_prior = np.dot(
            np.dot(self._process_model, self._state_covariances[-1]),
            np.transpose(self._process_model)
        ) + self._process_covariance
        # Update
        _innovation_estimate = observation - np.dot(
            self._observation_model, _state_estimate_prior
        )
        _innovation_covariance = np.dot(
            np.dot(self._observation_model, _state_covariance_prior),
            np.transpose(self._observation_model)
        ) + self._observation_covariance
        _inverse_innovation_covariance = (
            1 / _innovation_covariance if _innovation_covariance.ndim == 0
            else np.linalg.inv(_innovation_covariance)
        )
        _kalman_gain = np.dot(np.dot(
            _state_covariance_prior, np.transpose(self._observation_model)
        ), _inverse_innovation_covariance)
        # Correction
        _state_estimate_posterior = (
            _state_estimate_prior + np.dot(_kalman_gain, _innovation_estimate)
        )
        _state_covariance_posterior = np.dot(
            1 - np.dot(_kalman_gain, self._observation_model),
            _state_covariance_prior
        )
        self._state_estimates.append(_state_estimate_posterior)
        self._state_covariances.append(_state_covariance_posterior)
        return _state_estimate_posterior

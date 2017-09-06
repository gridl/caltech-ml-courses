"""Logger to log metrics during training"""
import json

# TODO: should be able to continue some logs
# TODO: SubLog.log should create metric if not exists previously
# TODO: metric acces should be easy. They should be created in some python
#       way as attributes and also as __getitem__ method


class SubLog:
    """Logger for some part of model - training, validation, test"""

    def __init__(self, name, logged_metrics=[]):
        """
        Args:
            name(str): name of the sub log
            logged_metrics(list): list of str of logged metrics, like
                ['accuracy', 'loss']
        """
        self._name = name
        self._metrics = {}
        for metric_name in logged_metrics:
            metric = []
            self._metrics[metric_name] = metric
            setattr(self, metric_name, metric)

    def __repr__(self):
        return 'SubLog: %s, metrics: %s' % (self._name, self.metrics_names)

    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._metrics

    @property
    def metrics_names(self):
        return list(self._metrics.keys())

    def __getitem__(self, key):
        return self._metrics[key]

    def add_metric(self, name):
        """Add metric for future monitoring
        """
        if name not in self._metrics:
            metric = []
            self._metrics[name] = metric
            setattr(self, name, metric)

    def log(self, name, value, step=None):
        """
        Args:
            name(str): name of the metric
            value(float or int): value of the metric
            step(int): optional step
        """
        if name not in self._metrics:
            self.add_metric(name)
        metric_to_log = self._metrics[name]
        if step is None:
            step = len(metric_to_log) - 1
        metric_to_log.append((step, value))


class MainLog:
    """Main logger for the full model"""

    def __init__(self, logged_states, logged_metrics=[]):
        """
        Args:
            logged_states(list): list of str of logged states, like
                ['train', 'val', 'test']
            logged_metrics(list): list of str of logged metrics, like
                ['accuracy', 'loss']
        """
        self.states = logged_states
        self._sub_logs = {}
        for state_name in logged_states:
            sub_log = SubLog(state_name, logged_metrics)
            self._sub_logs[state_name] = sub_log
            setattr(self, state_name, sub_log)

    @property
    def dict_logs(self):
        logs_dict = {sub_log.name: sub_log.data for sub_log in
                     self._sub_logs.values()}
        return logs_dict

    def __getitem__(self, key):
        return self._sub_logs[key]

    def dump_logs(self, path):
        with open(str(path), 'w') as f:
            json.dump(self.dict_logs, f, indent=2)

    def add_metric(self, name):
        """
        Add metric for future monitoring by all sub logs
        Args:
            name(str): name of new metric
        """
        for sub_log in self._sub_logs:
            sub_log.add_metric(name)

    def log(self, state, name, value, step=None):
        """
        Args:
            state(str): state that should be logged(train, test)
            name(str): name of the metric
            value(float or int): value of the metric
            step(int): optional step
        """
        getattr(self, state).log(name, value, step)

import deepsense.neptune as neptune
import tensorflow as tf


class TensorBoardLogger(object):
    def __init__(self, log_dir):
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, name, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self.writer.add_summary(summary, step)


class NeptuneLogger(object):
    def __init__(self, context):
        self.context = context
        self.channels = {}

    def add_channel(self, name):
        channel = self.context.job.create_channel(name=name, channel_type=neptune.ChannelType.NUMERIC)
        self.context.job.create_chart(name=name, series={name: channel})
        self.channels[name] = channel

    def log_scalar(self, name, value, step):
        if name not in self.channels:
            self.add_channel(name)
        self.channels[name].send(x=step, y=value)


class AggregatedLogger(object):
    def __init__(self, loggers):
        self.loggers = loggers

    def log_scalar(self, name, value, step):
        for logger in self.loggers:
            logger.log_scalar(name, value, step)

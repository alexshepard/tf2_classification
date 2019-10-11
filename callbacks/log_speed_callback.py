import tensorflow as tf
import datetime

class LogStepsPerSecCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_batch_freq, steps_per_epoch):
        self.last_batch_end_timestamp = None
        self.update_batch_freq = update_batch_freq
        self.steps_per_epoch = steps_per_epoch

    def on_batch_end(self, batch, logs=None):
        prev_batch_end = self.last_batch_end_timestamp
        now = datetime.datetime.now()
        if batch % self.update_batch_freq == 0:
            # log the calculated steps/sec
            if prev_batch_end != None:
                seconds_for_this_batch = (now - prev_batch_end).total_seconds()
                steps_per_second = 1. / seconds_for_this_batch
                log_step = (self.steps_per_epoch * self.current_epoch) + batch
                tf.summary.scalar('steps_per_sec', data=steps_per_second, step=log_step)
                tf.summary.flush()

        self.last_batch_end_timestamp = now
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

import tensorflow as tf
import optuna


class TensorFlowPruningHook(tf.train.SessionRunHook):

    def __init__(self, trial, estimator, metric, is_higher_better, run_every_steps):
        self.trial = trial
        self.estimator = estimator
        self.current_step = -1
        self.metric = metric
        self.is_higher_better = is_higher_better
        self._timer = tf.train.SecondOrStepTimer(every_secs=None, every_steps=run_every_steps)

    def after_run(self, run_context, run_values):
        global_step = run_values.results
        # Get eval metrics every n steps
        if self._timer.should_trigger_for_step(global_step):
            eval_metrics = tf.contrib.estimator.read_eval_metrics(self.estimator.eval_dir())
        else:
            eval_metrics = None
        if eval_metrics:
            step = next(reversed(eval_metrics))
            latest_eval_metrics = eval_metrics[step]
            # If there exists a new evaluation summary
            if step > self.current_step:
                if self.is_higher_better:
                    current_score = 1.0 - latest_eval_metrics[self.metric]
                else:
                    current_score = latest_eval_metrics[self.metric]
                self.trial.report(current_score, step=step)
                self.current_step = step
            if self.trial.should_prune(self.current_step):
                message = "Trial was pruned at iteration {}.".format(self.current_step)
                raise optuna.structs.TrialPruned(message)

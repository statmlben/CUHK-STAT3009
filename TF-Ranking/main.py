import tensorflow as tf
import tensorflow_ranking as tfr

_TRAIN_DATA_PATH="/data/train.txt"
_TEST_DATA_PATH="/data/test.txt"
_LOSS="approx_ndcg_loss"
_N_ASSETS=100
_N_FEATURES=16
_BATCH_SIZE=32
_HIDDEN_LAYER_DIMS=["20", "10"]


# Input Pipeline
def input_fn(path):
    data = tf.data.Dataset.from_generator(
        tfr.data.libsvm_generator(path, _N_FEATURES, _N_ASSETS),
        output_types=({str(k): tf.float32 for k in range(1,_N_FEATURES+1)}, tf.float32),
        output_shapes=(
            {str(k): tf.TensorShape([_N_ASSETS, 1]) for k in range(1,_N_FEATURES+1)},
            tf.TensorShape([_N_ASSETS])
        )
    )

    data = data.shuffle(1000).repeat().batch(_BATCH_SIZE)
  
    return data.make_one_shot_iterator().get_next()


def example_feature_columns():
    """Returns the example feature columns."""
    
    feature_names = ["%d" % (i + 1) for i in range(0, _N_FEATURES)]
    
    return {name: tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
            for name in feature_names}


# Scoring Function
def make_score_fn():
    """Returns a scoring function to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score assets."""
        
        del params
        del config
        # Define input layer.
        example_input = [
            tf.layers.flatten(group_features[name])
            for name in sorted(example_feature_columns())
        ]
        input_layer = tf.concat(example_input, 1)

        cur_layer = input_layer
        for i, layer_width in enumerate(int(d) for d in _HIDDEN_LAYER_DIMS):
            cur_layer = tf.layers.dense(
                cur_layer,
                units=layer_width,
                activation="tanh")

        logits = tf.layers.dense(cur_layer, units=1)
        
        return logits

    return _score_fn


# Evaluation Metric
def eval_metric_fns():
	"""Returns a dict from name to metric functions.
	Returns:
	A dict mapping from metric name to a metric function with above signature.
	"""

	metric_fns = {}
	metric_fns.update({
		"metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
			tfr.metrics.RankingMetricKey.NDCG, topn=topn)
		for topn in [1, 3, 5, 10]
	})

	return metric_fns


# Estimator
def get_estimator(hparams):
	"""Create a ranking estimator.
	Args:
	hparams: (tf.contrib.training.HParams) a hyperparameters object.
	Returns:
	tf.learn `Estimator`.
	"""

	def _train_op_fn(loss):
		"""Defines train op used in ranking head."""
		return tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.train.get_global_step(),
			learning_rate=hparams.learning_rate,
			optimizer="Adagrad")

	ranking_head = tfr.head.create_ranking_head(
		loss_fn=tfr.losses.make_loss_fn(_LOSS),
		eval_metric_fns=eval_metric_fns(),
		train_op_fn=_train_op_fn)

	return tf.estimator.Estimator(
		model_fn=tfr.model.make_groupwise_ranking_fn(
			group_score_fn=make_score_fn(),
			group_size=1,
			transform_fn=None,
			ranking_head=ranking_head),
		params=hparams)


# Initialize estimator
hparams = tf.contrib.training.HParams(learning_rate=0.05)
ranker = get_estimator(hparams)

# Train model
ranker.train(input_fn=lambda: input_fn(_TRAIN_DATA_PATH), steps=100)

# Evaluate model
ranker.evaluate(input_fn=lambda: input_fn(_TEST_DATA_PATH), steps=100)

# Visualize
ranker.model_dir

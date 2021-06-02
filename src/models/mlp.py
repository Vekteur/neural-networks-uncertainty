import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
	Layer, Input, Flatten, Dense, Concatenate,
	Dropout, Reshape
)

class MCDropout(Dropout):
	def call(self, inputs):
		return super().call(inputs, training=True)


def mlp(input_size=1, output_size=1, mixture_size=1, units_size=50, nb_hidden=1, 
		dropout_rate=0, add_std_output=False, dense_layer=Dense):

	inp = Input(shape=[input_size])
	x = inp
	x = Flatten()(x)
	for _ in range(nb_hidden):
		x = dense_layer(units_size, activation='relu')(x)
		if dropout_rate != 0:
			x = MCDropout(rate=dropout_rate)(x)

	mean = dense_layer(output_size * mixture_size)(x)
	output_layers = [mean]
	if add_std_output:
		std = dense_layer(output_size * mixture_size, activation='softplus')(x)
		std += 0.01 # Set a minimum std
		output_layers.append(std)
	
		mix = dense_layer(output_size * mixture_size)(x)
		mix = Reshape([output_size, mixture_size])(mix)
		mix = tf.nn.softmax(mix, axis=-1)
		mix = Flatten()(mix)
		output_layers.append(mix)

	output = tf.stack(output_layers, axis=1)
	assert output.shape.as_list() == [None, len(output_layers), output_size * mixture_size], output.shape
	model = Model(inp, output)
	return model
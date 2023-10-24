import type * as tf from '@tensorflow/tfjs';

export default class ManuelPredict {
	model: tf.Sequential | tf.LayersModel;
	input: tf.Tensor2D | tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[];
	constructor(model: tf.Sequential | tf.LayersModel, input: tf.Tensor2D) {
		this.model = model;
		this.input = input;
	}

	predict = () => {
		console.clear();
		for (let i = 0; i < this.model.layers.length; i++) {
			const layerType = this.model.layers[i].constructor.name;
			console.log(this.model.layers[i]);

			if (layerType == '_Conv2D') {
				const inputShape = this.model.layers[i].input.shape;
				const outputShape = this.model.layers[i].output.shape;
				inputShape[0] ??= 1;
				outputShape[0] ??= 1;
				const testxs = this.input.reshape(inputShape);
				this.input = this.model.layers[i].call(testxs, {});
				this.input.reshape(outputShape);
			} else {
				this.input = this.model.layers[i].call(this.input, {});
			}
		}
		console.log(this.input.argMax(-1).dataSync());
	};
}

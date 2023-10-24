import * as tf from '@tensorflow/tfjs';

export default class ManuelPredict {
	model: tf.Sequential | tf.LayersModel;
	input: tf.Tensor2D;
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
				let IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH;
				if (this.model.layers[i].batchInputShape) {
					IMAGE_WIDTH = this.model.layers[i].batchInputShape[1];
					IMAGE_HEIGHT = this.model.layers[i].batchInputShape[2];
					IMAGE_DEPTH = this.model.layers[i].batchInputShape[3];
				} else {
					IMAGE_WIDTH = this.model.layers[i].input.shape[1];
					IMAGE_HEIGHT = this.model.layers[i].input.shape[2];
					IMAGE_DEPTH = this.model.layers[i].input.shape[3];
				}

				const testxs = this.input.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]);
				this.input = this.model.layers[i].call(testxs);
				const outputShape = this.model.layers[i].output.shape;
				outputShape[0] ??= 1;
				this.input.reshape(outputShape);
			}
			if (layerType == 'MaxPooling2D') {
				// IMAGE_WIDTH = this.model.layers[i].poolSize[0];
				// IMAGE_HEIGHT = this.model.layers[i].poolSize[1];
				this.input = this.model.layers[i].call(this.input);
				console.log(this.input);
			}
			if (layerType == 'Flatten') {
				console.log('Flat');
			}
		}
	};
}

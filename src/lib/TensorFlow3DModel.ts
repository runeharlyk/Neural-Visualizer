import type * as tf from '@tensorflow/tfjs';
import { BoxGeometry, Color, Group, InstancedMesh, Matrix4, MeshBasicMaterial } from 'three';

export default class Tensorflow3DModel {
	model: tf.Sequential | tf.LayersModel;
	mesh: Group;
	squareSize: number;
	gridSpacing: number;
	input: tf.Tensor2D | tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[];
	layers: { [name: string]: Group } = {};
	output: any;
	constructor(
		model: tf.Sequential | tf.LayersModel,
		input: tf.Tensor2D | tf.Tensor<tf.Rank> | tf.Tensor<tf.Rank>[]
	) {
		this.model = model;
		this.input = input;
		this.mesh = new Group();
		this.squareSize = 0.1;
		this.gridSpacing = 0.05;
		this.displayLayers();
	}

	displayLayers = () => {
		let i = -15;
		this.model.layers.forEach((layer) => {
			const showInput = layer.constructor.name == '_Conv2D';
			i += showInput ? 4 : 2;
			this.makeDefaultGrid(layer, i, showInput);
		});
	};

	makeDefaultGrid = (layer: tf.layers.Layer, offset: number, showInput = false) => {
		const w = layer.output.shape[1] ?? 1;
		const h = layer.output.shape[2] ?? 1;
		const n = layer.output.shape[3] ?? 1;

		const meshGroup = this.generateGrid(w, h, n, offset);
		this.layers[layer.name] = meshGroup;
		this.mesh.add(meshGroup);

		if (!showInput) return;
		const wi = layer.input.shape[1];
		const hi = layer.input.shape[2];
		const ni = layer.input.shape[3];

		const meshGroupInput = this.generateGrid(wi, hi, ni, offset - 2);
		this.layers[layer.name + 'input'] = meshGroupInput;
		this.mesh.add(meshGroupInput);
	};

	generateGrid = (width: number, height: number, instances: number, zOffset: number) => {
		const geometry = new BoxGeometry(this.squareSize, this.squareSize, this.squareSize);
		const material = new MeshBasicMaterial({ color: new Color(0.5, 0.5, 0.5) });
		const nGrid = Math.ceil(Math.sqrt(instances));
		const mOffset = height * (this.squareSize + this.gridSpacing);
		const gridOffset = -(nGrid * mOffset) / 2;
		const meshGroup = new Group();
		for (let m = 0; m < instances; m++) {
			const mesh = new InstancedMesh(geometry, material, width * height);
			mesh.position.x = 0; //(width * this.squareSize + width * this.gridSpacing) / 4;
			mesh.position.y = 0; //(height * this.squareSize + height * this.gridSpacing) / 2;

			const matrix = new Matrix4();

			for (let x = 0; x < width; x++) {
				for (let y = 0; y < height; y++) {
					const i = x + y * height;
					matrix.setPosition(
						gridOffset +
							(m % nGrid) +
							(m % nGrid) * mOffset +
							x * this.squareSize +
							x * this.gridSpacing,
						gridOffset +
							Math.floor(m / nGrid) +
							Math.floor(m / nGrid) * mOffset +
							y * this.squareSize +
							y * this.gridSpacing,
						zOffset
					); // m%nGrid*mOffset+ m/4+m*(mOffset) +
					mesh.setMatrixAt(i, matrix);
					mesh.setColorAt(i, new Color(0, 0, 0));
				}
			}
			meshGroup.add(mesh);
		}
		return meshGroup;
	};

	makeConv2DGrid = (layer: tf.layers.Layer, offset: number) => {
		const w = layer.output.shape[1];
		const h = layer.output.shape[2];
		const n = layer.output.shape[3];

		const meshGroupOutput = this.generateGrid(w, h, n, offset + 1);
		this.layers[layer.name] = meshGroupOutput;
		this.mesh.add(meshGroupOutput);
	};

	updateDataGrid = (layerName: string) => {
		const output = this.input.dataSync();
		const layer = this.layers[layerName];
		if (!layer) {
			console.warn('Could not find layer', layerName);
			return;
		}

		for (let c = 0; c < layer.children.length; c++) {
			const gridSize = Math.sqrt(layer.children[c].count);
			for (let x = 0; x < gridSize; x++) {
				for (let y = 0; y < gridSize; y++) {
					const i = x + y * gridSize;
					const k = i + gridSize ** 2 * c;
					layer.children[c].setColorAt(i, new Color(output[k], output[k], output[k]));
				}
				layer.children[c].instanceColor.needsUpdate = true;
			}
		}
	};

	updateDataLayers = () => {
		for (let i = 0; i < this.model.layers.length; i++) {
			const layerType = this.model.layers[i].constructor.name;

			if (layerType == '_Conv2D') {
				const inputShape = this.model.layers[i].input.shape;
				const outputShape = this.model.layers[i].output.shape;
				inputShape[0] ??= 1;
				outputShape[0] ??= 1;
				const testxs = this.input.reshape(inputShape);
				this.updateDataGrid(this.model.layers[i].name + 'input');
				this.input = this.model.layers[i].call(testxs, {});
				this.updateDataGrid(this.model.layers[i].name);
				this.input.reshape(outputShape);
			} else {
				this.input = this.model.layers[i].call(this.input, {});
				this.updateDataGrid(this.model.layers[i].name);
			}
			if (i == this.model.layers.length - 1) {
				console.log('Prediction: ', this.input.argMax(-1).dataSync()[0]);
			}
		}
	};

	setData = (data: any) => {
		this.input = data;
		this.updateDataLayers();
	};
}

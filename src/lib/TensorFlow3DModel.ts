import * as tf from '@tensorflow/tfjs';
import { BoxGeometry, Color, Group, InstancedMesh, Matrix4, MeshBasicMaterial } from 'three';

export default class Tensorflow3DModel {
	model: tf.Sequential | tf.LayersModel;
	mesh: Group;
	squareSize: number;
	gridSpacing: number;
	grid: InstancedMesh<BoxGeometry, MeshBasicMaterial>;
	input: Float32Array | Int32Array | Uint8Array;
	layers: { [name: string]: Group } = {};
	data: any;
	output: any;
	constructor(
		model: tf.Sequential | tf.LayersModel,
		input: any //: Float32Array | Int32Array | Uint8Array
	) {
		this.model = model;
		this.input = input.xs.data();
		this.data = input;
		this.mesh = new Group();
		this.squareSize = 0.1;
		this.gridSpacing = 0.05;

		console.log(model);

		this.grid = this.makeInputGrid();
		this.mesh.add(this.grid);

		this.displayLayers();
	}

	modelWidthInput = () => this.model.inputLayers[0].batchInputShape[1] ?? 1;
	modelHeightInput = () => this.model.inputLayers[0].batchInputShape[2] ?? 1;

	layerInput = (layer: tf.layers.Layer) => layer.input.shape;
	layerOutput = (layer: tf.layers.Layer) => layer.input.output;

	displayLayers = () => {
		let i = 0;

		this.model.layers.forEach((layer) => {
			i += 5;

			switch (layer.constructor.name) {
				case '_Conv2D':
					this.makeConv2DGrid(layer, i);
					return;
				case 'MaxPooling2D':
					this.makeMaxPooling2D(layer, i);
					break;
				// case 'Flatten':
				//   console.log('Flatten');
				//   break
				// case 'Dense':
				//   console.log('Dense');
				//   break
			}
		});
	};

	makeMaxPooling2D = (layer: tf.layers.Layer, offset: number) => {
		const meshGroup = new Group();
		this.layers[layer.name] = meshGroup;
	};

	makeConv2DGrid = (layer: tf.layers.Layer, offset: number) => {
		const w = layer.output.shape[1];
		const h = layer.output.shape[2];
		const n = layer.output.shape[3];
		const k = layer?.filters;
		const kernels = layer?.kernelSize;
		const bias = layer?.useBias ? layer?.bias.shape : 0;
		const output = layer.output.shape;
		// console.log(
		// 	`Conv input=${w},${h},${n} numfilter=${k}, kernelSize=[${kernels}], bias=${bias}, output=${output}`
		// );

		const geometry = new BoxGeometry(this.squareSize, this.squareSize, this.squareSize);
		const material = new MeshBasicMaterial({ color: new Color(0.5, 0.5, 0.5) });
		const nGrid = Math.ceil(Math.sqrt(n));
		const mOffset = h * (this.squareSize + this.gridSpacing);
		const gridOffset = -(nGrid * mOffset) / 2;
		const meshGroup = new Group();
		this.layers[layer.name] = meshGroup;

		for (let m = 0; m < n; m++) {
			const mesh = new InstancedMesh(geometry, material, w * h);
			mesh.position.x -= (w * this.squareSize + w * this.gridSpacing) / 2;
			mesh.position.y -= (h * this.squareSize + h * this.gridSpacing) / 2;

			const matrix = new Matrix4();

			for (let x = 0; x < w; x++) {
				for (let y = 0; y < h; y++) {
					const i = x + y * h;
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
						offset
					); // m%nGrid*mOffset+ m/4+m*(mOffset) +
					mesh.setMatrixAt(i, matrix);
					mesh.setColorAt(i, new Color(0, 0, 0));
				}
			}
			meshGroup.add(mesh);
		}
		this.mesh.add(meshGroup);
	};

	makeInputGrid = () => {
		const width = this.modelWidthInput();
		const height = this.modelHeightInput();

		const geometry = new BoxGeometry(this.squareSize, this.squareSize, this.squareSize);
		const material = new MeshBasicMaterial({ color: new Color(0.5, 0.5, 0.5) });

		const mesh = new InstancedMesh(geometry, material, width * height);
		mesh.position.x -= (width * this.squareSize + width * this.gridSpacing) / 2;
		mesh.position.y -= (height * this.squareSize + height * this.gridSpacing) / 2;
		const matrix = new Matrix4();

		for (let x = 0; x < width; x++) {
			for (let y = 0; y < height; y++) {
				const i = x + y * height;
				matrix.setPosition(
					x * this.squareSize + x * this.gridSpacing,
					y * this.squareSize + y * this.gridSpacing,
					0
				);
				mesh.setMatrixAt(i, matrix);
				mesh.setColorAt(i, new Color(this.input[i], this.input[i], this.input[i]));
			}
		}
		return mesh;
	};

	imageColorFromTensor = (layerName: string, data: any) => {
		const desiredLayer = this.model.getLayer(layerName);
		const intermediateModel = tf.model({
			inputs: desiredLayer.input,
			outputs: desiredLayer.output
		});

		const IMAGE_WIDTH = desiredLayer.input.shape[1];
		const IMAGE_HEIGHT = desiredLayer.input.shape[2];
		const IMAGE_DEPTH = desiredLayer.input.shape[3];

		const testxs = data.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]);

		const outputTensor = intermediateModel.predict(testxs);

		return outputTensor;
	};

	updateDataLayers = () => {
		let gridSize = Math.sqrt(this.input.length);
		for (let x = 0; x < gridSize; x++) {
			for (let y = 0; y < gridSize; y++) {
				const i = x + y * gridSize;
				this.grid.setColorAt(i, new Color(this.input[i], this.input[i], this.input[i]));
			}
		}
		if (this.grid.instanceColor) this.grid.instanceColor.needsUpdate = true;
		this.output = this.data.xs;
		for (let l = 0; l < Object.keys(this.layers).length; l++) {
			const layerName = Object.keys(this.layers)[l];

			if (layerName != 'conv2d_Conv2D3') continue;
			const output = this.imageColorFromTensor(layerName, this.output);
			const fillData = output.dataSync();

			for (let c = 0; c < this.layers[layerName].children.length; c++) {
				gridSize = Math.sqrt(this.layers[layerName].children[c].count);
				for (let x = 0; x < gridSize; x++) {
					for (let y = 0; y < gridSize; y++) {
						const i = x + y * gridSize;
						const k = i + gridSize * gridSize * c;
						const color = fillData[k];
						this.layers[layerName].children[c].setColorAt(i, new Color(color, color, color));
					}
					this.layers[layerName].children[c].instanceColor.needsUpdate = true;
				}
			}
		}

		// this.layers.forEach((layerMesh) => {
		// 	layerMesh.children.forEach((grid) => {
		// 		console.log(grid);
		// 		gridSize = Math.sqrt(grid.count);
		// 		for (let x = 0; x < gridSize; x++) {
		// 			for (let y = 0; y < gridSize; y++) {
		// 				const i = x + y * gridSize;
		// 				grid.setColorAt(i, new Color(this.input[i], this.input[i], this.input[i]));
		// 			}
		// 		}
		// 	});
		// });
	};

	setData = (data: any) => {
		data.xs.data().then((data: any) => (this.input = data));
		this.data = data;
		this.updateDataLayers();
	};
}

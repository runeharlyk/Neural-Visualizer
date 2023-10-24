<script lang="ts" setup>
	import { onMount } from 'svelte';
  import { MnistData } from '../lib/data';
  import * as tf from '@tensorflow/tfjs';
  import SceneBuilder from '../lib/sceneBuilder';
	import { BoxGeometry, Color, Group, InstancedMesh, Matrix4, MeshBasicMaterial, Raycaster, Vector2 } from 'three';
  import '../app.css'

  let sceneManager:SceneBuilder
  let canvas:HTMLCanvasElement;
  let image_data

  let image_mesh:InstancedMesh<BoxGeometry, MeshBasicMaterial>

  const mouse = new Vector2(1, 1);
  const raycaster = new Raycaster();
  const color = new Color();
  const white = new Color().setHex( 0xffffff );

class tensorflow3DModel {
	model: tf.Sequential | tf.LayersModel;
  mesh: Group
	squareSize: number;
	gridSpacing: number;
	grid: InstancedMesh<BoxGeometry, MeshBasicMaterial>;
	//grid: InstancedMesh<BoxGeometry, MeshBasicMaterial>;
  constructor(model:tf.Sequential | tf.LayersModel) {
    this.model = model
    this.mesh = new Group()
    this.squareSize = 0.1;
    this.gridSpacing = 0.05;

    this.grid = this.makeInputGrid()
    this.mesh.add(this.grid)
    
    this.displayLayers()
  }

  modelWidthInput = () => this.model.inputLayers[0].batchInputShape[1] ?? 1
  modelHeightInput = () => this.model.inputLayers[0].batchInputShape[2] ?? 1

  layerInput = (layer: tf.layers.Layer) => layer.input.shape
  layerOutput = (layer: tf.layers.Layer) => layer.input.output

  displayLayers = () => {
    let i = 0

    this.model.layers.forEach(layer => {    
      i += 5
      
      switch(layer.constructor.name){
        case '_Conv2D':
          this.makeConv2DGrid(layer, i)
          // this.mesh.add(g)
          return
        case 'MaxPooling2D': 
          // console.log(layer);
          break
        // case 'Flatten': 
        //   console.log('Flatten');
        //   break
        // case 'Dense': 
        //   console.log('Dense');
        //   break 
      }
    })
  }

  makeConv2DGrid = (layer:tf.layers.Layer, offset:number) => {
    const w = layer.output.shape[1]
    const h = layer.output.shape[2] 
    const n = layer.output.shape[3] 
    const k = layer?.filters
    const kernels = layer?.kernelSize
    const bias = layer?.useBias ? layer?.bias.shape : 0
    const output = layer.output.shape
    console.log(`Conv input=${w},${h},${n} numfilter=${k}, kernelSize=[${kernels}], bias=${bias}, output=${output}`);   

    const geometry = new BoxGeometry(this.squareSize, this.squareSize, this.squareSize);
    const material = new MeshBasicMaterial({ color: new Color(0.5, 0.5, 0.5) });
    const nGrid = Math.ceil((Math.sqrt(n)))
    const mOffset = h * (this.squareSize + this.gridSpacing)
    const gridOffset = -(nGrid*mOffset)/2

    for (let m = 0; m < n; m++) {
      const mesh = new InstancedMesh(geometry, material, w*h);
      mesh.position.x -= (w * this.squareSize + w * this.gridSpacing) / 2
      mesh.position.y -= (h * this.squareSize + h * this.gridSpacing) / 2
  
      const matrix = new Matrix4();

      for (let x = 0; x < w; x++) {
        for (let y = 0; y < h; y++) {
          const i = x + y * h
          matrix.setPosition(gridOffset + m%nGrid + m%nGrid*mOffset + x * this.squareSize + x * this.gridSpacing, gridOffset + Math.floor(m/nGrid) + Math.floor(m/nGrid)*mOffset +  y * this.squareSize + y * this.gridSpacing, offset); // m%nGrid*mOffset+ m/4+m*(mOffset) +
          mesh.setMatrixAt(i, matrix);
          mesh.setColorAt(i, new Color(0, 0, 0));
        }
      }
      this.mesh.add(mesh)
    }
  }

  makeInputGrid = () => {
    //console.log(this.model);
    const width = this.modelWidthInput()
    const height = this.modelHeightInput()
    
    
    const geometry = new BoxGeometry(this.squareSize, this.squareSize, this.squareSize);
    const material = new MeshBasicMaterial({ color: new Color(0.5, 0.5, 0.5) });

    const mesh = new InstancedMesh(geometry, material, width*height);
    mesh.position.x -= (width * this.squareSize + width * this.gridSpacing) / 2
    mesh.position.y -= (height * this.squareSize + height * this.gridSpacing) / 2
    const matrix = new Matrix4();
    
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
          const i = x + y * height 
          matrix.setPosition(x * this.squareSize + x * this.gridSpacing, y * this.squareSize + y * this.gridSpacing, 0);
          mesh.setMatrixAt(i, matrix);
          mesh.setColorAt(i, new Color(0, 0, 0));
      }
    }
    return mesh
  }

  setData = (data:Float32Array | Int32Array | Uint8Array) => {
    const gridSize = Math.sqrt(data.length);
    for (let x = 0; x < gridSize; x++) {
      for (let y = 0; y < gridSize; y++) {
        const i = x + y * gridSize 
        this.grid.setColorAt(i, new Color(data[i], data[i], data[i]));
      }
    }
    if (this.grid.instanceColor) this.grid.instanceColor.needsUpdate = true;
  }
}

  onMount(async () => {
    createScene()
    const data = new MnistData();
    await data.load();

    console.log("Loading model");
    let model
    model = await tf.loadLayersModel('localstorage://demo');
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(1);
    image_data = await testData.xs.data()
    const testxs = testData.xs.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const preds = model.predict(testxs).argMax(-1);
    testxs.dispose();


    const tfModel = new tensorflow3DModel(model)
    sceneManager.scene.add(tfModel.mesh)
    tfModel.setData(image_data)
    // if(localStorage.getItem("demo") !== null){
    // } else {
    //   model = getModel();
    //   console.log("Starting training");
    //   await train(model, data);
    //   console.log("Saving model");
    //   await model.save('localstorage://demo');
    // }
    
    //image_mesh = makeGrid(image_data)
    //await model.save('downloads://demo');
  })

  const render = (timeStamp:number) => {
    if (start === undefined) start = timeStamp
    if(timeStamp - start > 1000) {
      const testData = data.nextTestBatch(1);
      tfModel.setData(testData)
      start = timeStamp
    }

    // if (!image_mesh) return 
    // raycaster.setFromCamera(mouse, sceneManager.camera);
    // const intersection = raycaster.intersectObject(image_mesh);
    // if (intersection.length > 0) {
    //   const instanceId = intersection[0].instanceId ?? 0;
    //   image_mesh.getColorAt( instanceId, color );
    //   image_mesh.setColorAt(instanceId, white);
    //   if (image_mesh.instanceColor) image_mesh.instanceColor.needsUpdate = true;
    // }
  }

  const createScene = () => {
    sceneManager = new SceneBuilder()
        .addRenderer({ antialias: true, canvas: canvas, alpha: true})
        .addPerspectiveCamera({x:0, y:0, z:1})
        .addOrbitControls(30, 50)
        .addGroundPlane({x:0, y:-2, z:0})
        .addAmbientLight({color:0xffffff, intensity:0.3})
        .addDirectionalLight({x:50, y:100, z:100, color:0xffffff, intensity:0.9})
        .addFogExp2(0xcccccc, 0.015)
        .handleResize()
        .addRenderCb(render)
        .startRenderLoop()
}

  const onResize = () => {
    canvas.width = window.innerWidth
    canvas.height = window.innerHeight
  }

  const onMouseMove = (event:MouseEvent) => {
    event.preventDefault();
    mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
  }

  async function train(model:tf.Sequential, data:MnistData) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
  //const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
  const BATCH_SIZE = 512;
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [
      d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    //callbacks: fitCallbacks
  });
}

  function getModel():tf.Sequential {
    const model = tf.sequential();
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;  
    
    // In the first layer of our convolutional neural network we have 
    // to specify the input shape. Then we specify some parameters for 
    // the convolution operation that takes place in this layer.
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
      kernelSize: 5,
      filters: 8,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));

    // The MaxPooling layer acts as a sort of downsampling using max values
    // in a region instead of averaging.  
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Repeat another conv2d + maxPooling stack. 
    // Note that we have more filters in the convolution.
    model.add(tf.layers.conv2d({
      kernelSize: 5,
      filters: 16,
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    
    // Now we flatten the output from the 2D filters into a 1D vector to prepare
    // it for input into our last layer. This is common practice when feeding
    // higher dimensional data to a final classification output layer.
    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
      units: NUM_OUTPUT_CLASSES,
      kernelInitializer: 'varianceScaling',
      activation: 'softmax'
    }));

    
    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    return model;
  }
  const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model:tf.Sequential, data:MnistData, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}
</script>

<svelte:body class="m-0" on:mousemove={onMouseMove}></svelte:body>
<svelte:window on:resize={onResize}></svelte:window>
<canvas bind:this={canvas} class="bg-indigo-700 w-full h-full"></canvas>

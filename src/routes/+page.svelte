<script lang="ts" setup>
	import { onMount } from 'svelte';
  import { MnistData } from '../lib/data';
  import * as tf from '@tensorflow/tfjs';
  import SceneBuilder from '../lib/sceneBuilder';
	import { BoxGeometry, BufferAttribute, BufferGeometry, Color, InstancedMesh, Matrix4, Mesh, MeshBasicMaterial } from 'three';

  let sceneManager:SceneBuilder
  let canvas:HTMLCanvasElement;
  let image_data

  onMount(async () => {
    createScene()
    const data = new MnistData();
    await data.load();

    console.log("Loading model");
    let model
    model = await tf.loadLayersModel('localstorage://demo');
    // if(localStorage.getItem("demo") !== null){
    // } else {
    //   model = getModel();
    //   console.log("Starting training");
    //   await train(model, data);
    //   console.log("Saving model");
    //   await model.save('localstorage://demo');
    // }
    
    console.log("Trained model loaded");
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(1);
    image_data = await testData.xs.data()
    const testxs = testData.xs.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    // console.log(testxs)
    console.log(model.predict(testxs))
    testxs.dispose();
    // console.log(labels, preds);
    makeGrid(image_data)

    
    //tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);
  })

  const render = () => {
    
  }

  const createScene = () => {
    sceneManager = new SceneBuilder()
        .addRenderer({ antialias: true, canvas: canvas, alpha: true})
        .addPerspectiveCamera({x:-0.5, y:0.5, z:1})
        .addOrbitControls(10, 30)
        .addGroundPlane({x:0, y:-2, z:0})
        .addGridHelper({size:250, divisions:125, y:-2})
        .addAmbientLight({color:0xffffff, intensity:0.3})
        .addDirectionalLight({x:50, y:100, z:100, color:0xffffff, intensity:0.9})
        .addFogExp2(0xcccccc, 0.015)
        .handleResize()
        .addRenderCb(render)
        .startRenderLoop()

}

  const makeGrid = (data:Float32Array | Int32Array | Uint8Array) => {

    const gridSize = 28;
    const squareSize = 0.1;
    const gridSpacing = 0.0001;
    const halfGridSize = (gridSize - 1) / 2;
    
    const geometry = new BoxGeometry(squareSize, squareSize, squareSize);
    const material = new MeshBasicMaterial({ color: new Color(0.5, 0.5, 0.5) });

    const mesh = new InstancedMesh( geometry, material, gridSize**2 );

    const matrix = new Matrix4();
    let i = 0;
    const offset = gridSpacing//( gridSize**2 - 1 ) / 2;
    
    for ( let x = 0; x < gridSize; x ++ ) {
      for ( let y = 0; y < gridSize; y ++ ) {
          matrix.setPosition( (x+offset*x)/4 - halfGridSize*offset, (offset*y + y/4),0);
          mesh.setMatrixAt( i, matrix );
          mesh.setColorAt( i, new Color(data[i],data[i],data[i]) );
          i ++;
      }
    }
    sceneManager.scene.add(mesh)
  }

  async function train(model, data) {
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

  function getModel() {
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

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}

</script>

<style>
  canvas {
    background-color: #2e019e;
  }
</style>

<canvas bind:this={canvas}></canvas>

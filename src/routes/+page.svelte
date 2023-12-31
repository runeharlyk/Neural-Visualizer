<script lang="ts" setup>
	import { onMount } from 'svelte';
  import { MnistData } from '../lib/data';
  import * as tf from '@tensorflow/tfjs';
  import SceneBuilder from '../lib/sceneBuilder';
	import { Vector2 } from 'three';
  import Tensorflow3DModel from '../lib/TensorFlow3DModel'
  import '../app.css'

  let sceneManager:SceneBuilder
  let canvas:HTMLCanvasElement;
  let start: number
  let tfModel: Tensorflow3DModel
  let data:MnistData

  const mouse = new Vector2(1, 1);

  onMount(async () => {
    createScene()
    data = new MnistData();
    await data.load();
    let model
    try {
      model = await tf.loadLayersModel('https://runeharlyk.github.io/Neural-Visualizer/model/demo.json');
    } catch {
      console.log("No model were found");
      model = getModel();
      console.log("Traning the model");
      await train(model, data);
      await model.save('localstorage://demo');
      console.log("Done saving the model");
    }
    
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(1);
    const testxs = testData.xs.reshape([1, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
    testxs.dispose();

    tfModel = new Tensorflow3DModel(model, testData.xs)

    sceneManager.scene.add(tfModel.mesh)
  })

  const render = (timeStamp:number) => {
    if (start === undefined) start = timeStamp
    if(timeStamp - start > 1000 && tfModel) {
      const testData = data.nextTestBatch(1);
      tfModel.setData(testData.xs)
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
        .addPerspectiveCamera({x:1, y:0.1, z:0})
        .addOrbitControls(30, 50)
        .addGroundPlane({x:0, y:0, z:0})
        .addAmbientLight({color:0xffffff, intensity:0.3})
        //.addGridHelper({divisions: 50, size:100})
        .addDirectionalLight({x:50, y:100, z:100, color:0xffffff, intensity:0.9})
        //.addFogExp2(0xcccccc, 0.015)
        .handleResize()
        .addRenderCb(render)
        .startRenderLoop()
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
</script>

<style>
  .purple {
    background: #7F00FF;  /* fallback for old browsers */
    background: -webkit-linear-gradient(to right, #E100FF, #7F00FF);  /* Chrome 10-25, Safari 5.1-6 */
    background: linear-gradient(to right, #E100FF, #7F00FF); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */
  }
</style>

<svelte:body class="m-0" on:mousemove={onMouseMove}></svelte:body>
<canvas bind:this={canvas} class="purple w-full h-full"></canvas>

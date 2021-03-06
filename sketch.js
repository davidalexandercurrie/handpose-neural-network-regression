let handpose;
let video;
let predictions = [];

let model;
let targetLabel = [];
let state = false;
// let state = 'prediction';

let nnResults;
let loopBroken = false;

let socket;

function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.size(width, height);

  socket = io.connect();

  handpose = ml5.handpose(video, modelReady);
  // This sets up an event that fills the global variable "predictions"
  // with an array every time new hand poses are detected
  handpose.on('predict', results => {
    predictions = results;
    if (predictions[0] != undefined) {
      let inputs = predictions[0].landmarks.flat();
      if (state == 'collection') {
        let target = targetLabel;
        if (targetLabel != undefined) {
          model.addData(inputs, target);
          console.log(`${inputs} recorded for label ${targetLabel}`);
        } else {
          console.log('Target label not set.');
        }
      }
    }
  });

  let options = {
    inputs: 63,
    outputs: 3,
    task: 'regression',
    debug: true,
  };

  model = ml5.neuralNetwork(options);
  // autoStartPredict();

  // Hide the video element, and just show the canvas
  video.hide();

  createButton('Load Model').mousePressed(onLoadModelClick);
  createButton('Start Prediction').mousePressed(onPredictClick);
  createButton('Toggle Collection').mousePressed(onCollectClick);
}

function autoStartPredict() {
  if (state == 'prediction') {
    onLoadModelClick();
    onPredictClick();
  }
}

function dataLoaded() {
  console.log(model.data);
}

function modelReady() {
  console.log('Model ready!');
}

function draw() {
  image(video, 0, 0, width, height);

  // We can call both functions to draw all keypoints
  drawKeypoints();
  restartPredictions();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  for (let i = 0; i < predictions.length; i += 1) {
    const prediction = predictions[i];

    for (let j = 0; j < prediction.landmarks.length; j += 1) {
      const keypoint = prediction.landmarks[j];
      fill(0, 255, 0);
      noStroke();
      ellipse(keypoint[0], keypoint[1], 10, 10);
    }
  }
}

function mousePressed() {
  if (predictions[0] != undefined) {
    let inputs = predictions[0].landmarks.flat();
    // if (state == 'collection') {
    //   let target = targetLabel;
    //   if (targetLabel != undefined) {
    //     model.addData(inputs, target);
    //     console.log(`${inputs} recorded for label ${targetLabel}`);
    //   } else {
    //     console.log('Target label not set.');
    //   }
    // } else
    if (state == 'prediction') {
      model.predict(inputs, gotResults);
    }
  }
}

function gotResults(error, results) {
  if (error) {
    console.error(error);
    return;
  }
  console.log(results);
  nnResults = results;
  sendToServer();
  predictValues();
}

function keyPressed() {
  if (key == 't') {
    console.log('starting training');
    model.normalizeData();
    state = 'training';
    let options = {
      epochs: 50,
      batchSize: 12,
    };
    model.train(options, whileTraining, finishedTraining);
  } else if (key == 's') {
    model.saveData();
  } else if (key == 'm') {
    model.save();
  } else if (key == 'r') {
    targetLabel = [round(random(255)), round(random(255)), round(random(255))];
    console.log(targetLabel);
  }
}

function whileTraining(epoch, loss) {
  console.log(epoch, loss);
}
function finishedTraining() {
  console.log('finished training');
}

function predictValues() {
  if (predictions[0] != undefined) {
    let inputs = predictions[0].landmarks.flat();
    model.predict(inputs, gotResults);
  } else {
    loopBroken = true;
  }
}

function onPredictClick() {
  state = 'prediction';
}
function onLoadModelClick() {
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  model.load(modelInfo, () => console.log('Model Loaded.'));
}

function restartPredictions() {
  if (loopBroken) {
    loopBroken = false;
    predictValues();
  }
}

const sendToServer = () => {
  socket.emit('handpose', nnResults);
};

const onCollectClick = () =>
  state === 'collection' ? (state = false) : (state = 'collection');

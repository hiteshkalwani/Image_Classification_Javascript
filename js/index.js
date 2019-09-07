let net;
const webcamElement = document.getElementById('webcam');
const webcamElement_custom_classification = document.getElementById('webcam_custom_classification');
const classifier = knnClassifier.create();

async function uploadImg() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load();
  console.log('Sucessfully loaded model');

  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('img');
  console.log(imgEl)
  const result = await net.classify(imgEl);
  console.log(result);
 
  document.getElementById('console_img').innerText = `
        prediction: ${result[0].className}\n
        probability: ${result[0].probability}
      `;
}

async function setupWebcam() {
    return new Promise((resolve, reject) => {
      const navigatorAny = navigator;
      navigator.getUserMedia = navigator.getUserMedia ||
          navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
          navigatorAny.msGetUserMedia;
      if (navigator.getUserMedia) {
        navigator.getUserMedia({video: true},
          stream => {
            webcamElement.srcObject = stream;
            webcamElement.addEventListener('loadeddata',  () => resolve(), false);
            webcamElement_custom_classification.srcObject = stream;
            webcamElement_custom_classification.addEventListener('loadeddata',  () => resolve(), false);
          },
          error => reject());
      } else {
        reject();
      }
    });
}

async function webcam_classification() {
    console.log('Loading mobilenet..');
  
    // Load the model.
    net = await mobilenet.load();
    console.log('Sucessfully loaded model');
    
    await setupWebcam();
    while (true) {
      const result = await net.classify(webcamElement);
  
      document.getElementById('console_webcam').innerText = `
        prediction: ${result[0].className}\n
        probability: ${result[0].probability}
      `;
  
      // Give some breathing room by waiting for the next animation frame to
      // fire.
      await tf.nextFrame();
    }
  }

// Custome Video classification
async function custom_classification() {
    console.log('Loading mobilenet..');
  
    // Load the model.
    net = await mobilenet.load();
    console.log('Sucessfully loaded model');
  
    //await setupWebcam();
  
    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = classId => {
      // Get the intermediate activation of MobileNet 'conv_preds' and pass that
      // to the KNN classifier.
      const activation = net.infer(webcamElement, 'conv_preds');
  
      // Pass the intermediate activation to the classifier.
      classifier.addExample(activation, classId);
    };
  
    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
  
    while (true) {
      if (classifier.getNumClasses() > 0) {
        // Get the activation from mobilenet from the webcam.
        const activation = net.infer(webcamElement_custom_classification, 'conv_preds');
        // Get the most likely class and confidences from the classifier module.
        const result = await classifier.predictClass(activation);
        console.log(result)
        const classes = ['Item_1', 'Item_2', 'Item_3'];
        document.getElementById('console_custom_classification').innerText = `
          prediction: ${classes[result.classIndex]}\n
          probability: ${result.confidences[result.classIndex]}
        `;
      }
  
      await tf.nextFrame();
    }
  }


  uploadImg()
  webcam_classification()
  custom_classification()

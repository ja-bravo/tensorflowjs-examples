import { Tensor } from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';
import path from 'path';
import { prepareImage } from './prepareImage';
import { Labels } from './train';
import fg from 'fast-glob';

const testImage = async () => {
  const model = await tf.loadLayersModel(`file://${path.join(__dirname, 'best_model/model.json')}`);

  const files = fg.sync(path.join(__dirname, './test-images/*.jpg'));

  const images = files.map((f) => ({
    image: prepareImage(f),
    path: f.split('/').at(-1),
  }));

  const stackedImages = tf.stack(images.map((i) => i.image))
  const result = (await model.predict(stackedImages)) as Tensor;
  const predictions = Array.from(result.dataSync())

  for (let i = 0; i < predictions.length; i++) {
    const prediction = predictions[i];
    const testImage = images[i];
    console.log(`${testImage.path} has been predicted as a ${prediction === Labels.Cat ? 'Cat' : 'Dog'}`);
  }
};

testImage();

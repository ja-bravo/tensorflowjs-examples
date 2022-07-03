import { Tensor } from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node-gpu';
import path from 'path';
import { prepareImage } from './prepareImage';
import { Labels } from './train';
import fg from 'fast-glob';

const testImage = async () => {
  const model = await tf.loadLayersModel(`file://${path.join(__dirname, 'best_model/model.json')}`);

  const files = fg.sync(path.join(__dirname, './test-images/*.jpg'));

  const images = files.map((f) => ({
    image: prepareImage(f),
    label: f.split('/').at(-1).includes('cat') ? 'Cat' : 'Dog',
    path: f.split('/').at(-1),
  }));

  const result = (await model.predict(tf.stack(images.map((i) => i.image)))) as Tensor;
  const predictions = result.arraySync() as number[][];

  for (let i = 0; i < predictions.length; i++) {
    const prediction = predictions[i];
    const testImage = images[i];
    const [predictedLabel] = prediction;
    console.log(`${testImage.path} has been predicted as a ${predictedLabel === Labels.Cat ? 'Cat' : 'Dog'}`);
  }
};

testImage();

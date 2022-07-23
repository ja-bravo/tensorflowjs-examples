import * as tf from '@tensorflow/tfjs-node';
import { Tensor3D } from '@tensorflow/tfjs-node';
import fg from 'fast-glob';
import path from 'path';
import { prepareImage } from './prepareImage';

export const enum Labels {
  Cat = 0,
  Dog = 1,
}

const run = async () => {
  const files = fg.sync(path.join(__dirname, './train/*.jpg'));
  const labels: Labels[] = [];
  const images: Tensor3D[] = [];

  for (const file of files) {
    const image = prepareImage(file);

    const fileName = file.split('/').at(-1);
    labels.push(fileName.includes('cat') ? Labels.Cat : Labels.Dog);
    images.push(image);
  }

  // Shuffle to help the model learn the features and not the order
  tf.util.shuffleCombo(images, labels);

  // Turn both the images and labels into tensors
  const imagesTensor = tf.stack(images);
  const labelsTensor = tf.stack(labels);

  // Normalize the data to the range [0, 1] as dense layers do not work well with denormalized features
  const normalizedImages = imagesTensor.div(255);

  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({ activation: 'relu', filters: 32, kernelSize: 3, padding: 'same', inputShape: [80, 80, 1] }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.dropout({ rate: 0.2 }),

      tf.layers.conv2d({ activation: 'relu', filters: 64, kernelSize: 3, padding: 'same' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.dropout({ rate: 0.2 }),

      tf.layers.conv2d({ activation: 'relu', filters: 128, kernelSize: 3, padding: 'same' }),
      tf.layers.maxPooling2d({ poolSize: 2 }),
      tf.layers.dropout({ rate: 0.2 }),

      tf.layers.flatten(),

      tf.layers.dense({ units: 512, activation: 'relu' }),
      tf.layers.dense({ units: 128, activation: 'relu' }),

      tf.layers.dropout({ rate: 0.2 }),
      tf.layers.dense({ units: 1, activation: 'sigmoid' }),
    ],
  });

  model.compile({ loss: 'binaryCrossentropy', optimizer: tf.train.adam(0.001), metrics: ['accuracy'] });

  let bestValAcc = 0;
  await model.fit(normalizedImages, labelsTensor, {
    epochs: 60,
    batchSize: 128,
    validationSplit: 0.2,
    callbacks: [
      tf.callbacks.earlyStopping({ monitor: 'val_acc', patience: 5 }),
      new tf.CustomCallback({
        onEpochEnd: async (_, logs: { val_acc: number }) => {
          if (logs.val_acc > bestValAcc) {
            console.log('Saving model', logs.val_acc * 100);
            model.save(`file://${path.join(__dirname, 'best_model')}`);
            bestValAcc = logs.val_acc;
          }
        },
      }),
    ],
  });

  tf.dispose([normalizedImages, imagesTensor, labelsTensor]);
  model.dispose();
};

run();

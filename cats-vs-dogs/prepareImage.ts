import * as tf from '@tensorflow/tfjs-node';
import { Tensor3D } from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';


export const prepareImage = (filePath: string) => {
  // Read the image as grayscale (1 channel)
  const image = tf.node.decodeImage(readFileSync(filePath), 1) as Tensor3D;

  // Resize the image to a fixed size to make it easier to process.
  const resized = tf.image.resizeBilinear(image, [80, 80]);
  tf.dispose([image]);

  return resized;
};

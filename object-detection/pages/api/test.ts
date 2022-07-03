import type { NextApiRequest, NextApiResponse } from 'next';

import * as tf from '@tensorflow/tfjs-node-gpu';
import { readFileSync } from 'fs';
import { INCEPTION_CLASSES } from './labels';
import { load } from '@tensorflow-models/coco-ssd';

const getPredictions = async () => {
  const model = await load();
  const img = tf.node.decodeImage(readFileSync(__dirname + '/test.jpg'));
  // const input = tf.image.resizeBilinear(img, [299, 299], true).div(255).reshape([1, 299, 299, 3]);
  const res = await model.detect(img as tf.Tensor3D);
  // const { values, indices } = tf.topk(res as any, 3);

  console.log(res);
  // const answers = indices.dataSync();

  return res;
};

export default async function handler(req: NextApiRequest, res: NextApiResponse<any>) {
  res.status(200).json(await getPredictions());
}

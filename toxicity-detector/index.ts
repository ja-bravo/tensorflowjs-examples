import '@tensorflow/tfjs-node';
import * as toxicity from '@tensorflow-models/toxicity';
import { printTable } from 'console-table-printer';

const run = async () => {
  const model = await toxicity.load(0.5, []);
  const sentences = ['Your opinion is irrelevant and you are dumb', 'Dogs are awesome'];
  const predictions = await model.classify(sentences);
  for (let i = 0; i < sentences.length; i++) {
    const sentence = sentences[i];
    console.log(sentence);
    printTable(
      predictions.map((prediction) => ({
        label: prediction.label,
        probability: `No: ${prediction.results[i].probabilities[0].toFixed(4)}. Yes: ${prediction.results[i].probabilities[1].toFixed(4)}`,
        match: prediction.results[i].match,
      })),
    );
  }
};
run();

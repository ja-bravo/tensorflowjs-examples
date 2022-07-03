import type { NextPage } from 'next';
import { load, ObjectDetection } from '@tensorflow-models/coco-ssd';
import { useEffect, useState, useRef } from 'react';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as tf from '@tensorflow/tfjs';

const Home: NextPage = () => {
  const inputRef = useRef<HTMLInputElement | null>();
  const [model, setModel] = useState<ObjectDetection>();
  useEffect(() => {
    const loadModel = async () => {
      setModel(await load());
    };
    loadModel();
  }, []);

  const handleImagePick = (imgSrc: string) => {
    const img = new Image();
    img.src = imgSrc;
    img.onload = async () => {
      const canvas = document.getElementById('canvas') as HTMLCanvasElement;
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(img, 0, 0);
      const imageTensor = await tf.browser.fromPixelsAsync(canvas, 3);

      const predictions = await model?.detect(imageTensor)!;
      alert(`I've found ${predictions.length} objects! Look in your console`);
      console.log(predictions.map((p) => `${p.class}: ${(p.score * 100).toFixed(2)}% probability`));
      ctx.lineWidth = 20;
      ctx.strokeStyle = '#4ade80';

      for (const prediction of predictions) {
        ctx.strokeRect(prediction.bbox[0], prediction.bbox[1], prediction.bbox[2], prediction.bbox[3]);
      }
      imageTensor.dispose();
    };
  };

  return (
    <div className="p-4">
      <h1 className="text-3xl text-center mb-8 mt-8 text-indigo-600">Image detection</h1>
      <button
        onClick={() => inputRef.current?.click()}
        type="button"
        className="mb-8 items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      >
        <span className="mt-2 block text-sm font-medium text-gray-900">Click to upload Image</span>
        <input
          ref={(ref) => (inputRef.current = ref)}
          className="absolute top-0 left-0 opacity-0 cursor-pointer"
          type="file"
          onChange={(e) => {
            const files = e.target.files;

            if (files) {
              const reader = new FileReader();
              reader.onload = () => {
                handleImagePick(reader.result as string);
              };
              reader.readAsDataURL(files[0]);
            }

            inputRef.current!.value = '';
            e.target.value = null as any;
          }}
        />
      </button>
      <div className="w-full flex justify-center">
        <canvas id="canvas" className="w-[1200px] bg-red-50" />
      </div>
    </div>
  );
};

export default Home;

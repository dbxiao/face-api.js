import * as tf from '@tensorflow/tfjs-core';

import { bufferToImage, Dimensions, isTensor3D, NetInput, Point, TMediaElement, toNetInput } from '../../../src';
import { FaceLandmarks68 } from '../../../src/classes/FaceLandmarks68';
import { createFaceLandmarkNet } from '../../../src/faceLandmarkNet';
import { FaceLandmark68TinyNet } from '../../../src/faceLandmarkNet/FaceLandmark68TinyNet';
import { describeWithNets, expectAllTensorsReleased, expectMaxDelta, expectPointClose } from '../../utils';

function getInputDims (input: tf.Tensor | TMediaElement): Dimensions {
  if (input instanceof tf.Tensor) {
    const [height, width] = input.shape.slice(isTensor3D(input) ? 0 : 1)
    return { width, height }
  }
  return input
}

describe('faceLandmark68TinyNet', () => {

  let imgEl1: HTMLImageElement
  let imgEl2: HTMLImageElement
  let imgElRect: HTMLImageElement
  let faceLandmarkPositions1: Point[]
  let faceLandmarkPositions2: Point[]
  let faceLandmarkPositionsRect: Point[]

  beforeAll(async () => {
    const img1 = await (await fetch('base/test/images/face1.png')).blob()
    imgEl1 = await bufferToImage(img1)
    const img2 = await (await fetch('base/test/images/face2.png')).blob()
    imgEl2 = await bufferToImage(img2)
    const imgRect = await (await fetch('base/test/images/face_rectangular.png')).blob()
    imgElRect = await bufferToImage(imgRect)
    faceLandmarkPositions1 = await (await fetch('base/test/data/faceLandmarkPositions1Tiny.json')).json()
    faceLandmarkPositions2 = await (await fetch('base/test/data/faceLandmarkPositions2Tiny.json')).json()
    faceLandmarkPositionsRect = await (await fetch('base/test/data/faceLandmarkPositionsRectTiny.json')).json()
  })

  describeWithNets('uncompressed weights', { withFaceLandmark68TinyNet: { quantized: false } }, ({ faceLandmark68TinyNet }) => {

    it('computes face landmarks for squared input', async () => {
      const { width, height } = imgEl1

      const result = await faceLandmark68TinyNet.detectLandmarks(imgEl1) as FaceLandmarks68
      expect(result.getImageWidth()).toEqual(width)
      expect(result.getImageHeight()).toEqual(height)
      expect(result.getShift().x).toEqual(0)
      expect(result.getShift().y).toEqual(0)
      result.getPositions().forEach((pt, i) => {
        const { x, y } = faceLandmarkPositions1[i]
        expectPointClose(pt, { x, y }, 5)
      })
    })

    it('computes face landmarks for rectangular input', async () => {
      const { width, height } = imgElRect

      const result = await faceLandmark68TinyNet.detectLandmarks(imgElRect) as FaceLandmarks68
      expect(result.getImageWidth()).toEqual(width)
      expect(result.getImageHeight()).toEqual(height)
      expect(result.getShift().x).toEqual(0)
      expect(result.getShift().y).toEqual(0)
      result.getPositions().forEach((pt, i) => {
        const { x, y } = faceLandmarkPositionsRect[i]
        expectPointClose(pt, { x, y }, 5)
      })
    })

  })

  describeWithNets('quantized weights', { withFaceLandmark68TinyNet: { quantized: true } }, ({ faceLandmark68TinyNet }) => {

    it('computes face landmarks for squared input', async () => {
      const { width, height } = imgEl1

      const result = await faceLandmark68TinyNet.detectLandmarks(imgEl1) as FaceLandmarks68
      expect(result.getImageWidth()).toEqual(width)
      expect(result.getImageHeight()).toEqual(height)
      expect(result.getShift().x).toEqual(0)
      expect(result.getShift().y).toEqual(0)
      result.getPositions().forEach((pt, i) => {
        const { x, y } = faceLandmarkPositions1[i]
        expectPointClose(pt, { x, y }, 5)
      })
    })

    it('computes face landmarks for rectangular input', async () => {
      const { width, height } = imgElRect

      const result = await faceLandmark68TinyNet.detectLandmarks(imgElRect) as FaceLandmarks68
      expect(result.getImageWidth()).toEqual(width)
      expect(result.getImageHeight()).toEqual(height)
      expect(result.getShift().x).toEqual(0)
      expect(result.getShift().y).toEqual(0)
      result.getPositions().forEach((pt, i) => {
        const { x, y } = faceLandmarkPositionsRect[i]
        expectPointClose(pt, { x, y }, 5)
      })
    })

  })

  describeWithNets('batch inputs', { withFaceLandmark68TinyNet: { quantized: false } }, ({ faceLandmark68TinyNet }) => {

    it('computes face landmarks for batch of image elements', async () => {
      const inputs = [imgEl1, imgEl2, imgElRect]

      const faceLandmarkPositions = [
        faceLandmarkPositions1,
        faceLandmarkPositions2,
        faceLandmarkPositionsRect
      ]

      const results = await faceLandmark68TinyNet.detectLandmarks(inputs) as FaceLandmarks68[]
      expect(Array.isArray(results)).toBe(true)
      expect(results.length).toEqual(3)
      results.forEach((result, batchIdx) => {
        const { width, height } = getInputDims(inputs[batchIdx])
        expect(result.getImageWidth()).toEqual(width)
        expect(result.getImageHeight()).toEqual(height)
        expect(result.getShift().x).toEqual(0)
        expect(result.getShift().y).toEqual(0)
        result.getPositions().forEach((pt, i) => {
          const { x, y } = faceLandmarkPositions[batchIdx][i]
          expectPointClose(pt, { x, y }, 5)
        })
      })
    })

    it('computes face landmarks for batch of tf.Tensor3D', async () => {
      const inputs = [imgEl1, imgEl2, imgElRect].map(el => tf.fromPixels(el))

      const faceLandmarkPositions = [
        faceLandmarkPositions1,
        faceLandmarkPositions2,
        faceLandmarkPositionsRect
      ]

      const results = await faceLandmark68TinyNet.detectLandmarks(inputs) as FaceLandmarks68[]
      expect(Array.isArray(results)).toBe(true)
      expect(results.length).toEqual(3)
      results.forEach((result, batchIdx) => {
        const { width, height } = getInputDims(inputs[batchIdx])
        expect(result.getImageWidth()).toEqual(width)
        expect(result.getImageHeight()).toEqual(height)
        expect(result.getShift().x).toEqual(0)
        expect(result.getShift().y).toEqual(0)
        result.getPositions().forEach((pt, i) => {
          const { x, y } = faceLandmarkPositions[batchIdx][i]
          expectPointClose(pt, { x, y }, 3)
        })
      })
    })

    it('computes face landmarks for batch of mixed inputs', async () => {
      const inputs = [imgEl1, tf.fromPixels(imgEl2), tf.fromPixels(imgElRect)]

      const faceLandmarkPositions = [
        faceLandmarkPositions1,
        faceLandmarkPositions2,
        faceLandmarkPositionsRect
      ]

      const results = await faceLandmark68TinyNet.detectLandmarks(inputs) as FaceLandmarks68[]
      expect(Array.isArray(results)).toBe(true)
      expect(results.length).toEqual(3)
      results.forEach((result, batchIdx) => {
        const { width, height } = getInputDims(inputs[batchIdx])
        expect(result.getImageWidth()).toEqual(width)
        expect(result.getImageHeight()).toEqual(height)
        expect(result.getShift().x).toEqual(0)
        expect(result.getShift().y).toEqual(0)
        result.getPositions().forEach((pt, i) => {
          const { x, y } = faceLandmarkPositions[batchIdx][i]
          expectPointClose(pt, { x, y }, 3)
        })
      })
    })

  })

  describeWithNets('no memory leaks', { withFaceLandmark68TinyNet: { quantized: true } }, ({ faceLandmark68TinyNet }) => {

    describe('NeuralNetwork, uncompressed model', () => {

      it('disposes all param tensors', async () => {
        await expectAllTensorsReleased(async () => {
          const res = await fetch('base/weights_uncompressed/face_landmark_68_model.weights')
          const weights = new Float32Array(await res.arrayBuffer())
          const net = createFaceLandmarkNet(weights)
          net.dispose()
        })
      })

    })

    describe('NeuralNetwork, quantized model', () => {

      it('disposes all param tensors', async () => {
        await expectAllTensorsReleased(async () => {
          const net = new FaceLandmark68TinyNet()
          await net.load('base/weights')
          net.dispose()
        })
      })

    })

    describe('forwardInput', () => {

      it('single image element', async () => {
        await expectAllTensorsReleased(async () => {
          const netInput = new NetInput([imgEl1])
          const outTensor = await faceLandmark68TinyNet.forwardInput(netInput)
          outTensor.dispose()
        })
      })

      it('multiple image elements', async () => {
        await expectAllTensorsReleased(async () => {
          const netInput = new NetInput([imgEl1, imgEl1, imgEl1])
          const outTensor = await faceLandmark68TinyNet.forwardInput(netInput)
          outTensor.dispose()
        })
      })

      it('single tf.Tensor3D', async () => {
        const tensor = tf.fromPixels(imgEl1)

        await expectAllTensorsReleased(async () => {
          const netInput = new NetInput([tensor])
          const outTensor = await faceLandmark68TinyNet.forwardInput(netInput)
          outTensor.dispose()
        })

        tensor.dispose()
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1].map(el => tf.fromPixels(el))

        await expectAllTensorsReleased(async () => {
          const netInput = new NetInput(tensors)
          const outTensor = await faceLandmark68TinyNet.forwardInput(netInput)
          outTensor.dispose()
        })

        tensors.forEach(t => t.dispose())
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => tf.fromPixels(imgEl1).expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          const outTensor = await faceLandmark68TinyNet.forwardInput(await toNetInput(tensor))
          outTensor.dispose()
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1]
          .map(el => tf.tidy(() => tf.fromPixels(el).expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          const outTensor = await faceLandmark68TinyNet.forwardInput(await toNetInput(tensors))
          outTensor.dispose()
        })

        tensors.forEach(t => t.dispose())
      })

    })

    describe('detectLandmarks', () => {

      it('single image element', async () => {
        await expectAllTensorsReleased(async () => {
          await faceLandmark68TinyNet.detectLandmarks(imgEl1)
        })
      })

      it('multiple image elements', async () => {
        await expectAllTensorsReleased(async () => {
          await faceLandmark68TinyNet.detectLandmarks([imgEl1, imgEl1, imgEl1])
        })
      })

      it('single tf.Tensor3D', async () => {
        const tensor = tf.fromPixels(imgEl1)

        await expectAllTensorsReleased(async () => {
          await faceLandmark68TinyNet.detectLandmarks(tensor)
        })

        tensor.dispose()
      })

      it('multiple tf.Tensor3Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1].map(el => tf.fromPixels(el))


        await expectAllTensorsReleased(async () => {
          await faceLandmark68TinyNet.detectLandmarks(tensors)
        })

        tensors.forEach(t => t.dispose())
      })

      it('single batch size 1 tf.Tensor4Ds', async () => {
        const tensor = tf.tidy(() => tf.fromPixels(imgEl1).expandDims()) as tf.Tensor4D

        await expectAllTensorsReleased(async () => {
          await faceLandmark68TinyNet.detectLandmarks(tensor)
        })

        tensor.dispose()
      })

      it('multiple batch size 1 tf.Tensor4Ds', async () => {
        const tensors = [imgEl1, imgEl1, imgEl1]
          .map(el => tf.tidy(() => tf.fromPixels(el).expandDims())) as tf.Tensor4D[]

        await expectAllTensorsReleased(async () => {
          await faceLandmark68TinyNet.detectLandmarks(tensors)
        })

        tensors.forEach(t => t.dispose())
      })

    })
  })

})


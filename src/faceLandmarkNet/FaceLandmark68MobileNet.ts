import * as tf from '@tensorflow/tfjs-core';
import { NetInput } from 'tfjs-image-recognition-base';

import { dephtwiseSeparableConv } from './dephtwiseSeparableConv';
import { extractMobilenetParams } from './extractMobilenetParams';
import { FaceLandmark68NetBase } from './FaceLandmark68NetBase';
import { fullyConnectedLayer } from './fullyConnectedLayer';
import { MobilenetParams } from './types';

export class FaceLandmark68MobileNet extends FaceLandmark68NetBase<MobilenetParams> {

  constructor() {
    super('FaceLandmark68MobileNet')
  }

  public runNet(input: NetInput): tf.Tensor2D {

    const { params } = this

    if (!params) {
      throw new Error('FaceLandmark68MobileNet - load model before inference')
    }

    return tf.tidy(() => {
      const batchTensor = input.toBatchTensor(112, true)

      let out = dephtwiseSeparableConv(batchTensor, params.conv0, [2, 2])
      out = dephtwiseSeparableConv(batchTensor, params.conv1, [1, 1])
      out = dephtwiseSeparableConv(batchTensor, params.conv2, [2, 2])
      out = dephtwiseSeparableConv(batchTensor, params.conv3, [1, 1])
      out = dephtwiseSeparableConv(batchTensor, params.conv4, [2, 2])
      out = dephtwiseSeparableConv(batchTensor, params.conv5, [1, 1])
      out = dephtwiseSeparableConv(batchTensor, params.conv6, [2, 2])
      out = dephtwiseSeparableConv(batchTensor, params.conv7, [1, 1])
      out = tf.avgPool(out, [7, 7], [1, 1], 'valid')

      return fullyConnectedLayer(out.as2D(out.shape[0], -1), params.fc)
    })
  }
/*
  protected loadQuantizedParams(uri: string | undefined) {
    return loadQuantizedParams(uri)
  }
*/
  protected extractParams(weights: Float32Array) {
    return extractMobilenetParams(weights)
  }
}
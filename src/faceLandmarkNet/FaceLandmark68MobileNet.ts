import * as tf from '@tensorflow/tfjs-core';
import { NetInput, normalize } from 'tfjs-image-recognition-base';

import { depthwiseSeparableConv } from './depthwiseSeparableConv';
import { extractMobilenetParams } from './extractMobilenetParams';
import { FaceLandmark68NetBase } from './FaceLandmark68NetBase';
import { MobilenetParams } from './types';
import { fullyConnectedLayer } from './fullyConnectedLayer';
import { SeparableConvParams } from 'tfjs-tiny-yolov2/build/tinyYolov2/types';

function residual(
  x: tf.Tensor4D,
  conv1Params: SeparableConvParams,
  conv2Params: SeparableConvParams
): tf.Tensor4D {
  return tf.tidy(() => {
    const out1 = depthwiseSeparableConv(x, conv1Params, [2, 2])
    const out2 = depthwiseSeparableConv(out1, conv2Params, [1, 1], false)
    return tf.relu(tf.add(out1, out2)) as tf.Tensor4D
  })
}

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
      const meanRgb = [122.782, 117.001, 104.298]
      const normalized = normalize(batchTensor, meanRgb).div(tf.scalar(255)) as tf.Tensor4D

      let out = residual(normalized, params.conv0,  params.conv1)
      out = residual(out, params.conv2,  params.conv3)
      out = residual(out, params.conv4,  params.conv5)
      out = tf.avgPool(out, [14, 14], [2, 2], 'valid')

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
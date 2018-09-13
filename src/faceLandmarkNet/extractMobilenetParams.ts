import * as tf from '@tensorflow/tfjs-core';
import { extractWeightsFactory, ParamMapping } from 'tfjs-image-recognition-base';
import { FCParams } from 'tfjs-tiny-yolov2';
import { SeparableConvParams } from 'tfjs-tiny-yolov2/build/tinyYolov2/types';

import { MobilenetParams } from './types';

export function extractMobilenetParams(weights: Float32Array): { params: MobilenetParams, paramMappings: ParamMapping[] } {

  const paramMappings: ParamMapping[] = []

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)

  function extractSeparableConvParams(channelsIn: number, channelsOut: number, mappedPrefix: string): SeparableConvParams {
    const depthwise_filter = tf.tensor4d(extractWeights(3 * 3 * channelsIn), [3, 3, channelsIn, 1])
    const pointwise_filter = tf.tensor4d(extractWeights(channelsIn * channelsOut), [1, 1, channelsIn, channelsOut])
    const bias = tf.tensor1d(extractWeights(channelsOut))

    paramMappings.push(
      { paramPath: `${mappedPrefix}/depthwise_filter` },
      { paramPath: `${mappedPrefix}/pointwise_filter` },
      { paramPath: `${mappedPrefix}/bias` }
    )

    return new SeparableConvParams(
      depthwise_filter,
      pointwise_filter,
      bias
    )
  }

  function extractFCParams(channelsIn: number, channelsOut: number, mappedPrefix: string): FCParams {
    const weights = tf.tensor2d(extractWeights(channelsIn * channelsOut), [channelsIn, channelsOut])
    const bias = tf.tensor1d(extractWeights(channelsOut))

    paramMappings.push(
      { paramPath: `${mappedPrefix}/weights` },
      { paramPath: `${mappedPrefix}/bias` }
    )

    return {
      weights,
      bias
    }
  }

  const conv0 = extractSeparableConvParams(3, 32, 'conv0')
  const conv1 = extractSeparableConvParams(32, 32, 'conv1')
  const conv2 = extractSeparableConvParams(32, 64, 'conv2')
  const conv3 = extractSeparableConvParams(64, 64, 'conv3')
  const conv4 = extractSeparableConvParams(64, 128, 'conv4')
  const conv5 = extractSeparableConvParams(128, 128, 'conv5')
  const fc = extractFCParams(128, 136, 'fc')

  if (getRemainingWeights().length !== 0) {
    throw new Error(`weights remaing after extract: ${getRemainingWeights().length}`)
  }

  return {
    paramMappings,
    params: {
      conv0,
      conv1,
      conv2,
      conv3,
      conv4,
      conv5,
      fc
    }
  }
}
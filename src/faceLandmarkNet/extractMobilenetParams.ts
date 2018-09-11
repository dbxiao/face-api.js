import { extractWeightsFactory, ParamMapping } from 'tfjs-image-recognition-base';
import { extractFCParamsFactory, extractSeparableConvParamsFactory } from 'tfjs-tiny-yolov2';

import { MobilenetParams } from './types';

export function extractMobilenetParams(weights: Float32Array): { params: MobilenetParams, paramMappings: ParamMapping[] } {

  const paramMappings: ParamMapping[] = []

  const {
    extractWeights,
    getRemainingWeights
  } = extractWeightsFactory(weights)
  const extractSeparableConvParams = extractSeparableConvParamsFactory(extractWeights, paramMappings)

  const extractFCParams = extractFCParamsFactory(extractWeights, paramMappings)

  const conv0 = extractSeparableConvParams(3, 32, 'conv0')
  const conv1 = extractSeparableConvParams(32, 64, 'conv1')
  const conv2 = extractSeparableConvParams(64, 128, 'conv2')
  const conv3 = extractSeparableConvParams(128, 128, 'conv3')
  const conv4 = extractSeparableConvParams(128, 256, 'conv4')
  const conv5 = extractSeparableConvParams(256, 256, 'conv5')
  const conv6 = extractSeparableConvParams(256, 512, 'conv6')
  const conv7 = extractSeparableConvParams(512, 512, 'conv7')
  const fc = extractFCParams(512, 136, 'fc')

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
      conv6,
      conv7,
      fc
    }
  }
}
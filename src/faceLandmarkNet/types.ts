import { ConvParams, FCParams } from 'tfjs-tiny-yolov2';
import { SeparableConvParams } from 'tfjs-tiny-yolov2/build/tinyYolov2/types';

export type NetParams = {
  conv0: ConvParams
  conv1: ConvParams
  conv2: ConvParams
  conv3: ConvParams
  conv4: ConvParams
  conv5: ConvParams
  conv6: ConvParams
  conv7: ConvParams
  fc0: FCParams
  fc1: FCParams
}

export type MobilenetParams = {
  conv0: SeparableConvParams
  conv1: SeparableConvParams
  conv2: SeparableConvParams
  conv3: SeparableConvParams
  conv4: SeparableConvParams
  conv5: SeparableConvParams
  conv6: SeparableConvParams
  conv7: SeparableConvParams
  fc: FCParams
}
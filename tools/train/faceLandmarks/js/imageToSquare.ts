import { createCanvas, createCanvasFromMedia, getContext2dOrThrow } from 'tfjs-image-recognition-base';
import { getMediaDimensions } from 'tfjs-tiny-yolov2';

// TODO, move this to tfjs-image-recognition-base
export function imageToSquare(input: HTMLImageElement | HTMLCanvasElement, inputSize: number, centerImage: boolean = false) {

  if (!(input instanceof HTMLImageElement || input instanceof HTMLCanvasElement)) {
    throw new Error('imageToSquare - expected arg0 to be HTMLImageElement | HTMLCanvasElement')
  }

  const dims = getMediaDimensions(input)
  const scale = inputSize / Math.max(dims.height, dims.width)
  const width = scale * dims.width
  const height = scale * dims.height

  const targetCanvas = createCanvas({ width: inputSize, height: inputSize })
  const inputCanvas = input instanceof HTMLCanvasElement ? input : createCanvasFromMedia(input)

  const offset = Math.abs(width - height) / 2
  const dx = centerImage && width < height ? offset : 0
  const dy = centerImage && height < width ? offset : 0
  getContext2dOrThrow(targetCanvas).drawImage(inputCanvas, dx, dy, width, height)

  return targetCanvas
}
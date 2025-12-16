import * as tf from '@tensorflow/tfjs';

// Checks if two tensors are the same
export function checkTensors(
  firstTensor: tf.Tensor,
  secondTensor: tf.Tensor,
  checkElements: boolean
): boolean {
  // Same shape
  if (
    JSON.stringify(firstTensor.shape) !== JSON.stringify(secondTensor.shape)
  ) {
    return false;
  }

  // Same element type
  if (firstTensor.dtype !== secondTensor.dtype) {
    return false;
  }

  // Same elements
  if (checkElements && firstTensor.equal(secondTensor).dataSync()[0] !== 1) {
    return false;
  }

  return true;
}

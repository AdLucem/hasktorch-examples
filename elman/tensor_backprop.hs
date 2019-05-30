-- Skeletal code for what I want to do

data SomeInput a = Tensor '[1, a]

tensorOp :: (KnownDim a, KnownNat a)
         => Tensor [1, a]
         -> SomeInput a
         -> Double

data TensorOp a = Tensor [1, a] -> SomeInput a -> Double

autograd :: TensorOp a
         -> Tensor '[1, a]
         -> Tensor '[1, a]

data InputTensor numInputs numFeatures = Tensor '[numInputs, numFeatures]

sgdOptimize :: TensorOp a
            -> InputTensor
            -> Tensor '[1, a] -- initial parameters
            -> Tensor '[1, a] -- optimized parameters










-- Skeletal code for linear regression

data SomeInput a = Tensor '[1, a]
type SomeOutput a = SomeInput a


-- a function involving Hasktorch tensors
tensorOp :: (KnownDim a, KnownNat a)
         => Tensor [1, a]  -- vector of parameters
         -> SomeInput a    -- input vector
         -> Double         -- output


data TensorOp a = Tensor '[1, a] -> SomeInput a -> Double
data TensorGrad a = Tensor '[1, a] -> Tensor '[1, a]


-- gradient of the tensor operation
tensorOpGrad :: Tensor '[1, a]  -- vector w.r.t which we take gradient
             -> Tensor '[1, a]  -- gradient vector


data InputTensor numInputs numFeatures = Tensor '[numInputs, numFeatures]
type OutputTensor = InputTensor


-- stochastic gradient descent for the above-defined tensor operation
sgdOptimize :: InputTensor a b
            -> Tensor '[1, a] -- initial parameters
            -> Tensor '[1, a] -- optimized parameters


-- explicitly a loss function
loss :: (KnownDim a, KnownNat a)
     => Tensor [1, a]  -- vector of parameters
     -> SomeInput a    -- input vector
     -> SomeOutput a   -- actual output
     -> Double         -- error value


-- gradient of loss function
lossGrad :: (KnownDim a, KnownNat a)
         => SomeInput a
         -> SomeOutput a
         -> Tensor [1, a]  -- vector of parameters
         -> Tensor '[1, a] -- gradient vector


-- loss function gradient, partially applied to input
-- and output
lossGradWithInput :: (KnownDim a, KnownNat a)
                  => SomeInput a
                  -> SomeOutput a
                  -> TensorGrad a
lossGradWithInput inp out = lossGrad inp out


-- generalize SGD function
sgdOptimize' :: TensorOp a
             -> TensorGrad a
             -> InputTensor a b
             -> OutputTensor a b
             -> Tensor '[1, a] -- initial parameters
             -> Tensor '[1, a] -- optimized parameters


-- linear regression function
linearRegression :: InputTensor a b
                 -> OutputTensor a b
                 -> Tensor '[a, b] -- input parameters
                 -> Tensor '[a, b] -- output parameters










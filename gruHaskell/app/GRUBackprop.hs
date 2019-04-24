{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module GRUBackprop where

import qualified Torch.Double as T
import Numeric.Backprop

-- | loss function- a simple sum of squares
loss :: (T.KnownDim a, T.KnownDim b) => T.Tensor '[a, b] -> T.Tensor '[a, b] -> Double
loss output target =
    let
        diff = target - output
    in
        T.dot diff diff


-- | GRU output layer gradient function placeholder.
-- | not actually a gradient function
{-# gruGradOutput ::
    (T.KnownDim a, T.KnownDim b, T.KnownDim c, T.KnownDim d) =>
    Double -> -- differential of loss function
    T.Tensor '[a, b] -> -- input-hidden resetWeights
    T.Tensor '[a, b] -> -- input-hidden updateWeights
    T.Tensor '[a, b] -> -- input-hidden weights
    T.Tensor '[c, d] -> -- hidden-hidden resetWeights
    T.Tensor '[c, d] -> -- hidden-hidden updateWeights
    T.Tensor '[c, d] -> -- hidden-hidden weights
    Double
gruGradOutput l rWih uWih nWih rWhh uWhh nWhh = T.sumall (rWih + uWih + nWih) / T.sumall (rWhh + uWhh + nWhh)
#-}
-- ihResetGrad ::
--    (T.KnownDim a, T.KnownDim b) =>
--    Double ->
--    T.Tensor '[a, b] ->
--    T.Tensor '[a, b] -> -- input-hidden resetBias. assumed to be 0 for now
--    T.Tensor '[a, b]

--ihUpdateGrad ::
--    (T.KnownDim a, T.KnownDim b) =>
--    Double ->
--    T.Tensor '[a, b] ->
--    T.Tensor '[a, b] -> -- input-hidden updateBias. assumed to be 0 for now
--    T.Tensor '[a, b]

-- ihNewGradOutput ::
--    (T.KnownDim a, T.KnownDim b) =>
--    Double ->
--    T.Tensor '[a, b] ->
--    T.Tensor '[a, b]
--ihNewGradOutput

-- update :: Tensor -> Tensor -> Int

-- unroll -> [GRNN] -> GRUTimeSteps








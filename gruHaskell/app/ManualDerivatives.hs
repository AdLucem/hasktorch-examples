-- manually writing the derivatives
-- until I figure out how to use `ad`
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module ManualDerivatives where

import Torch.Double as T

lossDiff ::
    (T.KnownDim a, T.KnownDim b) =>
    T.Tensor '[a, b] ->
    T.Tensor '[a, b] ->
    Double
lossDiff target output = T.sumall (target - output)


ihUpdateGradient ::
    (T.KnownDim a, T.KnownDim b) =>
    T.Tensor '[a, b] ->  -- output-loss gradient
    T.Tensor '[a, b] ->  -- updateGateOutput
    T.Tensor '[a, b] ->  -- newGateOutput
    T.Tensor '[a, b]
ihUpdateGradient og ig ng = og * (1 - ig) * (1 - ng**2)

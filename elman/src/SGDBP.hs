{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}

module SGDBP where

import qualified Torch.Double as T
import qualified Numeric.Dimensions as D
import GHC.TypeLits
import Numeric.Backprop
import Data.Typeable


type Tensor = T.Tensor '[1, 2]

type BTensor s = BVar s (T.Tensor '[1, 2])

type BDouble s = BVar s Double

n :: Double
n = 2.0


-- | returns the error between predicted values
-- | and actual values, when given inputs
-- | and a set of parameters for prediction
errBP :: Reifies s W
       => BTensor s
       -> BTensor s
       -> BTensor s
       -> BTensor s
errBP inp act params = ((inp * params) - act) ^ 2


-- | A backprop-able version of T.sumall
sumallBP :: Reifies s W
         => BTensor s
         -> BDouble s
sumallBP =
  liftOp1 . op1 $ \t -> (T.sumall t, dx)
  where
    dx :: T.HsAccReal -> Tensor
    dx x = T.constant x


lossBP :: Reifies s W
       => BTensor s
       -> BTensor s
       -> BTensor s
       -> BDouble s
lossBP inputs outputs params =
  0.5 * (sumallBP $ errBP inputs outputs params)


{-
-- TODO: type error in 's'
-- gradient of MSE function w.r.t every tensor element
lossGradBP :: Reifies s W
           => BTensor s
           -> BTensor s
           -> Tensor
           -> Tensor
lossGradBP inputVars outputVars params =
  gradBP (lossBP inputVars actualVars) params
-}



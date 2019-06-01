{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}

module SGD where

import qualified Torch.Double as T
import qualified Numeric.Dimensions as D
import GHC.TypeLits
import Numeric.Backprop
import Data.Typeable


type Toy = T.Tensor '[1, 2]

type BTensor s = BVar s (T.Tensor '[1, 2])

type BAccReal s = BVar s T.HsAccReal


-- | returns the error between predicted values
-- | and actual values, when given inputs
-- | and a set of parameters for prediction
err :: Reifies s W
    => BTensor s
    -> BTensor s
    -> BTensor s
    -> BTensor s
err inp act params = (inp * params) - act


-- takes in a set of inputs and returns a
-- partially applied function that can calculate
-- the difference between predicted and actual
errForInputs :: Reifies s W
             => BTensor s
             -> (BTensor s ->  BTensor s ->  BTensor s)
errForInputs inp = err inp


-- | returns the squared error between predicted values
-- | and actual values
sqErr :: Reifies s W
      => BTensor s
      -> BTensor s
      -> BTensor s
      -> BTensor s
sqErr inp act params =  ((inp * params) - act) ^ 2


-- | A backprop-able version of T.sumall
sumallBP :: Reifies s W
         => BTensor s
         -> BAccReal s
sumallBP =
  liftOp1 . op1 $ \t -> (T.sumall t, (dx t))
  where
    dx :: Toy -> T.HsAccReal -> Toy
    dx t x = T.cmul t (T.constant x)

{-
-- Takes the mean of the given value
meanSquareError :: Reifies s W
                => BAccReal s
                -> BAccReal s
                -> BAccReal s
meanSquareError n inp act params = val / n


step :: T.HsReal -> Toy -> Toy -> Toy
step eta t grad = T.csub t eta grad


sgdIter :: T.HsReal  -- learning rate
        -> Int       -- number of iterations
        -> Toy       -- input vector-- TODO: make stochastic, give it 2-dim tensor
        -> Toy       -- output vector -- TODO: same as above
        -> Toy       -- parameters to be iterated on
        -> Toy       -- final parameters
sgdIter eta 0 input actual params = params
sgdIter eta steps input actual params =
  sgdIter eta (steps - 1) input actual updated
  where
    actualVar = auto actual
    inputVar = auto input
    gradient = gradBP ((mean 2) . sumallBP . (sqErr inputVar actualVar)) params
    updated = step eta params gradient



sgdLoss :: T.HsReal -> T.HsReal -> Toy -> Toy -> Toy
sgdLoss eps eta actual params =
  if  (abs eval) > (abs eps)
  then
    sgdLoss eps eta actual updated
  else
    params
  where
    actualVar = auto actual
    eval = evalBP ((meanSquare 2) . (err actualVar)) params
    grad = gradBP ((meanSquare 2) . (err actualVar)) params
    updated = step eta params grad
-}

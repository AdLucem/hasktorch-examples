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



err :: Reifies s W
    => BTensor s
    -> BTensor s
    -> BTensor s
    -> BTensor s
err act inp params = (inp * params) - act


-- BUG: why does this function return [NaN, NaN]
-- when tensorA < tensorB ???
sqErr :: Reifies s W
      => BTensor s
      -> BTensor s
      -> BTensor s
      -> BTensor s
-- arguments: actual values, predicted values
-- returns a tensor composed of square errors
sqErr act inp params =  (err act inp params) ^ 2


-- | A backprop-able version of T.sumall
-- | that adds the squared version of the tensor
-- NOTE: am here 30/5/19, am trying to get the derivative
-- w.r.t both input and params
squaredSumallBP :: Reifies s W
                => BTensor s
                -> BAccReal s
squaredSumallBP =
  liftOp1 . op1 $ \t -> (T.sumall t ^ 2, (dx t))
  where
    dx :: Toy -> T.HsAccReal -> Toy
    dx t x = T.cmul t (T.constant x)

{-
-- Takes the mean of the square of the given tensor
meanSquare :: Reifies s W
     => BAccReal s
     -> BTensor s
     -> BAccReal s
meanSquare n t = (squaredSumallBP t) / n


step :: T.HsReal -> Toy -> Toy -> Toy
step eta t grad = T.csub t eta grad


sgdIter :: T.HsReal -> Int -> Toy -> Toy -> Toy
sgdIter eta 0 actual params = params
sgdIter eta steps actual params =
  sgdIter eta (steps - 1) actual updated
  where
    actualVar = auto actual
    grad = gradBP ((meanSquare 2) . (err actualVar)) params
    updated = step eta params grad


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
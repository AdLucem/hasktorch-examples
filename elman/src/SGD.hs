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


-- just takes the difference (error) between two tensors
err :: Reifies s W
    => BTensor s
    -> BTensor s
    -> BTensor s
err act pr = pr - act


-- BUG: why does this function return [NaN, NaN]
-- when tensorA < tensorB ???
sqErr :: Reifies s W
      => BTensor s
      -> BTensor s
      -> BTensor s
-- arguments: actual values, predicted values
-- returns a tensor composed of square errors
sqErr act pr =  (err act pr) ^ 2


-- | A backprop-able version of T.sumall
-- | that adds the squared version of the tensor
squaredSumallBP :: Reifies s W
                => BTensor s
                -> BAccReal s
squaredSumallBP =
  liftOp1 . op1 $ \t -> (T.sumall (t ^ 2), (dx t))
  where
    dx :: Toy -> T.HsAccReal -> Toy
    dx t x = T.cmul t (T.constant x)


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

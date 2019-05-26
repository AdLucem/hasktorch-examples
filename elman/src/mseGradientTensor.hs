{- Note: this right now is MSE optimization where all inputs = 1. i.e: the 'predicted' output, or the hypothesis, is only the weights. And both hypothesis and actual are integers, and learning rate is 1.0. Theoretically this should converge perfectly. -}


{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}


import qualified Torch.Double as T
import qualified Numeric.Dimensions as D
import GHC.TypeLits
import Numeric.Backprop
import Data.Typeable
--import Numeric.Backprop.Internal


type Toy = T.Tensor '[1, 2]

type BTensor s = BVar s (T.Tensor '[1, 2]) 

type BAccReal s = BVar s T.HsAccReal


-- | defining a tensor of size 1 x something
makeTensor :: (D.KnownDim a, KnownNat a) 
           => [Double] 
           -> IO (T.Tensor '[1, a])
makeTensor ls = do
    t <- T.fromList ls
    case t of
        Nothing -> return T.empty
        Just x -> return x

---------- Tensor Math Functions -----------


-- just takes the difference between two tensors
err :: Reifies s W
    => BTensor s
    -> BTensor s
    -> BTensor s
err act pr = pr - act


-- | 'xt' is the predicted
-- BUG: why does this function return [NaN, NaN]
-- when tensorA < tensorB ???
sqErr :: Reifies s W
      => BTensor s  
      -> BTensor s
      -> BTensor s
-- the infix power operator - ** - returned the above error
-- when used on a tensor with negative numbers
-- so here's my hack for squaring
sqErr act pr =  (err act pr) ^ 2


-- | A backprop-able version of T.sumall
-- | that adds the squared version of the tensor
sumSquaredBP :: Reifies s W
             => BTensor s
             -> BAccReal s
sumSquaredBP = 
  liftOp1 . op1 $ \t -> (T.sumall (t ^ 2), (dx t))
  where
    dx :: Toy -> T.HsAccReal -> Toy
    dx t x = T.cmul t (T.constant x)


-- Takes the mean of the square of the given tensor
meanSquare :: Reifies s W
     => BAccReal s
     -> BTensor s
     -> BAccReal s
meanSquare n t = (sumSquaredBP t) / n



main = do
    ta :: T.Tensor '[1, 2] <- makeTensor [6.0, 6.0]
    let bta = auto ta 
    tb :: T.Tensor '[1, 2] <- makeTensor [9.0, 9.0]
    -- so, we have a backprop-able version
    -- of a tensor function, but we had to specifically
    -- make a BP version of sum-squared, because I ran into 
    -- type matching hell when I tried to make a purely 
    -- BP-compatible version of =sumall=. The problem is
    -- internal derivatives- I have to explicitly define the
    -- internal derivative of the function, which leads to more
    -- arguments than I want to. This can be solved by simply
    -- using op2 or opN
    print $ evalBP ((mean 2) . (err bta)) tb 
    print $ gradBP ((mean 2) . (err bta)) tb
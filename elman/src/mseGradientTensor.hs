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
sqErr tensorA tensorB = (tensorB - tensorA) * (tensorB - tensorA)


-- | A backprop-able version of T.sumall
-- | with squared error
sumallBP :: Reifies s W
         => BTensor s
         -> BAccReal s
sumallBP = 
  liftOp1 . op1 $ \t -> (T.sumall t, (dx t))
  where
    dx :: Toy -> T.HsAccReal -> Toy
    dx t x = T.cmul t (T.constant x)


mean :: Reifies s W
     => BAccReal s
     -> BTensor s
     -> BAccReal s
mean n t = (sumallBP t) / n



main = do
    ta :: T.Tensor '[1, 2] <- makeTensor [6.0, 6.0]
    let bta = auto ta 
    tb :: T.Tensor '[1, 2] <- makeTensor [9.0, 9.0]
    -- so, we have a (theoretically) backprop-able version
    -- of a tensor function
    -- very theoretically, and this is not the nice type of "theory"
    print $ gradBP ((mean 2) . (sqErr bta)) tb
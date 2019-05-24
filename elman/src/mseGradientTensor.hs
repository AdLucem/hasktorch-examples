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
squareErr :: Reifies s W
          => BTensor s  
          -> BTensor s
          -> BTensor s
-- the infix power operator - ** - returned the above error
-- when used on a tensor with negative numbers
-- so here's my hack for squaring
squareErr tensorA tensorB = (tensorB - tensorA) * (tensorB - tensorA)


squareErr' :: Toy
           -> Toy  
           -> Toy
squareErr' actual xt = 
    gradBP (squareErr $ auto actual) xt


mean :: Reifies s W 
     => BVar s Double
     -> BTensor s  
     -> BAccReal s
mean n t = (sumallBP t) / n


-- | A backprop-able version of T.sumall
sumallBP :: Reifies s W
         => BTensor s
         -> BAccReal s
sumallBP = 
  liftOp1 . op1 $ \t -> (T.sumall t, (dx t))
  where
    dx :: Toy -> T.HsAccReal -> Toy
    -- the differential provided here is wrong, but I was trying to get the 
    -- types to line up
    -- the hack to multiply a tensor with a `HsReal` value: 
    -- (tensorA + c*tensorA) - tensorA = c*tensorA
    dx t x = T.csub (T.cadd t (T.acc2real x) t) 1.0 t


main = do
    ta :: T.Tensor '[1, 2] <- makeTensor [9.0, 9.0]
    let bta = auto ta 
    tb :: T.Tensor '[1, 2] <- makeTensor [7.0, 7.0]
    -- so, we have a (theoretically) backprop-able version
    -- of a tensor function
    -- very theoretically, and this is not the nice type of "theory"
    print $ evalBP ((mean 2) . squareErr bta) tb
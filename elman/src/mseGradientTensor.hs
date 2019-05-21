{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}


import qualified Torch.Double as T
import qualified Numeric.Dimensions as D
import GHC.TypeLits
import Numeric.Backprop
--import Numeric.Backprop.Internal


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


meanSquareErr :: Reifies s W 
              => BVar s Double
              -> BTensor s 
              -> BTensor s 
              -> BTensor s
meanSquareErr n xt yt = (xt - yt) ** 2


main = do
    ta :: T.Tensor '[1, 2] <- makeTensor [9, 9]
    let bta = auto ta 
    tb :: T.Tensor '[1, 2] <- makeTensor [7, 7]
    let id :: T.Tensor '[2, 1] = T.new
    T.fill_ id 1.0 
-- yes I went full-on lisp here because I ran out of patience
    print (
        evalBP (
            meanSquareErr 2 bta) 
        tb :: T.Tensor '[1, 2])
    print (
        gradBP (
            meanSquareErr 2 bta) 
        tb :: T.Tensor '[1, 2])
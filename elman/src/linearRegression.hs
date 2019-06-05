
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
import SGD
import SGDBP
import Control.Monad.Except


-- | defining a tensor of size 1 x something
makeTensor :: (D.KnownDim a, KnownNat a)
           => [Double]
           -> IO (T.Tensor '[1, a])
makeTensor ls = do
    t <- T.fromList ls
    case t of
        Nothing -> return T.empty
        Just x -> return x


makeMatrix :: [[Double]]
           -> IO (T.Tensor '[3, 2])
makeMatrix ls = do
    t <- runExceptT $ T.matrix ls
    case t of
        Left a -> return T.empty
        Right b -> return b


safeGetRow :: T.Tensor '[3, 2]
           -> Word
           -> T.Tensor '[1, 2]
safeGetRow t r = 
    case (T.getRow t r) of
        Nothing -> T.empty 
        Just x -> x


main = do
    actual :: T.Tensor '[3, 2] <- makeMatrix [[6.0, 6.0] | x <- [1..3]]
    let actualVar = auto actual
    input :: T.Tensor '[3, 2] <- makeMatrix [[2.0, 2.0] | x <- [1..3]]
    let inputVar = auto input
    params :: T.Tensor '[1, 2] <- makeTensor [5.0, 5.0]
    -- mat :: T.Tensor '[3, 2] <- makeMatrix [[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]
    print $ sgdOptimize loss lossGrad 0.1 0.0000001 input actual params
    -- print $ gdOptimizeBP 0.1 0.0000001 input actual params
    -- print $ T.getRow mat 0

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


-- | defining a tensor of size 1 x something
makeTensor :: (D.KnownDim a, KnownNat a)
           => [Double]
           -> IO (T.Tensor '[1, a])
makeTensor ls = do
    t <- T.fromList ls
    case t of
        Nothing -> return T.empty
        Just x -> return x


main = do
    actual :: T.Tensor '[1, 2] <- makeTensor [6.0, 6.0]
    let actualVar = auto actual
    input :: T.Tensor '[1, 2] <- makeTensor [2.0, 2.0]
    let inputVar = auto input
    params :: T.Tensor '[1, 2] <- makeTensor [4.0, 4.0]
    let eta = 1.0
    print $ evalBP (sqErr actualVar inputVar) params 

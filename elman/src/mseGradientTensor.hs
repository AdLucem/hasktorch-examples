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
import SGD
--import Numeric.Backprop.Internal


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
    params :: T.Tensor '[1, 2] <- makeTensor [9.0, 9.0]
    let eta = 1.0
    -- NOTE: so, we have a backprop-able version
    -- of a tensor function, but we had to specifically
    -- make a BP version of sum-squared, because I ran into
    -- type matching hell when I tried to make a purely
    -- BP-compatible version of =sumall=. The problem is
    -- internal derivatives- I have to explicitly define the
    -- internal derivative of the function, which leads to more
    -- arguments than I want to. This can be solved by simply
    -- using op2 or opN

    -- this is iterative SGD running for a fixed number
    -- of timesteps
    print $ sgdIter eta 10 actual params
    -- this is SGD running until the loss function has
    -- been minimized to a given amount
    print $ sgdLoss 0.000000005 eta actual params

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import qualified Torch.Double as T
import qualified Torch.Core.Random as RNG
import qualified Numeric.Dimensions as D
import GHC.TypeLits

import qualified GRU as GRU


-- | defining a tensor of size 1 x something
makeTensor :: (D.KnownDim a, KnownNat a) => [Double] -> IO (T.Tensor '[1, a])
makeTensor ls = do
    t <- T.fromList ls
    case t of
        Nothing -> return T.empty
        Just x -> return x


-- | make a RNN composed of a single GRU cell
--singleGRU = GRUCell 4 4

-- The point of defining something that is technically a GRU Cell
-- in the main function is to show that all the types line up
-- and that we can make an object of type GRUCell that can be manipulated
main = do
    i <- makeTensor [1, 2, 3, 4]
    h <- makeTensor [5, 6, 7, 8]
    let (w_i :: GRU.Weights '[1, 4]) = GRU.Weights i i i i i i
    let (w_h :: GRU.Weights '[1, 4]) = GRU.Weights h h h h h h
    let (gru_cell :: GRU.GRUCell '[1, 4] '[1, 4]) = GRU.GRUCell w_i w_h
    print $ 2.00000000 * i * h
    --let (rnnLayer1 :: GRU.GRNN '[1, 4] '[1, 4]) = GRU.Single gru_cell
    --let (rnnLayer2 :: GRU.GRNN '[1, 4] '[1, 4]) = GRU.Composed gru_cell rnnLayer1
    --putStrLn $ show rnnLayer2

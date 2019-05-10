----------------------------------------------------------------------------------
-- |
-- Module    :  Main
-- Maintainer:  AdLucem
-- Stability :  experimental
-- Portability: non-portable

-----------------------------------------------------------------------------------

-------------------------------------- IMPORTS ------------------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import qualified Data.Vector.Unboxed as V
import qualified Numeric.Dimensions as D
import GHC.TypeLits

import qualified Torch.Double as T
-- Local imports:

import DataLoader

-- Training a convolutional neural net to classify pictures from the fashion-mnist dataset

--------------------------------------- MAIN --------------------------------------

getLabels :: FilePath -> IO (V.Vector (Int))
getLabels filename = do
    l <- fetchLabelsFrom filename
    let l' = labelsProcess l
    return l'


-- | The driver function
main = do
    d <- getData "fashion-mnist/fashion/t10k-images-idx3-ubyte"
    (t :: [T.Tensor '[1, 784]]) <- getTensors d
    putStrLn "HELLO"
    --let dim = fst d
    --let feat = snd d
    --putStrLn $ show dim
    --putStrLn $ show $ length feat

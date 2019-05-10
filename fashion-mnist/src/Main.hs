----------------------------------------------------------------------------------
-- |
-- Module    :  Main
-- Maintainer:  AdLucem
-- Stability :  experimental
-- Portability: non-portable

-----------------------------------------------------------------------------------

-------------------------------------- IMPORTS ------------------------------------

module Main where

import qualified Data.Vector.Unboxed as V

-- Local imports:

import DataLoader

-- Training a convolutional neural net to classify pictures from the fashion-mnist dataset

--------------------------------------- MAIN --------------------------------------

getData :: FilePath -> IO ((V.Vector Int, V.Vector Double))
getData filename = do
    d <- fetchDataFrom filename
    let d' = dataProcess d
    return d'


getLabels :: FilePath -> IO (V.Vector (Int))
getLabels filename = do
    l <- fetchLabelsFrom filename
    let l' = labelsProcess l
    return l'


-- | The driver function
main = putStrLn "HELLO"

------------------------------------------------------------------------------
-- |
-- Module    :  DataLoader
-- Maintainer:  AdLucem
-- Stability :  experimental
-- Portability: non-portable

-- This is the module containing functions that loads, reads
-- and processes data from IDX-format files to tensors.

--------------------------------- IMPORTS -------------------------------------
{-# LANGUAGE DataKinds #-}

module DataLoader where

-- the library we're using to encode/decode the IDX data format is the mnist-idx library

import Data.IDX
import Data.IDX.Internal
import qualified Data.Vector.Unboxed as V
import qualified Data.Vector.Generic as V'

-- importing tensor and functions to convert to tensors

import Torch.Double

------------------------------- READING IN IDX DATA ----------------------------

-- First off, we write functions to read in the IDX data files, like so:


-- | This function reads the content of the file and
-- | stores it within the library's handy IDXData type
-- |which is encased within a Maybe.
fetchDataFrom :: FilePath -> IO (Maybe IDXData)
fetchDataFrom filename = do
    filedata  <- decodeIDXFile filename
    return filedata


-- | This function processes the raw IDXData type
-- | into a vector that gives us the dimensions of the data -
-- | (num_images, size_of_image_x, size_of_image_y) -
-- |and the raw data vector, separately.
dataProcess :: Maybe IDXData -> (V.Vector Int, V.Vector Double)
-- `empty` is a function from Vector.Unboxed that returns
-- a new empty vector, in case there is no data returned
dataProcess Nothing = (V.empty, V.empty)
-- but if there is data returned in the idx format, we
-- extract the dimensions and the raw data from the format
dataProcess (Just idxdata) =
    let
      dim = idxDimensions idxdata
      n = dim V.! 0
      features = toVector idxdata
    in
      (dim, features)


-- | This is a helper function to extract the
-- | raw data vector from the IDXData type.
toVector :: IDXData -> V.Vector Double
toVector (IDXDoubles _ _ vec) = vec
toVector (IDXInts _ _ vec) = V.map fromIntegral vec


-- | this is a helper function to convert
-- | from a vector to a tensor of dimensions (m, n)
--toTensor :: Int ->
--            Int ->
--            V.Vector Double ->
--            IO (Tensor '[2, 3])
--toTensor m n vec =
--  let
--    tens = fromList $ V'.toList vec
--  in
--    case tens of
--      Nothing -> return Torch.Double.empty
--      Just x  -> return x


------------------------------- READING IDX LABELS ----------------------------------

-- Similarly, we write functions to read in the IDX labels files.
-- These functions are separate because mnist-idx deals with
-- IDX labels data using a separate type, IDXLabels


 -- | This function reads in the labels file content and
 -- | converts it to mnist-idx's handy IDXLabels type
 -- | encased within a Maybe.
fetchLabelsFrom :: FilePath -> IO (Maybe IDXLabels)
fetchLabelsFrom filename = do
    labels <- decodeIDXLabelsFile filename
    return labels


-- | This function processes the raw IDXLabels type
-- | into a single vector of Ints, that is the labels vector
labelsProcess :: Maybe IDXLabels -> V.Vector Int
-- this function follows a similar pattern to DataProcess
labelsProcess Nothing = V.empty
labelsProcess (Just idxLabelsData) =
    let
      labels = toLabelsVector idxLabelsData
    in
      labels


-- | This is a helper function to extract
-- | the labels vector from the IDXLabels type.
toLabelsVector :: IDXLabels -> V.Vector Int
toLabelsVector (IDXLabels vec) = vec







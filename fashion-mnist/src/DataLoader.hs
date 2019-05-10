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
import qualified Numeric.Dimensions as D
import GHC.TypeLits

-- importing tensor and functions to convert to tensors

import qualified Torch.Double as T

------- Utils --------

-- | Divvy up a vector into segments of size 'n'
divvyN :: Int -> V.Vector Double -> [V.Vector Double]
divvyN n vec = 
  let
    slice = fst $ V.splitAt n vec
    remainder = snd $ V.splitAt n vec
  in
    if remainder == V.empty
    then slice : []
    else slice : (divvyN n remainder)


------- Vectors To Tensors ---------

-- mnist-idx stores the data as vectors, and we want to manipulate tensors,
-- which is why we make a helper function to convert to Tensors

-- | takes in a vector, and returns a (1, x) tensor 
-- | from the vector
toTensor :: (D.KnownDim a, KnownNat a) => V.Vector Double -> IO (T.Tensor '[1, a])
toTensor vec = do
    t <- T.fromList $ V.toList vec
    case t of
        Nothing -> return T.empty
        Just x -> return x

------- READING IN IDX DATA -------

-- First off, we write functions to read in the IDX data files, like so:


-- | This function reads the content of the file and
-- | stores it within the library's handy IDXData type
-- |which is encased within a Maybe.
getData :: FilePath -> IO (Maybe IDXData)
getData filename = do
    filedata  <- decodeIDXFile filename
    return filedata


-- | This function processes the raw IDXData type
-- | into a vector that gives us the dimensions of the data -
-- | (num_images, size_of_image_x, size_of_image_y)
getDimensions :: Maybe IDXData -> [Int]
-- returns empty list in case there is no data returned
getDimensions Nothing = []
-- but if there is data returned in the idx format, we
-- extract the dimensions from the format
getDimensions (Just idxdata) =
    let
      dim = idxDimensions idxdata
    in
      V.toList dim


----------------------------------------------------------------
-- | This function processes the raw IDXData type
-- | into a vector that gives us the 
-- | raw data tensors
getTensors 
  :: (D.KnownDim a, KnownNat a) 
  => Maybe IDXData 
  -> IO ([T.Tensor '[1, a]])
-- returns empty list in case there is no data returned
getTensors Nothing = mapM toTensor [V.empty]
-- but if there is data returned in the idx format, we
-- extract the data from the format and split the vector
-- into tensors of size <feature_list_size>
getTensors (Just idxdata) =
  let
    dim = idxDimensions idxdata
    a = dim V.! 1
    b = dim V.! 2
    imageSize = a * b
    featVectors = divvyN imageSize $ toVector idxdata
    featTensors = mapM toTensor featVectors
  in
    featTensors 
       
----------------------------------------------------------------
-- | This is a helper function to extract the
-- | raw data vector from the IDXData type.
toVector :: IDXData -> V.Vector Double
toVector (IDXDoubles _ _ vec) = vec
toVector (IDXInts _ _ vec) = V.map fromIntegral vec


-------- READING IDX LABELS -------

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







{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Torch.Double

-- main :: IO ()
main = do
  let Just (v :: Tensor '[2]) = fromList [1..2]
  print v

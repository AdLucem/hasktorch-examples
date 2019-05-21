{-# LANGUAGE FlexibleContexts #-}

import Numeric.Backprop

{- Let's get familiar with the type-level stuff important for understanding Justin Le's Backprop library. We'll start off with a simple numeric function. -} 

type BPair s a = BVar s (a, a)

type BList s a = BVar s [a] 

sqErr :: Reifies s W 
      => BVar s Double 
      -> BVar s Double 
      -> BVar s Double
sqErr x y = (x - y)**2


meanSquareErr :: Reifies s W 
              => BVar s Double
              -> BList s Double 
              -> BList s Double 
              -> BVar s Double
meanSquareErr n xs ys = (sum $ zipWith sqErr xs' ys') / n
    where
        xs' = sequenceVar xs
        ys' = sequenceVar ys


-- | The partially applied function trick
-- | since we only want grad w.r.t one output 

main = do
    let la = auto [9.0, 9.0]
    let lb = [7.0, 7.0]
    print (evalBP (meanSquareErr (fromIntegral $ length lb) la) lb :: Double)
    print (gradBP (meanSquareErr (fromIntegral $ length lb) la) lb :: [Double])

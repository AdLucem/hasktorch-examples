{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}

module SGD where

import qualified Torch.Double as T
import qualified Numeric.Dimensions as D
import GHC.TypeLits
import Numeric.Backprop
import Data.Typeable


type Tensor = T.Tensor '[1, 2]
type InputTensor = T.Tensor '[3, 2]
type OutputTensor = T.Tensor '[3, 2]


type BTensor s = BVar s (T.Tensor '[1, 2])

type BAccReal s = BVar s T.HsAccReal

n :: Double
n = 2.0

-- | utility function using getRow
getRow' :: T.Tensor '[3, 2]
        -> Word
        -> T.Tensor '[1, 2]
getRow' t r = 
    case (T.getRow t r) of
        Nothing -> T.empty 
        Just x -> x


-- | returns the error between predicted values
-- | and actual values, when given inputs
-- | and a set of parameters for prediction
err :: Tensor
    -> Tensor
    -> Tensor
    -> Tensor
err inp act params = ((inp * params) - act) ^ 2


loss :: Tensor
     -> Tensor
     -> Tensor
     -> Double
loss inputs outputs params =
  (T.sumall $ err inputs outputs params) / n


-- gradient of MSE function w.r.t every tensor element
lossGrad :: Tensor
         -> Tensor
         -> Tensor
         -> Tensor
lossGrad inputs outputs params =
  T.cdiv (2 * ((inputs * params) - outputs) * inputs) (T.constant n)


gdStep :: Double -- learning rate
       -> Tensor -- parameters
       -> Tensor -- gradient
       -> Tensor -- updated parameters
gdStep lr params gradient = T.csub params lr gradient


-- TODO: complete batch gradient descent 
gdOptimize :: (Tensor -> Tensor -> Tensor -> Double)  -- loss function
           -> (Tensor -> Tensor -> Tensor -> Tensor)   -- loss function gradient
           -> Double     -- learning rate
           -> Double     -- epsilon (amount gradient can be above zero)
           -> Tensor     -- single input to be repeated
           -> Tensor     -- single output to be repeated
           -> Tensor     -- initial parameters
           -> Tensor     -- optimized parameters
gdOptimize lossFunc lossGrad lr eps inputs outputs params =
  if (abs maxGradient) > eps
  then
    gdOptimize lossFunc lossGrad lr eps inputs outputs updatedParams
  else
    params
  where
    gradient = lossGrad inputs outputs params
    maxGradient = T.maxall gradient
    updatedParams = gdStep lr params gradient


-- stochastic gradient descent
sgdOptimize :: (Tensor -> Tensor -> Tensor -> Double)  -- loss function
            -> (Tensor -> Tensor -> Tensor -> Tensor)   -- loss function gradient
            -> Double     -- learning rate
            -> Double     -- epsilon (amount gradient can be above zero)
            -> InputTensor     -- input matrix
            -> OutputTensor     -- output matrix
            -> Tensor     -- initial parameters
            -> Tensor     -- optimized parameters
sgdOptimize lossFunc lossGrad lr eps inputs outputs params =
  sgdHelper numInputs lossFunc lossGrad lr eps inputs outputs params 
  where
    numInputs = T.size inputs 1


sgdHelper :: Word  -- iteration number
          -> (Tensor -> Tensor -> Tensor -> Double)  -- loss function
          -> (Tensor -> Tensor -> Tensor -> Tensor)   -- loss function gradient
          -> Double     -- learning rate
          -> Double     -- epsilon (amount gradient can be above zero)
          -> InputTensor     -- input matrix
          -> OutputTensor     -- output matrix
          -> Tensor     -- initial parameters
          -> Tensor     -- optimized parameters
sgdHelper 0 _ _ _ _ _ _ params = params
sgdHelper numIter lossFunc lossGrad lr eps inputs outputs params =
  sgdHelper (numIter - 1) lossFunc lossGrad lr eps inputs outputs updatedParams
  where
    inputVector = getRow' inputs numIter
    outputVector = getRow' outputs numIter
    gradient = lossGrad inputVector outputVector params
    maxGradient = T.maxall gradient
    updatedParams = gdStep lr params gradient


{-
-- Takes the mean of the given value
meanSquareError :: Reifies s W
                => BAccReal s
                -> BAccReal s
                -> BAccReal s
meanSquareError n inp act params = val / n


step :: T.HsReal -> Toy -> Toy -> Toy
step eta t grad = T.csub t eta grad


sgdIter :: T.HsReal  -- learning rate
        -> Int       -- number of iterations
        -> Toy       -- input vector-- TODO: make stochastic, give it 2-dim tensor
        -> Toy       -- output vector -- TODO: same as above
        -> Toy       -- parameters to be iterated on
        -> Toy       -- final parameters
sgdIter eta 0 input actual params = params
sgdIter eta steps input actual params =
  sgdIter eta (steps - 1) input actual updated
  where
    actualVar = auto actual
    inputVar = auto input
    gradient = gradBP ((mean 2) . sumallBP . (sqErr inputVar actualVar)) params
    updated = step eta params gradient



sgdLoss :: T.HsReal -> T.HsReal -> Toy -> Toy -> Toy
sgdLoss eps eta actual params =
  if  (abs eval) > (abs eps)
  then
    sgdLoss eps eta actual updated
  else
    params
  where
    actualVar = auto actual
    eval = evalBP ((meanSquare 2) . (err actualVar)) params
    grad = gradBP ((meanSquare 2) . (err actualVar)) params
    updated = step eta params grad
-}

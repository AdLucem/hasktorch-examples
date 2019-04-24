{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}

module GRU where

import Torch.Double
import GHC.Generics
import Numeric.Dimensions


-- | defining a type to hold a set of GRU weights and biases
-- |  for all the gates
data Weights size
    = Weights
                {   resetWeights  :: Tensor size,
                    resetBias     :: Tensor size,
                    updateWeights :: Tensor size,
                    updateBias    :: Tensor size,
                    weights       :: Tensor size,
                    bias          :: Tensor size
                } deriving (Eq, Generic)


-- show function for the Weights type
instance Show (Weights size) where
    show (Weights rW rB uW uB w b) =
        "Reset Gate:\n" ++
        "\t Weights:" ++ show rW ++ "\n" ++
        "\t Bias:" ++ show rB ++ "\n" ++
        "Update Gate:\n" ++
        "\t Weights:" ++ show uW ++ "\n" ++
        "\t Bias:" ++ show uB ++ "\n" ++
        "Input Gate:\n" ++
        "\t Weights:" ++ show w ++ "\n" ++
        "\t Bias:" ++ show b ++ "\n"


-- | Defining a GRU cell
-- TODO: it's a fully connected RNN as of now
-- that is a problem yes
data GRUCell inputSize hiddenSize
    = GRUCell
            {   inputHidden  :: Weights inputSize,
                hiddenHidden :: Weights hiddenSize
            } deriving (Eq, Generic)


-- | show instance from GRUCell
instance Show (GRUCell a b) where
    show (GRUCell iW hW) =
        "Input-Hidden:\n" ++ show iW ++ "Hidden-Hidden:\n" ++ show hW


-- | Composable GRU RNN
data GRNN a b = Single (GRUCell a b) | Composed (GRUCell a b) (GRNN a b)

-- | show instance for GRNN
-- TODO: standardize it acc. to the rest of the hasktorch library
instance Show (GRNN a b) where
    show (Single cell) = show cell
    show (Composed cell rnn) =
        show cell ++
        "\n----------------------------------------\n"
        ++ show rnn


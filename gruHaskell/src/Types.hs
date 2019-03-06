{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE CPP #-}

module Types where

import Torch.Double

data GRUCell = GRUCell { wReset  :: Tensor '[1, 32],
                         uReset  :: Tensor '[1, 32],
                         wUpdate :: Tensor '[1, 32],
                         uUpdate :: Tensor '[1, 32],
                         w       :: Tensor '[1, 32],
                         u       :: Tensor '[1, 32]
                       } deriving (Show, Read) 

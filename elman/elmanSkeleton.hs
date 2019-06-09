-- Skeletal code for Elman Cell

data ElmanSpec = 
	ElmanSpec { in_features  :: Int,
	            hidden_size  :: Int,
	            nonlinearitySpec :: Tensor -> Tensor
	          } deriving (Show)

data ElmanCell = ElmanCell { weight_ih :: Parameter,
                             bias_ih :: Parameter,
                             weight_hh :: Parameter,
                             bias_hh :: Parameter 
                           } deriving (Show)

data RunElman = [ElmanCell]

instance Randomizable ElmanSpec Elman

instance Parametrized Elman

run :: Elman -> Tensor -> Tensor  

runOverTimesteps :: Int   -- number of timesteps 
                 -> Elman  -- elman cell 
                 -> Tensor  -- input
                 -> Tensor  -- output (final hidden state)

data ElmanStates = [ElmanCell]

runOverTimesteps' :: Int  -- number of timesteps
                  -> Elman  -- elman cell
                  -> Tensor  -- input
                  -> ElmanStates  -- elman cell state at each timestep

-- if the libtorch grad functions don't work
singleLayerGradient :: ElmanCell    -- layer
                    -> OutputGradient  -- gradient of output to next layer
                    -> [Tensor]     -- gradient   

networkGradient :: ElmanStates
                -> [[Tensor]]


-- anyway, backprop-through-time
-- so the bit where I go from 'list of ElmanCells' to 'backprop-able list of parameters' is handwavy,
-- but it depends on whether I use `flattenParameters` and `grad` or handwritten gradient functions
-- (which, in turn, depends on if `grad`- the Torch function- works with a time-series of tensors)
-- so I'm leaving it like this for now
bptt :: ElmanStates  -- elman cell state at each timestep 
     -> [Tensor]     -- gradients at each timestep

update :: ElmanCell   -- elman Cell
       -> Tensor      -- final gradient at timestep '1'
       -> ElmanCell   -- updated cell
